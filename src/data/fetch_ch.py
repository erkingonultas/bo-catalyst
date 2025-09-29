# ──────────────────────────────────────────────────────────────────────────────
# Purpose: Fetch activity labels (ΔG_H* adsorption energies) from Catalysis-Hub
#          for Ni–Cu (111) with adsorbed H.
# CLI:
#   python -m src.data.fetch_ch --facet 111 --adsorbate H --elements Ni Cu \
#       --outfile data/raw_catalysis_hub.parquet
# Notes:
#   • Requires .env file for CATHUB_GRAPHQL_ENDPOINT (falls back to default)
#   • Uses Catalysis-Hub public GraphQL endpoint
#   • Paginates results; keeps a tidy schema; robust to partial fields
#   • pip install httpx pandas typer pyarrow tenacity python-dotenv
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import httpx
import pandas as pd
import typer
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pymatgen.core.composition import Composition

load_dotenv()

CH_ENDPOINT = os.getenv("CATHUB_GRAPHQL_ENDPOINT", "https://api.catalysis-hub.org/graphql")

ch_app = typer.Typer(help="Fetch adsorption energies from Catalysis-Hub via GraphQL.")

@dataclass
class CHAdsorptionRow:
    record_id: str
    reaction_energy_eV: Optional[float]
    # Context / metadata we expect to be commonly present
    surface_composition: Optional[str]
    bulk_composition: Optional[str]
    facet: Optional[str]
    site: Optional[str]
    adsorbates: Optional[str]
    publication_year: Optional[int]
    # Derived
    reduced_formula: Optional[str]
    x_Ni: Optional[float]
    x_Cu: Optional[float]


GRAPHQL_QUERY = """
query Adsorption($limit: Int!, $offset: Int!, $facet: String, $ads: String, $elements: [String!]) {
  reactions(
    first: $limit,
    offset: $offset,
    filter: {
      facet: $facet,
      adsorbates: $ads,
      elements: $elements
    }
  ) {
    totalCount
    edges {
      node {
        id
        reactionEnergy
        facet
        sites
        adsorbates
        surfaceComposition
        bulkComposition
        publicationYear
      }
    }
  }
}
"""


class GQLRequestError(RuntimeError):
    pass


@retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)))
async def _gql(client: httpx.AsyncClient, variables: Dict[str, Any]) -> Dict[str, Any]:
    resp = await client.post(CH_ENDPOINT, json={"query": GRAPHQL_QUERY, "variables": variables}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise GQLRequestError(str(data["errors"]))
    return data


@ch_app.command()
def main(
    elements: List[str] = typer.Option(["Ni", "Cu"], help="Elements that must appear in the system."),
    facet: str = typer.Option("111", help="Surface facet to filter."),
    adsorbate: str = typer.Option("H", help="Adsorbate identifier (e.g., H, O, OH)."),
    page_size: int = typer.Option(200, help="GraphQL page size."),
    max_pages: int = typer.Option(200, help="Safety cap on pages to fetch."),
    outfile: str = typer.Option("data/raw_catalysis_hub.parquet", help="Output Parquet path."),
    verbose: bool = typer.Option(True, help="Print progress."),
):
    """Fetch adsorption/reaction energies for given filters from Catalysis-Hub."""
    import anyio

    async def run() -> pd.DataFrame:
        async with httpx.AsyncClient() as client:
            # First call to get totalCount
            vars0 = {"limit": page_size, "offset": 0, "facet": facet, "ads": adsorbate, "elements": elements}
            data0 = await _gql(client, vars0)
            total = data0["data"]["reactions"]["totalCount"]
            if verbose:
                typer.echo(f"Matched {total} reactions in Catalysis-Hub.")
            pages = min(max_pages, math.ceil(total / page_size)) if total else 0

            all_edges = data0["data"]["reactions"]["edges"]
            # Fetch remaining pages concurrently
            async def fetch_page(p: int):
                if p == 0:
                    return []  # already have it
                vars_p = {"limit": page_size, "offset": p * page_size, "facet": facet, "ads": adsorbate, "elements": elements}
                data_p = await _gql(client, vars_p)
                return data_p["data"]["reactions"]["edges"]

            results = await anyio.gather(*[fetch_page(p) for p in range(1, pages)])
            for chunk in results:
                all_edges.extend(chunk)

        # Normalize into rows
        rows: list[CHAdsorptionRow] = []
        for edge in all_edges:
            n = edge.get("node", {})
            surf = n.get("surfaceComposition") or n.get("bulkComposition")
            reduced = None
            xNi = xCu = None
            if surf:
                try:
                    reduced = Composition(surf).reduced_formula
                    comp = Composition(surf).fractional_composition.get_el_amt_dict()
                    xNi = float(comp.get("Ni", 0.0))
                    xCu = float(comp.get("Cu", 0.0))
                except Exception:
                    pass
            rows.append(
                CHAdsorptionRow(
                    record_id=str(n.get("id")),
                    reaction_energy_eV=(float(n["reactionEnergy"]) if n.get("reactionEnergy") is not None else None),
                    surface_composition=surf,
                    bulk_composition=n.get("bulkComposition"),
                    facet=n.get("facet"),
                    site=(n.get("sites") if isinstance(n.get("sites"), str) else None),
                    adsorbates=n.get("adsorbates"),
                    publication_year=(int(n["publicationYear"]) if n.get("publicationYear") is not None else None),
                    reduced_formula=reduced,
                    x_Ni=xNi,
                    x_Cu=xCu,
                )
            )
        df = pd.DataFrame([r.__dict__ for r in rows])
        if not df.empty:
            df.insert(0, "source", "catalysis_hub")
        return df

    df = anyio.run(run)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_parquet(outfile, index=False)
    if verbose:
        typer.echo(f"Wrote {len(df)} rows → {outfile}")


if __name__ == "__main__":
    ch_app()

