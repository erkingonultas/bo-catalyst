# ──────────────────────────────────────────────────────────────────────────────
# Purpose: Fetch activity labels (reaction/adsorption energies) from Catalysis-Hub
#          for Ni–Cu surfaces, with robust pagination + JSON parsing.
# CLI:
#   python -m src.data.fetch_ch \
#                --facet "" -e Ni,Cu \
#                --no-strict-ads \
#                --require-both \
#                --element-source either \
#                --keep-unknown-comp \
#                --outfile data/ch_all_facets_either_keepUnknown_v2.parquet \
#                --verbose
#   # optional: --reactants "~H" (contains) or "H*" (exact-ish), --chem "NiCu"
# Notes:
#   • Requires .env (python-dotenv). Default endpoint: http://api.catalysis-hub.org/graphql
#   • Pagination uses `first` + `after` (cursor-based).
#   • `reactants`, `products`, `sites` are JSONString in the schema → parse safely.
#   • pip install httpx anyio tenacity pandas pyarrow typer python-dotenv pymatgen
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import anyio
import httpx
import pandas as pd
import typer
from dotenv import load_dotenv
from pymatgen.core.composition import Composition
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


load_dotenv()

# Per docs + tutorials, use HTTP endpoint; allow override via env
CH_ENDPOINT = os.getenv("CATHUB_GRAPHQL_ENDPOINT", "http://api.catalysis-hub.org/graphql")

app = typer.Typer(help="Fetch adsorption/reaction energies from Catalysis-Hub via GraphQL.")

@dataclass
class CHRow:
    record_id: str
    reaction_energy_eV: Optional[float]
    surface_composition: Optional[str]
    chemical_composition: Optional[str]
    facet: Optional[str]
    site: Optional[str]
    reactants: Optional[str]       # JSONString (raw)
    products: Optional[str]        # JSONString (raw)
    pub_id: Optional[str]
    publication_year: Optional[int]
    reduced_formula: Optional[str]
    x_Ni: Optional[float]
    x_Cu: Optional[float]
    adsorbates: Optional[str]      # derived short tag (e.g., "H*") for compatibility


GRAPHQL_QUERY = """
query Reactions(
  $first: Int!,
  $after: String,
  $facet: String,
  $chem: String,
  $reactants: String,
  $order: String
) {
  reactions(
    first: $first,
    after: $after,
    facet: $facet,
    chemicalComposition: $chem,
    reactants: $reactants,
    order: $order
  ) {
    totalCount
    pageInfo { endCursor hasNextPage }
    edges {
      node {
        id
        reactionEnergy
        facet
        sites
        reactants
        products
        chemicalComposition
        surfaceComposition
        publication { year }
      }
    }
  }
}
"""


class GQLError(RuntimeError):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
)
async def _gql(client: httpx.AsyncClient, variables: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    resp = await client.post(
        CH_ENDPOINT,
        json={"query": GRAPHQL_QUERY, "variables": variables},
        headers=headers,
        timeout=60,
    )
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        # Include response text for debug
        raise httpx.HTTPStatusError(
            f"{e} | body={resp.text[:400]}", request=e.request, response=e.response
        )
    data = resp.json()
    if "errors" in data:
        raise GQLError(str(data["errors"]))
    return data

def _parse_elements_csv(elements_csv: str) -> List[str]:
    return [e.strip() for e in (elements_csv or "").split(",") if e.strip()]


def _has_h(js: Optional[str], strict_ads: bool) -> bool:
    """Detect hydrogen presence in reactants/products JSONString.
    Broadened to catch H*, *H, H(ads), Hstar; falls back to substring checks."""
    if not js:
        return False
    def is_h_ads(key: str) -> bool:
        k = key.lower()
        return k.startswith("h") and ("star" in k or "*" in k or "(ads" in k or "adsorbed" in k)
    try:
        d = json.loads(js)  # keys like "Hstar", "H2gas", etc.
        keys = list(d.keys())
        if strict_ads:
            return any(is_h_ads(k) for k in keys)  # H* only
        return any(k.lower().startswith("h") for k in keys)  # any H species
    except Exception:
        s = (js or "").lower()
        if strict_ads:
            return ("h*" in s) or ("*h" in s) or ("hstar" in s) or ("h(ads" in s)
        return "h" in s

# ------------------------ Composition parsing helpers ------------------------
_COMPOSITION_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")

def _regex_parse_formula_to_fracs(s: str) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    """Parse like 'NiCu', 'Ni1Cu3', 'Ni0.25Cu0.75' → (reduced, fractions dict).
    Returns (None, None) on failure."""
    if not s or not isinstance(s, str):
        return None, None
    s_clean = s.strip()
    if not s_clean:
        return None, None
    tokens = _COMPOSITION_TOKEN.findall(s_clean)
    if not tokens:
        return None, None
    counts: Dict[str, float] = {}
    for el, num in tokens:
        if not el:
            continue
        try:
            val = float(num) if num else 1.0
        except Exception:
            return None, None
        counts[el] = counts.get(el, 0.0) + val
    total = sum(counts.values())
    if total <= 0:
        return None, None
    fracs = {el: cnt / total for el, cnt in counts.items()}
    # Simple display reduced formula (alphabetical order). Not a strict chemical reduction.
    def _fmt(cnt: float) -> str:
        if abs(cnt - round(cnt)) < 1e-6:
            return str(int(round(cnt)))
        return f"{cnt:.3g}"
    # scale so smallest nonzero ~1 for nicer integers where possible
    min_nz = min(fracs.values())
    scale = 1.0 / min_nz if min_nz > 0 else 1.0
    scaled = {el: fracs[el] * scale for el in fracs}
    if max(scaled.values()) > 50:  # avoid huge numbers
        reduced = "".join(f"{el}{_fmt(fracs[el])}" for el in sorted(fracs))
    else:
        reduced = "".join(f"{el}{_fmt(scaled[el])}" for el in sorted(scaled))
    return reduced, fracs

def _derive_composition_fields(surf: Optional[str], chem: Optional[str] = None) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Try multiple fields/strategies to recover reduced formula and Ni/Cu fractions.
    Priority: pymatgen on surface → pymatgen on chem → regex on either.
    """
    candidates = [surf, chem]
    # 1) Try pymatgen Composition on candidates
    for cand in candidates:
        if not cand:
            continue
        try:
            comp = Composition(str(cand))
            reduced = comp.reduced_formula
            frac = comp.fractional_composition.get_el_amt_dict()
            return reduced, float(frac.get("Ni")) if "Ni" in frac else None, float(frac.get("Cu")) if "Cu" in frac else None
        except Exception:
            pass
    # 2) Regex fallback on candidates
    for cand in candidates:
        if not cand:
            continue
        reduced, fracs = _regex_parse_formula_to_fracs(str(cand))
        if fracs:
            return reduced, fracs.get("Ni"), fracs.get("Cu")
    return None, None, None

# ------------------------ System-element extraction -------------------------

def _split_elements_from_string(s: str) -> set:
    """Extract element symbols like Ni, Cu, H from strings e.g. 'NiCuH', 'Ni Cu H'."""
    if not s:
        return set()
    return set(re.findall(r"[A-Z][a-z]?", s))


def _extract_system_elements(node: dict) -> set:
    """Collect elements from multiple possible fields on a reaction node."""
    els = set()
    for key in ("reactionSystems", "systems", "elements", "reactionSystem"):
        v = node.get(key)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    els |= _split_elements_from_string(item)
        elif isinstance(v, str):
            els |= _split_elements_from_string(v)
    return els

@app.command()
def main(
    elements: str = typer.Option("Ni,Cu", "--elements", "-e",
                                 help="Comma-separated elements used to focus client-side filtering (e.g., 'Ni,Cu')."),
    facet: str = typer.Option("111", help="Surface facet filter (String)."),
    reactants: str = typer.Option("~H", help="Server-side reactants filter; '~H' = contains H (docs cheat-sheet)."),
    chem: Optional[str] = typer.Option(None, help="Server-side chemicalComposition filter, e.g. 'NiCu' or '~Ni'."),
    order: str = typer.Option("chemicalComposition", help="Order field (String)."),
    page_size: int = typer.Option(200, help="Page size (first)."),
    max_pages: int = typer.Option(200, help="Safety cap on number of pages."),
    outfile: str = typer.Option("data/raw_catalysis_hub.parquet", help="Output Parquet path."),
    strict_ads: bool = typer.Option(True, help="Keep only reactions with adsorbed hydrogen (H*)."),
    require_both: bool = typer.Option(True, help="Require both elements from --elements to be present (>0)."),
    verbose: bool = typer.Option(True, help="Print progress."),
    keep_unknown_comp: bool = typer.Option(
        True, help="Keep rows with unknown composition; mark comp_known=False"
    ),
    element_source: str = typer.Option(
        "composition", help="Where to enforce element presence: composition|system|either"
    ),
):
    """
    Fetch reactions matching facet/chem/reactants via cursor pagination.
    Then filter client-side for Ni&Cu (from --elements) and H/H* if requested.
    """
    el_list = _parse_elements_csv(elements)
    want_Ni = "Ni" in el_list
    want_Cu = "Cu" in el_list

    # If chem arg not provided, derive a broad contains filter to reduce server load.
    chem_arg = chem
    if chem_arg is None and want_Ni:
        chem_arg = "~Ni"  # documented "~" contains operator in CLI/tutorials

    stats = {
        "total_edges": 0,
        "drop_no_H": 0,
        "drop_comp_parse": 0,
        "drop_elements": 0,
        "kept": 0,
        "drop_comp_unknown": 0,
        "kept_unknown_comp": 0,
    }

    async def run() -> pd.DataFrame:
        async with httpx.AsyncClient() as client:
            after: Optional[str] = None
            total: Optional[int] = None
            all_edges: List[Dict[str, Any]] = []

            for _ in range(max_pages):
                variables = {
                    "first": page_size,
                    "after": after,
                    "facet": str(facet) if facet else None,
                    "chem": chem_arg,
                    "reactants": reactants,
                    "order": order,
                }
                data = await _gql(client, variables)
                block = data["data"]["reactions"]
                if total is None:
                    total = block.get("totalCount")
                    if verbose:
                        typer.echo(f"Matched ~{total} reactions (server-side). Scanning pages…")
                edges = block.get("edges") or []
                all_edges.extend(edges)

                page_info = block.get("pageInfo") or {}
                if not page_info.get("hasNextPage"):
                    break
                after = page_info.get("endCursor")

        # Normalize + client-side filtering
        rows: List[CHRow] = []

        for edge in all_edges:
            stats["total_edges"] += 1
            n = edge.get("node", {})

            # H check
            react_s, prod_s = n.get("reactants"), n.get("products")
            if not (_has_h(react_s, strict_ads) or _has_h(prod_s, strict_ads)):
                stats["drop_no_H"] += 1
                continue

            # Choose best available composition strings
            surf_str = n.get("surfaceComposition")
            chem_str = n.get("chemicalComposition")
            reduced, xNi, xCu = _derive_composition_fields(surf_str, chem_str)
            if reduced is None:
                # we attempted parsing and failed (or had no strings)
                stats["drop_comp_parse"] += 1

            # Enforce element presence per flag
            comp_known = (xNi is not None) or (xCu is not None)

            # >>> IMPORTANT: use parsed element list (not raw string)
            want = set(el_list)  # e.g., {"Ni","Cu"}

            present_comp = {e: False for e in want}
            if comp_known:
                if "Ni" in want:
                    present_comp["Ni"] = (xNi or 0.0) > 0.0
                if "Cu" in want:
                    present_comp["Cu"] = (xCu or 0.0) > 0.0

            present_sys = {e: False for e in want}
            sys_els = _extract_system_elements(n)
            for e in want:
                present_sys[e] = e in sys_els

            def ok_with(present: dict) -> bool:
                return all(present[e] for e in want) if require_both else any(present[e] for e in want)

            ok_comp = ok_with(present_comp) if comp_known else False
            ok_sys = ok_with(present_sys)

            use = element_source.lower().strip()
            if use == "composition":
                if not ok_comp:
                    if comp_known:
                        stats["drop_elements"] += 1
                        continue
                    else:
                        if keep_unknown_comp:
                            stats["kept_unknown_comp"] += 1  # keep, tagged as unknown
                        else:
                            stats["drop_comp_unknown"] += 1
                            continue
            elif use == "system":
                if not ok_sys:
                    stats["drop_elements"] += 1
                    continue
            else:  # "either"
                # Keep if EITHER composition or system satisfies. If comp is unknown, rely on system.
                if not (ok_comp or ok_sys):
                    if not comp_known and keep_unknown_comp:
                        stats["kept_unknown_comp"] += 1
                    else:
                        stats["drop_elements"] += 1
                        continue

            pub = n.get("publication") or {}
            pub_year = pub.get("year") if isinstance(pub, dict) else None

            stats["kept"] += 1
            rows.append(
                CHRow(
                    record_id=str(n.get("id")),
                    reaction_energy_eV=(float(n["reactionEnergy"]) if n.get("reactionEnergy") is not None else None),
                    surface_composition=surf_str,
                    chemical_composition=chem_str,
                    facet=n.get("facet"),
                    site=(n.get("sites") if isinstance(n.get("sites"), str) else None),
                    reactants=react_s,
                    products=prod_s,
                    pub_id=n.get("pubId"),
                    publication_year=(int(pub_year) if pub_year is not None else None),
                    reduced_formula=reduced,
                    x_Ni=(float(xNi) if xNi is not None else None),
                    x_Cu=(float(xCu) if xCu is not None else None),
                    adsorbates=("H*" if _has_h(react_s, True) or _has_h(prod_s, True) else None),
                )
            )

        df = pd.DataFrame([r.__dict__ for r in rows])
        if not df.empty:
            df.insert(0, "source", "catalysis_hub")
            # Normalize facet like "111"
            if "facet" in df.columns:
                df["facet"] = df["facet"].astype(str).str.replace(r"[^0-9]", "", regex=True)
            # Ensure numeric types for composition fractions
            if "x_Ni" in df.columns:
                df["x_Ni"] = pd.to_numeric(df["x_Ni"], errors="coerce")
            if "x_Cu" in df.columns:
                df["x_Cu"] = pd.to_numeric(df["x_Cu"], errors="coerce")
        return df

    df = anyio.run(run)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_parquet(outfile, index=False)

    if verbose:
        typer.echo(f"Wrote {len(df)} rows → {outfile}")
        typer.echo(f"Filter stats:\n" + str(stats))
        if not df.empty:
            dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
            preview_cols = ["record_id", "reduced_formula", "facet", "reaction_energy_eV", "x_Ni", "x_Cu", "adsorbates"]
            typer.echo("Schema:\n" + json.dumps(dtypes, indent=2))
            typer.echo("Preview cols: " + ", ".join([c for c in preview_cols if c in df.columns]))


if __name__ == "__main__":
    app()
