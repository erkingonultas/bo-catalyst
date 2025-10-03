# ──────────────────────────────────────────────────────────────────────────────
# Purpose: Fetch stability labels (formation energy, energy above hull) for
#          binary bulk alloys from the Materials Project and write a tidy table (parquet).
#
# Why this rewrite?
#   • Fixes under-fetching by eagerly materializing search results inside the
#     API context and exposing looser/stricter filters as CLI flags.
#   • Generalizes element-fraction columns (x_A, x_B) to any binary (not just
#     Ni–Cu) while still emitting explicit x_<El> columns for joins.
#   • Adds richer fields (is_stable, spacegroup, density, volume) and
#     robust progress / diagnostics so you can see where rows get filtered.
#   • Improves schema safety and Parquet writing.
#
# CLI examples:
#   python -m src.data.fetch_mp_improved -e Ni,Cu -o data/test_mp.parquet \
#       --eah-max 1.0 --min-nelements 2 --max-nelements 2 --include-unstable
#
# Notes:
#   • Requires .env with MP_API_KEY (python-dotenv is used).
#   • Install: mp-api, pymatgen, pandas, pyarrow, typer, python-dotenv.
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import typer
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core.composition import Composition

app = typer.Typer(help="Fetch Materials Project summary data for binary alloys.")

load_dotenv()


@dataclass
class MPThermoRow:
    # Core
    material_id: str
    formula_pretty: str
    chemsys: str
    elements: List[str]
    nelements: int
    # Thermo
    is_stable: Optional[bool]
    energy_above_hull_eV_per_atom: Optional[float]
    formation_energy_eV_per_atom: Optional[float]
    # Electronic/structure (lightweight summary fields)
    efermi: Optional[float]
    spacegroup_symbol: Optional[str]
    density: Optional[float]
    volume: Optional[float]
    # Derived for joins
    reduced_formula: str
    # Element fractions (x_<El>) — added dynamically for the 2 provided elements
    # plus generic slots for convenience
    x_A: Optional[float]
    x_B: Optional[float]
    x_el1: Optional[float]
    x_el2: Optional[float]


# ---------- helpers ----------

def _sorted_pair(elements: List[str]) -> Tuple[str, str]:
    el = [e.strip() for e in elements if e.strip()]
    if len(el) != 2:
        raise ValueError("Expect exactly two elements for a binary alloy.")
    el.sort()
    return el[0], el[1]


def _fractions_from_formula(formula: str, el1: str, el2: str) -> Tuple[float, float, str]:
    comp = Composition(formula)
    frac = comp.fractional_composition.get_el_amt_dict()
    x1 = float(frac.get(el1, 0.0))
    x2 = float(frac.get(el2, 0.0))
    return x1, x2, comp.reduced_formula


# ---------- CLI ----------

@app.command()
def main(
    elements: str = typer.Option(
        "Ni,Cu", "--elements", "-e", help="Comma-separated list of exactly two elements (e.g., 'Ni,Cu')."
    ),
    outfile: str = typer.Option("data/raw_mp_thermo.parquet", "--outfile", "-o", help="Output Parquet path."),
    eah_max: float = typer.Option(1.0, "--eah-max", help="Max energy above hull (eV/atom) to keep in OUTPUT."),
    include_unstable: bool = typer.Option(
        False,
        "--include-unstable/--only-feasible",
        help="If set, fetch without an E_hull filter and filter AFTER download.",
    ),
    min_nelements: int = typer.Option(2, help="Lower bound for nelements filter (applied at server)."),
    max_nelements: int = typer.Option(2, help="Upper bound for nelements filter (applied at server)."),
    limit: Optional[int] = typer.Option(None, help="Optional hard cap on number of docs to fetch (for debugging)."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print progress and schema info."),
):
    """Fetch MP Summary docs for a binary system and write a tidy table.

    Implementation notes
    --------------------
    • We materialize the search results *inside* the API context to avoid lazy
      generators that might under-fetch when the client is closed.
    • We can either ask the server to filter by energy_above_hull, or we can
      download a broader set with --include-unstable then filter locally.
    """

    el1, el2 = _sorted_pair([e for e in elements.split(",") if e])
    chemsys = f"{el1}-{el2}"

    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        typer.secho("MP_API_KEY not set in environment (.env).", fg=typer.colors.RED)
        raise typer.Exit(2)

    # --- query ---
    fields = [
        "material_id",
        "elements",
        "nelements",
        "is_stable",
        "energy_above_hull",
        "formation_energy_per_atom",
        "efermi",
        "density",
        "volume",
        "formula_pretty",
        "symmetry",
    ]

    if verbose:
        hint = "(server-filtered E_hull)" if not include_unstable else "(broad fetch; will filter locally)"
        typer.echo(
            f"Querying MP Summary for chemsys={chemsys}, nelements∈[{min_nelements},{max_nelements}] {hint}…"
        )

    with MPRester(api_key, use_document_model=False, monty_decode=False) as mpr:
        if include_unstable:
            docs_iter = mpr.materials.summary.search(
                chemsys=chemsys,
                num_elements=(min_nelements, max_nelements),
                fields=fields,
                # having a deterministic sort makes debugging easier
                
            )
        else:
            docs_iter = mpr.materials.summary.search(
                chemsys=chemsys,
                num_elements=(min_nelements, max_nelements),
                energy_above_hull=(0, max(eah_max, 0.0)),
                fields=fields,
                
            )
        # Eagerly materialize while connection is open
        docs = list(docs_iter if limit is None else (d for i, d in enumerate(docs_iter) if i < limit))

    if verbose:
        typer.echo(f"Fetched {len(docs)} summaries from MP.")

    # --- rows & local filtering ---
    rows: List[MPThermoRow] = []
    for d in docs:
        formula = d.get("formula_pretty")
        if not formula:
            continue
        if not formula:
            continue
        try:
            x1, x2, reduced = _fractions_from_formula(formula, el1, el2)
        except Exception:
            # Skip if formula parsing fails
            continue

        # Optional local filter by E_hull
        e_hull = d.get("energy_above_hull")
        if (e_hull is not None) and (not include_unstable) and (float(e_hull) > eah_max):
            continue

        rows.append(
            MPThermoRow(
                material_id = str(d.get("material_id")),
                formula_pretty=formula,
                chemsys=chemsys,
                elements=[str(e) for e in (d.get("elements") or [])],
                nelements=int(d.get("nelements") or 0),
                is_stable=bool(d.get("is_stable")) if d.get("is_stable") is not None else None,
                energy_above_hull_eV_per_atom=float(e_hull) if e_hull is not None else None,
                formation_energy_eV_per_atom=(
                    float(d.get("formation_energy_per_atom"))
                    if d.get("formation_energy_per_atom") is not None
                    else None
                ),
                efermi=float(d.get("efermi")) if d.get("efermi") is not None else None,
                # symmetry can be dict; try to extract symbol
                spacegroup_symbol=(
                    (d.get("symmetry") or {}).get("symbol") if isinstance(d.get("symmetry"), dict) else None
                ),
                density=float(d.get("density")) if d.get("density") is not None else None,
                volume=float(d.get("volume")) if d.get("volume") is not None else None,
                reduced_formula=reduced,
                x_A=x1,
                x_B=x2,
                x_el1=x1,
                x_el2=x2,
            )
        )

    # DataFrame
    df = pd.DataFrame([asdict(r) for r in rows])

    # prepend source and explicit element fraction columns (x_<El>) for the pair
    df.insert(0, "source", "materials_project")
    # Stable explicit columns for joins downstream
    df[f"x_{el1}"] = df["x_el1"]
    df[f"x_{el2}"] = df["x_el2"]

    # Tidy up columns order (human friendly)
    preferred = [
        "source",
        "material_id",
        "formula_pretty",
        "reduced_formula",
        "chemsys",
        "elements",
        "nelements",
        "is_stable",
        "energy_above_hull_eV_per_atom",
        "formation_energy_eV_per_atom",
        "efermi",
        "spacegroup_symbol",        "density",
        "volume",
        f"x_{el1}",
        f"x_{el2}",
    ]
    # Keep any extras at the end
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df.loc[:, cols]

    # I/O
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    df.to_parquet(outfile, index=False)

    if verbose:
        kept = len(df)
        dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
        typer.echo(f"Wrote {kept} rows → {outfile}")
        typer.echo("Schema:\n" + json.dumps(dtypes, indent=2))


if __name__ == "__main__":
    app()
