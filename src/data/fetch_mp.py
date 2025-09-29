# ──────────────────────────────────────────────────────────────────────────────
# Purpose: Fetch stability labels (formation energy, energy above hull) for
#          Ni–Cu bulk alloys from the Materials Project.
# CLI:
#   python -m src.data.fetch_mp --elements Ni,Cu --outfile data/raw_mp_thermo.parquet
# Notes:
#   • Requires .env file with MP_API_KEY (use python-dotenv)
#   • Uses mp-api official client (pip install mp-api pymatgen pyarrow typer python-dotenv)
#   • Outputs a tidy Parquet with a stable schema documented below.
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd
import typer
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core.composition import Composition

app = typer.Typer(help="Fetch stability data from Materials Project (MP) for binary alloys.")

load_dotenv() # Load keys from .env file

@dataclass
class MPThermoRow:
    material_id: str
    formula_pretty: str
    chemsys: str
    elements: List[str]
    nelements: int
    energy_above_hull_eV_per_atom: float
    formation_energy_eV_per_atom: Optional[float]
    efermi: Optional[float]
    # Derived keys for joins
    reduced_formula: str
    x_Ni: Optional[float]
    x_Cu: Optional[float]


def _fraction_from_formula(formula: str, el: str) -> Optional[float]:
    try:
        comp = Composition(formula).fractional_composition
        return float(comp.get_el_amt_dict().get(el, 0.0))
    except Exception:
        return None


@app.command()
def main(
    elements: str = typer.Option("Ni,Cu", "--elements", "-e", help="Comma-separated list of exactly two elements, e.g. 'Ni,Cu' or 'Ni, Cu'"),
    outfile: str = typer.Option("data/raw_mp_thermo.parquet", help="Output Parquet path."),
    eah_max: float = typer.Option(1.0, help="Max energy above hull (eV/atom) filter for retrieval."),
    verbose: bool = typer.Option(True, help="Print progress."),
):
    """Fetch MP stability data for the given binary alloy system."""
    el_list = [e.strip() for e in elements.split(",") if e.strip()]
    if len(el_list) != 2:
        typer.secho("This script is currently tailored for binaries (exactly two elements).", fg=typer.colors.RED)
        raise typer.Exit(2)

    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        typer.secho("MP_API_KEY not set in environment (.env).", fg=typer.colors.RED)
        raise typer.Exit(2)

    elems_sorted = sorted(el_list)
    chemsys = "-".join(elems_sorted)

    rows: list[MPThermoRow] = []
    with MPRester(api_key, use_document_model=False, monty_decode=False) as mpr:
        # mp-api Summary endpoint includes thermo info including energy_above_hull, formation_energy_per_atom
        # Filter by chemical system and an upper bound on E_above_hull for practicality
        if verbose:
            typer.echo(f"Querying MP summary for chemsys={chemsys}, E_hull<={eah_max} eV/atom…")

        docs = mpr.materials.summary.search(
            chemsys=chemsys,
            energy_above_hull=(0, eah_max),
            fields=[
                "material_id", "formula_pretty", "nelements", "elements",
                "energy_above_hull", "formation_energy_per_atom", "efermi"
            ]
        )

    for d in docs:
        # --- safe gets for dict-style docs ---
        formula = d.get("formula_pretty")
        if not formula:
            # skip rows without a valid formula
            continue

        # --- composition-derived fields ---
        comp = Composition(formula)
        reduced_formula = comp.reduced_formula  # e.g., "NiCu"
        frac_dict = comp.fractional_composition.get_el_amt_dict()
        x_Ni = float(frac_dict.get("Ni", 0.0))
        x_Cu = float(frac_dict.get("Cu", 0.0))

        # --- elements: ensure Arrow-friendly (list[str]) ---
        raw_elems = d.get("elements") or []
        elements_list = [str(el) for el in raw_elems]

        # --- numeric fields with defensive casts ---
        e_hull = d.get("energy_above_hull")
        fe_per_atom = d.get("formation_energy_per_atom")
        efermi = d.get("efermi")

        rows.append(MPThermoRow(
            material_id=str(d.get("material_id")),
            formula_pretty=d.get("formula_pretty") or d.get("formula_prety", formula),
            chemsys=chemsys,
            elements=elements_list,
            nelements=int(d.get("nelements") or len(elements_list)),
            energy_above_hull_eV_per_atom=float(e_hull) if e_hull is not None else None,
            formation_energy_eV_per_atom=float(fe_per_atom) if fe_per_atom is not None else None,
            efermi=float(efermi) if efermi is not None else None,
            reduced_formula=reduced_formula,
            x_Ni=x_Ni,
            x_Cu=x_Cu,
        ))


    df = pd.DataFrame([asdict(r) for r in rows])
    df.insert(0, "source", "materials_project")

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_parquet(outfile, index=False)

    if verbose:
        typer.echo(f"Wrote {len(df)} rows → {outfile}")
        # Show a compact schema preview
        dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
        typer.echo("Schema:" + "\n" + json.dumps(dtypes, indent=2))


if __name__ == "__main__":
    app()