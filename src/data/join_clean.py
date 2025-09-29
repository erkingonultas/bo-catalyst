# ──────────────────────────────────────────────────────────────────────────────
# Purpose: Join Catalysis-Hub activity data with MP stability data; standardize
#          sign/units; compute feasibility flag; emit a tidy table ready for BO.
# CLI:
#   python -m data.join_clean \
#       --ch data/raw_catalysis_hub.parquet \
#       --mp data/raw_mp_thermo.parquet \
#       --outfile data/processed.parquet
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd
import typer

join_app = typer.Typer(help="Join & clean raw Catalysis-Hub and MP tables into one tidy dataset.")


def _normalize_adsorbate_sign(df: pd.DataFrame, energy_col: str) -> pd.DataFrame:
    """Ensure adsorption energies follow the convention: more negative = stronger binding.
    If the source already provides reaction/adsorption energies in eV (final - initial),
    leave as-is. This hook is provided to apply custom flips if needed in the future."""
    # For now, we trust Catalysis-Hub reactionEnergy as adsorption energy in eV.
    return df.rename(columns={energy_col: "E_ads_eV"})


@join_app.command()
def main(
    ch: str = typer.Option("data/raw_catalysis_hub.parquet", help="Input Parquet from fetch_ch.py"),
    mp: str = typer.Option("data/raw_mp_thermo.parquet", help="Input Parquet from fetch_mp.py"),
    outfile: str = typer.Option("data/processed.parquet", help="Output Parquet path."),
    e_hull_feasible: float = typer.Option(0.2, help="Feasibility threshold on energy_above_hull (eV/atom)."),
    verbose: bool = typer.Option(True, help="Print progress."),
):
    df_ch = pd.read_parquet(ch)
    df_mp = pd.read_parquet(mp)

    # Clean Catalysis-Hub table
    df_ch = _normalize_adsorbate_sign(df_ch, energy_col="reaction_energy_eV")
    # Normalize facet text (e.g., "(111)" → "111")
    if "facet" in df_ch.columns:
        df_ch["facet"] = df_ch["facet"].astype(str).str.replace(r"[^0-9]", "", regex=True)

    # Build join keys
    def make_key(df: pd.DataFrame, formula_col: str) -> pd.Series:
        return df[formula_col].fillna("").str.replace(r"\s+", "", regex=True).str.upper()

    ch_key = make_key(df_ch, "reduced_formula") if "reduced_formula" in df_ch else make_key(df_ch, "surface_composition")
    mp_key = make_key(df_mp, "reduced_formula") if "reduced_formula" in df_mp else make_key(df_mp, "formula_pretty")

    df_ch = df_ch.assign(__key=ch_key)
    df_mp = df_mp.assign(__key=mp_key)

    # Left-join CH → MP (some CH alloys may lack MP entries; keep CH rows)
    keep_cols_mp = [
        "material_id",
        "energy_above_hull_eV_per_atom",
        "formation_energy_eV_per_atom",
        "efermi",
        "x_Ni",
        "x_Cu",
        "reduced_formula",
        "formula_pretty",
    ]
    keep_cols_mp = [c for c in keep_cols_mp if c in df_mp.columns]

    merged = (
        df_ch.merge(df_mp[keep_cols_mp + ["__key"]], on="__key", how="left", suffixes=("_ch", "_mp"))
    )

    # Final schema
    out = pd.DataFrame({
        "catalyst_id": merged.get("record_id"),
        "composition": merged.get("reduced_formula_ch", merged.get("reduced_formula")),
        "facet": merged.get("facet"),
        "site": merged.get("site"),
        "adsorbate": merged.get("adsorbates"),
        "E_ads_eV": merged.get("E_ads_eV"),
        "formation_energy_eV_per_atom": merged.get("formation_energy_eV_per_atom"),
        "energy_above_hull_eV": merged.get("energy_above_hull_eV_per_atom"),
        "feasible": (merged.get("energy_above_hull_eV_per_atom") <= e_hull_feasible), # type: ignore
        "x_Ni": merged.get("x_Ni_ch", merged.get("x_Ni")),
        "x_Cu": merged.get("x_Cu_ch", merged.get("x_Cu")),
        "mp_material_id": merged.get("material_id"),
    })

    # Ensure numeric types
    num_cols = ["E_ads_eV", "formation_energy_eV_per_atom", "energy_above_hull_eV", "x_Ni", "x_Cu"]
    for c in num_cols:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop rows missing core labels
    out = out.dropna(subset=["composition", "E_ads_eV"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    out.to_parquet(outfile, index=False)

    if verbose:
        typer.echo(f"Wrote {len(out)} rows → {outfile}")
        typer.echo("Columns: " + ", ".join(out.columns))


if __name__ == "__main__":
    join_app()
