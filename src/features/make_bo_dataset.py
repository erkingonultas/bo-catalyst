# ──────────────────────────────────────────────────────────────────────────────
# Purpose: Transform data/processed.parquet → data/bo_dataset.parquet suitable
#          for multi‑objective BO (activity vs. stability)
# CLI:
#   python -m src.features.make_bo_dataset \
#       --infile data/processed.parquet \
#       --outfile data/bo_dataset.parquet \
#       --target-ads 0.0 \
#       --hull-col energy_above_hull_eV \
#       --ads-col E_ads_eV
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd
import typer

app = typer.Typer(help="Prepare a tidy BO dataset (features + objectives) from processed.parquet.")


@app.command()
def main(
    infile: str = typer.Option("data/processed.parquet", help="Input parquet from join_clean.py"),
    outfile: str = typer.Option("data/bo_dataset.parquet", help="Output parquet path."),
    target_ads: float = typer.Option(0.0, help="Target ΔG_H* (eV) for activity optimum (Sabatier)."),
    hull_col: str = typer.Option("energy_above_hull_eV", help="Column name for stability metric."),
    ads_col: str = typer.Option("E_ads_eV", help="Column name for adsorption energy."),
    min_rows: int = typer.Option(5, help="Require at least this many rows to proceed."),
    verbose: bool = typer.Option(True, help="Print progress."),
):
    df = pd.read_parquet(infile)
    # choose composition feature: x_Ni (since x_Cu = 1 - x_Ni)
    needed = ["x_Ni", ads_col, hull_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {infile}: {missing}")

    # Keep only valid rows
    df = df.dropna(subset=["x_Ni", ads_col, hull_col]).copy()
    df = df[(df["x_Ni"] >= 0.0) & (df["x_Ni"] <= 1.0)]

    # Objectives: maximize y1 = -|ΔG_H* - target| (closer to target is better)
    #             maximize y2 = -E_hull (more stable)
    df["y1_activity"] = -(df[ads_col] - float(target_ads)).abs()
    df["y2_stability"] = -df[hull_col]

    # Keep lightweight schema
    out = df[["x_Ni", ads_col, hull_col, "y1_activity", "y2_stability"]].rename(columns={
        "x_Ni": "x",
        ads_col: "E_ads_eV",
        hull_col: "E_hull_eV",
    })

    if len(out) < min_rows:
        raise SystemExit(f"Only {len(out)} usable rows (<{min_rows}).")

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    out.to_parquet(outfile, index=False)

    if verbose:
        schema = {k: str(v) for k, v in out.dtypes.to_dict().items()}
        print(f"Wrote {len(out)} rows → {outfile}")
        print("Schema:\n" + json.dumps(schema, indent=2))


if __name__ == "__main__":
    app()