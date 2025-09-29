# ──────────────────────────────────────────────────────────────────────────────
# Purpose: Visualize dataset Pareto structure and suggested compositions.
#
# CLI:
# python -m src.bo.visualize --data data/bo_dataset.parquet --suggestions results/bo/suggestions.parquet --figdir results/figures --target-ads 0.0
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

viz = typer.Typer(help="Plot Pareto structure and BO suggestions.")

def _pareto_mask(costs: np.ndarray) -> np.ndarray:
    """Return boolean mask of non‑dominated points for minimization in all dims.
    costs: shape (N, M)
    """
    n = costs.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # any point strictly better in all objectives?
        dominates = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        # if someone dominates i, mark i as dominated
        if np.any(dominates):
            mask[i] = False
    return mask

@viz.command()
def main(
    data: str = typer.Option("data/bo_dataset.parquet", help="Input BO dataset (from make_bo_dataset)."),
    suggestions: str = typer.Option("results/bo/suggestions.parquet", help="BO suggestions parquet."),
    figdir: str = typer.Option("results/figures", help="Directory to save figures."),
    target_ads: float = typer.Option(0.0, help="Target ΔG_H* for activity visualization."),
):
    os.makedirs(figdir, exist_ok=True)
    df = pd.read_parquet(data)


    # Compute costs for Pareto (minimize |ΔG_H* - target|, minimize E_hull)
    df = df.copy()
    df["cost_activity"] = (df["E_ads_eV"] - target_ads).abs()
    df["cost_stability"] = df["E_hull_eV"]


    costs = df[["cost_activity", "cost_stability"]].to_numpy()
    mask = _pareto_mask(costs)


    # Plot 1: Pareto scatter in cost space (lower‑left is better)
    plt.figure()
    plt.scatter(df["cost_activity"], df["cost_stability"], label="All points")
    plt.scatter(df.loc[mask, "cost_activity"], df.loc[mask, "cost_stability"], marker="x", s=64, label="Pareto set")
    plt.xlabel(f"|ΔG_H* − {target_ads:.2f}| (eV)")
    plt.ylabel("E_hull (eV/atom)")
    plt.title("Dataset Pareto structure (minimize both)")
    plt.legend()
    pareto_path = os.path.join(figdir, "pareto.png")
    plt.tight_layout()
    plt.savefig(pareto_path, dpi=160)
    plt.close()


    # Plot 2: Suggested compositions on [0,1]
    sug = pd.read_parquet(suggestions)
    xs = sug["x_Ni"].to_numpy()
    plt.figure()
    plt.scatter(xs, np.zeros_like(xs))
    plt.yticks([])
    plt.xlim(0, 1)
    plt.xlabel("x_Ni (composition)")
    plt.title("Suggested compositions (qEHVI)")
    sug_path = os.path.join(figdir, "suggestions.png")
    plt.tight_layout()
    plt.savefig(sug_path, dpi=160)
    plt.close()


    # Small manifest JSON for convenience
    manifest = {"pareto": pareto_path, "suggestions": sug_path}
    with open(os.path.join(figdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    viz()