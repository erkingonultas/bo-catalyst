# ──────────────────────────────────────────────────────────────────────────────
# Purpose: Fit a 2‑objective GP (y1_activity, y2_stability) on data/bo_dataset.parquet
# and propose new compositions x∈[0,1] via qEHVI. Save results under results/.
#
# CLI:
# python -m src.bo.optimize --infile data/bo_dataset.parquet --suggestions 5 --outfile results/bo/suggestions.parquet
#
# Deps:
# pip install botorch gpytorch torch pandas pyarrow typer
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
import os
from typing import Tuple

import pandas as pd
import torch
import typer
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch.models.transforms import Standardize, Normalize

bo_app = typer.Typer(help="Run multi‑objective BO on Ni–Cu composition (x in [0,1]).")


def _to_tensor(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(df[["x"]].values, dtype=torch.double)
    Y = torch.tensor(df[["y1_activity", "y2_stability"]].values, dtype=torch.double)
    return X, Y


@bo_app.command()
def main(
    infile: str = typer.Option("data/bo_dataset.parquet", help="Input from make_bo_dataset.py"),
    outfile: str = typer.Option("results/bo/suggestions.parquet", help="Parquet with suggested candidates."),
    suggestions: int = typer.Option(5, min=1, max=20, help="Number of suggestions (q)."),
    seed: int = typer.Option(42, help="Random seed for reproducibility."),
):
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.double)

    df = pd.read_parquet(infile)
    if df.empty:
        raise SystemExit("Empty BO dataset.")

    X, Y = _to_tensor(df)

    # Bounds for x (composition): [0, 1]
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    # Two independent GPs (one per objective)
    m1 = SingleTaskGP(X, Y[..., [0]], input_transform=Normalize(1), outcome_transform=Standardize(1))
    m2 = SingleTaskGP(X, Y[..., [1]], input_transform=Normalize(1), outcome_transform=Standardize(1))
    model = ModelListGP(m1, m2)

    # Use the official GPyTorch SumMarginalLogLikelihood for ModelListGP
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)


    # Reference point must be strictly dominated; use observed mins minus a small margin
    y_min = Y.min(dim=0).values
    ref_point = (y_min - 1e-3).tolist()  # 2‑dim vector

    partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point, dtype=torch.double), Y=Y)

    qehvi = qExpectedHypervolumeImprovement(model=model, ref_point=ref_point, partitioning=partitioning)

    # Optimize acquisition over [0,1]; repeat restarts for robustness
    candidates, _ = optimize_acqf(
        acq_function=qehvi,
        bounds=bounds,
        q=suggestions,
        num_restarts=8,
        raw_samples=64,
        options={"batch_limit": 5, "maxiter": 200},
    )

    X_new = candidates.detach().cpu().numpy().reshape(-1)
    out = pd.DataFrame({"x_Ni": X_new, "x_Cu": 1.0 - X_new})

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    out.to_parquet(outfile, index=False)
    print(f"Wrote {len(out)} suggestions → {outfile}")

if __name__ == "__main__":
    bo_app()
