"""Paired patient bootstrap for the assay-aware model versus CORAL-PCA-HGB.

The two prediction tables are matched on the fixed external evaluation cases
for each direction and seed.  The bootstrap resamples patients within that
fixed set and reports descriptive 95% intervals for candidate-minus-control
metric differences.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
GPU = pd.read_csv(ROOT / "results/confirmatory_gpu/external_predictions.csv")
CORAL = pd.read_csv(ROOT / "results/coral_cpu/external_predictions.csv")
OUT = ROOT / "results/analysis"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (11, 23, 47)
N_BOOT = 2000


def metrics(y, p):
    return np.array([
        np.sqrt(mean_squared_error(y, p)),
        mean_absolute_error(y, p),
        spearmanr(y, p).statistic,
        r2_score(y, p),
    ], dtype=float)


def main():
    rng = np.random.default_rng(20261005)
    rows = []
    for direction in ("TCGA_to_CPTAC", "CPTAC_to_TCGA"):
        for seed in SEEDS:
            candidate = GPU.query("direction == @direction and seed == @seed")
            control = CORAL.query("direction == @direction and seed == @seed")
            z = candidate[["case_id", "true", "pred"]].rename(columns={"pred": "candidate"}).merge(
                control[["case_id", "pred"]].rename(columns={"pred": "coral"}),
                on="case_id", how="inner", validate="one_to_one")
            y = z.true.to_numpy(float)
            pc = z.candidate.to_numpy(float)
            pb = z.coral.to_numpy(float)
            observed = metrics(y, pc) - metrics(y, pb)
            boot = np.empty((N_BOOT, 4), dtype=float)
            for i in range(N_BOOT):
                ix = rng.integers(0, len(z), len(z))
                boot[i] = metrics(y[ix], pc[ix]) - metrics(y[ix], pb[ix])
            lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
            rows.append({
                "direction": direction, "seed": int(seed), "n": int(len(z)),
                "baseline": "coral_pca_hgb",
                "delta_rmse": float(observed[0]), "delta_mae": float(observed[1]),
                "delta_spearman": float(observed[2]), "delta_r2": float(observed[3]),
                "bootstrap_replicates": N_BOOT,
                "bootstrap_ci95": [[float(x) for x in lo], [float(x) for x in hi]],
            })
    (OUT / "bootstrap_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
