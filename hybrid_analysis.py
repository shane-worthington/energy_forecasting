"""
hybrid_analysis.py

Sweeps the SARIMA/XGBoost blend weight and reports the error-minimising
weight and the correlation between the two components' errors
(Section 4.6; Figure 4.6).

Requires: Python 3.10, pandas, numpy, matplotlib.
Reads:  results/<dataset>/unified_comparison.csv.
Writes: results/hybrid_analysis.txt and results/fig_4_6_hybrid_weight.png.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
BLUE = "#33507a"
AMBER = "#b5791f"
DPI = 300
RESULTS = Path("results")
OUT_TXT = RESULTS / "hybrid_analysis.txt"
OUT_FIG = RESULTS / "fig_4_6_hybrid_weight.png"
DATASETS = [("spain", "National (Spain)", "Test MAE (MW)"),
            ("ee", "Western Sydney substation", "Test MAE (MW)")]
N_GRID = 201  # weight grid resolution


def mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))


def analyse(ds):
    df = pd.read_csv(RESULTS / ds / "unified_comparison.csv")
    a = df["actual"].values
    s, x, h = df["sarima_pred"].values, df["xgb_pred"].values, df["hybrid_pred"].values

    recon_max_diff = float(np.max(np.abs(0.5 * s + 0.5 * x - h)))  # check hybrid equals the 50/50 blend
    es, ex = a - s, a - x
    corr = float(np.corrcoef(es, ex)[0, 1])

    weights = np.linspace(0.0, 1.0, N_GRID)          # weight on SARIMA
    maes = np.array([mae(a, w * s + (1 - w) * x) for w in weights])
    i_opt = int(np.argmin(maes))
    w_opt, mae_opt = float(weights[i_opt]), float(maes[i_opt])

    return {
        "ds": ds, "n": len(df), "recon_max_diff": recon_max_diff, "corr": corr,
        "mae_sarima": mae(a, s), "mae_xgb": mae(a, x), "mae_hybrid": mae(a, h),
        "weights": weights, "maes": maes, "w_opt": w_opt, "mae_opt": mae_opt,
    }


results = [analyse(ds) for ds, _, _ in DATASETS]

# Results file
with open(OUT_TXT, "w") as f:
    f.write("Hybrid ensemble analysis (Section 4.6)\n")
    f.write("=" * 60 + "\n\n")
    f.write("Hybrid = equal-weight (0.50 SARIMA + 0.50 XGBoost) one-step blend.\n")
    f.write("Weight below = weight on SARIMA; (1 - weight) on XGBoost.\n\n")
    for r in results:
        f.write(f"--- {r['ds']} (n = {r['n']:,}) ---\n")
        f.write(f"  hybrid == 0.5*SARIMA + 0.5*XGB : max|diff| = {r['recon_max_diff']:.3e}\n")
        f.write(f"  MAE  SARIMA   = {r['mae_sarima']:.4f}\n")
        f.write(f"  MAE  XGBoost  = {r['mae_xgb']:.4f}\n")
        f.write(f"  MAE  Hybrid   = {r['mae_hybrid']:.4f}  (equal weight)\n")
        f.write(f"  one-step error correlation corr(e_SARIMA, e_XGB) = {r['corr']:.3f}\n")
        f.write(f"  error-minimising SARIMA weight = {r['w_opt']:.2f}  ->  MAE {r['mae_opt']:.4f}\n")
        gain_vs_half = 100 * (r["mae_hybrid"] - r["mae_opt"]) / r["mae_hybrid"]
        gain_vs_xgb = 100 * (r["mae_xgb"] - r["mae_opt"]) / r["mae_xgb"]
        f.write(f"  optimum improves on equal-weight by {gain_vs_half:.1f}%  "
                f"and on XGBoost-only by {gain_vs_xgb:.1f}%\n\n")
print(f"Saved numbers: {OUT_TXT}")

# Figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, r, (ds, title, ylabel) in zip(axes, results, DATASETS):
    ax.plot(r["weights"], r["maes"], color=BLUE, lw=2, zorder=2)
    # equal-weight (the fixed hybrid) and error-minimising weight
    ax.scatter([0.5], [r["mae_hybrid"]], color=AMBER, s=55, zorder=3,
               label=f"Equal weight (0.50), MAE {r['mae_hybrid']:.3g}")
    ax.axvline(r["w_opt"], color=BLUE, ls="--", lw=1.2, alpha=0.7, zorder=1,
               label=f"Error-minimising weight ({r['w_opt']:.2f})")
    ax.set_title(title)
    ax.set_xlabel("Weight on SARIMA  (1 - weight on XGBoost)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", frameon=False)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=DPI, bbox_inches="tight")
print(f"Saved figure:  {OUT_FIG}")