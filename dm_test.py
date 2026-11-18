"""
dm_test.py

Diebold-Mariano test for the accuracy difference between the top-two models
on each dataset, using absolute-error loss and a Newey-West (HAC) variance
(Sections 3.9 and 4.7).

Requires: Python 3.10, pandas, numpy, scipy.
Reads:  results/<dataset>/unified_comparison.csv.
Writes: prints the test result to the console.
"""
import pandas as pd
import numpy as np
from scipy import stats


def dm_test(actual, pred1, pred2, loss="abs", nw_lags=None):
    a = np.asarray(actual, float)
    e1 = a - np.asarray(pred1, float)
    e2 = a - np.asarray(pred2, float)
    d = (np.abs(e1) - np.abs(e2)) if loss == "abs" else (e1**2 - e2**2)
    n = len(d)
    dbar = d.mean()
    d0 = d - dbar

    # Newey-West HAC long-run variance (Bartlett kernel).
    if nw_lags is None:
        nw_lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))  # Newey-West (1994) rule
    gamma0 = np.sum(d0 * d0) / n
    lrv = gamma0
    for k in range(1, nw_lags + 1):
        gk = np.sum(d0[k:] * d0[:n - k]) / n
        w = 1.0 - k / (nw_lags + 1.0)
        lrv += 2.0 * w * gk

    dm = dbar / np.sqrt(lrv / n)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    acf1 = np.sum(d0[1:] * d0[:-1]) / np.sum(d0 * d0)
    return {"dm": dm, "p": p, "dbar": dbar, "n": n, "acf1": acf1, "nw_lags": nw_lags}


MODELS = {"arima_pred": "ARIMA", "sarima_pred": "SARIMA", "xgb_pred": "XGBoost",
          "lstm_pred": "LSTM", "hybrid_pred": "Hybrid"}

# (dataset folder, label, (model1_col, model2_col)) -- top two by MAE on each set
CASES = [
    ("spain", "Spain (national)", ("xgb_pred", "hybrid_pred")),
    ("ee", "Western Sydney substation", ("hybrid_pred", "sarima_pred")),
]

for ds, name, (m1, m2) in CASES:
    df = pd.read_csv(f"results/{ds}/unified_comparison.csv")
    a = df["actual"].values
    maes = {m: np.mean(np.abs(a - df[m].values)) for m in MODELS}
    rank = sorted(maes, key=maes.get)
    r = dm_test(a, df[m1].values, df[m2].values, loss="abs")
    better = MODELS[m1] if r["dbar"] < 0 else MODELS[m2]

    print(f"=== {name} ===  n = {r['n']:,}")
    print("  MAE rank:", ", ".join(f"{MODELS[m]} {maes[m]:.4f}" for m in rank))
    print(f"  Top-two tested: {MODELS[m1]} vs {MODELS[m2]}  (lower MAE: {better})")
    print(f"  mean |e| difference = {r['dbar']:.5f}")
    print(f"  loss-diff ACF(1) = {r['acf1']:.3f}   Newey-West lags = {r['nw_lags']}")
    print(f"  DM statistic (HAC) = {r['dm']:.3f}   p-value (two-sided) = {r['p']:.4g}")
    verdict = "significant at 5%" if r["p"] < 0.05 else "NOT significant at 5% (indistinguishable)"
    print(f"  => {verdict}\n")