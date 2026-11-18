"""
xgboost_model.py

One-step-ahead XGBoost forecasts for a chosen dataset, with a randomised
hyperparameter search under time-series cross-validation (Section 3.8.3).

Requires: Python 3.10, pandas, numpy, xgboost, scikit-learn, scipy,
          matplotlib, data_loader, evaluation.
Reads:  the chosen dataset via data_loader.
Writes: results/<dataset>/xgboost_forecast.csv, xgboost_results.txt and plots.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import (
    load_data, get_splits, choose_dataset, get_results_dir, TARGET_COL,
)
from evaluation import compute_metrics, format_metrics

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("xgboost not installed. Run: pip install xgboost") from e

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform


# Configuration

TUNING_ITERS = 30
TUNING_CV_SPLITS = 3
EARLY_STOP_ROUNDS = 50
SKIP_TUNING = False

LAG_HOURS = [1, 2, 3, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]
SEASONAL_PERIOD = 24  # daily seasonality for MASE seasonal-naive baseline

ENGINEERED_PREFIXES = ("lag_", "roll_", "hour_sin", "hour_cos",
                       "dow_sin", "dow_cos")


# Feature engineering

def build_features(df, target_col=TARGET_COL):
    """
    Add lag, rolling and cyclical calendar features.

    The target for each row is load[t]. Lag and rolling features use values
    at t-1 or earlier. Shifts are by row, so across a gap in the timestamp
    index a lag refers to the nearest earlier row rather than the exact
    clock hour. Rows with a missing feature or target (the burn-in period
    at the start of the series) are dropped.
    """
    out = df.copy()

    for lag in LAG_HOURS:
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    # Rolling statistics over past values only (shift, then roll). Partial
    # windows at the start of the series are left as NaN and dropped.
    shifted = out[target_col].shift(1)
    for w in ROLLING_WINDOWS:
        out[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=w).mean()
        out[f"roll_std_{w}"] = shifted.rolling(w, min_periods=w).std()

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    engineered = [c for c in out.columns if c.startswith(ENGINEERED_PREFIXES)]
    return out.dropna(subset=engineered + [target_col])


def get_feature_columns(df, target_col=TARGET_COL):
    """All columns except the target and the non-numeric provenance column."""
    exclude = {target_col, "source"}
    return [c for c in df.columns if c not in exclude]


# Tuning

def tune_xgboost(X_train, y_train, n_iter=TUNING_ITERS, cv_splits=TUNING_CV_SPLITS):
    print(f"\n-- Tuning XGBoost ({n_iter} iters, {cv_splits}-fold TimeSeriesSplit) --")

    param_dist = {
        "n_estimators":     randint(200, 800),
        "max_depth":        randint(3, 10),
        "learning_rate":    uniform(0.01, 0.19),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "reg_alpha":        uniform(0, 1.0),
        "reg_lambda":       uniform(0, 2.0),
    }

    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=n_iter, cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=1, random_state=42, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  Best CV MAE: {-search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    return search.best_params_


# Train

def train_model(train, val, feature_cols, params, target_col=TARGET_COL):
    """Train a single one-step model with early stopping on the val set."""
    print("\n-- Training one-step XGBoost --")
    Xt, yt = train[feature_cols], train[target_col]
    Xv, yv = val[feature_cols], val[target_col]

    model = xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=42, n_jobs=-1,
        early_stopping_rounds=EARLY_STOP_ROUNDS, **params,
    )
    model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)

    val_pred = model.predict(Xv)
    val_mae = float(np.mean(np.abs(val_pred - yv.values)))
    print(f"  best_iteration = {model.best_iteration}")
    print(f"  validation MAE = {val_mae:.4f}")
    return model


# One-step-ahead forecast

def one_step_forecast(model, test, feature_cols):
    """Predict load[t] for every test row from that row's features."""
    print("\n-- One-step-ahead forecast --")
    preds = model.predict(test[feature_cols])
    return np.asarray(preds, dtype=np.float64)


# Main

def main():
    print("=" * 65)
    print("XGBoost one-step-ahead pipeline")
    print("=" * 65)

    dataset = choose_dataset()
    out_dir = get_results_dir(dataset)

    df = load_data(dataset, include_weather=True)

    print("\n-- Engineering features --")
    df_feat = build_features(df)
    feature_cols = get_feature_columns(df_feat)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Rows after feature engineering: {len(df_feat):,} "
          f"(dropped {len(df) - len(df_feat)} for lag/rolling/target NaN)")

    train, val, test = get_splits(df_feat)

    if SKIP_TUNING:
        print("\n-- Skipping tuning, using sensible defaults --")
        best_params = {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1.0,
        }
    else:
        best_params = tune_xgboost(train[feature_cols], train[TARGET_COL])

    model = train_model(train, val, feature_cols, best_params)

    pred = one_step_forecast(model, test, feature_cols)
    actual = test[TARGET_COL].values

    # MASE is scaled on the training split only (Hyndman & Koehler 2006).
    train_for_mase = train[TARGET_COL].values

    metrics = compute_metrics(
        actual, pred,
        train_series=train_for_mase, seasonal_period=SEASONAL_PERIOD,
    )

    print("\n" + "=" * 65)
    print(f"Test-set metrics ({dataset}) - one-step-ahead")
    print("=" * 65)
    print(f"  {format_metrics(metrics, dataset_hint=dataset)}")
    print(f"  n_used = {metrics['n_used']:,}")

    # Text report
    results_file = out_dir / "xgboost_results.txt"
    with open(results_file, "w") as f:
        f.write(f"XGBoost results - {dataset}\n")
        f.write("=" * 65 + "\n\n")
        f.write("Strategy: one-step-ahead only.\n")
        f.write("Target = load[t]; features use values at t-1 or earlier.\n")
        f.write("Prediction uses only information observed before t,\n")
        f.write("matching the one-step convention of the other models.\n\n")
        f.write(f"Train rows: {len(train):,} | {train.index.min()} to {train.index.max()}\n")
        f.write(f"Val rows  : {len(val):,} | {val.index.min()} to {val.index.max()}\n")
        f.write(f"Test rows : {len(test):,} | {test.index.min()} to {test.index.max()}\n\n")
        f.write(f"Tuning: RandomizedSearchCV, {TUNING_ITERS} iters, "
                f"{TUNING_CV_SPLITS}-fold TimeSeriesSplit, scored on MAE\n")
        f.write(f"Best params: {best_params}\n\n")
        f.write(f"MASE scaling: training series only, seasonal_period={SEASONAL_PERIOD}\n\n")
        f.write(f"Features ({len(feature_cols)}):\n")
        for c in feature_cols:
            f.write(f"  - {c}\n")
        f.write("\n" + "=" * 65 + "\n")
        f.write("Metrics\n")
        f.write("=" * 65 + "\n")
        f.write(f"  {format_metrics(metrics, dataset_hint=dataset)}\n")
        f.write(f"  n_used = {metrics['n_used']:,}\n")
    print(f"\nSaved: {results_file}")

    # Forecast CSV
    forecast_file = out_dir / "xgboost_forecast.csv"
    pd.DataFrame({
        "datetime": test.index,
        "actual": actual,
        "predicted": pred,
    }).to_csv(forecast_file, index=False)
    print(f"Saved: {forecast_file}")

    # Forecast plot (first 7 days)
    fig, ax = plt.subplots(figsize=(14, 5))
    n_plot = min(24 * 7, len(test))
    idx = test.index[:n_plot]
    ax.plot(idx, actual[:n_plot], label="Actual", color="black", lw=1.4)
    ax.plot(idx, pred[:n_plot],
            label=f"One-step (MAE {metrics['MAE']:.2f})",
            color="#1f77b4", lw=1, alpha=0.9)
    ax.set_ylabel("Load (MW)")
    ax.set_title(f"{dataset.upper()} - XGBoost one-step-ahead "
                 f"(first {n_plot // 24} days of test)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_file = out_dir / "xgboost_forecast.png"
    plt.savefig(plot_file, dpi=110, bbox_inches="tight")
    print(f"Saved: {plot_file}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_cols) * 0.3)))
    importance.plot.barh(ax=ax, color="#2ca02c")
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title(f"{dataset.upper()} - XGBoost feature importance")
    plt.tight_layout()
    imp_file = out_dir / "xgboost_feature_importance.png"
    plt.savefig(imp_file, dpi=110, bbox_inches="tight")
    print(f"Saved: {imp_file}")


if __name__ == "__main__":
    main()