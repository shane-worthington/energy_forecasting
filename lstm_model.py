"""
lstm_model.py

One-step-ahead LSTM forecasts for a chosen dataset, with a validation-based
grid search over sequence length, layer sizes and dropout (Section 3.8.4).

Requires: Python 3.10, pandas, numpy, tensorflow, scikit-learn, matplotlib,
          data_loader, evaluation.
Reads:  the chosen dataset via data_loader.
Writes: results/<dataset>/lstm_forecast.csv, lstm_results.txt and plots.
"""

import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data, get_splits, choose_dataset, get_results_dir, TARGET_COL
from evaluation import compute_metrics, format_metrics

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError as e:
    raise ImportError("tensorflow not installed. Run: pip install tensorflow") from e

from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 1e-3
SEASONAL_PERIOD = 24

GRID_SEQ_LENS = [24, 48, 168]
GRID_UNITS = [[32, 16], [64, 32]]
GRID_DROPOUTS = [0.1, 0.2]

INPUT_FEATURES = [
    TARGET_COL,
    "temp", "humidity", "wind_speed", "precipitation",
    "hour", "day_of_week", "month", "is_weekend",
]


def make_grid():
    return [{"seq_len": sl, "units": u, "dropout": dr}
            for sl in GRID_SEQ_LENS for u in GRID_UNITS for dr in GRID_DROPOUTS]


def build_sequences(feature_arr, target_arr, seq_len):
    T = len(feature_arr); n = T - seq_len
    if n <= 0:
        raise ValueError(f"Series too short: need > {seq_len} rows, got {T}")
    F = feature_arr.shape[1]
    X = np.zeros((n, seq_len, F), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        X[i] = feature_arr[i:i + seq_len]
        y[i] = target_arr[i + seq_len]
    return X, y


def build_lstm_model(seq_len, n_features, units, dropout, learning_rate=LEARNING_RATE):
    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(units[0], return_sequences=True, dropout=dropout),
        layers.LSTM(units[1], return_sequences=False, dropout=dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse", metrics=["mae"])
    return model


def train_one(config, train_feat_s, train_tgt_s, val_feat_s, val_tgt_s, n_features):
    sl = config["seq_len"]
    X_train, y_train = build_sequences(train_feat_s, train_tgt_s, sl)
    X_val, y_val = build_sequences(val_feat_s, val_tgt_s, sl)
    np.random.seed(SEED); tf.random.set_seed(SEED)
    model = build_lstm_model(sl, n_features, config["units"], config["dropout"])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_mae", patience=EARLY_STOP_PATIENCE,
                                      restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5,
                                          min_lr=1e-5, verbose=0),
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=callbacks, verbose=0)
    return model, history, min(history.history["val_mae"])


def one_step_forecast(model, train_tail, test, feature_cols, x_scaler, y_scaler, seq_len):
    combined = pd.concat([train_tail.tail(seq_len), test])
    feat = combined[feature_cols].values.astype(np.float32)
    feat_scaled = x_scaler.transform(feat)
    n_windows = len(feat_scaled) - seq_len
    assert n_windows == len(test), f"window count {n_windows} != n_test {len(test)}"
    X = np.zeros((n_windows, seq_len, feat_scaled.shape[1]), dtype=np.float32)
    for i in range(n_windows):
        X[i] = feat_scaled[i:i + seq_len]
    yhat_scaled = model.predict(X, batch_size=256, verbose=0).ravel()
    return y_scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).ravel().astype(np.float64)


def main():
    print("=" * 65)
    print("LSTM pipeline (one-step-ahead, 12-config validation grid)")
    print("=" * 65)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPU available: {[g.name for g in gpus]}" if gpus
          else "  No GPU detected, running on CPU (will be slower)")

    dataset = choose_dataset()
    out_dir = get_results_dir(dataset)

    df = load_data(dataset, include_weather=True)
    missing = [c for c in INPUT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataset {dataset}: {missing}")
    df = df[INPUT_FEATURES].copy()

    train, val, test = get_splits(df)

    x_scaler = StandardScaler().fit(train[INPUT_FEATURES].values)
    y_scaler = StandardScaler().fit(train[[TARGET_COL]].values)
    train_feat_s = x_scaler.transform(train[INPUT_FEATURES].values)
    val_feat_s = x_scaler.transform(val[INPUT_FEATURES].values)
    train_tgt_s = y_scaler.transform(train[[TARGET_COL]].values).ravel()
    val_tgt_s = y_scaler.transform(val[[TARGET_COL]].values).ravel()
    n_features = len(INPUT_FEATURES)
    y_scale = y_scaler.scale_[0]

    grid = make_grid()
    print(f"\n-- Grid search: {len(grid)} configs, selected on validation MAE --")
    print(f"   (test set is NOT used for selection)\n")

    grid_results = []
    best = None

    for k, config in enumerate(grid, 1):
        model, history, best_val_scaled = train_one(
            config, train_feat_s, train_tgt_s, val_feat_s, val_tgt_s, n_features)
        val_mae_mw = best_val_scaled * y_scale
        epochs_run = len(history.history["loss"])
        grid_results.append({"config": config, "val_mae_mw": val_mae_mw, "epochs": epochs_run})
        tag = ""
        if best is None or val_mae_mw < best[0]:
            best = (val_mae_mw, config, model, history)
            tag = "  <- best so far"
        print(f"  [{k:2d}/{len(grid)}] seq={config['seq_len']:>3} "
              f"units={str(config['units']):<9} drop={config['dropout']} "
              f"-> val MAE {val_mae_mw:.4f} MW ({epochs_run} ep){tag}")

    best_val_mae_mw, best_config, best_model, best_history = best
    print(f"\n-- Best config (by validation MAE) --")
    print(f"  seq_len={best_config['seq_len']}, units={best_config['units']}, "
          f"dropout={best_config['dropout']}")
    print(f"  validation MAE = {best_val_mae_mw:.4f} MW")

    train_tail = pd.concat([train, val])
    predictions = one_step_forecast(best_model, train_tail, test, INPUT_FEATURES,
                                    x_scaler, y_scaler, best_config["seq_len"])

    train_for_mase = train[TARGET_COL].values
    print("\n" + "=" * 65)
    print(f"Test-set metrics ({dataset}) - one-step-ahead (best config)")
    print("=" * 65)
    m = compute_metrics(test[TARGET_COL].values, predictions,
                        train_series=train_for_mase, seasonal_period=SEASONAL_PERIOD)
    print(f"\nLSTM:")
    print(f"  {format_metrics(m, dataset_hint=dataset)}")
    print(f"  n_used = {m['n_used']:,}")

    results_file = out_dir / "lstm_results.txt"
    with open(results_file, "w") as f:
        f.write(f"LSTM results - {dataset}\n")
        f.write("=" * 65 + "\n\n")
        f.write("Strategy: one-step-ahead (window of actuals ending at t-1).\n")
        f.write("Hyperparameters selected by grid search on validation MAE;\n")
        f.write("best config evaluated once on test (no test-set leakage).\n\n")
        f.write(f"Grid (seq_len x units x dropout = "
                f"{len(GRID_SEQ_LENS)}x{len(GRID_UNITS)}x{len(GRID_DROPOUTS)} "
                f"= {len(grid)} configs):\n")
        f.write(f"{'seq_len':>8}{'units':>12}{'dropout':>9}{'val_MAE_MW':>13}{'epochs':>8}\n")
        for r in sorted(grid_results, key=lambda x: x["val_mae_mw"]):
            c = r["config"]
            f.write(f"{c['seq_len']:>8}{str(c['units']):>12}{c['dropout']:>9}"
                    f"{r['val_mae_mw']:>13.4f}{r['epochs']:>8}\n")
        f.write(f"\nSelected (lowest val MAE): seq_len={best_config['seq_len']}, "
                f"units={best_config['units']}, dropout={best_config['dropout']}\n")
        f.write(f"Validation MAE: {best_val_mae_mw:.4f} MW\n\n")
        f.write(f"Architecture: 2-layer LSTM ({best_config['units'][0]} -> "
                f"{best_config['units'][1]}) + Dense(32) + Dense(1)\n")
        f.write(f"Batch size: {BATCH_SIZE}, max epochs: {MAX_EPOCHS}, "
                f"patience: {EARLY_STOP_PATIENCE}\n")
        f.write(f"Input features ({len(INPUT_FEATURES)}): {INPUT_FEATURES}\n")
        f.write(f"Train rows: {len(train):,}, Val rows: {len(val):,}, "
                f"Test rows: {len(test):,}\n")
        f.write(f"MASE scaling: training series only, seasonal_period={SEASONAL_PERIOD}\n\n")
        f.write(f"TEST metrics (best config):\n  {format_metrics(m, dataset_hint=dataset)}\n")
        f.write(f"  n_used = {m['n_used']:,}\n")
    print(f"\nSaved: {results_file}")

    forecast_file = out_dir / "lstm_forecast.csv"
    pd.DataFrame({"datetime": test.index, "actual": test[TARGET_COL].values,
                  "lstm_pred": predictions}).to_csv(forecast_file, index=False)
    print(f"Saved: {forecast_file}")

    fig, ax = plt.subplots(figsize=(14, 5))
    n_plot = min(24 * 14, len(test))
    ax.plot(test.index[:n_plot], test[TARGET_COL].values[:n_plot],
            label="Actual", color="black", lw=1.2)
    ax.plot(test.index[:n_plot], predictions[:n_plot],
            label=f"LSTM (seq={best_config['seq_len']})", color="#9467bd", lw=1, alpha=0.85)
    ax.set_ylabel("Load (MW)")
    ax.set_title(f"{dataset.upper()} - LSTM one-step-ahead (first {n_plot//24} days)")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plot_file = out_dir / "lstm_forecast_plot.png"
    plt.savefig(plot_file, dpi=110, bbox_inches="tight")
    print(f"Saved: {plot_file}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_history.history["loss"], label="train loss (MSE, scaled)", lw=1.2)
    ax.plot(best_history.history["val_loss"], label="val loss (MSE, scaled)", lw=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (scaled)")
    ax.set_title(f"{dataset.upper()} - LSTM training curve (best config)")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    curve_file = out_dir / "lstm_training_curve.png"
    plt.savefig(curve_file, dpi=110, bbox_inches="tight")
    print(f"Saved: {curve_file}")


if __name__ == "__main__":
    main()