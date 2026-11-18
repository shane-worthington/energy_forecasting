"""
data_loader.py

Loads, cleans and chronologically splits (70/15/15) the two demand datasets
('spain' and 'ee') into one common schema, so every model sees identical data.

Requires: Python 3.10, pandas, numpy.
Provides: load_data, get_splits, get_results_dir, choose_dataset, imported by
          the model and analysis scripts. Reads the CSV files under data/.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration

DATA_DIR = Path("data")  # relative to project root

TARGET_COL = "load_mw"
DATETIME_COL = "datetime"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # the remaining 15% is the test set

DATASETS = {
    "spain": {
        "load_file":   "energy_dataset.csv",
        "weather_file": "weather_features.csv",
        "timezone":    "Europe/Madrid",
        "description": "Kaggle Spain Electricity Load (2015-2018, ~35k hours)",
    },
    "ee": {
        "load_file":   "substation_hourly_merged.csv",
        "weather_file": "substation_weather.csv",
        "timezone":    "Australia/Sydney",
        "description": "Endeavour Energy Western Sydney zone substation (Jan 2017 - Dec 2020, ~35k hours)",
    },
}

# Interactive prompt

def choose_dataset(default="ee"):
    """Interactive prompt. Returns 'spain' or 'ee'."""
    print("\nAvailable datasets:")
    for key, meta in DATASETS.items():
        marker = " (default)" if key == default else ""
        print(f"  {key:6s} - {meta['description']}{marker}")

    try:
        choice = input(f"\nWhich dataset? [{'/'.join(DATASETS.keys())}] (default: {default}): ").strip().lower()
    except EOFError:
        choice = ""

    if not choice:
        choice = default
    if choice not in DATASETS:
        print(f"Invalid choice '{choice}'. Using default: {default}")
        choice = default

    print(f"Selected: {choice}")
    return choice

# Shared helpers

def _add_calendar_features(df):
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df

# Spain loader

def _load_spain(include_weather=True):
    """
    Load Kaggle Spain dataset and normalise to unified schema.

    energy_dataset.csv: 'time' (UTC), 'total load actual' (target).
    weather_features.csv: 5 cities, aggregated to a single series here.
    """
    load_path = DATA_DIR / DATASETS["spain"]["load_file"]
    weather_path = DATA_DIR / DATASETS["spain"]["weather_file"]

    if not load_path.exists():
        raise FileNotFoundError(f"Spain load file not found: {load_path.resolve()}")

    raw = pd.read_csv(load_path, parse_dates=["time"])
    df = pd.DataFrame({
        "datetime": pd.to_datetime(raw["time"], utc=True),
        TARGET_COL: raw["total load actual"],
    })
    # Convert UTC -> Madrid local wall-clock (consistent with EE convention)
    df["datetime"] = df["datetime"].dt.tz_convert(DATASETS["spain"]["timezone"])
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if include_weather:
        if not weather_path.exists():
            raise FileNotFoundError(f"Spain weather file not found: {weather_path.resolve()}")

        w = pd.read_csv(weather_path, parse_dates=["dt_iso"])
        w["dt_iso"] = pd.to_datetime(w["dt_iso"], utc=True).dt.tz_convert(
            DATASETS["spain"]["timezone"]).dt.tz_localize(None)

        # Identify rain column (varies by Kaggle file version)
        rain_col = next((c for c in ["rain_1h", "rain_3h", "precip", "precipitation"]
                         if c in w.columns), None)

        agg_cols = ["temp", "humidity", "wind_speed"]
        if rain_col:
            agg_cols.append(rain_col)

        # Unweighted mean across the five cities
        w_agg = w.groupby("dt_iso")[agg_cols].mean()
        if rain_col:
            w_agg = w_agg.rename(columns={rain_col: "precipitation"})
        else:
            w_agg["precipitation"] = 0.0

        # Kelvin -> Celsius (Spain Kaggle stores temp in K)
        if w_agg["temp"].mean() > 100:
            w_agg["temp"] = w_agg["temp"] - 273.15

        before = len(df)
        df = df.join(w_agg, how="inner")
        if len(df) < before:
            print(f"  Note: Spain load+weather join trimmed {before - len(df)} rows.")

    df = _add_calendar_features(df)
    return df

# EE loader

def _load_ee(include_weather=True):
    """Load the merged substation dataset."""
    load_path = DATA_DIR / DATASETS["ee"]["load_file"]
    weather_path = DATA_DIR / DATASETS["ee"]["weather_file"]

    if not load_path.exists():
        raise FileNotFoundError(f"EE load file not found: {load_path.resolve()}")

    df = pd.read_csv(load_path, parse_dates=[DATETIME_COL]).set_index(DATETIME_COL).sort_index()
    keep = [TARGET_COL, "source"]
    df = df[[c for c in keep if c in df.columns]]

    if include_weather:
        if not weather_path.exists():
            raise FileNotFoundError(f"EE weather file not found: {weather_path.resolve()}")
        w = pd.read_csv(weather_path, parse_dates=[DATETIME_COL]).set_index(DATETIME_COL).sort_index()
        before = len(df)
        df = df.join(w, how="inner")
        if len(df) < before:
            print(f"  Note: EE load+weather join trimmed {before - len(df)} rows.")

    df = _add_calendar_features(df)
    return df

# Unified entry point

def load_data(dataset, include_weather=True, drop_missing=True):
    """Load and normalise the chosen dataset."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(DATASETS)}")

    print(f"\nLoading {dataset} dataset...")
    loader = _load_spain if dataset == "spain" else _load_ee
    df = loader(include_weather=include_weather)

    if drop_missing:
        before = len(df)
        df = df.dropna(subset=[TARGET_COL])
        dropped = before - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} row(s) with missing {TARGET_COL}.")

    print(f"  Loaded: {len(df):,} rows | {df.index.min()} to {df.index.max()}")
    return df

# Split

def get_splits(df, target_only=False, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, verbose=True):
    """Chronological 70/15/15 train/val/test split."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    if verbose:
        print(f"\nSplit summary (n={n:,}):")
        for name, part in [("Train", train), ("Val  ", val), ("Test ", test)]:
            print(f"  {name}: {len(part):>6,} rows | {part.index.min()} to {part.index.max()}")

    if target_only:
        return train[TARGET_COL], val[TARGET_COL], test[TARGET_COL]
    return train, val, test

# Output dir helper

def get_results_dir(dataset):
    """Return Path to results/<dataset>/, creating it if needed."""
    p = Path("results") / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p

# Sanity check when run directly

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else choose_dataset()
    df = load_data(dataset)

    print(f"\nColumns: {list(df.columns)}")
    print(f"\nLoad summary:")
    print(df[TARGET_COL].describe().round(2))

    weather_cols = [c for c in ["temp", "humidity", "wind_speed", "precipitation"] if c in df.columns]
    if weather_cols:
        print(f"\nWeather summary:")
        print(df[weather_cols].describe().round(2))

    train, val, test = get_splits(df)
    print(f"\nResults dir: {get_results_dir(dataset).resolve()}")