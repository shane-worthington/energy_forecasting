"""
dataset_summary.py

Descriptive statistics for each dataset's load series (Tables 3.1 and 3.2).

Requires: Python 3.10, pandas, numpy, data_loader.
Reads:  the load series via data_loader.
Writes: results/<dataset>/dataset_summary.txt and results/dataset_summary.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_data, TARGET_COL

# Data source label per dataset
SOURCE = {
    "spain": "Kaggle (Spain Electricity Load & Weather)",
    "ee": "Endeavour Energy (Western Sydney Zone Substation)",
}

# Row order for the output table
ROW_ORDER = [
    "Source",
    "Period covered",
    "Temporal resolution",
    "Number of observations",
    "Minimum (MW)",
    "Maximum (MW)",
    "Mean (MW)",
    "Standard deviation (MW)",
    "Missing values",
    "Missing timestamps (gaps)",
]


def infer_resolution(idx):
    """Most common spacing between consecutive timestamps."""
    diffs = pd.Series(idx.sort_values()).diff().dropna()
    if diffs.empty:
        return "n/a"
    mode = diffs.mode().iloc[0]
    hours = mode.total_seconds() / 3600.0
    if abs(hours - 1.0) < 1e-9:
        return "Hourly (1 h)"
    if hours >= 1:
        return f"{hours:g} h"
    return f"{mode.total_seconds() / 60:g} min"


def summarise(dataset):
    """Return an ordered dict of statistics for one dataset."""
    df = load_data(dataset, include_weather=False, drop_missing=False)
    s = df[TARGET_COL]
    idx = df.index

    n_obs = len(s)
    missing = int(s.isna().sum())

    # Gap diagnostic: how many hourly timestamps are absent over the period.
    full = pd.date_range(idx.min(), idx.max(), freq="h")
    gaps = len(full) - len(idx.unique())

    return {
        "Source": SOURCE.get(dataset, dataset),
        "Period covered": f"{idx.min():%Y-%m-%d %H:%M} to {idx.max():%Y-%m-%d %H:%M}",
        "Temporal resolution": infer_resolution(idx),
        "Number of observations": f"{n_obs:,}",
        "Minimum (MW)": f"{s.min():,.2f}",
        "Maximum (MW)": f"{s.max():,.2f}",
        "Mean (MW)": f"{s.mean():,.2f}",
        "Standard deviation (MW)": f"{s.std():,.2f}",  # sample std, ddof=1
        "Missing values": f"{missing:,}",
        "Missing timestamps (gaps)": f"{gaps:,}",
    }


def render_table(title, stats):
    width = max(len(k) for k in ROW_ORDER)
    lines = [title, "=" * len(title), ""]
    lines.append(f"{'Statistic':<{width}}  Value")
    lines.append(f"{'-' * width}  {'-' * 40}")
    for k in ROW_ORDER:
        lines.append(f"{k:<{width}}  {stats.get(k, '')}")
    return "\n".join(lines)


def main():
    datasets = [sys.argv[1].lower()] if len(sys.argv) > 1 else ["spain", "ee"]
    results = {}

    for d in datasets:
        stats = summarise(d)
        results[d] = stats

        table = render_table(f"Load summary: {d}", stats)
        print("\n" + table + "\n")

        out_dir = Path("results") / d
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dataset_summary.txt").write_text(table + "\n")

    # Combined CSV: rows = statistics, columns = datasets
    combined = pd.DataFrame(
        {d: [results[d][k] for k in ROW_ORDER] for d in datasets},
        index=ROW_ORDER,
    )
    combined.index.name = "Statistic"
    Path("results").mkdir(exist_ok=True)
    combined.to_csv("results/dataset_summary.csv")
    print(f"Saved: results/dataset_summary.csv")
    for d in datasets:
        print(f"Saved: results/{d}/dataset_summary.txt")


if __name__ == "__main__":
    main()
