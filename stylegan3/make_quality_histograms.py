#!/usr/bin/env python3
"""Build histogram tables from quality_filter_insightface.py output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def categorical_hist(df: pd.DataFrame, column: str) -> list[dict[str, object]]:
    rows = []
    total = len(df)
    counts = df[column].value_counts(dropna=False).sort_index()
    cumulative = 0
    for value, count in counts.items():
        cumulative += int(count)
        rows.append({
            'metric': column,
            'bin_label': str(value),
            'bin_start': value,
            'bin_end': value,
            'count': int(count),
            'ratio': int(count) / total if total else 0.0,
            'cumulative_count': cumulative,
            'cumulative_ratio': cumulative / total if total else 0.0,
        })
    return rows


def numeric_hist(df: pd.DataFrame, column: str, bins: np.ndarray) -> list[dict[str, object]]:
    rows = []
    values = pd.to_numeric(df[column], errors='coerce').fillna(0).to_numpy(dtype=np.float64)
    total = len(values)
    counts, edges = np.histogram(values, bins=bins)
    cumulative = 0
    for idx, count in enumerate(counts):
        start = float(edges[idx])
        end = float(edges[idx + 1])
        cumulative += int(count)
        right_bracket = ']' if idx == len(counts) - 1 else ')'
        rows.append({
            'metric': column,
            'bin_label': f'[{start:.1f},{end:.1f}{right_bracket}',
            'bin_start': start,
            'bin_end': end,
            'count': int(count),
            'ratio': int(count) / total if total else 0.0,
            'cumulative_count': cumulative,
            'cumulative_ratio': cumulative / total if total else 0.0,
        })
    return rows


def detected_numeric_hist(df: pd.DataFrame, column: str, bins: np.ndarray) -> list[dict[str, object]]:
    detected = df[pd.to_numeric(df['face_detected'], errors='coerce').fillna(0).astype(int) == 1].copy()
    rows = numeric_hist(detected, column, bins)
    for row in rows:
        row['metric'] = f'detected_only_{column}'
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--out-csv', type=Path, required=True)
    parser.add_argument('--bin-size', type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    bins = np.arange(0.0, 1.0 + args.bin_size, args.bin_size)
    if bins[-1] < 1.0:
        bins = np.append(bins, 1.0)

    rows = []
    rows.extend(categorical_hist(df, 'face_detected'))
    rows.extend(numeric_hist(df, 'det_score', bins))
    rows.extend(numeric_hist(df, 'quality_score', bins))
    rows.extend(detected_numeric_hist(df, 'det_score', bins))
    rows.extend(detected_numeric_hist(df, 'quality_score', bins))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['metric', 'bin_label', 'bin_start', 'bin_end', 'count', 'ratio', 'cumulative_count', 'cumulative_ratio']
    with args.out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'[DONE] wrote {args.out_csv}')


if __name__ == '__main__':
    main()
