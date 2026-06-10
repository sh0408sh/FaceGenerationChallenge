#!/usr/bin/env python3
"""Summarize StyleGAN metric JSONL files across training run directories."""

import argparse
import csv
import json
from pathlib import Path


DEFAULT_ROOTS = [
    "training-runs-celebvhq-nocrop-scratch",
    "training-runs-celebvhq-nocrop-continue",
    "training-runs-celebvhq-nocrop-ffhqu-t-finetune",
    "training-runs-celebvhq-nocrop-r-scratch",
    "training-runs-celebvhq-nocrop-r-continue",
]


METRIC_COLUMNS = [
    "fid50k_full",
    "kid50k_full",
    "is50k_mean",
    "is50k_std",
    "pr50k3_full_precision",
    "pr50k3_full_recall",
]


def load_jsonl(path):
    rows = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_options(run_dir):
    path = run_dir / "training_options.json"
    if not path.exists():
        return {}
    with path.open("rt", encoding="utf-8") as f:
        return json.load(f)


def snapshot_to_kimg(snapshot):
    if not snapshot:
        return ""
    stem = Path(snapshot).stem
    if stem.startswith("network-snapshot-"):
        try:
            return int(stem.rsplit("-", 1)[1])
        except ValueError:
            return ""
    return ""


def collect_run(run_dir):
    options = read_options(run_dir)
    rows_by_snapshot = {}

    for metric_path in sorted(run_dir.glob("metric-*.jsonl")):
        for row in load_jsonl(metric_path):
            snapshot = row.get("snapshot_pkl", "")
            out = rows_by_snapshot.setdefault(
                snapshot,
                {
                    "run_dir": str(run_dir),
                    "run_name": run_dir.name,
                    "snapshot": snapshot,
                    "snapshot_kimg": snapshot_to_kimg(snapshot),
                },
            )
            for key, value in row.get("results", {}).items():
                out[key] = value

    common = {
        "cfg": infer_cfg(options, run_dir.name),
        "resume": options.get("resume_pkl", ""),
        "dataset": Path(options.get("training_set_kwargs", {}).get("path", "")).name,
        "dataset_size": options.get("training_set_kwargs", {}).get("max_size", ""),
        "batch": options.get("batch_size", ""),
        "gamma": options.get("loss_kwargs", {}).get("r1_gamma", ""),
        "total_kimg": options.get("total_kimg", ""),
    }

    rows = []
    for row in rows_by_snapshot.values():
        full = dict(common)
        full.update(row)
        rows.append(full)
    return rows


def infer_cfg(options, run_name):
    class_name = options.get("G_kwargs", {}).get("class_name", "")
    if "networks_stylegan2" in class_name:
        return "stylegan2"
    if "stylegan3-r" in run_name:
        return "stylegan3-r"
    if "stylegan3-t" in run_name:
        return "stylegan3-t"
    return ""


def sort_key(row):
    fid = row.get("fid50k_full")
    fid_key = float(fid) if isinstance(fid, (int, float)) else float("inf")
    return (fid_key, row.get("run_name", ""), row.get("snapshot_kimg") or 0)


def format_value(value):
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_markdown(rows, limit):
    columns = [
        "rank",
        "cfg",
        "run_name",
        "snapshot",
        "fid50k_full",
        "kid50k_full",
        "is50k_mean",
        "pr50k3_full_precision",
        "pr50k3_full_recall",
        "resume",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for idx, row in enumerate(rows[:limit], start=1):
        values = []
        for col in columns:
            if col == "rank":
                values.append(str(idx))
            else:
                values.append(format_value(row.get(col, "")))
        print("| " + " | ".join(values) + " |")


def write_csv(rows, path):
    columns = [
        "cfg",
        "run_name",
        "snapshot",
        "snapshot_kimg",
        "dataset",
        "dataset_size",
        "batch",
        "gamma",
        "total_kimg",
        *METRIC_COLUMNS,
        "resume",
        "run_dir",
    ]
    with path.open("wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help="Run root directories to scan. Defaults to known CelebV-HQ run roots.",
    )
    parser.add_argument("--limit", type=int, default=30, help="Rows to print in markdown output.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    args = parser.parse_args()

    all_rows = []
    for root in args.roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for run_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            all_rows.extend(collect_run(run_dir))

    rows = sorted(all_rows, key=sort_key)
    if args.csv:
        write_csv(rows, args.csv)
    print_markdown(rows, args.limit)


if __name__ == "__main__":
    main()
