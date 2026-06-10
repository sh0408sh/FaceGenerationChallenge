#!/usr/bin/env python3
"""Create high/low quality image datasets from quality CSV bins."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import zipfile
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_zip(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(src_dir.rglob('*')):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir).as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--quality-csv', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--metric', default='quality_score')
    parser.add_argument('--bin-start', type=float, default=0.0)
    parser.add_argument('--bin-end', type=float, default=0.1)
    parser.add_argument('--make-zip', action='store_true')
    return parser.parse_args()


def format_bin_value(value: float) -> str:
    return f'{value:g}'.replace('.', 'p')


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    quality_csv = args.quality_csv.resolve()
    out_root = args.out_root.resolve()
    bin_start = format_bin_value(args.bin_start)
    bin_end = format_bin_value(args.bin_end)
    low_dir = out_root / f'nocrop_{args.metric}_{bin_start}_{bin_end}_only'
    filtered_dir = out_root / f'nocrop_{args.metric}_ge_{bin_end}_filtered'
    low_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    low_count = 0
    filtered_count = 0
    missing = []
    with quality_csv.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row['path']
            src = source_dir / rel
            if not src.is_file():
                missing.append(rel)
                continue
            value = float(row[args.metric])
            in_bin = args.bin_start <= value < args.bin_end
            if in_bin:
                link_or_copy(src, low_dir / rel)
                low_count += 1
            else:
                link_or_copy(src, filtered_dir / rel)
                filtered_count += 1

    summary = {
        'source_dir': str(source_dir),
        'quality_csv': str(quality_csv),
        'metric': args.metric,
        'bin_start': args.bin_start,
        'bin_end': args.bin_end,
        'low_only_dir': str(low_dir),
        'filtered_dir': str(filtered_dir),
        'low_only_count': low_count,
        'filtered_count': filtered_count,
        'missing_count': len(missing),
        'missing_examples': missing[:20],
    }

    if args.make_zip:
        low_zip = low_dir.with_suffix('.zip')
        filtered_zip = filtered_dir.with_suffix('.zip')
        write_zip(low_dir, low_zip)
        write_zip(filtered_dir, filtered_zip)
        summary['low_only_zip'] = str(low_zip)
        summary['filtered_zip'] = str(filtered_zip)

    range_label = f'{bin_start}_{bin_end}'
    summary_path = out_root / f'nocrop_{args.metric}_{range_label}_split_summary.json'
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f'[DONE] low-only: {low_count} images -> {low_dir}')
    print(f'[DONE] filtered: {filtered_count} images -> {filtered_dir}')
    if args.make_zip:
        print(f'[DONE] low-only zip: {summary["low_only_zip"]}')
        print(f'[DONE] filtered zip: {summary["filtered_zip"]}')
    print(f'[DONE] summary: {summary_path}')


if __name__ == '__main__':
    main()
