#!/usr/bin/env python3
"""Generate/evaluate all network snapshots against one real dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path

import dnnlib
import numpy as np
import torch

import generate_seed_eval as seed_eval
import legacy


def snapshot_label(path: Path) -> str:
    match = re.search(r'network-snapshot-(\d+)\.pkl$', path.name)
    return match.group(1) if match else path.stem


def write_combined_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        'snapshot',
        'row_type',
        'seed',
        'num_images',
        'image_dir',
        'fid',
        'is',
        'is_std',
        'kid',
        'precision',
        'recall',
        'toppr',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_snapshots(text: str | None) -> set[str] | None:
    if text is None or not text.strip():
        return None
    labels = set()
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        labels.add(f'{int(part):06d}')
    return labels


def write_metric_tables(outdir: Path, combined_rows: list[dict[str, object]]) -> None:
    seed_rows = [row for row in combined_rows if row.get('row_type') == 'seed']
    mean_rows = {row['snapshot']: row for row in combined_rows if row.get('row_type') == 'mean'}
    std_rows = {row['snapshot']: row for row in combined_rows if row.get('row_type') == 'std'}
    snapshots = sorted({str(row['snapshot']) for row in seed_rows})
    seeds = sorted({int(row['seed']) for row in seed_rows})
    metrics = ['fid', 'is', 'kid', 'toppr']

    for metric in metrics:
        path = outdir / f'{metric}_by_snapshot_seed.csv'
        fields = ['snapshot'] + [f'seed_{seed}' for seed in seeds] + ['mean(std)']
        values = {(str(row['snapshot']), int(row['seed'])): row for row in seed_rows}
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for snapshot in snapshots:
                row: dict[str, object] = {'snapshot': snapshot}
                for seed in seeds:
                    item = values.get((snapshot, seed), {})
                    row[f'seed_{seed}'] = item.get(metric, '')
                if snapshot in mean_rows and snapshot in std_rows:
                    row['mean(std)'] = f"{float(mean_rows[snapshot][metric]):.6f} ({float(std_rows[snapshot][metric]):.6f})"
                else:
                    row['mean(std)'] = ''
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--num-images', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--trunc', type=float, default=1.0)
    parser.add_argument('--noise-mode', choices=['const', 'random', 'none'], default='const')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--snapshots', default=None, help='Comma-separated kimg labels, e.g. 200,250,300. Defaults to all snapshots.')
    parser.add_argument('--delete-images', action='store_true', help='Delete generated seed image directories after evaluation.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    run_dir = args.run_dir.resolve()
    data = args.data.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    selected_snapshots = parse_snapshots(args.snapshots)
    networks = sorted(run_dir.glob('network-snapshot-*.pkl'))
    if selected_snapshots is not None:
        networks = [path for path in networks if snapshot_label(path) in selected_snapshots]
    if not networks:
        raise FileNotFoundError(f'no snapshots found in {run_dir}')
    if not data.exists():
        raise FileNotFoundError(f'data not found: {data}')

    seeds = seed_eval.parse_seeds(args.seeds)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f'[INFO] Found {len(networks)} snapshots in {run_dir}')
    print(f'[INFO] Eval data: {data}')
    print(f'[INFO] Output: {outdir}')
    print(f'[INFO] Seeds={seeds}; num_images={args.num_images}; trunc={args.trunc}; noise={args.noise_mode}')

    real_inception_stats = None
    real_vgg_stats = None
    combined_rows: list[dict[str, object]] = []
    combined_csv = outdir / 'snapshot_seed_metrics.csv'
    combined_json = outdir / 'snapshot_seed_metrics.json'
    start_time = time.time()

    for snap_idx, network in enumerate(networks, start=1):
        label = snapshot_label(network)
        snapshot_outdir = outdir / f'snapshot_{label}'
        snapshot_outdir.mkdir(parents=True, exist_ok=True)
        print(f'[{snap_idx}/{len(networks)}] Loading network: {network}')
        with dnnlib.util.open_url(str(network)) as handle:
            G = legacy.load_network_pkl(handle)['G_ema'].eval().requires_grad_(False).to(device)

        if real_inception_stats is None or real_vgg_stats is None:
            print(f'[INFO] Computing/loading real stats once: {data}')
            real_inception_stats, real_vgg_stats = seed_eval.compute_real_stats(
                data=data,
                resolution=G.img_resolution,
                device=device,
            )

        metric_rows: list[dict[str, object]] = []
        snapshot_rows: list[dict[str, object]] = []
        for seed_index, seed in enumerate(seeds, start=1):
            seed_dir = snapshot_outdir / f'seed_{seed:02d}'
            print(f'[{snap_idx}/{len(networks)}][{seed_index}/{len(seeds)}] snapshot={label} seed={seed}: generating')
            seed_eval.generate_images(
                G=G,
                seed=seed,
                outdir=seed_dir,
                num_images=args.num_images,
                batch_size=args.batch_size,
                device=device,
                truncation_psi=args.trunc,
                noise_mode=args.noise_mode,
                jpeg_quality=args.jpeg_quality,
            )
            print(f'[{snap_idx}/{len(networks)}][{seed_index}/{len(seeds)}] snapshot={label} seed={seed}: evaluating')
            metrics = seed_eval.evaluate_fake_dir(
                fake_dir=seed_dir,
                real_inception_stats=real_inception_stats,
                real_vgg_stats=real_vgg_stats,
                resolution=G.img_resolution,
                device=device,
                num_images=args.num_images,
            )
            row: dict[str, object] = {
                'row_type': 'seed',
                'seed': seed,
                'num_images': args.num_images,
                'image_dir': str(seed_dir.relative_to(root)),
                **metrics,
            }
            print(json.dumps({'snapshot': label, **row}, sort_keys=True))
            if args.delete_images:
                shutil.rmtree(seed_dir)
                row['image_dir'] = ''

            metric_rows.append(row)
            snapshot_rows = list(metric_rows)
            seed_eval.add_summary_rows(snapshot_rows, metric_rows=metric_rows, num_images=args.num_images)
            seed_eval.write_csv(snapshot_outdir / 'seed_metrics.csv', snapshot_rows)
            (snapshot_outdir / 'seed_metrics.json').write_text(
                json.dumps(snapshot_rows, indent=2, sort_keys=True),
                encoding='utf-8',
            )

            combined_rows = [item for item in combined_rows if item.get('snapshot') != label]
            combined_rows.extend({'snapshot': label, **item} for item in snapshot_rows)
            write_combined_csv(combined_csv, combined_rows)
            write_metric_tables(outdir, combined_rows)
            combined_json.write_text(json.dumps(combined_rows, indent=2, sort_keys=True), encoding='utf-8')

        del G
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f'[DONE] all snapshots evaluated in {elapsed:.1f}s')
    print(f'[DONE] combined csv: {combined_csv}')
    print(f'[DONE] combined json: {combined_json}')


if __name__ == '__main__':
    main()
