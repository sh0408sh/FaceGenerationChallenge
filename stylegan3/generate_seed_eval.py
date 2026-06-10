#!/usr/bin/env python3
"""Generate image sets for multiple seeds and evaluate each saved set."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import dnnlib
import numpy as np
import PIL.Image
import scipy.linalg
import torch

import legacy
from metrics import metric_utils


INCEPTION_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
VGG16_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/vgg16.pkl"
DATA_LOADER_KWARGS = dict(pin_memory=True, num_workers=0)


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def toppr(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def make_opts(dataset_path: Path, resolution: int, device: torch.device, cache: bool) -> metric_utils.MetricOptions:
    return metric_utils.MetricOptions(
        dataset_kwargs=dnnlib.EasyDict(
            class_name="training.dataset.ImageFolderDataset",
            path=str(dataset_path),
            resolution=resolution,
            use_labels=False,
            xflip=False,
        ),
        num_gpus=1,
        rank=0,
        device=device,
        progress=metric_utils.ProgressMonitor(verbose=False),
        cache=cache,
    )


def compute_real_stats(data: Path, resolution: int, device: torch.device):
    opts = make_opts(data, resolution=resolution, device=device, cache=True)
    inception_stats = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=INCEPTION_URL,
        detector_kwargs=dict(return_features=True),
        data_loader_kwargs=DATA_LOADER_KWARGS,
        capture_all=True,
        capture_mean_cov=True,
        max_items=None,
    )
    vgg_stats = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=VGG16_URL,
        detector_kwargs=dict(return_features=True),
        data_loader_kwargs=DATA_LOADER_KWARGS,
        capture_all=True,
        max_items=None,
    )
    return inception_stats, vgg_stats


def compute_fake_stats(fake_dir: Path, resolution: int, device: torch.device, num_images: int):
    opts = make_opts(fake_dir, resolution=resolution, device=device, cache=False)
    inception_stats = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=INCEPTION_URL,
        detector_kwargs=dict(return_features=True),
        data_loader_kwargs=DATA_LOADER_KWARGS,
        capture_all=True,
        capture_mean_cov=True,
        max_items=num_images,
    )
    prob_stats = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=INCEPTION_URL,
        detector_kwargs=dict(no_output_bias=True),
        data_loader_kwargs=DATA_LOADER_KWARGS,
        capture_all=True,
        max_items=num_images,
    )
    vgg_stats = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=VGG16_URL,
        detector_kwargs=dict(return_features=True),
        data_loader_kwargs=DATA_LOADER_KWARGS,
        capture_all=True,
        max_items=num_images,
    )
    return inception_stats, prob_stats, vgg_stats


def fid_from_stats(real_stats, fake_stats) -> float:
    mu_real, sigma_real = real_stats.get_mean_cov()
    mu_fake, sigma_fake = fake_stats.get_mean_cov()
    m = np.square(mu_fake - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
    return float(np.real(m + np.trace(sigma_fake + sigma_real - s * 2)))


def kid_from_features(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    n = real_features.shape[1]
    m = min(min(real_features.shape[0], fake_features.shape[0]), 1000)
    total = 0.0
    rng = np.random.RandomState(0)
    for _subset_idx in range(10):
        x = fake_features[rng.choice(fake_features.shape[0], m, replace=False)]
        y = real_features[rng.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        total += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return float(total / 10 / m)


def inception_score_from_probs(probs: np.ndarray, num_images: int) -> tuple[float, float]:
    scores = []
    for split_idx in range(10):
        part = probs[split_idx * num_images // 10 : (split_idx + 1) * num_images // 10]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


def compute_pr_from_features(real_features_np: np.ndarray, fake_features_np: np.ndarray, device: torch.device) -> tuple[float, float]:
    real_features = torch.from_numpy(real_features_np).to(torch.float16).to(device)
    fake_features = torch.from_numpy(fake_features_np).to(torch.float16).to(device)

    def manifold_membership(manifold: torch.Tensor, probes: torch.Tensor) -> float:
        kth = []
        for manifold_batch in manifold.split(10000):
            dist = torch.cdist(manifold_batch.unsqueeze(0), manifold.unsqueeze(0))[0]
            kth.append(dist.to(torch.float32).kthvalue(4).values.to(torch.float16))
        kth_all = torch.cat(kth)

        pred = []
        for probe_batch in probes.split(10000):
            dist = torch.cdist(probe_batch.unsqueeze(0), manifold.unsqueeze(0))[0]
            pred.append((dist <= kth_all).any(dim=1))
        return float(torch.cat(pred).to(torch.float32).mean().cpu())

    precision = manifold_membership(real_features, fake_features)
    recall = manifold_membership(fake_features, real_features)
    return precision, recall


def generate_images(
    G,
    seed: int,
    outdir: Path,
    num_images: int,
    batch_size: int,
    device: torch.device,
    truncation_psi: float,
    noise_mode: str,
    jpeg_quality: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    label = torch.zeros([batch_size, G.c_dim], device=device)

    if hasattr(G.synthesis, "input") and hasattr(G.synthesis.input, "transform"):
        G.synthesis.input.transform.copy_(torch.eye(3, device=device))

    generated = 0
    while generated < num_images:
        batch = min(batch_size, num_images - generated)
        z = torch.from_numpy(rng.randn(batch, G.z_dim)).to(device)
        c = label[:batch]
        with torch.no_grad():
            img = G(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        for idx in range(batch):
            image = PIL.Image.fromarray(img[idx].cpu().numpy(), "RGB")
            image.save(outdir / f"img_{generated + idx:04d}.jpg", quality=jpeg_quality, optimize=True)
        generated += batch


def evaluate_fake_dir(
    fake_dir: Path,
    real_inception_stats,
    real_vgg_stats,
    resolution: int,
    device: torch.device,
    num_images: int,
) -> dict[str, float]:
    fake_inception_stats, fake_prob_stats, fake_vgg_stats = compute_fake_stats(
        fake_dir=fake_dir,
        resolution=resolution,
        device=device,
        num_images=num_images,
    )
    fid = fid_from_stats(real_inception_stats, fake_inception_stats)
    kid = kid_from_features(real_inception_stats.get_all(), fake_inception_stats.get_all())
    is_mean, is_std = inception_score_from_probs(fake_prob_stats.get_all(), num_images=num_images)
    precision, recall = compute_pr_from_features(real_vgg_stats.get_all(), fake_vgg_stats.get_all(), device=device)
    return {
        "fid": fid,
        "is": is_mean,
        "is_std": is_std,
        "kid": kid,
        "precision": precision,
        "recall": recall,
        "toppr": toppr(precision, recall),
    }


def compute_generator_stats(G, dataset_path: Path, resolution: int, device: torch.device, num_images: int, batch_size: int, seed: int, truncation_psi: float, noise_mode: str):
    opts = make_opts(dataset_path, resolution=resolution, device=device, cache=False)
    opts.G = G
    opts.G_kwargs = dnnlib.EasyDict(truncation_psi=truncation_psi, noise_mode=noise_mode)
    torch.manual_seed(seed)
    np.random.seed(seed)
    inception_stats = metric_utils.compute_feature_stats_for_generator(
        opts=opts,
        detector_url=INCEPTION_URL,
        detector_kwargs=dict(return_features=True),
        capture_all=True,
        capture_mean_cov=True,
        max_items=num_images,
        batch_size=batch_size,
        batch_gen=min(batch_size, 4),
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    prob_stats = metric_utils.compute_feature_stats_for_generator(
        opts=opts,
        detector_url=INCEPTION_URL,
        detector_kwargs=dict(no_output_bias=True),
        capture_all=True,
        max_items=num_images,
        batch_size=batch_size,
        batch_gen=min(batch_size, 4),
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    vgg_stats = metric_utils.compute_feature_stats_for_generator(
        opts=opts,
        detector_url=VGG16_URL,
        detector_kwargs=dict(return_features=True),
        capture_all=True,
        max_items=num_images,
        batch_size=batch_size,
        batch_gen=min(batch_size, 4),
    )
    return inception_stats, prob_stats, vgg_stats


def evaluate_generator(
    G,
    dataset_path: Path,
    real_inception_stats,
    real_vgg_stats,
    resolution: int,
    device: torch.device,
    num_images: int,
    batch_size: int,
    seed: int,
    truncation_psi: float,
    noise_mode: str,
) -> dict[str, float]:
    fake_inception_stats, fake_prob_stats, fake_vgg_stats = compute_generator_stats(
        G=G,
        dataset_path=dataset_path,
        resolution=resolution,
        device=device,
        num_images=num_images,
        batch_size=batch_size,
        seed=seed,
        truncation_psi=truncation_psi,
        noise_mode=noise_mode,
    )
    fid = fid_from_stats(real_inception_stats, fake_inception_stats)
    kid = kid_from_features(real_inception_stats.get_all(), fake_inception_stats.get_all())
    is_mean, is_std = inception_score_from_probs(fake_prob_stats.get_all(), num_images=num_images)
    precision, recall = compute_pr_from_features(real_vgg_stats.get_all(), fake_vgg_stats.get_all(), device=device)
    return {
        "fid": fid,
        "is": is_mean,
        "is_std": is_std,
        "kid": kid,
        "precision": precision,
        "recall": recall,
        "toppr": toppr(precision, recall),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "row_type",
        "seed",
        "num_images",
        "image_dir",
        "fid",
        "is",
        "is_std",
        "kid",
        "precision",
        "recall",
        "toppr",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_summary_rows(rows: list[dict[str, object]], metric_rows: list[dict[str, object]], num_images: int) -> None:
    metric_names = ["fid", "is", "is_std", "kid", "precision", "recall", "toppr"]
    for row_type, reducer in [("mean", np.mean), ("std", np.std)]:
        row: dict[str, object] = {
            "row_type": row_type,
            "seed": "",
            "num_images": num_images,
            "image_dir": "",
        }
        for name in metric_names:
            values = np.array([float(item[name]) for item in metric_rows], dtype=np.float64)
            row[name] = float(reducer(values, ddof=1)) if row_type == "std" and len(values) > 1 else float(reducer(values))
        rows.append(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trunc", type=float, default=1.0)
    parser.add_argument("--noise-mode", choices=["const", "random", "none"], default="const")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    network = (root / args.network).resolve() if not args.network.is_absolute() else args.network.resolve()
    data = (root / args.data).resolve() if not args.data.is_absolute() else args.data.resolve()
    outdir = (root / args.outdir).resolve() if not args.outdir.is_absolute() else args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not network.is_file():
        raise FileNotFoundError(f"network not found: {network}")
    if not data.exists():
        raise FileNotFoundError(f"data not found: {data}")

    seeds = parse_seeds(args.seeds)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"Loading network: {network}")
    with dnnlib.util.open_url(str(network)) as handle:
        G = legacy.load_network_pkl(handle)["G_ema"].eval().requires_grad_(False).to(device)

    print(f"Computing/loading real stats: {data}")
    real_inception_stats, real_vgg_stats = compute_real_stats(data=data, resolution=G.img_resolution, device=device)

    csv_path = outdir / "seed_metrics.csv"
    json_path = outdir / "seed_metrics.json"
    metric_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    start_time = time.time()

    for index, seed in enumerate(seeds, start=1):
        seed_dir = outdir / f"seed_{seed:02d}"
        print(f"[{index}/{len(seeds)}] seed={seed}: generating {args.num_images} jpg images in {seed_dir}")
        generate_images(
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
        print(f"[{index}/{len(seeds)}] seed={seed}: evaluating saved images")
        metrics = evaluate_fake_dir(
            fake_dir=seed_dir,
            real_inception_stats=real_inception_stats,
            real_vgg_stats=real_vgg_stats,
            resolution=G.img_resolution,
            device=device,
            num_images=args.num_images,
        )
        row: dict[str, object] = {
            "row_type": "seed",
            "seed": seed,
            "num_images": args.num_images,
            "image_dir": str(seed_dir.relative_to(root)),
            **metrics,
        }
        metric_rows.append(row)
        rows = list(metric_rows)
        add_summary_rows(rows, metric_rows=metric_rows, num_images=args.num_images)
        write_csv(csv_path, rows)
        json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(row, sort_keys=True))

    elapsed = time.time() - start_time
    print(f"done in {elapsed:.1f}s")
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()
