"""Evaluate discriminator scores for real images and generator samples."""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

import dnnlib
import legacy
from training.dataset import ImageFolderDataset


def load_pkl(path, device):
    with dnnlib.util.open_url(path) as f:
        data = legacy.load_network_pkl(f)
    return {key: value.eval().requires_grad_(False).to(device) for key, value in data.items() if key in ["G_ema", "D"]}


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-values))
    return {
        "count": int(values.size),
        "logit_mean": float(np.mean(values)),
        "logit_std": float(np.std(values)),
        "logit_min": float(np.min(values)),
        "logit_p05": float(np.percentile(values, 5)),
        "logit_p25": float(np.percentile(values, 25)),
        "logit_median": float(np.percentile(values, 50)),
        "logit_p75": float(np.percentile(values, 75)),
        "logit_p95": float(np.percentile(values, 95)),
        "logit_max": float(np.max(values)),
        "prob_real_mean": float(np.mean(probs)),
        "prob_real_std": float(np.std(probs)),
        "real_decision_rate_logit_gt_0": float(np.mean(values > 0.0)),
        "fake_decision_rate_logit_lt_0": float(np.mean(values < 0.0)),
    }


def write_summary(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def score_real_dataset(D, dataset_path, num, batch, seed, device, raw_writer):
    dataset = ImageFolderDataset(path=dataset_path, resolution=D.img_resolution, use_labels=False, max_size=num, random_seed=seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    logits_all = []
    seen = 0
    for images, _labels in loader:
        images = images.to(device, non_blocking=True).to(torch.float32) / 127.5 - 1.0
        c = torch.zeros([images.shape[0], D.c_dim], device=device)
        logits = D(images, c).detach().flatten().cpu().numpy()
        for idx, logit in enumerate(logits):
            prob = float(1.0 / (1.0 + np.exp(-float(logit))))
            raw_writer.writerow({
                "group": "real_dataset",
                "index": seen + idx,
                "source": "real",
                "logit": float(logit),
                "prob_real": prob,
                "decision": "real" if logit > 0 else "fake",
            })
        logits_all.extend(logits.tolist())
        seen += len(logits)
        if seen >= num:
            break
    dataset.close()
    return logits_all[:num]


@torch.no_grad()
def score_generator(D, G, group, num, batch, seed, truncation_psi, noise_mode, device, raw_writer, save_real_dir=None):
    logits_all = []
    seen = 0
    saved = 0
    while seen < num:
        cur = min(batch, num - seen)
        rng = np.random.RandomState(seed + seen)
        z = torch.from_numpy(rng.randn(cur, G.z_dim)).to(device)
        c = torch.zeros([cur, G.c_dim], device=device)
        images = G(z, c, truncation_psi=truncation_psi, noise_mode=noise_mode)
        d_c = torch.zeros([cur, D.c_dim], device=device)
        logits = D(images, d_c).detach().flatten().cpu().numpy()
        for idx, logit in enumerate(logits):
            prob = float(1.0 / (1.0 + np.exp(-float(logit))))
            raw_writer.writerow({
                "group": group,
                "index": seen + idx,
                "source": f"seed_base_{seed}",
                "logit": float(logit),
                "prob_real": prob,
                "decision": "real" if logit > 0 else "fake",
            })
            if save_real_dir is not None and logit > 0:
                import PIL.Image
                img = (images[idx].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img.cpu().numpy(), "RGB").save(
                    os.path.join(save_real_dir, f"{group}_idx{seen + idx:06d}_logit{float(logit):+.4f}_prob{prob:.4f}.png")
                )
                saved += 1
        logits_all.extend(logits.tolist())
        seen += cur
    return logits_all, saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disc-network", required=True)
    parser.add_argument("--real-data", required=True)
    parser.add_argument("--snap-g-network", required=True)
    parser.add_argument("--ffhq-network", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--num", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trunc", type=float, default=1.0)
    parser.add_argument("--noise-mode", default="const", choices=["const", "random", "none"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-snap-real-decisions", action="store_true", help="Save snap generated images that D classifies as real.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    started = time.time()

    print(f'Loading discriminator from "{args.disc_network}"')
    disc_data = load_pkl(args.disc_network, device)
    D = disc_data["D"]

    print(f'Loading snap generator from "{args.snap_g_network}"')
    snap_G = load_pkl(args.snap_g_network, device)["G_ema"]

    print(f'Loading FFHQ pretrained generator from "{args.ffhq_network}"')
    ffhq_G = load_pkl(args.ffhq_network, device)["G_ema"]

    raw_path = os.path.join(args.outdir, "discriminator_scores_raw.csv")
    summary_path = os.path.join(args.outdir, "discriminator_scores_summary.csv")
    json_path = os.path.join(args.outdir, "discriminator_scores_summary.json")

    rows = []
    with open(raw_path, "w", newline="") as f:
        raw_writer = csv.DictWriter(f, fieldnames=["group", "index", "source", "logit", "prob_real", "decision"])
        raw_writer.writeheader()

        print(f"Scoring {args.num} real dataset images...")
        real_logits = score_real_dataset(D, args.real_data, args.num, args.batch, args.seed, device, raw_writer)
        rows.append({"group": "real_dataset", **summarize(real_logits)})

        save_snap_dir = None
        if args.save_snap_real_decisions:
            save_snap_dir = os.path.join(args.outdir, "snap1000_generated_D_real_images")
            os.makedirs(save_snap_dir, exist_ok=True)

        print(f"Scoring {args.num} images from snap generator...")
        snap_logits, snap_saved = score_generator(D, snap_G, "snap1000_generated", args.num, args.batch, args.seed, args.trunc, args.noise_mode, device, raw_writer, save_real_dir=save_snap_dir)
        rows.append({"group": "snap1000_generated", "saved_real_decision_images": int(snap_saved), **summarize(snap_logits)})

        print(f"Scoring {args.num} images from FFHQ pretrained generator...")
        ffhq_logits, _ffhq_saved = score_generator(D, ffhq_G, "ffhq_pretrained_generated", args.num, args.batch, args.seed, args.trunc, args.noise_mode, device, raw_writer)
        rows.append({"group": "ffhq_pretrained_generated", "saved_real_decision_images": 0, **summarize(ffhq_logits)})

    write_summary(summary_path, rows)
    with open(json_path, "w") as f:
        json.dump({"summary": rows, "args": vars(args), "elapsed_sec": time.time() - started}, f, indent=2)

    print(f"Wrote raw scores: {raw_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
