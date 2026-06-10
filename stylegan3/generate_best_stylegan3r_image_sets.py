#!/usr/bin/env python3
"""Regenerate selected StyleGAN3-R image sets from evaluation seeds."""

from pathlib import Path

import dnnlib
import torch

import generate_seed_eval as seed_eval
import legacy


ROOT = Path("/home/sh0408sh/FaceGeneration/CelebV-HQ/stylegan3")
RUN_DIR = ROOT / "training-runs-celebvhq-stylegan3-r-ffhqu-qscore-ge0p5-clean-metrics50k/00000-stylegan3-r-nocrop_quality_score_ge_0p5_filtered-gpus2-batch32-gamma2-stylegan3-r-ffhqu-qscore-ge0p5-clean-l2sp0.001-snap50-metrics5000-300kimg"
OUTDIR = ROOT / "generation-tests/stylegan3r_qscore_ge0p5_snap200_250_300_seeds0_5_num1000/generated_images_best"
JOBS = [(250, 3), (300, 4)]


def main() -> None:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    for kimg, seed in JOBS:
        network = RUN_DIR / f"network-snapshot-{kimg:06d}.pkl"
        outdir = OUTDIR / f"snapshot_{kimg:06d}_seed_{seed:02d}_num1000"
        print(f"[LOAD] {network}", flush=True)
        with dnnlib.util.open_url(str(network)) as handle:
            G = legacy.load_network_pkl(handle)["G_ema"].eval().requires_grad_(False).to(device)
        print(f"[GENERATE] snapshot={kimg:06d} seed={seed} out={outdir}", flush=True)
        seed_eval.generate_images(
            G=G,
            seed=seed,
            outdir=outdir,
            num_images=1000,
            batch_size=16,
            device=device,
            truncation_psi=1.0,
            noise_mode="const",
            jpeg_quality=95,
        )
        del G
        torch.cuda.empty_cache()
    print(f"[DONE] {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
