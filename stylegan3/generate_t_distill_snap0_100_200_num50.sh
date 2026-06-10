#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/sh0408sh/FaceGeneration/CelebV-HQ/stylegan3"
GPU="${GPU:-5}"
SEEDS="${SEEDS:-0-49}"
TRUNC="${TRUNC:-1.0}"
NOISE_MODE="${NOISE_MODE:-const}"
OUTROOT="${OUTROOT:-${ROOT}/generation-tests/t_distill_snap0_100_200_num50}"

RUN_LABELS=(
  "t1400_distill"
  "t1000_distill"
)

RUN_DIRS=(
  "${ROOT}/training-runs-celebvhq-nocrop-t-distill/00000-stylegan3-t-image_dataset_from_videos_256_nocrop_png-gpus2-batch32-gamma2-t1400-ffhq-vgg-freq-l2sp-weak-200kimg"
  "${ROOT}/training-runs-celebvhq-nocrop-t-distill/00001-stylegan3-t-image_dataset_from_videos_256_nocrop_png-gpus2-batch32-gamma2-t1000-ffhq-vgg-freq-l2sp-weak-200kimg"
)

SNAPS=(
  "000000"
  "000100"
  "000200"
)

source /home/sh0408sh/miniconda3/etc/profile.d/conda.sh
conda activate stylegan2-dev

cd "$ROOT"
mkdir -p "$OUTROOT"

echo "Output root: $OUTROOT"
echo "GPU: $GPU"
echo "Seeds: $SEEDS"

for idx in "${!RUN_LABELS[@]}"; do
  label="${RUN_LABELS[$idx]}"
  run_dir="${RUN_DIRS[$idx]}"

  for snap in "${SNAPS[@]}"; do
    network="${run_dir}/network-snapshot-${snap}.pkl"
    outdir="${OUTROOT}/${label}/snapshot_${snap}"

    if [[ ! -f "$network" ]]; then
      echo "Missing snapshot: $network" >&2
      exit 1
    fi

    mkdir -p "$outdir"
    echo "Generating ${label} ${snap} -> ${outdir}"
    CUDA_VISIBLE_DEVICES="$GPU" python3 gen_images.py \
      --network "$network" \
      --outdir "$outdir" \
      --seeds "$SEEDS" \
      --trunc "$TRUNC" \
      --noise-mode "$NOISE_MODE"
  done
done

echo "Generated image counts:"
for idx in "${!RUN_LABELS[@]}"; do
  label="${RUN_LABELS[$idx]}"
  for snap in "${SNAPS[@]}"; do
    outdir="${OUTROOT}/${label}/snapshot_${snap}"
    count=$(find "$outdir" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | wc -l)
    printf "%s snapshot_%s: %s\n" "$label" "$snap" "$count"
  done
done
