#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sh0408sh/FaceGeneration/CelebV-HQ/stylegan3"
RUN_DIR="${ROOT}/training-runs-celebvhq-stylegan3-r-ffhqu-qscore-ge0p5-clean-metrics50k/00000-stylegan3-r-nocrop_quality_score_ge_0p5_filtered-gpus2-batch32-gamma2-stylegan3-r-ffhqu-qscore-ge0p5-clean-l2sp0.001-snap50-metrics5000-300kimg"
OUT_ROOT="${ROOT}/generation-tests/ffhqu_qscore_ge0p5_metrics50k_snap0_150_300_num50"
GPU_ID="${GPU_ID:-1}"
SEEDS="${SEEDS:-0-49}"
CONDA_ENV="${CONDA_ENV:-stylegan2-dev}"

cd "${ROOT}"

set +u
source /home/sh0408sh/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
set -u

for snap in 000000 000150 000300; do
  network="${RUN_DIR}/network-snapshot-${snap}.pkl"
  outdir="${OUT_ROOT}/snapshot_${snap}"
  if [[ ! -f "${network}" ]]; then
    echo "[ERROR] Missing network: ${network}" >&2
    exit 1
  fi
  mkdir -p "${outdir}"
  echo "[INFO] Generating ${SEEDS} from ${network}"
  echo "[INFO] Output: ${outdir}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 gen_images.py \
    --network="${network}" \
    --seeds="${SEEDS}" \
    --trunc=1 \
    --noise-mode=const \
    --outdir="${outdir}"
done

echo "[done] ${OUT_ROOT}"
