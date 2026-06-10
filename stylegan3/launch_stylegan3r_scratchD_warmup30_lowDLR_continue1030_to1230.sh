#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sh0408sh/FaceGeneration/CelebV-HQ/stylegan3"
LOG="${ROOT}/run_stylegan3r_scratchD_warmup30_lowDLR_continue1030_to1230_$(date +%Y%m%d_%H%M%S).log"

setsid bash -lc "
set -eo pipefail
source /home/sh0408sh/miniconda3/etc/profile.d/conda.sh
conda activate stylegan2-dev
cd ${ROOT}
STYLEGAN_METRIC_NUM_GEN=5000 CUDA_VISIBLE_DEVICES=5,7 python3 train.py \
  --outdir=${ROOT}/training-runs-celebvhq-stylegan3-r-qscore-ge0p5-scratchD-warmup-lowDLR-continue \
  --cfg=stylegan3-r \
  --data=/home/sh0408sh/FaceGeneration/CelebV-HQ/downloaded_celebvhq/quality_splits/nocrop_quality_score_ge_0p5_filtered.zip \
  --gpus=2 \
  --batch=32 \
  --gamma=2 \
  --cbase=16384 \
  --dcbase=32768 \
  --cmax=512 \
  --resume=${ROOT}/training-runs-celebvhq-stylegan3-r-qscore-ge0p5-scratchD-warmup-lowDLR/00001-stylegan3-r-nocrop_quality_score_ge_0p5_filtered-gpus2-batch32-gamma2-stylegan3-r-bestfid-scratchD1400-warmup30-lowDLR0.0010-l2sp0.0005-1030kimg/network-snapshot-001030.pkl \
  --resume-g-only=false \
  --resume-kimg=1030 \
  --l2sp-anchor=${ROOT}/pretrained/stylegan3-r-ffhqu-256x256.pkl \
  --mirror=true \
  --aug=ada \
  --target=0.5 \
  --freezed=0 \
  --glr=0.0005 \
  --dlr=0.0010 \
  --lambda-l2sp=0.0005 \
  --kimg=1230 \
  --tick=5 \
  --snap=5 \
  --seed=0 \
  --metrics=fid_kid_is_pr_full \
  --metric-data=/home/sh0408sh/FaceGeneration/CelebV-HQ/downloaded_celebvhq/image_dataset_from_videos_256_nocrop_png \
  --desc=stylegan3-r-scratchD-warmup30-lowDLR0.0010-continue1030to1230-l2sp0.0005
" >"${LOG}" 2>&1 < /dev/null &

echo "[started] pid=$!"
echo "[log] ${LOG}"
