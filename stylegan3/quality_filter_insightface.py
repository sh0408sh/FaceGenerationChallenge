#!/usr/bin/env python3
from __future__ import annotations

"""Score real images for quality filtering using InsightFace detection.

The script writes per-image scores and a compact summary table.  It does not
delete or move images; thresholding can be changed later without recomputing
detections.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob('*') if path.suffix.lower() in IMAGE_EXTS)


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def percentile_norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return clamp01((value - lo) / (hi - lo))


def face_size_score(area_ratio: float, min_area: float, ideal_area: float, max_area: float) -> float:
    if area_ratio <= 0 or area_ratio < min_area or area_ratio > max_area:
        return 0.0
    if area_ratio <= ideal_area:
        return percentile_norm(area_ratio, min_area, ideal_area)
    return percentile_norm(max_area - area_ratio, 0.0, max_area - ideal_area)


def landmark_valid_score(kps: np.ndarray | None, width: int, height: int) -> tuple[int, float, float]:
    if kps is None or np.asarray(kps).shape != (5, 2):
        return 0, 0.0, 0.0
    pts = np.asarray(kps, dtype=np.float32)
    if not np.isfinite(pts).all():
        return 0, 0.0, 0.0
    left_eye, right_eye, nose, left_mouth, right_mouth = pts
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    diag = float(math.sqrt(width * width + height * height))
    eye_distance = eye_dist / max(diag, 1.0)

    checks = [
        left_eye[0] < right_eye[0],
        left_mouth[0] < right_mouth[0],
        nose[1] > min(left_eye[1], right_eye[1]),
        nose[1] < max(left_mouth[1], right_mouth[1]),
        left_mouth[1] > min(left_eye[1], right_eye[1]),
        right_mouth[1] > min(left_eye[1], right_eye[1]),
        eye_distance > 0.04,
    ]
    valid = int(all(bool(item) for item in checks))
    landmark_score = sum(bool(item) for item in checks) / len(checks)
    return valid, float(landmark_score), float(eye_distance)


def image_stats(img_bgr: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    return blur, brightness, contrast


def score_image(app: FaceAnalysis, path: Path, root: Path) -> dict[str, object]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    rel = str(path.relative_to(root))
    if img is None:
        return {
            'path': rel, 'face_detected': 0, 'num_faces': 0, 'det_score': 0.0,
            'bbox_area_ratio': 0.0, 'center_offset': 1.0, 'eye_distance': 0.0,
            'landmark_valid': 0, 'landmark_score': 0.0, 'blur': 0.0,
            'brightness': 0.0, 'contrast': 0.0, 'read_error': 1,
        }
    height, width = img.shape[:2]
    blur, brightness, contrast = image_stats(img)
    faces = app.get(img)
    if len(faces) == 0:
        return {
            'path': rel, 'face_detected': 0, 'num_faces': 0, 'det_score': 0.0,
            'bbox_area_ratio': 0.0, 'center_offset': 1.0, 'eye_distance': 0.0,
            'landmark_valid': 0, 'landmark_score': 0.0, 'blur': blur,
            'brightness': brightness, 'contrast': contrast, 'read_error': 0,
        }

    def sort_key(face):
        x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float32)
        area = max(float(x2 - x1), 0.0) * max(float(y2 - y1), 0.0)
        return float(face.det_score) * area

    face = max(faces, key=sort_key)
    x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float32)
    x1, y1 = max(float(x1), 0.0), max(float(y1), 0.0)
    x2, y2 = min(float(x2), float(width)), min(float(y2), float(height))
    bbox_w = max(x2 - x1, 0.0)
    bbox_h = max(y2 - y1, 0.0)
    bbox_area_ratio = (bbox_w * bbox_h) / max(float(width * height), 1.0)
    center_x = (x1 + x2) * 0.5 / max(float(width), 1.0)
    center_y = (y1 + y2) * 0.5 / max(float(height), 1.0)
    center_offset = float(math.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) / math.sqrt(0.5))
    landmark_valid, landmark_score, eye_distance = landmark_valid_score(getattr(face, 'kps', None), width, height)

    return {
        'path': rel,
        'face_detected': 1,
        'num_faces': int(len(faces)),
        'det_score': float(face.det_score),
        'bbox_area_ratio': float(bbox_area_ratio),
        'center_offset': float(center_offset),
        'eye_distance': float(eye_distance),
        'landmark_valid': int(landmark_valid),
        'landmark_score': float(landmark_score),
        'blur': blur,
        'brightness': brightness,
        'contrast': contrast,
        'read_error': 0,
    }


def add_quality_scores(rows: list[dict[str, object]], keep_quantile: float, det_threshold: float) -> None:
    detected = [row for row in rows if int(row['face_detected']) == 1]
    blur_values = np.array([float(row['blur']) for row in detected], dtype=np.float64)
    contrast_values = np.array([float(row['contrast']) for row in detected], dtype=np.float64)
    brightness_values = np.array([float(row['brightness']) for row in detected], dtype=np.float64)

    blur_lo, blur_hi = np.percentile(blur_values, [5, 95]) if len(blur_values) else (0.0, 1.0)
    contrast_lo, contrast_hi = np.percentile(contrast_values, [5, 95]) if len(contrast_values) else (0.0, 1.0)
    bright_lo, bright_hi = np.percentile(brightness_values, [5, 95]) if len(brightness_values) else (0.0, 255.0)

    quality_values = []
    for row in rows:
        det_score = float(row['det_score'])
        area_ratio = float(row['bbox_area_ratio'])
        center_offset = float(row['center_offset'])
        landmark_score = float(row['landmark_score'])
        blur_score = percentile_norm(float(row['blur']), blur_lo, blur_hi)
        contrast_score = percentile_norm(float(row['contrast']), contrast_lo, contrast_hi)
        brightness_mid = (bright_lo + bright_hi) * 0.5
        brightness_half = max((bright_hi - bright_lo) * 0.5, 1.0)
        brightness_score = clamp01(1.0 - abs(float(row['brightness']) - brightness_mid) / brightness_half)
        size_score = face_size_score(area_ratio, min_area=0.05, ideal_area=0.36, max_area=0.90)
        center_score = clamp01(1.0 - center_offset / 0.45)
        single_face_score = 1.0 if int(row['num_faces']) == 1 else 0.75 if int(row['num_faces']) > 1 else 0.0

        quality = (
            0.35 * det_score +
            0.20 * blur_score +
            0.15 * size_score +
            0.10 * center_score +
            0.10 * landmark_score +
            0.05 * contrast_score +
            0.03 * brightness_score +
            0.02 * single_face_score
        )
        if int(row['face_detected']) == 0 or int(row['read_error']) == 1:
            quality = 0.0
        row['quality_score'] = float(quality)
        row['keep'] = 0
        quality_values.append(float(quality))

    threshold = float(np.quantile(np.asarray(quality_values, dtype=np.float64), 1.0 - keep_quantile)) if quality_values else 1.0
    for row in rows:
        row['keep'] = int(
            int(row['face_detected']) == 1 and
            float(row['det_score']) >= det_threshold and
            float(row['quality_score']) >= threshold
        )
    return {
        'blur_p05': float(blur_lo), 'blur_p95': float(blur_hi),
        'contrast_p05': float(contrast_lo), 'contrast_p95': float(contrast_hi),
        'brightness_p05': float(bright_lo), 'brightness_p95': float(bright_hi),
        'quality_keep_quantile': float(keep_quantile),
        'quality_threshold': float(threshold),
        'det_threshold': float(det_threshold),
    }


def summarize(rows: list[dict[str, object]], aux: dict[str, float]) -> dict[str, object]:
    total = len(rows)
    detected = [row for row in rows if int(row['face_detected']) == 1]
    kept = [row for row in rows if int(row['keep']) == 1]

    def stats(name: str, subset: list[dict[str, object]]):
        values = np.array([float(row[name]) for row in subset], dtype=np.float64)
        if values.size == 0:
            return {f'{name}_{key}': None for key in ['mean', 'std', 'min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max']}
        return {
            f'{name}_mean': float(values.mean()),
            f'{name}_std': float(values.std(ddof=1)) if values.size > 1 else 0.0,
            f'{name}_min': float(values.min()),
            f'{name}_p05': float(np.percentile(values, 5)),
            f'{name}_p25': float(np.percentile(values, 25)),
            f'{name}_p50': float(np.percentile(values, 50)),
            f'{name}_p75': float(np.percentile(values, 75)),
            f'{name}_p95': float(np.percentile(values, 95)),
            f'{name}_max': float(values.max()),
        }

    summary: dict[str, object] = {
        'total_images': total,
        'face_detected_count': len(detected),
        'face_detected_rate': len(detected) / total if total else 0.0,
        'kept_count': len(kept),
        'kept_rate': len(kept) / total if total else 0.0,
        'multi_face_count': sum(int(row['num_faces']) > 1 for row in rows),
        'multi_face_rate': sum(int(row['num_faces']) > 1 for row in rows) / total if total else 0.0,
        **aux,
    }
    for name in ['det_score', 'quality_score', 'blur', 'brightness', 'contrast', 'bbox_area_ratio', 'center_offset']:
        summary.update(stats(name, rows))
        summary.update({f'detected_{key}': value for key, value in stats(name, detected).items()})
        summary.update({f'kept_{key}': value for key, value in stats(name, kept).items()})
    return summary


def write_summary_table(summary: dict[str, object], path: Path) -> None:
    rows = [
        ('total_images', summary['total_images']),
        ('face_detected_count', summary['face_detected_count']),
        ('face_detected_rate', summary['face_detected_rate']),
        ('kept_count', summary['kept_count']),
        ('kept_rate', summary['kept_rate']),
        ('multi_face_count', summary['multi_face_count']),
        ('multi_face_rate', summary['multi_face_rate']),
        ('det_score_mean', summary['det_score_mean']),
        ('det_score_p05', summary['det_score_p05']),
        ('det_score_p50', summary['det_score_p50']),
        ('det_score_p95', summary['det_score_p95']),
        ('detected_det_score_mean', summary['detected_det_score_mean']),
        ('detected_det_score_p05', summary['detected_det_score_p05']),
        ('detected_det_score_p50', summary['detected_det_score_p50']),
        ('detected_det_score_p95', summary['detected_det_score_p95']),
        ('quality_score_mean', summary['quality_score_mean']),
        ('quality_score_p05', summary['quality_score_p05']),
        ('quality_score_p50', summary['quality_score_p50']),
        ('quality_score_p95', summary['quality_score_p95']),
        ('detected_quality_score_mean', summary['detected_quality_score_mean']),
        ('detected_quality_score_p05', summary['detected_quality_score_p05']),
        ('detected_quality_score_p50', summary['detected_quality_score_p50']),
        ('detected_quality_score_p95', summary['detected_quality_score_p95']),
        ('quality_threshold', summary['quality_threshold']),
        ('det_threshold', summary['det_threshold']),
    ]
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out-csv', type=Path, required=True)
    parser.add_argument('--summary-csv', type=Path, required=True)
    parser.add_argument('--summary-json', type=Path, required=True)
    parser.add_argument('--model-name', default='buffalo_l')
    parser.add_argument('--allowed-modules', default='detection', help='Comma-separated InsightFace modules, e.g. detection or detection,recognition')
    parser.add_argument('--det-size', type=int, default=640)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--keep-quantile', type=float, default=0.8)
    parser.add_argument('--det-threshold', type=float, default=0.75)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    paths = list_images(data_dir)
    if args.limit > 0:
        paths = paths[:args.limit]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.gpu >= 0 else ['CPUExecutionProvider']
    allowed_modules = [item.strip() for item in args.allowed_modules.split(',') if item.strip()]
    app = FaceAnalysis(name=args.model_name, allowed_modules=allowed_modules, providers=providers)
    app.prepare(ctx_id=args.gpu, det_size=(args.det_size, args.det_size))

    print(f'[INFO] data_dir={data_dir}')
    print(f'[INFO] num_images={len(paths)}')
    print(f'[INFO] model={args.model_name}, allowed_modules={allowed_modules}, det_size={args.det_size}, gpu={args.gpu}')
    start = time.time()
    rows = []
    for index, path in enumerate(paths, start=1):
        rows.append(score_image(app, path, data_dir))
        if index == 1 or index % 1000 == 0 or index == len(paths):
            elapsed = time.time() - start
            print(f'[INFO] processed {index}/{len(paths)} images ({index / max(elapsed, 1e-6):.1f} img/s)')

    aux = add_quality_scores(rows, keep_quantile=args.keep_quantile, det_threshold=args.det_threshold)
    summary = summarize(rows, aux)

    fieldnames = [
        'path', 'face_detected', 'num_faces', 'det_score', 'bbox_area_ratio',
        'center_offset', 'eye_distance', 'landmark_valid', 'blur', 'brightness',
        'contrast', 'quality_score', 'keep',
    ]
    with args.out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    write_summary_table(summary, args.summary_csv)
    with args.summary_json.open('w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f'[DONE] wrote {args.out_csv}')
    print(f'[DONE] wrote {args.summary_csv}')
    print(f'[DONE] wrote {args.summary_json}')
    print(f'[SUMMARY] detected={summary["face_detected_count"]}/{summary["total_images"]} ({summary["face_detected_rate"]:.4f}), kept={summary["kept_count"]}/{summary["total_images"]} ({summary["kept_rate"]:.4f})')


if __name__ == '__main__':
    main()
