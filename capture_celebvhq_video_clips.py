"""
Build an image dataset from already-downloaded CelebV-HQ clip videos.

For each clip in celebvhq_info.json, this script:
1. Finds <clip_id>.mp4 under downloaded_celebvhq/videos/videos.
2. Captures the middle frame of the local clip with ffmpeg.
3. Resizes the full frame to a fixed square image.
5. Writes a CSV row with image path, bbox, and clip attributes.
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = BASE_DIR / "celebvhq_info.json"
DEFAULT_VIDEO_ROOT = BASE_DIR / "downloaded_celebvhq" / "videos" / "videos"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "downloaded_celebvhq" / "image_dataset_from_videos_512"
DEFAULT_CSV_PATH = BASE_DIR / "downloaded_celebvhq" / "image_dataset_from_videos_512.csv"
DEFAULT_FAILURE_CSV_PATH = (
    BASE_DIR / "downloaded_celebvhq" / "image_dataset_from_videos_512_failures.csv"
)
DEFAULT_FFMPEG = Path("/home/sh0408sh/miniconda3/bin/ffmpeg")
DEFAULT_FFPROBE = Path("/home/sh0408sh/miniconda3/bin/ffprobe")


def log(message):
    print(message, flush=True)


def expand_bbox(bbox, ratio=0.02):
    top, bottom, left, right = bbox
    return (
        max(top - ratio, 0.0),
        min(bottom + ratio, 1.0),
        max(left - ratio, 0.0),
        min(right + ratio, 1.0),
    )


def denorm_bbox(bbox, height, width):
    top, bottom, left, right = bbox
    return (
        round(top * height),
        round(bottom * height),
        round(left * width),
        round(right * width),
    )


def to_square_and_clip(bbox, height, width):
    top, bottom, left, right = bbox
    h = bottom - top
    w = right - left
    side = min(h, w)

    center_y = (top + bottom) // 2
    center_x = (left + right) // 2
    half = side // 2

    top = max(center_y - half, 0)
    bottom = min(center_y + half, height)
    left = max(center_x - half, 0)
    right = min(center_x + half, width)

    if bottom - top != right - left:
        side = min(bottom - top, right - left)
        bottom = top + side
        right = left + side

    return top, bottom, left, right


def run_cmd(cmd, timeout=None):
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def get_duration_sec(video_path, ffprobe):
    cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    result = run_cmd(cmd, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    text = result.stdout.decode("utf-8", errors="replace").strip()
    duration = float(text)
    if duration <= 0:
        raise RuntimeError(f"invalid duration: {duration}")
    return duration


def read_frame_png(video_path, target_sec, ffmpeg):
    cmd = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{target_sec:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    result = run_cmd(cmd, timeout=60)
    if result.returncode != 0 or not result.stdout:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or "ffmpeg returned no frame")
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")


def load_tasks(json_path, video_root, output_root, image_ext):
    with open(json_path, "r") as f:
        data = json.load(f)

    appearance_names = data["meta_info"]["appearance_mapping"]
    action_names = data["meta_info"]["action_mapping"]
    tasks = []

    for clip_idx, (clip_id, item) in enumerate(data["clips"].items()):
        bbox = [
            item["bbox"]["top"],
            item["bbox"]["bottom"],
            item["bbox"]["left"],
            item["bbox"]["right"],
        ]
        tasks.append(
            {
                "clip_idx": clip_idx,
                "clip_id": clip_id,
                "ytb_id": item["ytb_id"],
                "video_path": str(video_root / f"{clip_id}.mp4"),
                "image_path": str(output_root / f"{clip_id}.{image_ext}"),
                "source_start_sec": item["duration"]["start_sec"],
                "source_end_sec": item["duration"]["end_sec"],
                "bbox": bbox,
                "appearance": item["attributes"].get("appearance", []),
                "action": item["attributes"].get("action", []),
                "emotion": item["attributes"].get("emotion", {}),
                "version": item.get("version", ""),
            }
        )

    return tasks, appearance_names, action_names


def emotion_at_midpoint(emotion_info, duration_sec):
    if not isinstance(emotion_info, dict):
        return ""
    labels = emotion_info.get("labels")
    if isinstance(labels, str):
        return labels
    if not isinstance(labels, list) or not labels:
        return ""

    mid = duration_sec / 2.0 if duration_sec > 0 else None
    if mid is not None:
        for label in labels:
            start = float(label.get("start_sec", 0))
            end = float(label.get("end_sec", 0))
            if start <= mid <= end:
                return label.get("emotion", "")
    return labels[0].get("emotion", "")


def row_from_task(task, appearance_names, action_names, image_size, duration_sec, crop_box):
    row = {
        "image_path": task["image_path"],
        "image_name": Path(task["image_path"]).name,
        "clip_id": task["clip_id"],
        "clip_idx": task["clip_idx"],
        "ytb_id": task["ytb_id"],
        "video_path": task["video_path"],
        "source_start_sec": task["source_start_sec"],
        "source_end_sec": task["source_end_sec"],
        "local_clip_duration_sec": round(duration_sec, 6),
        "image_size": image_size,
        "version": task["version"],
        "bbox_top": task["bbox"][0],
        "bbox_bottom": task["bbox"][1],
        "bbox_left": task["bbox"][2],
        "bbox_right": task["bbox"][3],
        "crop_top_px": crop_box[0],
        "crop_bottom_px": crop_box[1],
        "crop_left_px": crop_box[2],
        "crop_right_px": crop_box[3],
        "emotion_sep_flag": task["emotion"].get("sep_flag", "")
        if isinstance(task["emotion"], dict)
        else "",
        "emotion_label": emotion_at_midpoint(task["emotion"], duration_sec),
    }

    for idx, name in enumerate(appearance_names):
        row[f"appearance_{name}"] = task["appearance"][idx] if idx < len(task["appearance"]) else ""
    for idx, name in enumerate(action_names):
        row[f"action_{name}"] = task["action"][idx] if idx < len(task["action"]) else ""

    return row


def process_task(task, appearance_names, action_names, args_dict):
    video_path = Path(task["video_path"])
    image_path = Path(task["image_path"])

    if not video_path.exists():
        return {
            "ok": False,
            "clip_id": task["clip_id"],
            "video_path": str(video_path),
            "reason": "missing_video",
            "message": "video file does not exist",
        }

    try:
        if image_path.exists() and args_dict["skip_existing"]:
            duration_sec = get_duration_sec(video_path, args_dict["ffprobe"])
            crop_box = ("", "", "", "")
            row = row_from_task(
                task,
                appearance_names,
                action_names,
                args_dict["image_size"],
                duration_sec,
                crop_box,
            )
            return {"ok": True, "skipped_existing": True, "row": row}

        duration_sec = get_duration_sec(video_path, args_dict["ffprobe"])
        target_sec = max(0.0, duration_sec / 2.0)
        frame = read_frame_png(video_path, target_sec, args_dict["ffmpeg"])
        width, height = frame.size
        crop_box = (0, height, 0, width)

        image = frame.resize(
            (args_dict["image_size"], args_dict["image_size"]),
            Image.Resampling.LANCZOS,
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, quality=args_dict["jpeg_quality"], optimize=True)

        row = row_from_task(
            task,
            appearance_names,
            action_names,
            args_dict["image_size"],
            duration_sec,
            crop_box,
        )
        return {"ok": True, "skipped_existing": False, "row": row}
    except Exception as exc:
        return {
            "ok": False,
            "clip_id": task["clip_id"],
            "video_path": str(video_path),
            "reason": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }


def write_header_if_needed(path, fieldnames):
    exists = path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer


def load_completed_clip_ids(csv_path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    try:
        with open(csv_path, "r", newline="") as f:
            return {row["clip_id"] for row in csv.DictReader(f) if row.get("clip_id")}
    except Exception:
        log(f"[Warning] Could not read existing CSV for resume: {csv_path}")
        return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--failure-csv-path", type=Path, default=DEFAULT_FAILURE_CSV_PATH)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--bbox-expand", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--image-ext", default="jpg")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    start = time.time()
    log(f"[Start] {datetime.now().isoformat(timespec='seconds')}")
    log(f"[Video root] {args.video_root}")
    log(f"[Output root] {args.output_root}")
    log(f"[CSV] {args.csv_path}")
    log(f"[Workers] {args.workers}")

    tasks, appearance_names, action_names = load_tasks(
        args.json_path, args.video_root, args.output_root, args.image_ext
    )
    completed_clip_ids = load_completed_clip_ids(args.csv_path)
    if completed_clip_ids:
        before = len(tasks)
        tasks = [
            task
            for task in tasks
            if not (
                task["clip_id"] in completed_clip_ids
                and Path(task["image_path"]).exists()
            )
        ]
        log(f"[Resume] skipped completed tasks from CSV: {before - len(tasks)}")
    if args.limit > 0:
        tasks = tasks[: args.limit]

    log(f"[Tasks] {len(tasks)}")
    if not tasks:
        log("[Done] no remaining tasks")
        return
    fieldnames = list(
        row_from_task(tasks[0], appearance_names, action_names, args.image_size, 0.0, ("", "", "", "")).keys()
    )
    failure_fields = ["clip_id", "video_path", "reason", "message", "traceback"]

    csv_file, writer = write_header_if_needed(args.csv_path, fieldnames)
    failure_file, failure_writer = write_header_if_needed(args.failure_csv_path, failure_fields)

    ok_count = 0
    skip_count = 0
    fail_count = 0
    args_dict = {
        "ffmpeg": args.ffmpeg,
        "ffprobe": args.ffprobe,
        "image_size": args.image_size,
        "jpeg_quality": args.jpeg_quality,
        "bbox_expand": args.bbox_expand,
        "skip_existing": args.skip_existing,
    }

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_task, task, appearance_names, action_names, args_dict)
                for task in tasks
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result["ok"]:
                    writer.writerow(result["row"])
                    ok_count += 1
                    if result.get("skipped_existing"):
                        skip_count += 1
                else:
                    failure_writer.writerow({k: result.get(k, "") for k in failure_fields})
                    fail_count += 1

                if done % args.progress_every == 0 or done == len(tasks):
                    csv_file.flush()
                    failure_file.flush()
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = (len(tasks) - done) / rate if rate > 0 else 0
                    log(
                        "[Progress] "
                        f"done={done}/{len(tasks)} ok={ok_count} "
                        f"skipped_existing={skip_count} failed={fail_count} "
                        f"rate={rate:.2f}/sec eta_sec={remaining:.0f}"
                    )
    finally:
        csv_file.close()
        failure_file.close()

    elapsed = time.time() - start
    log(
        f"[Done] ok={ok_count} skipped_existing={skip_count} failed={fail_count} "
        f"elapsed_sec={elapsed:.2f}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("[Interrupted]")
        sys.exit(130)
