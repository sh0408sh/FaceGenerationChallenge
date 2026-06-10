"""
CelebV-HQ YouTube raw downloader + middle-frame face cropper.

Compared with mk_image6.py:
1. Save video IDs skipped because all required images already exist.
2. Load saved permanent failures and saved skip cases before processing.
3. If a ytb_id is already permanent-failed or skip-saved, skip download immediately.
"""

import json
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime

import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "celebvhq_info.json")
COOKIES_PATH = os.path.join(BASE_DIR, "cookies.txt")
RAW_VID_ROOT = os.path.join(BASE_DIR, "downloaded_celebvhq", "raw")
IMAGE_ROOT = os.path.join(BASE_DIR, "downloaded_celebvhq", "image_dataset")
PERMANENT_FAIL_PATH = os.path.join(
    BASE_DIR, "downloaded_celebvhq", "permanent_failures.json"
)
TEMPORARY_FAIL_PATH = os.path.join(
    BASE_DIR, "downloaded_celebvhq", "temporary_failures.json"
)
DOWNLOAD_SKIP_PATH = os.path.join(
    BASE_DIR, "downloaded_celebvhq", "download_skip_cases.json"
)

FRAMES_PER_CLIP = 1
IMAGE_SIZE = 512
TEMPORARY_FAILURE_LIMIT = 20
DELETE_RAW_AFTER_PROCESS = True
PROXY = None


def log(msg):
    print(msg, flush=True)


def load_saved_items(path):
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        log(f"[Load state failed] path={path}")
        log(traceback.format_exc())
        return {}

    if isinstance(data, dict) and isinstance(data.get("items"), dict):
        return data["items"]

    if isinstance(data, dict):
        return data

    return {}


def write_json_atomic(path, payload):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def save_state(permanent_failures, temporary_failures, download_skip_cases):
    os.makedirs(os.path.dirname(PERMANENT_FAIL_PATH), exist_ok=True)
    payload_common = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "temporary_failure_limit": TEMPORARY_FAILURE_LIMIT,
    }

    payloads = [
        (
            PERMANENT_FAIL_PATH,
            {
                **payload_common,
                "count": len(permanent_failures),
                "items": permanent_failures,
            },
        ),
        (
            TEMPORARY_FAIL_PATH,
            {
                **payload_common,
                "count": len(temporary_failures),
                "items": temporary_failures,
            },
        ),
        (
            DOWNLOAD_SKIP_PATH,
            {
                **payload_common,
                "count": len(download_skip_cases),
                "items": download_skip_cases,
            },
        ),
    ]

    try:
        for path, payload in payloads:
            write_json_atomic(path, payload)
            log(f"[Saved state] {path}")
    except Exception:
        log("[Save state failed]")
        log(traceback.format_exc())
        raise


def classify_download_error(output):
    text = output.lower()

    temporary_keywords = [
        "bot",
        "sign in to confirm you",
        "not a bot",
        "rate-limited",
        "rate limited",
        "rate limit",
        "try again later",
        "too many requests",
        "http error 403",
        "403 forbidden",
        " 403",
    ]
    permanent_keywords = [
        "private video",
        "this video is private",
        "account has been terminated",
        "account terminated",
        "terminated",
        "removed by the uploader",
        "has been removed",
        "no longer available",
        "blocked it on copyright grounds",
        "copyright grounds",
        "not made this video available in your country",
        "video unavailable",
        "unavailable",
        "age-restricted",
        "age restricted",
        "age_restricted",
    ]

    for keyword in temporary_keywords:
        if keyword in text:
            return "temporary", keyword

    for keyword in permanent_keywords:
        if keyword in text:
            return "permanent", keyword

    return "temporary", "unknown_download_error"


def run_cmd(cmd):
    log(" ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


def download(video_path, ytb_id, proxy=None):
    if os.path.exists(video_path):
        log(f"[Download skip existing] ytb_id={ytb_id}, path={video_path}")
        return True, None, "already_exists"

    cmd = [
        "yt-dlp",
        "--cookies",
        COOKIES_PATH,
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio",
        "--skip-unavailable-fragments",
        "--merge-output-format",
        "mp4",
        "https://www.youtube.com/watch?v=" + ytb_id,
        "--output",
        video_path,
        "--external-downloader",
        "aria2c",
        "--external-downloader-args",
        "aria2c:-x 16 -k 1M",
    ]

    if proxy is not None:
        cmd.insert(1, "--proxy")
        cmd.insert(2, proxy)

    download_start = datetime.now()
    log(
        f"[Download start] ytb_id={ytb_id}, "
        f"started_at={download_start.isoformat(timespec='seconds')}"
    )
    result = run_cmd(cmd)
    download_end = datetime.now()
    elapsed_sec = (download_end - download_start).total_seconds()
    log(
        f"[Download end] ytb_id={ytb_id}, returncode={result.returncode}, "
        f"ended_at={download_end.isoformat(timespec='seconds')}, "
        f"elapsed_sec={elapsed_sec:.2f}"
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")

    if result.returncode == 0 and os.path.exists(video_path):
        return True, None, "downloaded"

    kind, reason = classify_download_error(output)
    log(f"[Download failed] ytb_id={ytb_id}, kind={kind}, reason={reason}")
    log("[Download error output begin]")
    log(output.rstrip())
    log("[Download error output end]")
    return False, kind, reason, output


def expand_bbox(bbox, ratio=0.02):
    top, bottom, left, right = bbox
    return (
        max(top - ratio, 0),
        min(bottom + ratio, 1),
        max(left - ratio, 0),
        min(right + ratio, 1),
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

    return top, bottom, left, right


def expected_image_paths(image_root, idx, frames_per_clip):
    return [
        os.path.join(image_root, f"frame_{idx * frames_per_clip + i:06d}.jpg")
        for i in range(frames_per_clip)
    ]


def all_expected_images_exist(clips, image_root, frames_per_clip):
    for idx, _, _, _ in clips:
        expected_images = expected_image_paths(image_root, idx, frames_per_clip)
        if not all(os.path.exists(p) for p in expected_images):
            return False
    return True


def capture_face_frames(
    raw_vid_path,
    image_root,
    frame_start_idx,
    bbox,
    time_range,
    num_frames=1,
    image_size=512,
):
    os.makedirs(image_root, exist_ok=True)

    cap = cv2.VideoCapture(raw_vid_path)

    if not cap.isOpened():
        log(f"[Cannot open video] {raw_vid_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        log(f"[Invalid video metadata] {raw_vid_path}")
        cap.release()
        return 0

    start_sec, end_sec = time_range

    if num_frames == 1:
        target_secs = [(start_sec + end_sec) / 2]
    else:
        target_secs = [
            start_sec + i * (end_sec - start_sec) / (num_frames - 1)
            for i in range(num_frames)
        ]

    saved = 0

    for j, target_sec in enumerate(target_secs):
        image_idx = frame_start_idx + j
        out_path = os.path.join(image_root, f"frame_{image_idx:06d}.jpg")

        if os.path.exists(out_path):
            log(f"[Skip image exists] {out_path}")
            continue

        frame_idx = int(target_sec * fps)
        frame_idx = max(0, min(frame_idx, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if not ok:
            log(
                f"[Frame read failed] video={raw_vid_path}, "
                f"sec={target_sec}, frame_idx={frame_idx}"
            )
            continue

        height, width = frame.shape[:2]

        expanded = expand_bbox(bbox, ratio=0.02)
        pixel_bbox = denorm_bbox(expanded, height, width)
        top, bottom, left, right = to_square_and_clip(pixel_bbox, height, width)

        if bottom <= top or right <= left:
            log(f"[Invalid crop] {raw_vid_path}")
            continue

        face = frame[top:bottom, left:right]

        if face.size == 0:
            log(f"[Empty crop] {raw_vid_path}")
            continue

        face = cv2.resize(face, (image_size, image_size))

        try:
            ok = cv2.imwrite(out_path, face)
        except Exception:
            log(f"[Image write exception] {out_path}")
            log(traceback.format_exc())
            continue

        if ok:
            log(f"[Saved image] {out_path}")
            saved += 1
        else:
            log(f"[Image write failed] {out_path}")

    cap.release()
    return saved


def remove_raw_video(raw_vid_path):
    if not os.path.exists(raw_vid_path):
        log(f"[Delete raw skip missing] path={raw_vid_path}")
        return
    delete_start = datetime.now()
    log(
        f"[Delete raw start] path={raw_vid_path}, "
        f"started_at={delete_start.isoformat(timespec='seconds')}"
    )
    os.remove(raw_vid_path)
    delete_end = datetime.now()
    elapsed_sec = (delete_end - delete_start).total_seconds()
    log(
        f"[Delete raw end] path={raw_vid_path}, "
        f"ended_at={delete_end.isoformat(timespec='seconds')}, "
        f"elapsed_sec={elapsed_sec:.2f}"
    )


def load_data(file_path):
    with open(file_path, "r") as f:
        data_dict = json.load(f)

    items = []

    for key, val in data_dict["clips"].items():
        save_name = key + ".mp4"
        ytb_id = val["ytb_id"]

        time_range = (
            val["duration"]["start_sec"],
            val["duration"]["end_sec"],
        )

        bbox = [
            val["bbox"]["top"],
            val["bbox"]["bottom"],
            val["bbox"]["left"],
            val["bbox"]["right"],
        ]

        items.append((ytb_id, save_name, time_range, bbox))

    return items


def group_by_ytb_id(data):
    grouped = defaultdict(list)

    for idx, (ytb_id, save_vid_name, time_range, bbox) in enumerate(data):
        grouped[ytb_id].append((idx, save_vid_name, time_range, bbox))

    return grouped


def make_record(ytb_id, clips, status, reason=None, message=None):
    return {
        "ytb_id": ytb_id,
        "num_clips": len(clips),
        "status": status,
        "reason": reason,
        "message": message,
    }


def process_one_group(ytb_id, clips, permanent_failures, download_skip_cases):
    if ytb_id in permanent_failures:
        log(f"[Skip known permanent] ytb_id={ytb_id}, clips={len(clips)}")
        return {
            "status": "pre_skipped",
            "skip_kind": "permanent",
            "reason": "known_permanent_failure",
            "saved_images": 0,
        }

    if ytb_id in download_skip_cases:
        log(f"[Skip known complete] ytb_id={ytb_id}, clips={len(clips)}")
        return {
            "status": "pre_skipped",
            "skip_kind": "complete_images",
            "reason": "known_download_skip_case",
            "saved_images": 0,
        }

    raw_vid_path = os.path.join(RAW_VID_ROOT, ytb_id + ".mp4")

    if all_expected_images_exist(clips, IMAGE_ROOT, FRAMES_PER_CLIP):
        log(f"[Skip complete images] ytb_id={ytb_id}, clips={len(clips)}")
        return {
            "status": "download_skipped",
            "skip_kind": "complete_images",
            "reason": "all_expected_images_exist",
            "saved_images": 0,
        }

    downloaded = download(raw_vid_path, ytb_id, PROXY)
    if downloaded[0] is not True:
        _, kind, reason, message = downloaded
        return {
            "status": "download_failed",
            "failure_kind": kind,
            "reason": reason,
            "message": message,
        }

    group_success = False
    saved_total = 0

    for idx, save_vid_name, time_range, bbox in clips:
        expected_images = expected_image_paths(IMAGE_ROOT, idx, FRAMES_PER_CLIP)

        if all(os.path.exists(p) for p in expected_images):
            log(f"[Skip existing clip] idx={idx}, ytb_id={ytb_id}, clip={save_vid_name}")
            group_success = True
            continue

        frame_start_idx = idx * FRAMES_PER_CLIP
        saved = capture_face_frames(
            raw_vid_path=raw_vid_path,
            image_root=IMAGE_ROOT,
            frame_start_idx=frame_start_idx,
            bbox=bbox,
            time_range=time_range,
            num_frames=FRAMES_PER_CLIP,
            image_size=IMAGE_SIZE,
        )

        saved_total += saved
        if saved > 0:
            group_success = True

    if DELETE_RAW_AFTER_PROCESS:
        remove_raw_video(raw_vid_path)

    if not group_success:
        return {
            "status": "processing_failed",
            "failure_kind": "permanent",
            "reason": "no_image_saved",
            "message": "download succeeded, but no image was saved",
            "saved_images": 0,
        }

    return {
        "status": "success",
        "reason": "processed",
        "saved_images": saved_total,
    }


def main():
    os.makedirs(RAW_VID_ROOT, exist_ok=True)
    os.makedirs(IMAGE_ROOT, exist_ok=True)

    data = load_data(JSON_PATH)
    grouped = group_by_ytb_id(data)
    tasks = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=False)

    permanent_failures = load_saved_items(PERMANENT_FAIL_PATH)
    download_skip_cases = load_saved_items(DOWNLOAD_SKIP_PATH)
    temporary_failures = {}

    log(f"num clips: {len(data)}")
    log(f"num unique YouTube videos: {len(tasks)}")
    log(f"loaded permanent failures: {len(permanent_failures)}")
    log(f"loaded download skip cases: {len(download_skip_cases)}")
    log(f"temporary failure limit: {TEMPORARY_FAILURE_LIMIT}")

    success_groups = 0
    pre_skipped_groups = 0
    download_skipped_groups = 0
    saved_images = 0

    for group_idx, (ytb_id, clips) in enumerate(tasks, start=1):
        log(f"[Group {group_idx}/{len(tasks)}] ytb_id={ytb_id}, clips={len(clips)}")
        result = process_one_group(
            ytb_id=ytb_id,
            clips=clips,
            permanent_failures=permanent_failures,
            download_skip_cases=download_skip_cases,
        )

        if result["status"] == "pre_skipped":
            pre_skipped_groups += 1
            continue

        if result["status"] == "download_skipped":
            download_skipped_groups += 1
            download_skip_cases[ytb_id] = make_record(
                ytb_id=ytb_id,
                clips=clips,
                status=result["status"],
                reason=result.get("reason"),
            )
            save_state(permanent_failures, temporary_failures, download_skip_cases)
            continue

        if result["status"] == "success":
            success_groups += 1
            saved_images += result.get("saved_images", 0)
            continue

        record = make_record(
            ytb_id=ytb_id,
            clips=clips,
            status=result["status"],
            reason=result.get("reason"),
            message=result.get("message"),
        )

        if result.get("failure_kind") == "permanent":
            permanent_failures[ytb_id] = record
        else:
            temporary_failures[ytb_id] = record

        save_state(permanent_failures, temporary_failures, download_skip_cases)

        log(
            f"[Failure count] permanent={len(permanent_failures)}, "
            f"temporary={len(temporary_failures)}, "
            f"download_skip={len(download_skip_cases)}"
        )

        if len(temporary_failures) >= TEMPORARY_FAILURE_LIMIT:
            log("[Stop] temporary errors reached limit. Saving state and exiting.")
            save_state(permanent_failures, temporary_failures, download_skip_cases)
            sys.exit(1)

    save_state(permanent_failures, temporary_failures, download_skip_cases)
    log(
        "Done. "
        f"success_groups={success_groups}, "
        f"pre_skipped_groups={pre_skipped_groups}, "
        f"download_skipped_groups={download_skipped_groups}, "
        f"permanent_failures={len(permanent_failures)}, "
        f"temporary_failures={len(temporary_failures)}, "
        f"download_skip_cases={len(download_skip_cases)}, "
        f"saved_images={saved_images}"
    )
    log(f"Permanent failures saved to {PERMANENT_FAIL_PATH}")
    log(f"Temporary failures saved to {TEMPORARY_FAIL_PATH}")
    log(f"Download skip cases saved to {DOWNLOAD_SKIP_PATH}")


if __name__ == "__main__":
    main()
