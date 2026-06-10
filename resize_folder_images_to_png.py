#!/usr/bin/env python3
"""Resize images in a folder and save them as PNG files in another folder."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def convert_one(src_path: str, input_root: str, output_root: str, size: tuple[int, int]) -> tuple[bool, str]:
    src = Path(src_path)
    rel = src.relative_to(input_root)
    dst = Path(output_root) / rel.with_suffix(".png")

    if dst.exists():
        return True, "skipped_existing"

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in {"RGB", "L"}:
                img = img.convert("RGB")
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(dst, format="PNG", compress_level=0, optimize=False)
        return True, "converted"
    except Exception as exc:
        return False, f"{src}: {type(exc).__name__}: {exc}"


def iter_images(input_root: Path) -> list[str]:
    return [
        str(path)
        for path in sorted(input_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if input_root == output_root:
        raise ValueError("input_root and output_root must be different paths")

    images = iter_images(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"input={input_root}", flush=True)
    print(f"output={output_root}", flush=True)
    print(f"images={len(images)} workers={args.workers} size={args.width}x{args.height}", flush=True)

    start = time.time()
    converted = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                convert_one,
                src,
                str(input_root),
                str(output_root),
                (args.width, args.height),
            )
            for src in images
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            ok, status = future.result()
            if ok and status == "converted":
                converted += 1
            elif ok and status == "skipped_existing":
                skipped += 1
            else:
                failed += 1
                print(f"[Failed] {status}", flush=True)

            if done % args.progress_every == 0 or done == len(images):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(images) - done) / rate if rate > 0 else 0
                print(
                    f"[Progress] done={done}/{len(images)} converted={converted} "
                    f"skipped={skipped} failed={failed} rate={rate:.2f}/sec eta_sec={remaining:.0f}",
                    flush=True,
                )

    elapsed = time.time() - start
    print(
        f"[Done] converted={converted} skipped={skipped} failed={failed} elapsed_sec={elapsed:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
