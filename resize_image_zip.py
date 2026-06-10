#!/usr/bin/env python3
"""Resize all images in a zip archive and write them to a new zip archive."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resize_zip(input_zip: Path, output_zip: Path, size: tuple[int, int]) -> tuple[int, int]:
    resized = 0
    copied = 0
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_zip, "r") as src, zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as dst:
        members = src.infolist()
        total = len(members)

        for idx, info in enumerate(members, start=1):
            name = info.filename

            if info.is_dir():
                dst.writestr(info, b"")
                continue

            ext = Path(name).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                dst.writestr(info, src.read(info))
                copied += 1
                continue

            with src.open(info) as fp:
                with Image.open(fp) as img:
                    img = ImageOps.exif_transpose(img)
                    resized_img = img.resize(size, Image.Resampling.LANCZOS)

                    out = io.BytesIO()
                    save_kwargs = {}
                    fmt = img.format or ("JPEG" if ext in {".jpg", ".jpeg"} else ext.lstrip(".").upper())

                    if fmt.upper() == "JPEG":
                        if resized_img.mode not in {"RGB", "L"}:
                            resized_img = resized_img.convert("RGB")
                        save_kwargs.update({"quality": 95, "optimize": True})

                    resized_img.save(out, format=fmt, **save_kwargs)

            new_info = zipfile.ZipInfo(filename=name, date_time=info.date_time)
            new_info.compress_type = zipfile.ZIP_DEFLATED
            new_info.external_attr = info.external_attr
            dst.writestr(new_info, out.getvalue())
            resized += 1

            if resized % 500 == 0:
                print(f"resized {resized} images ({idx}/{total} zip entries)")

    return resized, copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_zip",
        nargs="?",
        default="downloaded_celebvhq/image_dataset.zip",
        type=Path,
        help="Path to the source zip archive.",
    )
    parser.add_argument(
        "output_zip",
        nargs="?",
        default="downloaded_celebvhq/image_dataset_216.zip",
        type=Path,
        help="Path for the resized output zip archive.",
    )
    parser.add_argument("--width", type=int, default=216)
    parser.add_argument("--height", type=int, default=216)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_zip = args.input_zip.resolve()
    output_zip = args.output_zip.resolve()

    if input_zip == output_zip:
        raise ValueError("input_zip and output_zip must be different paths")

    resized, copied = resize_zip(input_zip, output_zip, (args.width, args.height))
    print(f"done: resized={resized}, copied_non_images={copied}, output={output_zip}")


if __name__ == "__main__":
    main()
