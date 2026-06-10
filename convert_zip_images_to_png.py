#!/usr/bin/env python3
"""Convert all images in a zip archive to PNG files in a new zip archive."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def png_name(name: str) -> str:
    path = Path(name)
    return str(path.with_suffix(".png")).replace("\\", "/")


def convert_zip_to_png(input_zip: Path, output_zip: Path, compress_level: int) -> tuple[int, int]:
    converted = 0
    copied = 0
    used_names: set[str] = set()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_zip, "r") as src, zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_STORED
    ) as dst:
        members = src.infolist()
        total = len(members)

        for idx, info in enumerate(members, start=1):
            if info.is_dir():
                continue

            name = info.filename
            ext = Path(name).suffix.lower()

            if ext not in IMAGE_EXTENSIONS:
                new_info = zipfile.ZipInfo(filename=name, date_time=info.date_time)
                new_info.compress_type = zipfile.ZIP_STORED
                new_info.external_attr = info.external_attr
                dst.writestr(new_info, src.read(info))
                copied += 1
                continue

            out_name = png_name(name)
            if out_name in used_names:
                raise ValueError(f"duplicate PNG output path: {out_name}")
            used_names.add(out_name)

            with src.open(info) as fp:
                with Image.open(fp) as img:
                    img = ImageOps.exif_transpose(img)
                    if img.mode not in {"RGB", "L"}:
                        img = img.convert("RGB")

                    out = io.BytesIO()
                    img.save(out, format="PNG", compress_level=compress_level, optimize=False)

            new_info = zipfile.ZipInfo(filename=out_name, date_time=info.date_time)
            new_info.compress_type = zipfile.ZIP_STORED
            new_info.external_attr = info.external_attr
            dst.writestr(new_info, out.getvalue())
            converted += 1

            if converted % 500 == 0:
                print(f"converted {converted} images ({idx}/{total} zip entries)", flush=True)

    return converted, copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_zip",
        nargs="?",
        default="downloaded_celebvhq/image_dataset_256.zip",
        type=Path,
        help="Path to the source zip archive.",
    )
    parser.add_argument(
        "output_zip",
        nargs="?",
        default="downloaded_celebvhq/image_dataset_256_png.zip",
        type=Path,
        help="Path for the PNG output zip archive.",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=0,
        choices=range(10),
        metavar="[0-9]",
        help="PNG compression level. 0 matches StyleGAN2-ADA dataset_tool speed preference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_zip = args.input_zip.resolve()
    output_zip = args.output_zip.resolve()

    if input_zip == output_zip:
        raise ValueError("input_zip and output_zip must be different paths")

    converted, copied = convert_zip_to_png(input_zip, output_zip, args.compress_level)
    print(f"done: converted={converted}, copied_non_images={copied}, output={output_zip}")


if __name__ == "__main__":
    main()
