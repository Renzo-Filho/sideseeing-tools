from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import argparse
import cv2
import numpy as np


ImageShape = tuple[int, ...]
BoundingBox = tuple[int, int, int, int]
ProcessResult = Literal["processed", "skipped", "failed"]

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
ANONYMIZATION_MODES = ("pixel", "blur")
OUTPUT_FORMATS = ("original", "jpg", "png")
SAVE_PRESETS = ("default", "tiny")
CASCADE_PATH = Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(str(CASCADE_PATH))
PIXEL_DIVISIONS = 4
MIN_PIXEL_BLOCK_SIZE = 25
MIN_FACE_HEIGHT = 20
MIN_OBJECT_SIZE = 5
EXISTING_FACE_OVERLAP_THRESHOLD = 10
HEAD_REGION_RATIO = 0.20
BLUR_KERNEL_FACTOR = 0.5
DEFAULT_JPEG_QUALITY = 95
TINY_JPEG_QUALITY = 20
DEFAULT_PNG_COMPRESSION = 9
TINY_SAVE_BLUR_SIGMA = 0.7


@dataclass(frozen=True)
class AnonymizationConfig:
    input_root: Path
    face_masks_root: Path
    person_masks_root: Path
    plate_masks_root: Path | None
    output_root: Path
    mode: str
    output_format: str
    save_preset: str
    jpeg_quality: int
    png_compression: int
    save_blur_sigma: float


@dataclass
class ProcessingStats:
    processed_images: int = 0
    skipped_images: int = 0
    failed_images: int = 0


def empty_mask(target_shape: ImageShape) -> Any:
    return np.zeros(target_shape[:2], dtype=np.uint8)


def load_mask(
    mask_root: Path | None,
    relative_dir: Path,
    filename_stem: str,
    target_shape: ImageShape,
) -> Any:
    if mask_root is None:
        return empty_mask(target_shape)

    mask_path = mask_root / relative_dir / f"{filename_stem}_mask.png"
    if not mask_path.exists():
        return empty_mask(target_shape)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return empty_mask(target_shape)

    if mask.shape != target_shape[:2]:
        mask = cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask


def detect_faces_in_region(gray_region: Any) -> list[BoundingBox]:
    detections = FACE_CASCADE.detectMultiScale(
        gray_region,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(10, 10),
    )
    faces: list[BoundingBox] = []
    for detection in detections:
        face_x, face_y, face_width, face_height = (int(value) for value in detection)
        faces.append((face_x, face_y, face_width, face_height))
    return faces


def recover_missed_faces(
    image: Any,
    person_mask: Any,
    existing_face_mask: Any,
) -> Any:
    recovered_face_mask = np.zeros_like(person_mask)
    contours, _ = cv2.findContours(
        person_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        person_blob = np.zeros_like(person_mask)
        cv2.drawContours(person_blob, [contour], -1, 255, -1)

        overlap = cv2.bitwise_and(person_blob, existing_face_mask)
        if cv2.countNonZero(overlap) > EXISTING_FACE_OVERLAP_THRESHOLD:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if height < MIN_FACE_HEIGHT:
            continue

        gray_region = cv2.cvtColor(
            image[y : y + height, x : x + width],
            cv2.COLOR_BGR2GRAY,
        )
        faces = detect_faces_in_region(gray_region)
        if faces:
            for face_x, face_y, face_width, face_height in faces:
                top_left = (x + face_x, y + face_y)
                bottom_right = (x + face_x + face_width, y + face_y + face_height)
                cv2.rectangle(recovered_face_mask, top_left, bottom_right, 255, -1)
            continue

        head_height = int(height * HEAD_REGION_RATIO)
        head_region_mask = np.zeros_like(person_mask)
        cv2.rectangle(head_region_mask, (x, y), (x + width, y + head_height), 255, -1)
        recovered_head = cv2.bitwise_and(head_region_mask, person_blob)
        recovered_face_mask = cv2.bitwise_or(recovered_face_mask, recovered_head)

    return recovered_face_mask


def apply_pixelation(region: Any) -> Any:
    region_height, region_width = region.shape[:2]
    min_dimension = min(region_width, region_height)
    block_size = int(min_dimension / PIXEL_DIVISIONS)
    block_size = max(MIN_PIXEL_BLOCK_SIZE, block_size)
    block_size = min(min_dimension, block_size)
    block_size = max(1, block_size)

    downscaled = cv2.resize(
        region,
        (max(1, region_width // block_size), max(1, region_height // block_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(
        downscaled,
        (region_width, region_height),
        interpolation=cv2.INTER_NEAREST,
    )


def apply_blur(region: Any) -> Any:
    min_dimension = min(region.shape[1], region.shape[0])
    kernel_size = int(min_dimension * BLUR_KERNEL_FACTOR)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)
    return cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)


def apply_adaptive_anonymization(
    image: Any,
    combined_mask: Any,
    mode: str,
) -> Any:
    if not np.any(combined_mask > 0):
        return image

    output_image = image.copy()
    contours, _ = cv2.findContours(
        combined_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        region = output_image[y : y + height, x : x + width]
        min_dimension = min(region.shape[1], region.shape[0])
        if min_dimension < MIN_OBJECT_SIZE:
            continue

        processed_region = apply_pixelation(region) if mode == "pixel" else apply_blur(region)
        output_image[y : y + height, x : x + width] = processed_region

    return output_image


def build_combined_mask(
    image: Any,
    relative_image_path: Path,
    config: AnonymizationConfig,
) -> Any:
    relative_dir = relative_image_path.parent
    filename_stem = relative_image_path.stem

    face_mask = load_mask(config.face_masks_root, relative_dir, filename_stem, image.shape)
    person_mask = load_mask(config.person_masks_root, relative_dir, filename_stem, image.shape)
    plate_mask = load_mask(config.plate_masks_root, relative_dir, filename_stem, image.shape)
    recovered_face_mask = recover_missed_faces(image, person_mask, face_mask)

    combined_mask = cv2.bitwise_or(face_mask, plate_mask)
    return cv2.bitwise_or(combined_mask, recovered_face_mask)


def iter_input_images(input_root: Path):
    for image_path in sorted(input_root.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def resolve_output_suffix(source_suffix: str, output_format: str) -> str:
    if output_format == "original":
        return source_suffix.lower()
    if output_format == "jpg":
        return ".jpg"
    return ".png"


def build_output_path(relative_image_path: Path, config: AnonymizationConfig) -> Path:
    output_suffix = resolve_output_suffix(relative_image_path.suffix, config.output_format)
    return config.output_root / relative_image_path.with_suffix(output_suffix)


def prepare_image_for_save(image: Any, config: AnonymizationConfig) -> Any:
    if config.save_blur_sigma <= 0:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigmaX=config.save_blur_sigma, sigmaY=config.save_blur_sigma)


def build_imwrite_params(output_path: Path, config: AnonymizationConfig) -> list[int]:
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return [
            cv2.IMWRITE_JPEG_QUALITY,
            config.jpeg_quality,
            cv2.IMWRITE_JPEG_PROGRESSIVE,
            1,
            cv2.IMWRITE_JPEG_OPTIMIZE,
            1,
        ]
    if suffix == ".png":
        return [cv2.IMWRITE_PNG_COMPRESSION, config.png_compression]
    return []


def anonymize_image(image_path: Path, config: AnonymizationConfig) -> ProcessResult:
    image = cv2.imread(str(image_path))
    if image is None:
        return "skipped"

    relative_image_path = image_path.relative_to(config.input_root)
    combined_mask = build_combined_mask(image, relative_image_path, config)
    anonymized_image = apply_adaptive_anonymization(image, combined_mask, mode=config.mode)
    encoded_image = prepare_image_for_save(anonymized_image, config)

    output_path = build_output_path(relative_image_path, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), encoded_image, build_imwrite_params(output_path, config)):
        return "failed"
    return "processed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply adaptive anonymization to face and plate regions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Directory containing input images.")
    parser.add_argument("--faces", required=True, help="Directory containing face masks.")
    parser.add_argument("--persons", required=True, help="Directory containing person masks.")
    parser.add_argument("--plates", help="Directory containing plate masks.")
    parser.add_argument("--output", required=True, help="Directory for anonymized images.")
    parser.add_argument(
        "--mode",
        choices=ANONYMIZATION_MODES,
        default="pixel",
        help="Anonymization mode applied to the detected regions.",
    )
    parser.add_argument(
        "--save-preset",
        choices=SAVE_PRESETS,
        default="default",
        help="Output preset. 'tiny' exports highly compressed images for smaller files.",
    )
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMATS,
        default="original",
        help="Output image format. 'original' preserves the input extension unless a preset overrides it.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        help="JPEG quality from 0 to 100. Lower values produce smaller files.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        help="PNG compression from 0 to 9. Higher values usually produce smaller files.",
    )
    parser.add_argument(
        "--save-blur-sigma",
        type=float,
        help="Optional blur applied to the whole image before saving to reduce file size.",
    )
    return parser


def resolve_existing_directory(
    parser: argparse.ArgumentParser,
    raw_path: str,
    argument_name: str,
) -> Path:
    resolved_path = Path(raw_path).expanduser().resolve()
    if not resolved_path.is_dir():
        parser.error(f"{argument_name} must point to an existing directory: {resolved_path}")
    return resolved_path


def resolve_optional_directory(
    parser: argparse.ArgumentParser,
    raw_path: str | None,
    argument_name: str,
) -> Path | None:
    if not raw_path:
        return None
    return resolve_existing_directory(parser, raw_path, argument_name)


def resolve_output_directory(
    parser: argparse.ArgumentParser,
    raw_path: str,
    argument_name: str,
) -> Path:
    resolved_path = Path(raw_path).expanduser().resolve()
    if resolved_path.exists() and not resolved_path.is_dir():
        parser.error(f"{argument_name} must point to a directory path: {resolved_path}")
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def validate_jpeg_quality(parser: argparse.ArgumentParser, jpeg_quality: int) -> int:
    if 0 <= jpeg_quality <= 100:
        return jpeg_quality
    parser.error(f"--jpeg-quality must be between 0 and 100: {jpeg_quality}")


def validate_png_compression(parser: argparse.ArgumentParser, png_compression: int) -> int:
    if 0 <= png_compression <= 9:
        return png_compression
    parser.error(f"--png-compression must be between 0 and 9: {png_compression}")


def validate_save_blur_sigma(parser: argparse.ArgumentParser, save_blur_sigma: float) -> float:
    if save_blur_sigma >= 0:
        return save_blur_sigma
    parser.error(f"--save-blur-sigma must be greater than or equal to 0: {save_blur_sigma}")


def resolve_output_format(save_preset: str, output_format: str) -> str:
    if save_preset == "tiny" and output_format == "original":
        return "jpg"
    return output_format


def parse_args() -> AnonymizationConfig:
    parser = build_parser()
    args = parser.parse_args()
    jpeg_quality = args.jpeg_quality
    if jpeg_quality is None:
        jpeg_quality = TINY_JPEG_QUALITY if args.save_preset == "tiny" else DEFAULT_JPEG_QUALITY

    png_compression = args.png_compression
    if png_compression is None:
        png_compression = DEFAULT_PNG_COMPRESSION

    save_blur_sigma = args.save_blur_sigma
    if save_blur_sigma is None:
        save_blur_sigma = TINY_SAVE_BLUR_SIGMA if args.save_preset == "tiny" else 0.0

    return AnonymizationConfig(
        input_root=resolve_existing_directory(parser, args.input, "--input"),
        face_masks_root=resolve_existing_directory(parser, args.faces, "--faces"),
        person_masks_root=resolve_existing_directory(parser, args.persons, "--persons"),
        plate_masks_root=resolve_optional_directory(parser, args.plates, "--plates"),
        output_root=resolve_output_directory(parser, args.output, "--output"),
        mode=args.mode,
        output_format=resolve_output_format(args.save_preset, args.output_format),
        save_preset=args.save_preset,
        jpeg_quality=validate_jpeg_quality(parser, jpeg_quality),
        png_compression=validate_png_compression(parser, png_compression),
        save_blur_sigma=validate_save_blur_sigma(parser, save_blur_sigma),
    )


def validate_runtime() -> None:
    if FACE_CASCADE.empty():
        raise RuntimeError("Unable to load the Haar cascade classifier.")


def run(config: AnonymizationConfig) -> ProcessingStats:
    stats = ProcessingStats()
    for image_path in iter_input_images(config.input_root):
        result = anonymize_image(image_path, config)
        if result == "processed":
            stats.processed_images += 1
        elif result == "failed":
            stats.failed_images += 1
        else:
            stats.skipped_images += 1
    return stats


def main() -> int:
    config = parse_args()
    validate_runtime()

    print(f"Starting adaptive anonymization ({config.mode.upper()} mode)...")
    stats = run(config)
    print(
        "Completed. "
        f"Processed: {stats.processed_images}, "
        f"Skipped: {stats.skipped_images}, "
        f"Failed: {stats.failed_images}"
    )
    return 0 if stats.failed_images == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
