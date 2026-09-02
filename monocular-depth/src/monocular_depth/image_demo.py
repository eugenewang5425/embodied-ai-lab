from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from .config import PROJECT_ROOT, VARIANTS
from .io import read_image, save_result
from .model import load_model
from .pipeline import DepthPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run relative or metric depth on one RGB image")
    parser.add_argument("--input", type=Path, required=True, help="input image path")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "images")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--variant", choices=VARIANTS, default="relative")
    parser.add_argument(
        "--calibration", type=Path, help="accepted calibration for THIS camera mode"
    )
    args = parser.parse_args()

    image = read_image(args.input)
    if image is None:
        raise SystemExit(f"Could not read image: {args.input}")

    model, device = load_model(variant=args.variant)
    pipeline = DepthPipeline(model, args.variant, args.input_size, args.calibration)
    started = perf_counter()
    corrected, depth, metadata = pipeline.process(image)
    elapsed = perf_counter() - started
    metadata["source_image"] = str(args.input.resolve())
    paths = save_result(
        args.output_dir,
        f"{args.input.stem}_{args.variant}",
        corrected,
        depth,
        variant=args.variant,
        metadata=metadata,
        raw_image=image if args.calibration else None,
    )

    print(f"Device: {device}")
    print(f"Preprocessing + inference (single cold frame): {elapsed * 1000:.1f} ms")
    print(f"Variant: {args.variant}; metric accuracy is NOT validated")
    for path in paths:
        print(path.resolve())
