"""Lesson 24 bench: real Depth Anything V2 relative depth on the lesson-22 scene.

Renders the same 640x480 pinhole scene as lessons 22/23 (ground grid + one
vertical pole, K=[600,600,320,240], eye=(2.0,1.5,1.4) -> target=(3.0,3.0,0.0),
near plane 0.5 m) by reusing the main repo's per-pixel analytic ray caster,
shades it into a simple Lambertian RGB image (checkerboard ground, shaded pole
cylinder, optional distance fog), then runs the REAL Depth Anything V2 Small
checkpoint with the monocular-depth lab's own load/infer path (input size 518).

The metric depth Z, the rendered RGB and the model's relative map r_pred are
archived into outputs/affine_check_pinhole_<UTC stamp>/ for the main-repo
analysis:

  uv run python -m embodied_learning.experiments.real_depth_affine \
      --input <this npz> --output results/real_depth_affine_<date>

Run with the monocular-depth venv, NOT the main repo venv (torch stays out of
the main repo's test env):

  monocular-depth/.venv/Scripts/python.exe monocular-depth/bench/affine_check_pinhole.py

r_pred is the raw model output in arbitrary units. Depth Anything V2 is
expected to emit larger values for closer surfaces (disparity-like), i.e.
r ~ a*(1/Z) + b with a > 0; the sign convention is NOT pre-forced here - the
main-repo fit will report what the model actually does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]  # monocular-depth/
MAIN_SRC = PROJECT.parent / "src"  # main repo src/ (embodied_learning)
for _entry in (str(PROJECT / "src"), str(MAIN_SRC)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from embodied_learning.experiments.monocular_metric import (
    NEAR_PLANE_M,
    POLE_X_M,
    POLE_Y_M,
    render_depth_map,
)
from embodied_learning.experiments.pinhole_projection import (
    EYE,
    TARGET,
    look_at,
    unproject,
)

from monocular_depth.config import CHECKPOINT_PATH, UPSTREAM_COMMIT
from monocular_depth.model import comparison_view, infer, load_model

MODEL_NAME = "depth_anything_v2_vits"
DEFAULT_INPUT_SIZE = 518
# Lambertian shading: one directional light + ambient, plus optional distance
# fog. The clean synthetic look is intentional - it IS the domain under test.
CHECKER_SIZE_M = 0.5
GROUND_ALBEDO_A = 0.78
GROUND_ALBEDO_B = 0.42
POLE_ALBEDO = (0.52, 0.58, 0.66)  # slightly blue-gray metal
AMBIENT = 0.35
DIFFUSE = 0.65
LIGHT_DIRECTION = np.array([0.35, -0.5, 1.0])
FOG_TAU_M = 9.0
FOG_COLOR = np.array([0.62, 0.66, 0.72])


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shade_scene(rendered: dict, fog_tau: float | None) -> np.ndarray:
    """Lambert RGB (RGB order, uint8) of the lesson-22 scene from its depth map."""
    depth, valid, pole = rendered["depth_m"], rendered["valid"], rendered["pole"]
    height, width = depth.shape
    image = np.tile(FOG_COLOR, (height, width, 1))  # background (no hit) = fog color
    flat = np.flatnonzero(valid.ravel())
    pixels = np.column_stack(np.unravel_index(flat, depth.shape))[:, ::-1].astype(float)
    rotation, translation = look_at(EYE, TARGET)
    world = unproject(pixels, depth.ravel()[flat], rotation, translation)
    normals = np.zeros((flat.size, 3))
    normals[:, 2] = 1.0  # ground plane z=0
    is_pole = pole.ravel()[flat]
    if is_pole.any():
        radial = world[is_pole][:, :2] - np.array([POLE_X_M, POLE_Y_M])
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        normals[is_pole, :2] = radial
        normals[is_pole, 2] = 0.0
    light = LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)
    lambert = AMBIENT + DIFFUSE * np.maximum(normals @ light, 0.0)
    ground_cell = (
        np.floor(world[:, 0] / CHECKER_SIZE_M).astype(int)
        + np.floor(world[:, 1] / CHECKER_SIZE_M).astype(int)
    ) % 2
    albedo = np.where(
        (ground_cell == 0)[:, None],
        np.array([GROUND_ALBEDO_A] * 3),
        np.array([GROUND_ALBEDO_B] * 3),
    )
    albedo[is_pole] = np.asarray(POLE_ALBEDO)
    color = albedo * lambert[:, None]
    if fog_tau is not None and fog_tau > 0.0:
        fog = np.exp(-depth.ravel()[flat] / fog_tau)
        color = color * fog[:, None] + FOG_COLOR * (1.0 - fog)[:, None]
    image = image.reshape(-1, 3)
    image[flat] = color
    return (np.clip(image, 0.0, 1.0).reshape(height, width, 3) * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--fog-tau", type=float, default=FOG_TAU_M)
    parser.add_argument("--no-fog", action="store_true")
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="subdirectory name under outputs/; default affine_check_pinhole_<UTC stamp>",
    )
    args = parser.parse_args()

    rendered = render_depth_map()
    depth, valid, pole = rendered["depth_m"], rendered["valid"], rendered["pole"]
    fog_tau = None if args.no_fog else args.fog_tau
    rgb = shade_scene(rendered, fog_tau)
    print(f"scene rendered: valid={int(valid.sum())} pole={int(pole.sum())} "
          f"depth {depth[valid].min():.2f}-{depth[valid].max():.2f} m")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"checkpoint missing: {CHECKPOINT_PATH}")
    checkpoint_sha = sha256_of(CHECKPOINT_PATH)
    model, device = load_model()  # relative variant, exactly like the lab pipeline
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    started = time.perf_counter()
    r_pred = infer(model, bgr, args.input_size)
    runtime = time.perf_counter() - started
    import torch

    on_valid = r_pred[valid]
    print(f"DA2 inference: device={device} input_size={args.input_size} "
          f"runtime={runtime:.2f}s r_pred[valid] {on_valid.min():.4f}-{on_valid.max():.4f}")

    meta = {
        "model": MODEL_NAME,
        "checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_sha256": checkpoint_sha,
        "upstream_commit": UPSTREAM_COMMIT,
        "input_size": args.input_size,
        "device": str(device),
        "inference_runtime_s": round(runtime, 3),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canvas_px": [int(depth.shape[1]), int(depth.shape[0])],
        "eye_m": EYE.tolist(),
        "target_m": TARGET.tolist(),
        "near_plane_m": NEAR_PLANE_M,
        "fog_tau_m": fog_tau,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "r_convention": (
            "raw model output, arbitrary units; DA2 is expected to emit larger "
            "values for closer surfaces (r ~ a*(1/Z) + b with a > 0), not forced here"
        ),
    }
    name = args.output_name or f"affine_check_pinhole_{meta['timestamp_utc'].replace(':', '')}"
    out_dir = PROJECT / "outputs" / name
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out_dir / "affine_check_pinhole.npz",
        depth_m=depth,
        valid=valid,
        pole=pole,
        rgb=rgb,
        r_pred=r_pred.astype(np.float32),
        meta_json=np.array(json.dumps(meta)),
    )
    # cv2.imwrite cannot write non-ASCII (Chinese) paths on Windows: encode and
    # write the bytes ourselves.
    ok, encoded = cv2.imencode(".png", comparison_view(bgr, r_pred))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    (out_dir / "rgb_vs_pred.png").write_bytes(encoded.tobytes())
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
