"""Lesson 27 bench: MobileSAM proposals on the landmark scene for grounding.

Renders the lesson-22/24 rendering pipeline (640x480 pinhole, K=[600,600,320,240],
analytic ray casting, Lambert shading, distance fog) with THREE vertical
cylinders as known landmarks: L1 sits at the lesson-22 pole position (4.2, 3.6),
L2=(3.0, 4.2) and L3=(4.0, 2.7) complete a non-collinear triplet. All three
share the lesson-22 pole radius (0.06 m); the height is unified to 1.2 m so the
non-anchor views keep every mask inside the frame (the 2.0 m lesson-22 pole top
leaves the image in ALL poses - the anchor view still clips the mask tops at
the upper border, kept as an honest visibility artifact). Camera 0 is the
unchanged lesson-22/24 look-at pose; cameras 1-3 circle the landmark cluster so
that each landmark is visible without border clipping.

For each pose the REAL MobileSAM checkpoint (official ChaoningZhang/MobileSAM
weights) runs an automatic mask generator over the rendered RGB. The bench does
NOT assign identities: it archives every candidate mask. The main-repo analysis
(uv run python -m embodied_learning.experiments.visual_grounding --input <npz>)
back-projects mask centroids through the rendered depth, assigns identities by
nearest-neighbour matching against the known landmark coordinates, and feeds
the observations into the lesson-18 Procrustes localization chain.

Run with the monocular-depth venv, NOT the main repo venv (torch stays out of
the main repo's test env):

  monocular-depth/.venv/Scripts/python.exe monocular-depth/bench/grounding_inference.py
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

from embodied_learning.experiments.pinhole_projection import (
    HEIGHT_PX,
    K_INTRINSIC,
    NEAR_PLANE_M,
    WIDTH_PX,
    look_at,
    unproject,
)

# ------------------------------------------------------------------- scene
# Ground extent of the lesson-22/23 grid (monocular_metric scene_points()).
GRID_MIN_M, GRID_MAX_M = 0.75, 5.25
# L1 = lesson-22 pole position; L2/L3 complete the triplet inside camera 0's
# field of view (half-FOV 28.1 deg: off-axis -12.6 / +13.4 / -25.3 deg).
LANDMARK_XY = ((4.2, 3.6), (3.0, 4.2), (4.0, 2.7))
LANDMARK_RADIUS_M = 0.06  # lesson-22 pole radius
LANDMARK_HEIGHT_M = 1.2  # unified so poses 1-3 keep every mask inside the frame
POSES = (
    ((2.0, 1.5, 1.4), (3.0, 3.0, 0.0)),  # lesson-22/24 anchor (tops clipped)
    ((1.27, 1.78, 1.4), (3.73, 3.5, 0.5)),
    ((4.76, 0.68, 1.4), (3.73, 3.5, 0.5)),
    ((2.64, 6.51, 1.4), (3.73, 3.5, 0.5)),
)
# Lambert shading: identical constants to affine_check_pinhole.py. All three
# cylinders share ONE albedo on purpose: identity must come from geometry
# (back-projected mask position), never from appearance.
CHECKER_SIZE_M = 0.5
GROUND_ALBEDO_A = 0.78
GROUND_ALBEDO_B = 0.42
CYLINDER_ALBEDO = (0.52, 0.58, 0.66)  # lesson-22 pole blue-gray metal
AMBIENT = 0.35
DIFFUSE = 0.65
LIGHT_DIRECTION = np.array([0.35, -0.5, 1.0])
FOG_TAU_M = 9.0
FOG_COLOR = np.array([0.62, 0.66, 0.72])

# MobileSAM: official repository (package) and official master weights.
MODEL_NAME = "MobileSAM (Tiny-ViT encoder, automatic mask generator)"
MODEL_PACKAGE_SOURCE = "git+https://github.com/ChaoningZhang/MobileSAM.git"
MODEL_WEIGHTS_SOURCE = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
CHECKPOINT_PATH = PROJECT / "checkpoints" / "mobile_sam.pt"
AMG_PARAMS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.9,
    "crop_n_layers": 0,
    "min_mask_region_area": 250,
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_landmark_scene(rotation, translation, eye):
    """Per-pixel depth + truth landmark labels of the three-cylinder scene.

    Same conventions as lessons 23/26: depth = camera-frame z, a pixel is valid
    when its nearest surface hit lies beyond the near plane; label 0 = ground,
    1..3 = pixels whose nearest hit is that cylinder (occlusion = nearer wins).
    """
    eye = np.asarray(eye, dtype=float)
    cols = np.arange(WIDTH_PX, dtype=float)
    rows = np.arange(HEIGHT_PX, dtype=float)
    uu, vv = np.meshgrid(cols, rows)
    homogeneous = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)])
    rays = rotation.T @ (np.linalg.inv(K_INTRINSIC) @ homogeneous)
    wx, wy, wz = rays
    depth = np.full(uu.size, np.nan)
    label = np.zeros(uu.size, dtype=np.int8)
    towards_ground = wz < -1e-12
    s_ground = np.full(uu.size, np.nan)
    s_ground[towards_ground] = -eye[2] / wz[towards_ground]
    ground_x = eye[0] + s_ground * wx
    ground_y = eye[1] + s_ground * wy
    on_grid = (
        towards_ground
        & (ground_x >= GRID_MIN_M)
        & (ground_x <= GRID_MAX_M)
        & (ground_y >= GRID_MIN_M)
        & (ground_y <= GRID_MAX_M)
    )
    depth[on_grid] = s_ground[on_grid]
    for index, (lx, ly) in enumerate(LANDMARK_XY, start=1):
        offset_x, offset_y = eye[0] - lx, eye[1] - ly
        quad_a = wx * wx + wy * wy
        quad_b = 2.0 * (wx * offset_x + wy * offset_y)
        quad_c = offset_x * offset_x + offset_y * offset_y - LANDMARK_RADIUS_M**2
        disc = quad_b * quad_b - 4.0 * quad_a * quad_c
        hits = (quad_a > 1e-12) & (disc >= 0.0)
        s_cyl = np.full(uu.size, np.nan)
        s_cyl[hits] = (-quad_b[hits] - np.sqrt(disc[hits])) / (2.0 * quad_a[hits])
        with np.errstate(invalid="ignore"):
            cyl_z = eye[2] + s_cyl * wz
        on_cyl = hits & (s_cyl > NEAR_PLANE_M) & (cyl_z >= 0.0) & (cyl_z <= LANDMARK_HEIGHT_M)
        nearer = on_cyl & (np.isnan(depth) | (s_cyl < depth))
        depth[nearer] = s_cyl[nearer]
        label[nearer] = index
    depth[depth <= NEAR_PLANE_M] = np.nan  # near-plane cull for ground hits
    valid = np.isfinite(depth)
    return {
        "depth_m": depth.reshape(HEIGHT_PX, WIDTH_PX),
        "valid": valid.reshape(HEIGHT_PX, WIDTH_PX),
        "label": label.reshape(HEIGHT_PX, WIDTH_PX),
    }


def shade_scene(rendered: dict, rotation, translation, fog_tau: float | None) -> np.ndarray:
    """Lambert RGB (RGB order, uint8) of the three-cylinder scene."""
    depth, valid, label = rendered["depth_m"], rendered["valid"], rendered["label"]
    height, width = depth.shape
    image = np.tile(FOG_COLOR, (height, width, 1))  # background (no hit) = fog color
    flat = np.flatnonzero(valid.ravel())
    pixels = np.column_stack(np.unravel_index(flat, depth.shape))[:, ::-1].astype(float)
    world = unproject(pixels, depth.ravel()[flat], rotation, translation)
    normals = np.zeros((flat.size, 3))
    normals[:, 2] = 1.0  # ground plane z=0
    is_cylinder = label.ravel()[flat] > 0
    if is_cylinder.any():
        axes = np.asarray(LANDMARK_XY, dtype=float)[label.ravel()[flat][is_cylinder] - 1]
        radial = world[is_cylinder][:, :2] - axes
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
        normals[is_cylinder, :2] = radial
        normals[is_cylinder, 2] = 0.0
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
    albedo[is_cylinder] = np.asarray(CYLINDER_ALBEDO)
    color = albedo * lambert[:, None]
    if fog_tau is not None and fog_tau > 0.0:
        fog = np.exp(-depth.ravel()[flat] / fog_tau)
        color = color * fog[:, None] + FOG_COLOR * (1.0 - fog)[:, None]
    image = image.reshape(-1, 3)
    image[flat] = color
    return (np.clip(image, 0.0, 1.0).reshape(height, width, 3) * 255.0).astype(np.uint8)


def draw_mask_overlay(rgb: np.ndarray, label: np.ndarray, masks: list[dict]) -> np.ndarray:
    """Truth label edges (green) under all model-mask edges (red) for preview."""
    overlay = rgb.copy()
    truth = (label > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (120, 255, 120), 1)
    canvas = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.uint8)
    for proposal in masks:
        canvas |= proposal["segmentation"].astype(np.uint8) * 255
    contours, _ = cv2.findContours(canvas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (60, 60, 255), 1)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fog-tau", type=float, default=FOG_TAU_M)
    parser.add_argument("--no-fog", action="store_true")
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="subdirectory name under outputs/; default grounding_inference_<UTC stamp>",
    )
    args = parser.parse_args()

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"MobileSAM checkpoint missing: {CHECKPOINT_PATH}\n"
            f"Download from the official repository: {MODEL_WEIGHTS_SOURCE}"
        )
    checkpoint_sha = sha256_of(CHECKPOINT_PATH)

    import torch
    from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

    poses = [look_at(np.asarray(eye, float), np.asarray(target, float)) for eye, target in POSES]
    rendered_views = []
    for (eye, _target), (rotation, translation) in zip(POSES, poses):
        rendered = render_landmark_scene(rotation, translation, np.asarray(eye, float))
        rendered_views.append(rendered)
    fog_tau = None if args.no_fog else args.fog_tau
    rgbs = [
        shade_scene(r, rotation, translation, fog_tau)
        for r, (rotation, translation) in zip(rendered_views, poses)
    ]
    truth_counts = [[int((r["label"] == i).sum()) for i in (1, 2, 3)] for r in rendered_views]
    print(f"scene rendered: {len(poses)} poses; truth label px per landmark: {truth_counts}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT_PATH))
    sam.to(device=device)
    sam.eval()
    generator = SamAutomaticMaskGenerator(sam, **AMG_PARAMS)

    masks: list[dict] = []
    runtimes = []
    for view_index, bgr in enumerate([np.ascontiguousarray(rgb[:, :, ::-1]) for rgb in rgbs]):
        started = time.perf_counter()
        proposals = generator.generate(bgr)
        runtime = time.perf_counter() - started
        runtimes.append(round(runtime, 3))
        print(f"pose {view_index}: MobileSAM returned {len(proposals)} masks in {runtime:.2f}s")
        for proposal in proposals:
            masks.append(
                {
                    "pose": view_index,
                    "segmentation": proposal["segmentation"].astype(bool),
                    "area": int(proposal["area"]),
                    "predicted_iou": float(proposal["predicted_iou"]),
                    "stability_score": float(proposal["stability_score"]),
                    "bbox": [float(v) for v in proposal["bbox"]],
                }
            )

    meta = {
        "model": MODEL_NAME,
        "model_package_source": MODEL_PACKAGE_SOURCE,
        "model_weights_source": MODEL_WEIGHTS_SOURCE,
        "checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_sha256": checkpoint_sha,
        "amg_params": AMG_PARAMS,
        "device": str(device),
        "inference_runtime_s_per_pose": runtimes,
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canvas_px": [WIDTH_PX, HEIGHT_PX],
        "intrinsic": K_INTRINSIC.tolist(),
        "near_plane_m": NEAR_PLANE_M,
        "grid_extent_m": [GRID_MIN_M, GRID_MAX_M],
        "landmark_xy": [list(xy) for xy in LANDMARK_XY],
        "landmark_radius_m": LANDMARK_RADIUS_M,
        "landmark_height_m": LANDMARK_HEIGHT_M,
        "pose_eye_m": [list(eye) for eye, _ in POSES],
        "pose_target_m": [list(target) for _, target in POSES],
        "fog_tau_m": fog_tau,
        "truth_label_px": truth_counts,
        "mask_count": len(masks),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "mask_convention": (
            "model_mask[k] is a boolean proposal on pose mask_pose[k]; the bench "
            "does NOT assign identities - the main-repo analysis does that from "
            "the rendered depth by nearest-neighbour matching"
        ),
    }
    name = args.output_name or f"grounding_inference_{meta['timestamp_utc'].replace(':', '')}"
    out_dir = PROJECT / "outputs" / name
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    pose_of = np.array([m["pose"] for m in masks], dtype=np.int64)
    np.savez_compressed(
        out_dir / "grounding_masks.npz",
        rgb=np.stack(rgbs).astype(np.uint8),
        depth_m=np.stack([r["depth_m"] for r in rendered_views]),
        valid=np.stack([r["valid"] for r in rendered_views]),
        landmark_label=np.stack([r["label"] for r in rendered_views]).astype(np.int8),
        model_mask=(
            np.stack([m["segmentation"] for m in masks])
            if masks
            else np.zeros((0, HEIGHT_PX, WIDTH_PX), dtype=bool)
        ),
        mask_pose=pose_of,
        mask_area=np.array([m["area"] for m in masks], dtype=np.int64),
        mask_score=np.array(
            [[m["predicted_iou"], m["stability_score"]] for m in masks], dtype=np.float64
        ).reshape(len(masks), 2),
        mask_bbox=np.array([m["bbox"] for m in masks], dtype=np.float64).reshape(len(masks), 4),
        pose_rotation=np.stack([p[0] for p in poses]),
        pose_translation=np.stack([p[1] for p in poses]),
        landmark_xy=np.asarray(LANDMARK_XY, dtype=float),
        landmark_radius_m=np.array(LANDMARK_RADIUS_M),
        landmark_height_m=np.array(LANDMARK_HEIGHT_M),
        meta_json=np.array(json.dumps(meta)),
    )
    overlays = [
        draw_mask_overlay(rgb, r["label"], [m for m in masks if m["pose"] == i])
        for i, (rgb, r) in enumerate(zip(rgbs, rendered_views))
    ]
    grid = np.full((HEIGHT_PX * 2 + 12, WIDTH_PX * 2 + 12, 3), 255, dtype=np.uint8)
    for index, overlay in enumerate(overlays):
        row, col = divmod(index, 2)
        grid[
            row * (HEIGHT_PX + 12) : row * (HEIGHT_PX + 12) + HEIGHT_PX,
            col * (WIDTH_PX + 12) : col * (WIDTH_PX + 12) + WIDTH_PX,
        ] = overlay
    # cv2.imwrite cannot write non-ASCII (Chinese) paths on Windows: encode and
    # write the bytes ourselves.
    ok, encoded = cv2.imencode(".png", grid)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    (out_dir / "mask_preview.png").write_bytes(encoded.tobytes())
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {out_dir.resolve()} ({len(masks)} masks total)")


if __name__ == "__main__":
    main()
