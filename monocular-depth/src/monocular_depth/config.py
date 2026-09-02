from __future__ import annotations

import os
import tomllib
from pathlib import Path


def find_project_root() -> Path:
    def is_lab(path: Path) -> bool:
        manifest = path / "pyproject.toml"
        if not manifest.is_file():
            return False
        with manifest.open("rb") as stream:
            return tomllib.load(stream).get("project", {}).get("name") == "monocular-depth-lab"

    configured = os.environ.get("MONOCULAR_DEPTH_HOME")
    if configured:
        root = Path(configured).resolve()
        if not is_lab(root):
            raise RuntimeError("MONOCULAR_DEPTH_HOME must point to the monocular-depth lab")
        return root

    source_root = Path(__file__).resolve().parents[2]
    if is_lab(source_root):
        return source_root

    working_root = Path.cwd().resolve()
    if is_lab(working_root):
        return working_root

    raise RuntimeError(
        "Run this command from the monocular-depth directory or set MONOCULAR_DEPTH_HOME."
    )


PROJECT_ROOT = find_project_root()
# Verified maximum DirectShow mode of the user's EMEET SmartCam C60E 4K.
# These are capture dimensions, independent of the neural network input size.
DEFAULT_CAMERA_WIDTH = 3840
DEFAULT_CAMERA_HEIGHT = 2160
DEFAULT_CAMERA_FPS = 30
DEFAULT_CAMERA_FOURCC = "MJPG"
DEFAULT_UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "Depth-Anything-V2"
UPSTREAM_ROOT = Path(os.environ.get("DEPTH_ANYTHING_REPO", DEFAULT_UPSTREAM_ROOT))
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "depth_anything_v2_vits.pth"

MODEL_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
    "depth_anything_v2_vits.pth"
)
UPSTREAM_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"

MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}

VARIANTS = ("relative", "metric")
METRIC_CHECKPOINT_PATH = CHECKPOINT_DIR / "depth_anything_v2_metric_hypersim_vits.pth"
METRIC_REVISION = "3bc65d4e14a6786a61acec16453c50e12bf5f338"
METRIC_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/"
    f"{METRIC_REVISION}/depth_anything_v2_metric_hypersim_vits.pth"
)
METRIC_SHA256 = "b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545"


def checkpoint_for(variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown depth variant: {variant}")
    return METRIC_CHECKPOINT_PATH if variant == "metric" else CHECKPOINT_PATH
