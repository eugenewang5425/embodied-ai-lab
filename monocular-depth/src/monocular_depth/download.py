from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .config import METRIC_SHA256, METRIC_URL, MODEL_URL, VARIANTS, checkpoint_for


def download_checkpoint(
    destination: Path | None = None, force: bool = False, variant: str = "relative"
) -> Path:
    default = checkpoint_for(variant)
    destination = destination or default
    destination = destination.resolve()
    if destination.exists() and destination.stat().st_size > 0 and not force:
        verify_checkpoint(destination, variant)
        print(f"Checkpoint already exists: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Depth Anything V2 Small to {destination}")

    url = METRIC_URL if variant == "metric" else MODEL_URL
    request = urllib.request.Request(url, headers={"User-Agent": "monocular-depth-lab"})
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".part", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        verify_checkpoint(temporary, variant)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Downloaded {destination.stat().st_size / (1024**2):.1f} MiB")
    return destination


def verify_checkpoint(path: Path, variant: str) -> None:
    if variant == "metric":
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != METRIC_SHA256:
                raise ValueError("Metric checkpoint SHA256 mismatch; download was not accepted")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Depth Anything V2 Small checkpoint")
    parser.add_argument("--force", action="store_true", help="replace an existing checkpoint")
    parser.add_argument("--variant", choices=VARIANTS, default="relative")
    args = parser.parse_args()
    download_checkpoint(force=args.force, variant=args.variant)
