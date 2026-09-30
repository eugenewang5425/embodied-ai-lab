"""Vendor the pinned Apache-2.0 SO101 model, without installing another simulator."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

COMMIT = "4d038b3feae26ec82b46a4d586379114012a8ac7"
BASE = (
    f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{COMMIT}/robotstudio_so101"
)
DESTINATION = Path(__file__).resolve().parents[1] / "src/embodied_learning/assets/so101"


def download(name: str) -> bytes:
    request = Request(f"{BASE}/{name}", headers={"User-Agent": "embodied-ai-lab"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    xml = download("so101.xml")
    root = ET.fromstring(xml)
    meshes = sorted({f"assets/{node.attrib['file']}" for node in root.findall("asset/mesh")})
    names = ["LICENSE", "README.md", "CHANGELOG.md", *meshes]
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(zip(names, pool.map(download, names), strict=True))
    files["so101.xml"] = xml
    manifest = {
        "repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "directory": "robotstudio_so101",
        "commit": COMMIT,
        "license": "Apache-2.0",
        "files": {},
    }
    # Check all existing files before writing; never overwrite local model edits.
    for name, content in files.items():
        path = DESTINATION / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Local model differs from pinned upstream: {path}")
        manifest["files"][name] = {
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    for name, content in files.items():
        path = DESTINATION / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (DESTINATION / "provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SO101 {COMMIT}: {len(files)} verified files, {sum(map(len, files.values()))} bytes")


if __name__ == "__main__":
    main()
