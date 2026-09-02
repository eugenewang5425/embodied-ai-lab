from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
