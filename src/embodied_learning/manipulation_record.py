"""Validated immutable record adapter for reusable manipulation panels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.so101 import ASSETS, CONTROL_DT, DT, SceneConfig, verify_assets


class ManipulationRecord:
    def __init__(self, directory):
        verify_assets()
        self.directory = Path(directory)
        self.summary = json.loads((self.directory / "summary.json").read_text(encoding="utf-8"))
        summary = self.summary
        if (
            summary.get("experiment") != "so101_physical_grasping"
            or summary.get("schema_version") != 1
        ):
            raise ValueError("Expected SO101 physical grasping schema version 1")
        manifest_hash = hashlib.sha256((ASSETS / "provenance.json").read_bytes()).hexdigest()
        if summary.get("model_manifest_sha256") != manifest_hash:
            raise ValueError("Record uses a different SO101 model")
        if summary["physics_dt_s"] != DT or summary["control_dt_s"] != CONTROL_DT:
            raise ValueError("Unexpected simulation timestep")
        archive_path = self.directory / "trajectories.npz"
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != summary["archive_sha256"]:
            raise ValueError("Trajectory archive hash mismatch")
        for name, digest in summary["source_sha256"].items():
            if Path(name).name != name:
                raise ValueError("Invalid snapshot path")
            if (
                hashlib.sha256((self.directory / "source" / name).read_bytes()).hexdigest()
                != digest
            ):
                raise ValueError("Source snapshot hash mismatch")
        self.rows = {row["key"]: row for row in summary["episodes"]}
        if len(self.rows) != len(summary["episodes"]) or not self.rows:
            raise ValueError("Duplicate or missing episode keys")
        self.positions = sorted({row["position_key"] for row in self.rows.values()})
        self.arrays = {}
        with np.load(archive_path, allow_pickle=False) as archive:
            for key, row in self.rows.items():
                SceneConfig(**row["config"])
                values = {
                    name.split("__", 1)[1]: archive[name].copy()
                    for name in archive.files
                    if name.startswith(key + "__")
                }
                self._validate(values)
                for value in values.values():
                    value.flags.writeable = False
                self.arrays[key] = values

    @staticmethod
    def _validate(values):
        n = len(values["timestamps"])
        if (
            n < 2
            or abs(values["timestamps"][0]) > 1e-12
            or not np.allclose(np.diff(values["timestamps"]), CONTROL_DT, atol=1e-10)
        ):
            raise ValueError("Invalid control timestamps")
        for name, width in (
            ("qpos", 13),
            ("qvel", 12),
            ("commands", 6),
            ("tcp_xyz", 3),
            ("object_xyz", 3),
            ("jaw_normal_n", 3),
            ("waypoint_xyz", 3),
        ):
            if values[name].shape != (n, width):
                raise ValueError(f"Invalid {name} shape")
        m = (n - 1) * round(CONTROL_DT / DT)
        if values["phase"].shape != (n,) or np.any((values["phase"] < 0) | (values["phase"] > 10)):
            raise ValueError("Invalid phase identifiers")
        for name, shape in (
            ("physics_time", (m,)),
            ("physics_qpos", (m, 13)),
            ("physics_qvel", (m, 12)),
            ("physics_jaw_n", (m, 3)),
            ("physics_other_contact_n", (m,)),
            ("physics_penetration_m", (m,)),
        ):
            if values[name].shape != shape:
                raise ValueError(f"Invalid {name} shape")
        if not np.allclose(values["physics_time"], np.arange(1, m + 1) * DT, atol=1e-9) or any(
            not np.isfinite(value).all() for value in values.values()
        ):
            raise ValueError("Nonfinite or inconsistent physical trace")
        if not np.allclose(np.linalg.norm(values["qpos"][:, 9:13], axis=1), 1.0, atol=1e-6):
            raise ValueError("Invalid object quaternion")
        stride = round(CONTROL_DT / DT)
        if not np.allclose(
            values["physics_qpos"][stride - 1 :: stride], values["qpos"][1:]
        ) or not np.allclose(values["physics_qvel"][stride - 1 :: stride], values["qvel"][1:]):
            raise ValueError("Control frames disagree with physical trace")

    def select(self, position, method):
        key = f"{position}_{method}"
        return self.rows[key], self.arrays[key]
