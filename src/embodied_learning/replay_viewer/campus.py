"""Load immutable independent closed-loop patrol records into a shared timeline."""

import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.campus_patrol import COLORS, METHODS
from embodied_learning.experiments.grid_nav import Obstacle

from .model import CameraSpec, ReplayEntry, ReplayRecord, Track


def read_checked(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"record hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


class CampusSource:
    title = "园区巡检 · 看懂定位与任务结果"
    variant_label = "实验版本"

    def __init__(self, directory):
        self.directory = Path(directory)
        self.summary = json.loads((self.directory / "summary.json").read_text(encoding="utf-8"))
        p = self.summary["protocol"]
        protocol_file = self.directory / "protocol.json"
        if (
            hashlib.sha256(protocol_file.read_bytes()).hexdigest()
            != self.summary["protocol_sha256"]
        ):
            raise ValueError("campus protocol hash mismatch")
        if json.loads(protocol_file.read_text(encoding="utf-8")) != p:
            raise ValueError("summary embeds a different protocol")
        if p["experiment"] != "campus_patrol":
            raise ValueError("wrong experiment")
        self.entries = tuple(
            ReplayEntry(str(seed), "v1", f"园区第 {seed + 1} 轮 · 随机种子 {seed}")
            for seed in p["seeds"]
        )
        self.cache = {}

    def load(self, case, variant):
        if (case, variant) in self.cache:
            return self.cache[case, variant]
        protocol = self.summary["protocol"]
        rows = {r["method"]: r for r in self.summary["rows"] if r["seed"] == int(case)}
        if set(rows) != set(METHODS):
            raise ValueError("incomplete paired campus record")
        arrays = {
            key: read_checked(self.directory / row["file"], row["sha256"])
            for key, row in rows.items()
        }
        n = max(len(a["truth"]) for a in arrays.values())

        def pad(array):
            return np.concatenate((array, np.repeat(array[-1:], n - len(array), axis=0)))

        maps = read_checked(self.directory / "maps.npz", self.summary["maps_sha256"])
        tracks, lidar, goals, missions, plans, lengths, methods = [], {}, {}, {}, {}, {}, {}
        for key, (label, description) in METHODS.items():
            a, row = arrays[key], rows[key]
            truth_key = "truth" if key == "truth" else key + "_truth"
            tracks.append(
                Track(truth_key, label + "·实际", COLORS[key], pad(a["truth"]), truth_key, "truth")
            )
            if key != "truth":
                tracks.append(Track(key, label, COLORS[key], pad(a["estimate"]), truth_key))
            lengths[key] = len(a["truth"])
            lidar[key] = {
                "ranges": pad(a["ranges"]),
                "hits": pad(a["hits"]),
                "provenance": "recorded_simulated_scan",
            }
            goals[key], missions[key], plans[key] = (
                pad(a["goal"]),
                pad(a["mission"]),
                pad(a["plan"]),
            )
            methods[key] = {
                "label": label,
                "description": description,
                "result": row,
                "display_map": "轮子推算绘制的地图" if key == "self" else "参考障碍地图",
            }
        rates = []
        for key, (label, _) in METHODS.items():
            rr = [r for r in self.summary["rows"] if r["method"] == key]
            rates.append(
                f"{label} {sum(r['status'] == 'completed' for r in rr)}/{len(rr)} 轮全完成"
            )
        config = protocol["config"]
        record = ReplayRecord(
            title=f"街区式园区 / 种子 {case}",
            timestamps=np.arange(n) * protocol["dt_s"],
            tracks=tuple(tracks),
            maps={"参考障碍地图": maps["reference"], "轮子推算绘制的地图": maps["built"]},
            resolution_m=protocol["resolution_m"],
            primitives=tuple(Obstacle(**o) for o in protocol["geometry"]),
            wall_count=4,
            camera=CameraSpec(0.55, 0.18, 65),
            lidar=lidar,
            goals=goals,
            missions=missions,
            plans=plans,
            lengths=lengths,
            metadata={
                "case": case,
                "variant": variant,
                "source": str(self.directory.resolve()),
                "methods": methods,
                "task_names": [t[0] for t in protocol["task_points"]],
                "task_points": protocol["task_points"],
                "scene_style": protocol["styles"],
                "scene_kind": "campus",
                "robot_radius_m": protocol.get("robot_radius_m", 0.25),
                "scene_note": "24×24 m 街区园区；建筑、树干和车辆参与测距/碰撞",
                "lidar_mode": "recorded",
                "lidar_note": "青点：本次存档雷达回波，按所选方法的估计位置放到地图上",
                "sensor_note": "雷达为实验存档；相机按实际位姿重渲染，未参与定位",
                "protocol": "各方法独立规划和执行同一巡检任务；短记录结束后保持末帧，不计入误差均值",
                "settings": f"轮子读数统一偏大 {config['wheel_bias']:.0%}；{config['particles']} 粒子；64 束雷达 / 8 m",
                "map_metrics": "实线=实际走过的位置，虚线=算法认为的位置；越贴合，定位越准",
                "cohort": "；".join(rates),
                "hashes": {k: r["sha256"] for k, r in rows.items()},
            },
        )
        for key, row in rows.items():
            if key != "truth" and not np.isclose(
                record.statistics[key]["mean_m"], row["mean_m"], atol=1e-10
            ):
                raise ValueError("campus scoring mismatch")
        self.cache[case, variant] = record
        return record
