"""Checked adapters for body-aware avoidance and frozen-map repair studies."""

import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.campus_patrol import TASKS
from embodied_learning.experiments.grid_nav import Obstacle
from embodied_learning.navigation_geometry import NavigationRig

from .campus import read_checked
from .model import CameraSpec, ReplayEntry, ReplayRecord, Track

LABELS = {
    "stop": ("只会停车", "前方雷达太近就停车；不会绕路，低障碍和悬空杆可能漏检。", "#df852c"),
    "lidar": (
        "雷达＋车身",
        "把雷达回波和车身包络结合，更新临时障碍并重新找路；单层雷达有高度盲区。",
        "#8b5cf6",
    ),
    "depth": (
        "雷达＋深度＋车身",
        "加入深度相机的三维障碍和观测记忆；看过的低障碍进入近场盲区后仍会避让。",
        "#119c74",
    ),
    "raw": ("原始错图", "轮子读数偏大，墙面落错位置；使用原图定位，也在原图上规划。", "#dc7840"),
    "rigid": ("整图平移旋转", "只能整体搬动地图，不能修复不同路段的形变。", "#8b5cf6"),
    "repaired": (
        "扫描校正后重建",
        "从旧雷达扫描估计轮子尺度，再重放扫描生成新图；本轮独立巡检不参与修图。",
        "#119c74",
    ),
    "reference": (
        "准确地图参照",
        "准确几何图供定位与规划；仍受定位和规划器限制，并非保证完成。",
        "#247cc1",
    ),
}
LAYOUTS = {"plaza": "开阔场地", "street": "街区通道", "narrow": "两米窄通道"}
OBSTACLES = {"crate": "普通箱体", "low": "14厘米路障", "beam": "悬空横杆"}


class NavigationStudySource:
    title = "67–68 课 · 车身标定、避障与错图修复"
    variant_label = "课程"

    def __init__(self, avoidance="results/observed_navigation_v1", repair="results/map_repair_v1"):
        self.directories = {"67": Path(avoidance), "68": Path(repair)}
        self.summaries, self.entries = {}, []
        self.cache = {}
        for lesson, directory in self.directories.items():
            summary = json.loads((directory / "summary.json").read_text(encoding="utf8"))
            protocol = (directory / "protocol.json").read_bytes()
            if hashlib.sha256(protocol).hexdigest() != summary["protocol_sha256"]:
                raise ValueError("study protocol hash mismatch")
            if json.loads(protocol) != summary["protocol"]:
                raise ValueError("study protocol differs from summary")
            self.summaries[lesson] = summary
            seen = set()
            for row in summary["rows"]:
                case = self.case_id(lesson, row)
                if case in seen:
                    continue
                seen.add(case)
                label = (
                    f"67 · {LAYOUTS[row['layout']]} / {OBSTACLES[row['obstacle']]}"
                    if lesson == "67"
                    else "68 · 园区错图独立巡检"
                )
                self.entries.append(ReplayEntry(case, lesson, f"{label} / 种子 {row['seed']}"))
        self.entries = tuple(self.entries)

    @staticmethod
    def case_id(lesson, row):
        return (
            f"67_{row['layout']}_{row['obstacle']}_{row['seed']}"
            if lesson == "67"
            else f"68_{row['seed']}"
        )

    def load(self, case, variant):
        if (case, variant) in self.cache:
            return self.cache[case, variant]
        summary, directory = self.summaries[variant], self.directories[variant]
        rows = {r["method"]: r for r in summary["rows"] if self.case_id(variant, r) == case}
        keys = (
            ("stop", "lidar", "depth")
            if variant == "67"
            else ("raw", "rigid", "repaired", "reference")
        )
        if set(rows) != set(keys):
            raise ValueError("incomplete study comparison")
        arrays = {k: read_checked(directory / rows[k]["file"], rows[k]["sha256"]) for k in keys}
        n = max(2, max(len(a["truth"]) for a in arrays.values()))

        def pad(array):
            return np.concatenate((array, np.repeat(array[-1:], n - len(array), axis=0)))

        tracks, lidar, goals, missions, plans, lengths, methods, depth, telemetry = (
            [],
            {},
            {},
            {},
            {},
            {},
            {},
            {},
            {},
        )
        maps = {}
        for i, key in enumerate(keys):
            a = arrays[key]
            label, description, color = LABELS[key]
            true_key = "truth" if i == 0 else key + "_truth"
            tracks += [
                Track(true_key, label + "·实际", color, pad(a["truth"]), true_key, "truth"),
                Track(key, label, color, pad(a["estimate"]), true_key),
            ]
            lengths[key] = len(a["truth"])
            lidar[key] = {"ranges": pad(a["ranges"]), "hits": pad(a["hits"])}
            goals[key], missions[key], plans[key] = (
                pad(a["goal"]),
                pad(a["mission"]),
                pad(a["plan"]),
            )
            methods[key] = {"label": label, "description": description, "result": rows[key]}
            if variant == "67":
                depth[key] = {"values": pad(a["depth"]), "valid": pad(a["depth_valid"])}
                telemetry[key] = {k: pad(a[k]) for k in ("commands", "reason", "safety_clearance")}
                # Never show the final accumulated obstacle map as a live-time map.
                # The shared prior is fixed; live returns and plans remain time-aligned.
                methods[key]["display_map"] = "实验开始时的先验地图"
            else:
                methods[key]["display_map"] = label
        metadata = {
            "case": case,
            "variant": variant,
            "source": str(directory.resolve()),
            "methods": methods,
            "lidar_mode": "recorded",
            "hashes": {k: r["sha256"] for k, r in rows.items()},
            "protocol": "各方法独立驾驶；结束后保持末帧，保持帧不计分。",
            "map_metrics": "实线=实际轨迹；虚线=估计轨迹；金线=当前规划；青点=当前雷达回波。",
            "cohort": "；".join(
                f"{LABELS[k][0]} {sum(r['status'] == 'completed' for r in summary['rows'] if r['method'] == k)}/{sum(r['method'] == k for r in summary['rows'])} 回合完成"
                for k in keys
            ),
        }
        if variant == "67":
            rig = NavigationRig(**summary["protocol"]["rig"])
            boxes = arrays[keys[0]]["actual_boxes"]
            primitives = tuple(
                Obstacle(
                    "rect",
                    (b[0] + b[1]) / 2,
                    (b[2] + b[3]) / 2,
                    (b[1] - b[0]) / 2,
                    (b[3] - b[2]) / 2,
                )
                for b in boxes
            )
            styles = [
                {
                    "height": float(b[5] - b[4]),
                    "bottom_m": float(b[4]),
                    "color": ".73 .42 .22 1" if j == len(boxes) - 1 else ".72 .78 .83 1",
                }
                for j, b in enumerate(boxes)
            ]
            maps["实验开始时的先验地图"] = arrays[keys[0]]["prior_map"]
            metadata.update(
                rig=rig.to_dict(),
                body_parts=rig.parts.tolist(),
                lidar_offset_xy=[rig.lidar_x_m, 0],
                scene_style=styles,
                scene_kind="body_navigation",
                task_names=["通道终点"],
                sensor_note="深度/雷达来自同步存档；RGB 按实际位姿重渲染。当前没有视觉定位。",
                settings="55×44 cm车身，高58 cm；6 cm余量，预留0.2 s延迟；零定位误差是受控条件",
                scene_note="12×12 m；新障碍未画入先验地图；悬空杆只占其实际高度",
                lidar_note="青点来自高32 cm雷达；没有点不等于全车高度内没有障碍。",
            )
            camera = CameraSpec(rig.camera_z_m, rig.camera_x_m, rig.vfov_deg)
            dt = summary["protocol"]["dt_s"]
        else:
            grids = read_checked(directory / "maps.npz", summary["maps_sha256"])
            maps = {LABELS[k][0]: grids[k] for k in keys}
            primitives = tuple(Obstacle(**o) for o in summary["geometry"])
            metadata.update(
                scene_kind="campus",
                scene_style=summary["styles"],
                robot_radius_m=0.25,
                task_names=[t[0] for t in TASKS],
                task_points=TASKS,
                sensor_note="第66课雷达/PF控制器；未接入第67课深度避障；RGB为重渲染。",
                settings="24×24 m；300粒子；64束雷达；2%公共刻度偏差；每组用自己的图规划",
                scene_note="冻结地图后的独立巡检；修图仅用旧扫描和里程计",
                lidar_note="雷达点按当前估计投到地图上；落错墙可能来自错图或定位偏差。",
            )
            camera, dt = CameraSpec(0.55, 0.18, 65), 0.12
        record = ReplayRecord(
            next(e.label for e in self.entries if e.case == case),
            np.arange(n) * dt,
            tuple(tracks),
            maps,
            0.1,
            primitives,
            4,
            metadata=metadata,
            camera=camera,
            lidar=lidar,
            goals=goals,
            missions=missions,
            plans=plans,
            lengths=lengths,
            depth=depth,
            telemetry=telemetry,
        )
        for key, row in rows.items():
            if not np.isclose(record.statistics[key]["mean_m"], row["mean_m"], atol=1e-9):
                raise ValueError("study scoring differs from replay")
        self.cache.clear()  # Depth archives are large; retain only the visible comparison.
        self.cache[case, variant] = record
        return record
