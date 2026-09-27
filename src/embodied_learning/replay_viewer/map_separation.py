"""Adapter for the recorded 100/400-particle map-separation pilot."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from embodied_learning.experiments.aniso_env import _build_world
from embodied_learning.experiments.grid_nav import DT, build_walls, true_occupancy
from embodied_learning.experiments.map_localization import MapLocConfig
from embodied_learning.map_localization_demo import load_replays

from .model import ReplayEntry, ReplayRecord, Track

TRACKS = (
    ("truth", "实际位置（对照答案）", "#1684b8", "truth_chain"),
    ("odom", "只靠轮子推算", "#d67600", "est_chain"),
    ("ideal", "雷达＋准确地图", "#168b58", "pf_true_chain"),
    ("projected", "雷达＋按真实位置建图", "#6552c8", "pf_sensor_truth_chain"),
    ("self", "雷达＋自建地图", "#cd3c88", "pf_self_chain"),
)
DESCRIPTIONS = {
    "truth": "仿真中机器人实际在哪里，相当于对照答案；算法不能直接读取它。这批记录是一次运动的多种位置估计。",
    "odom": "只累计轮子转了多少来推算位置。轮子读数有偏差时，推算的位置会逐渐偏离实际位置。",
    "ideal": "用激光测到的墙壁/障碍物和准确地图对齐，修正轮子推算。PF（粒子滤波）就是保留许多位置猜测，再按雷达证据筛选。",
    "projected": "同一批扫描按机器人真实位置画到地图上，去掉位置不准造成的画歪，但仍保留测距噪声和漏扫。这是仿真对照，部署时不能偷用真实位置。",
    "self": "把扫描按有误差的轮子推算位置画成地图，再用这张图定位。它检验地图画歪后，雷达还能否纠正位置。",
}


class MapSeparationSource:
    title = "地图分离实验 · 同步回放"
    variant_label = "粒子预算"

    def __init__(self, hundred, four_hundred):
        self.roots = {"100": Path(hundred), "400": Path(four_hundred)}
        self.rows = {}
        self.summaries = {}
        self._geometry_cache = {}
        self._record_cache = OrderedDict()
        entries = []
        for variant, root in self.roots.items():
            report = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            if report.get("experiment") != "navigation_map_localization_benchmark":
                raise ValueError(f"not a paired localization benchmark: {root}")
            if report["plan"]["particles"] != int(variant):
                raise ValueError(f"wrong particle budget in {root}")
            self.summaries[variant] = report
            for row in report["rows"]:
                if row["status"] != "ok":
                    continue
                case = f"{row['arm'].lower()}_w{row['world_seed']}_s{row['sensor_seed']}"
                self.rows[case, variant] = row
                label = f"{row['arm']} · 场景 {row['world_seed']} · 种子 {row['sensor_seed']}"
                entries.append(ReplayEntry(case, variant, label))
        if not entries:
            raise ValueError("benchmark has no valid episodes")
        self.entries = tuple(sorted(entries, key=lambda e: (e.case, int(e.variant))))

    def _geometry(self, arm, world_seed):
        key = (arm, world_seed)
        if key in self._geometry_cache:
            return self._geometry_cache[key]
        cfg = MapLocConfig(arm=arm, seed=world_seed, run_kidnap=False)
        obstacles, walls = _build_world(arm, world_seed, cfg.base)
        walls = build_walls(cfg.base.world_size_m, cfg.base.res_m) if walls is None else walls
        grid = true_occupancy(obstacles, walls, cfg.base.grid_cells, cfg.base.res_m)
        result = cfg, tuple(walls) + tuple(obstacles), len(walls), grid
        self._geometry_cache[key] = result
        return result

    def load(self, case, variant):
        if (case, variant) in self._record_cache:
            self._record_cache.move_to_end((case, variant))
            return self._record_cache[case, variant]
        row = self.rows[case, variant]
        directory = self.roots[variant] / case
        path = directory / "trajectories.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["trajectory_sha256"]:
            raise ValueError(f"benchmark/archive SHA-256 mismatch: {directory}")
        data = load_replays(directory)
        cfg, primitives, n_walls, grid = self._geometry(row["arm"], row["world_seed"])
        if not np.array_equal(grid, data["true_grid"]):
            raise ValueError("current scene footprints differ from archived occupancy")
        metadata = {
            "case": case,
            "variant": variant,
            "source": str(directory.resolve()),
            "sha256": digest,
            "scene_note": "存档平面几何的 3D 展示；障碍高度仅示意",
            "methods": {
                key: {
                    "label": label,
                    "description": DESCRIPTIONS[key],
                    "display_map": {
                        "self": "里程计自建图",
                        "projected": "真值投影图",
                    }.get(key, "理想占据图"),
                }
                for key, label, _, _ in TRACKS
            },
            "lidar_mode": "geometry_recast",
            "lidar_note": "青点：按存档真值重算的几何雷达点（旧记录未保存原始扫描）",
            "sensor_note": "相机为真值位姿重渲染；本批没有存档 RGB-D / 原始扫描",
            "protocol": "真值控制巡游回放；地图为采集后固定结果",
            "settings": f"代码协议：右轮偏差 +{cfg.matcher.slam.encoder_bias:.0%}；测距 σ={cfg.range_noise_sigma_m:.2f} m；PF {variant} 粒子 / {cfg.pf_rays} 线",
            "map_metrics": f"本回合地图对齐 P/R={row['map_alignment_precision']:.3f}/{row['map_alignment_recall']:.3f}；中位偏移 {row['map_median_offset_m']:.2f} m（0.20 m 容差）",
        }
        comparison = []
        for budget, summary in self.summaries.items():
            rows = [r for r in summary["rows"] if r["status"] == "ok"]
            median = float(np.median([r["pf_self_mean_m"] for r in rows]))
            comparison.append(f"{budget}p：自建图均值的中位数 {median:.3f} m / {len(rows)} 回合")
        metadata["cohort"] = "；".join(comparison)
        record = ReplayRecord(
            title=f"{row['arm']} / 场景 {row['world_seed']} / 种子 {row['sensor_seed']} / {variant} 粒子",
            timestamps=np.arange(len(data["truth_chain"])) * DT,
            tracks=tuple(Track(k, label, color, data[field]) for k, label, color, field in TRACKS),
            maps={
                "理想占据图": data["true_grid"],
                "真值投影图": data["sensor_truth_grid"],
                "里程计自建图": data["built_grid"],
            },
            resolution_m=cfg.base.res_m,
            primitives=primitives,
            wall_count=n_walls,
            metadata=metadata,
        )
        for key, stored, mean_key in (
            ("odom", "est_err_curve", "est_mean_m"),
            ("ideal", "pf_true_err", "pf_true_mean_m"),
            ("projected", "pf_sensor_truth_err", "pf_sensor_truth_mean_m"),
            ("self", "pf_self_err", "pf_self_mean_m"),
        ):
            if not np.allclose(record.errors[key], data[stored], atol=1e-9, rtol=0):
                raise ValueError(f"stored error disagrees with poses: {key}")
            if not np.isclose(record.statistics[key]["mean_m"], row[mean_key], atol=1e-9, rtol=0):
                raise ValueError(f"benchmark mean disagrees with poses: {key}")
        self._record_cache[case, variant] = record
        if len(self._record_cache) > 4:
            self._record_cache.popitem(last=False)
        return record
