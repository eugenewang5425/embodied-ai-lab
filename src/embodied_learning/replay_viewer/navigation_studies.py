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
    "legacy_brake": (
        "旧版制动参照",
        "第二轮组合控制器，保留旧预测；补录失败后实际减速尾段。",
        "#94a3b8",
    ),
    "discrete": (
        "一致预测＋独立减速",
        "预测和执行统一到10Hz；线速度与角速度分别减到零。四组对照的基线。",
        "#dc7840",
    ),
    "coupled": (
        "仅同比例制动",
        "停车时线速度与角速度按相同比例降低，保持当前转弯半径；正常速度规则不变。",
        "#119c74",
    ),
    "speed": (
        "仅曲率限速",
        "根据前方路线弯曲程度降低速度；仍独立减速，检验仅慢下来是否足够。",
        "#8b5cf6",
    ),
    "joint_brake": (
        "同比例制动＋限速",
        "同时采用同比例停车和曲率速度预算；是否更好需看独立回合结果。",
        "#247cc1",
    ),
    "original": (
        "旧跟踪＋旧裁剪",
        "追35厘米外的路径点；保留路线时直接连到前视点。第一轮矩形组的相同基线。",
        "#dc7840",
    ),
    "tangent": (
        "仅换切线跟踪",
        "根据路线朝向、弯曲程度和横向偏差调整转向；仍使用旧裁剪。",
        "#8b5cf6",
    ),
    "retained": (
        "仅保留原曲线",
        "从最近线段开始保留原折线，避免把弯路切成直线；仍用旧跟踪。",
        "#247cc1",
    ),
    "combined": (
        "切线跟踪＋原曲线",
        "同时保留原曲线并纠正跟踪偏差；仍检查相同6厘米余量与整车刹停空间。",
        "#119c74",
    ),
    "circle_grid": (
        "旧圆形栅格规划",
        "把整个车身包成圆，再按10厘米格子膨胀；沿用第67课融合观测。",
        "#dc7840",
    ),
    "circle_metric": (
        "连续观测＋圆形车身",
        "保留观测的米制坐标；仍按圆形车身检查，用来分离取整和形状的影响。",
        "#8b5cf6",
    ),
    "rectangle": (
        "连续观测＋朝向足印",
        "同一观测下，检查矩形车身的位置和朝向；实际运动仍由原控制器执行。",
        "#119c74",
    ),
    "baseline": (
        "原判据／无路即停",
        "估计距离小于30厘米并连续5帧就宣布到达；规划无路时直接终止。",
        "#dc7840",
    ),
    "uncertainty": (
        "加入位置误差预算",
        "估计距离加粒子95%半径及10厘米系统预算，不超过45厘米才计入连续停留。内部预算不是误差保证。",
        "#8b5cf6",
    ),
    "joint": (
        "预算＋停稳联合判断",
        "在位置预算条件上，额外要求上一帧执行线速度和角速度接近零。",
        "#119c74",
    ),
    "recovery": (
        "观测约束低速恢复",
        "仅修复附近的规划起点失效，雷达检查完整圆盘后实际低速移动；不移动估计坐标冒充恢复。",
        "#119c74",
    ),
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
LAYOUTS = {
    "plaza": "开阔场地",
    "street": "街区通道",
    "narrow": "两米窄通道",
    "offset_narrow": "留出偏置窄弯",
    "blocked": "预留空间不足的窄口",
}
OBSTACLES = {"crate": "普通箱体", "low": "14厘米路障", "beam": "悬空横杆"}


GROUP_NAMES = {
    "zero": "零扰动桥接",
    "y_plus": "位置向地图上方偏",
    "y_minus": "位置向地图下方偏",
    "yaw_plus": "朝向逆时针偏",
    "yaw_minus": "朝向顺时针偏",
    "delay": "执行延迟",
}


def condition_label(c):
    colors = ["#119c74", "#247cc1", "#8b5cf6", "#d68026", "#c53e51"]
    if c["dy_m"]:
        value = c["dy_m"] * 100
        return (
            f"y{value:+g}cm",
            "同一控制器；估计位置的世界y坐标恒加此误差，地图和目标不移动。",
            colors[[0.5, 1, 2, 4].index(abs(value)) + 1],
        )
    if c["yaw_deg"]:
        value = c["yaw_deg"]
        return (
            f"朝向{value:+g}°",
            "同一控制器；估计朝向恒加此角度。正为逆时针，负为顺时针；不是位置误差。",
            colors[[0.5, 1, 2, 4].index(abs(value)) + 1],
        )
    if c["delay_steps"]:
        value = c["delay_steps"]
        return (
            f"延迟{value * 0.1:.1f}s",
            "同一控制器；运动和制动请求都排队后执行，传感器即时、位姿准确。",
            colors[value],
        )
    return (
        "零扰动参照",
        "同一同比例制动控制器；精确定位、立即执行。是受控参照，不是定位算法的成绩。",
        colors[0],
    )


class NavigationStudySource:
    title = "67–68 课 · 车身标定、避障与错图修复"
    variant_label = "课程"

    def __init__(
        self,
        avoidance="results/observed_navigation_v1",
        repair="results/map_repair_v1",
        followups=None,
    ):
        self.directories = {
            k: Path(v) for k, v in {"67": avoidance, "68": repair}.items() if v is not None
        }
        self.directories.update({k: Path(v) for k, v in (followups or {}).items()})
        if followups:
            self.title = "67–71课 · 身体、规划、到点与恢复"
        self.summaries, self.entries = {}, []
        self.cache = {}
        self.robust_views = {}
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
                if summary["protocol"].get("round") == 4:
                    available = {
                        r["method"] for r in summary["rows"] if self.case_id(lesson, r) == case
                    }
                    groups = {
                        name: [k for k in keys if k in available]
                        for name, keys in summary["protocol"]["view_groups"].items()
                        if any(k in available for k in keys if k != "zero")
                    } or {"zero": ["zero"]}
                    for group, keys in groups.items():
                        view = case + "__" + group
                        self.robust_views[lesson, view] = (case, keys, group)
                        self.entries.append(
                            ReplayEntry(
                                view,
                                lesson,
                                f"69 · {GROUP_NAMES[group]} / {LAYOUTS[row['layout']]} / {OBSTACLES[row['obstacle']]} / 种子 {row['seed']}",
                            )
                        )
                    continue
                label = (
                    f"{lesson} · {LAYOUTS[row['layout']]} / {OBSTACLES[row['obstacle']]}"
                    if lesson in ("67", "69")
                    else (
                        "68 · 园区错图独立巡检"
                        if lesson == "68"
                        else f"{lesson} · {'到点判据' if lesson == '70' else '无路恢复'} / {LABELS[row['map_kind']][0]}"
                    )
                )
                self.entries.append(ReplayEntry(case, lesson, f"{label} / 种子 {row['seed']}"))
        self.entries = tuple(self.entries)

    @staticmethod
    def case_id(lesson, row):
        return (
            f"{lesson}_{row['layout']}_{row['obstacle']}_{row['seed']}"
            if lesson in ("67", "69")
            else (
                f"68_{row['seed']}"
                if lesson == "68"
                else f"{lesson}_{row['map_kind']}_{row['seed']}"
            )
        )

    def load(self, case, variant):
        if (case, variant) in self.cache:
            return self.cache[case, variant]
        summary, directory = self.summaries[variant], self.directories[variant]
        base_case, selected, group = self.robust_views.get((variant, case), (case, None, None))
        rows = {
            r["method"]: r
            for r in summary["rows"]
            if self.case_id(variant, r) == base_case
            and (selected is None or r["method"] in selected)
        }
        keys = {
            "67": ("stop", "lidar", "depth"),
            "68": ("raw", "rigid", "repaired", "reference"),
            "69": tuple(summary["protocol"]["methods"])
            if summary["protocol"].get("round", 1) >= 2
            else ("circle_grid", "circle_metric", "rectangle"),
            "70": ("baseline", "uncertainty", "joint"),
            "71": ("baseline", "recovery"),
        }[variant]
        keys = tuple(selected) if selected is not None else keys
        labels = (
            {k: condition_label(summary["protocol"]["conditions"][k]) for k in keys}
            if selected is not None
            else LABELS
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
            label, description, color = labels[key]
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
            if variant in ("67", "69"):
                depth[key] = {"values": pad(a["depth"]), "valid": pad(a["depth_valid"])}
                telemetry[key] = {k: pad(a[k]) for k in ("commands", "reason", "safety_clearance")}
                if "tracking_cross_m" in a:
                    telemetry[key].update(
                        {
                            k: pad(a[k])
                            for k in (
                                "tracking_cross_m",
                                "tracking_heading_rad",
                                "initial_cross_m",
                                "initial_heading_rad",
                                "expanded_gap_m",
                                "prior_expanded_gap_m",
                                "path_curvature_m_inv",
                                "tracking_angular_raw",
                            )
                        }
                    )
                if "braking_active" in a:
                    telemetry[key].update(
                        {
                            k: pad(a[k])
                            for k in (
                                "braking_active",
                                "incoming_velocity",
                                "phase",
                                "speed_cap",
                                "preview_curvature",
                                "nominal_command",
                                "stop_clearance",
                            )
                        }
                    )
                if "requested_command" in a:
                    telemetry[key].update(
                        {
                            k: pad(a[k])
                            for k in (
                                "requested_command",
                                "executed_target",
                                "executed_braking",
                                "executed_request_frame",
                                "pending_motion",
                            )
                        }
                    )
                # Never show the final accumulated obstacle map as a live-time map.
                # The shared prior is fixed; live returns and plans remain time-aligned.
                methods[key]["display_map"] = "实验开始时的先验地图"
            else:
                methods[key]["display_map"] = (
                    label if variant == "68" else LABELS[rows[key]["map_kind"]][0]
                )
                if variant in ("70", "71"):
                    telemetry[key] = {
                        k: pad(a[k])
                        for k in (
                            "commands",
                            "reason",
                            "arrival_distance",
                            "arrival_r95",
                            "arrival_budget",
                            "arrival_limit",
                            "arrival_ready",
                            "observed_stopped",
                            "recovery_active",
                        )
                    }
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
                f"{labels[k][0]} {sum(r['status'] == 'completed' for r in summary['rows'] if r['method'] == k)}/{sum(r['method'] == k for r in summary['rows'])} 回合完成"
                for k in keys
            ),
        }
        if variant in ("67", "69"):
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
                depth_used_by=list(keys) if variant == "69" else ["depth"],
            )
            camera = CameraSpec(rig.camera_z_m, rig.camera_x_m, rig.vfov_deg)
            dt = summary["protocol"]["dt_s"]
            if summary["protocol"].get("round", 1) >= 2:
                cohort = rows[keys[0]]["cohort"]
                round_number = summary["protocol"]["round"]
                metadata.update(
                    tracking_round=round_number,
                    settings=f"69课第{round_number}轮；55×44cm车身、6cm余量不变；零定位误差是设定，跟踪偏差另见诊断页",
                    cohort={
                        "fixed": "固定批次",
                        "heldout": "留出噪声种子",
                        "heldout_noise": "新噪声留出",
                        "heldout_layout": "偏置布局留出",
                        "negative": "过窄负例",
                        "development": "开发预试",
                        "bridge": "旧批次桥接",
                        "scan": "新种子单变量扫描",
                        "environment": "同条件环境参照",
                    }[cohort]
                    + "："
                    + "；".join(
                        f"{labels[k][0]} {sum(r['reached'] for r in summary['rows'] if r['method'] == k and r['cohort'] == cohort)}/{sum(r['method'] == k and r['cohort'] == cohort for r in summary['rows'])} 回合到达"
                        for k in keys
                    ),
                )
                if round_number == 3:
                    metadata["braking_models"] = {
                        k: (
                            "legacy"
                            if k == "legacy_brake"
                            else "coupled"
                            if k in ("coupled", "joint_brake")
                            else "independent"
                        )
                        for k in keys
                    }
                    metadata["sensor_note"] = (
                        "雷达/深度为存档；RGB为重渲染，无视觉定位。失败后实际刹停。"
                    )
                    aliases = dict(
                        zip(keys, ("旧参照", "独立", "同比例", "限速", "组合"), strict=True)
                    )
                    metadata["cohort"] = (
                        metadata["cohort"].split("：")[0]
                        + "到达："
                        + "；".join(
                            f"{aliases[k]} {sum(r['reached'] for r in summary['rows'] if r['method'] == k and r['cohort'] == cohort)}/{sum(r['method'] == k and r['cohort'] == cohort for r in summary['rows'])}"
                            for k in keys
                        )
                    )
                if round_number == 4:
                    metadata.update(
                        protocol="同一同比例制动算法，分别注入一种误差；不是不同定位算法。结束后保持末帧，保持帧不计分。",
                        settings=f"69课第四轮 · {GROUP_NAMES[group]}；车身和6cm余量不变；0.2s预测预留≠本轮实际延迟",
                        sensor_note="雷达/深度存档；RGB按真值重渲染。没有视觉定位。",
                        lidar_note="青点按有偏估计投影；点与墙错开可能是注入的位姿误差。无回波≠无障碍。",
                        braking_models={k: "coupled" for k in keys},
                        robustness_group=group,
                        cohort="本回合到达且余量合格："
                        + "；".join(
                            f"{labels[k][0]} {int(rows[k]['safe_completed'])}/1" for k in keys
                        ),
                    )
        else:
            grids = read_checked(directory / "maps.npz", summary["maps_sha256"])
            maps = (
                {LABELS[k][0]: grids[k] for k in keys}
                if variant == "68"
                else {LABELS[rows[keys[0]]["map_kind"]][0]: grids[rows[keys[0]]["map_kind"]]}
            )
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
            if variant in ("70", "71"):
                kind = rows[keys[0]]["map_kind"]
                metadata.update(
                    sensor_note="64束雷达与PF参与控制；RGB按实际位姿重渲染，未接入深度避障。",
                    scene_note="冻结同一张地图、同一粒子滤波；独立比较到点或恢复策略。",
                    settings="25cm半径；300粒子；真实45cm到点线不变；粒子集中不等于真实误差小。",
                    cohort="；".join(
                        f"{labels[k][0]} {sum(r['status'] == 'completed' for r in summary['rows'] if r['method'] == k and r['map_kind'] == kind)}/{sum(r['method'] == k and r['map_kind'] == kind for r in summary['rows'])} 轮完成"
                        for k in keys
                    ),
                )
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
