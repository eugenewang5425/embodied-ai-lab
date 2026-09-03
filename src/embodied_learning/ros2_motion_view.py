"""Motion-first rendering of frozen ROS receipts; no simulation or estimator writes."""

import numpy as np

from embodied_learning.differential_drive import compose, to_parent
from embodied_learning.landmark_localization import LANDMARKS, inverse_pose
from embodied_learning.odometry import heading_error

BLUE, PURPLE, ORANGE, GREEN = "#2563eb", "#9333ea", "#ea580c", "#15803d"


def motion_snapshot(rows, index, before=False):
    """Recover same-time pre-update pose from the received odometry increment.

    Ground truth is used ONLY for rendering/evaluation, never to predict a pose.
    No future measurement or regenerated noise is used.
    """
    row = rows[index]
    truth, odom, fused = (np.asarray(row[key]) for key in ("truth", "odom", "fused"))
    prior = fused.copy()
    action = "起点：准备按预设轮速出发"
    if index:
        previous = rows[index - 1]
        increment = compose(inverse_pose(previous["odom"]), odom)
        prior = compose(previous["fused"], increment)
        distance = np.linalg.norm(truth[:2] - np.asarray(previous["truth"])[:2])
        turn = heading_error(truth[2], previous["truth"][2])
        if distance < 1e-8 and abs(turn) > 1e-8:
            action = "原地左转：左右轮反向转动" if turn > 0 else "原地右转：左右轮反向转动"
        elif distance > 1e-8 and abs(turn) < 1e-8:
            action = "直行：左右轮同向、同速转动"
        elif distance > 1e-8:
            action = "沿弧线运动：左右轮速度不同"
        else:
            action = "停在原地"
    if index == len(rows) - 1:
        action = "预设动作执行完毕；不是反馈导航到达"
    showing_prior = bool(before and row["observation"])
    displayed = prior if showing_prior else fused
    return {
        "truth": truth,
        "odom": odom,
        "fused": displayed,
        "prior": prior,
        "posterior": fused,
        "before": showing_prior,
        "action": action,
        "error_before_cm": float(np.linalg.norm(prior[:2] - truth[:2]) * 100),
        "error_after_cm": float(np.linalg.norm(fused[:2] - truth[:2]) * 100),
    }


class MotionView:
    """One world view and one fixed-scale, truth-centred error magnifier."""

    def __init__(self, world, zoom, rows):
        self.world, self.zoom, self.rows = world, zoom, rows
        self.paths = {
            key: np.array([row[key] for row in rows]) for key in ("truth", "odom", "fused")
        }
        points = np.vstack([LANDMARKS] + [p[:, :2] for p in self.paths.values()])
        self.low, self.high = points.min(0) - 0.35, points.max(0) + 0.35
        offsets = [self.paths[k][:, :2] - self.paths["truth"][:, :2] for k in ("odom", "fused")]
        offsets += [
            np.array(
                [
                    motion_snapshot(rows, i)["prior"][:2] - rows[i]["truth"][:2]
                    for i in range(len(rows))
                ]
            )
        ]
        self.extent_cm = max(5.0, float(np.ceil(np.max(np.abs(np.vstack(offsets))) * 125 / 5) * 5))
        self.world_scale, self.zoom_scale = 1.0, 1.0

    def draw(self, index, before=False):
        state = motion_snapshot(self.rows, index, before)
        self.draw_world(index, state)
        self.draw_zoom(index, state)
        return state

    def draw_world(self, index, state):
        c = self.world
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 100:
            return
        self.world_scale = min(
            (w - 65) / np.ptp([self.low[0], self.high[0]]), (h - 85) / (self.high[1] - self.low[1])
        )

        def xy(point):
            return np.array([w / 2, h / 2 + 15]) + (
                np.asarray(point) - (self.low + self.high) / 2
            ) * [self.world_scale, -self.world_scale]

        font = ("Microsoft YaHei", 10)
        c.create_text(12, 12, anchor="nw", text="运动俯视图｜X 向右、Y 向上｜方格 0.5 m", font=font)
        for axis in (0, 1):
            for value in np.arange(np.ceil(self.low[axis] * 2) / 2, self.high[axis], 0.5):
                a, b = self.low.copy(), self.high.copy()
                a[axis] = b[axis] = value
                c.create_line(*xy(a), *xy(b), fill="#e2e8f0")
        # The full reference is explicitly labelled, not presented as travelled path.
        c.create_line(
            *[v for p in self.paths["truth"][:, :2] for v in xy(p)],
            fill="#cbd5e1",
            width=4,
            dash=(5, 5),
            tags="reference_route",
        )
        for key, color in (("truth", BLUE), ("odom", PURPLE), ("fused", ORANGE)):
            if index:
                trail = self.paths[key][: index + 1, :2].copy()
                if key == "fused":
                    trail[-1] = state["fused"][:2]
                c.create_line(
                    *[v for p in trail for v in xy(p)],
                    fill=color,
                    width=2,
                    dash=() if key == "truth" else (4, 3),
                    tags=f"{key}_trail",
                )
        for number, landmark in enumerate(LANDMARKS, 1):
            x, y = xy(landmark)
            c.create_polygon(x, y - 9, x - 8, y + 7, x + 8, y + 7, fill=GREEN, tags="landmark")
            c.create_text(x + 12, y, text=f"P{number}", anchor="w", fill=GREEN, font=font)
            if self.rows[index]["observation"]:
                c.create_line(
                    *xy(state["truth"][:2]), x, y, fill=GREEN, dash=(3, 6), tags="observation_ray"
                )
        # Body outline is a schematic; wheel separation follows the existing 0.30 m model.
        pose = state["truth"]
        body = [[-0.16, -0.11], [0.16, -0.11], [0.16, 0.11], [-0.16, 0.11]]
        c.create_polygon(
            *[v for p in body for v in xy(to_parent(pose, p))],
            fill="#dbeafe",
            outline=BLUE,
            width=3,
            tags="true_body",
        )
        for side in (-0.15, 0.15):
            c.create_line(
                *xy(to_parent(pose, [-0.075, side])),
                *xy(to_parent(pose, [0.075, side])),
                fill="#0f172a",
                width=5,
                tags="true_wheel",
            )
        c.create_line(
            *xy(pose[:2]),
            *xy(to_parent(pose, [0.30, 0])),
            fill=BLUE,
            width=3,
            arrow="last",
            tags="true_heading",
        )
        for key, color, radius in (("odom", PURPLE, 8), ("fused", ORANGE, 5)):
            x, y = xy(state[key][:2])
            c.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                outline=color,
                width=2,
                tags=f"{key}_marker",
            )
        x, y = xy(self.paths["truth"][0, :2])
        c.create_text(x, y + 29, text="起点", fill=BLUE, font=font)
        c.create_text(
            12,
            h - 12,
            anchor="sw",
            text="淡灰虚线＝预设路线；绿连线仅示意观测对象"
            if self.rows[index]["observation"]
            else "淡灰虚线＝预设路线；彩色线＝到当前为止的轨迹",
            font=font,
            fill="#475569",
        )

    def draw_zoom(self, index, state):
        c = self.zoom
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 100:
            return
        self.zoom_scale = min(w - 100, h - 150) / (2 * self.extent_cm)
        center = np.array([w / 2, h / 2 + 10])

        def xy(pose):
            return center + (np.asarray(pose)[:2] - state["truth"][:2]) * [
                100 * self.zoom_scale,
                -100 * self.zoom_scale,
            ]

        font = ("Microsoft YaHei", 10)
        c.create_text(12, 12, text="定位误差放大镜｜蓝点始终居中", anchor="nw", font=font)
        c.create_text(
            12,
            38,
            text="坐标轴仍与地图一致；标记不是车体尺寸",
            anchor="nw",
            font=font,
            fill="#475569",
        )
        span = self.extent_cm * self.zoom_scale
        for value in np.linspace(-self.extent_cm, self.extent_cm, 5):
            x, y = center + [value * self.zoom_scale, -value * self.zoom_scale]
            c.create_line(x, center[1] - span, x, center[1] + span, fill="#e2e8f0")
            c.create_line(center[0] - span, y, center[0] + span, y, fill="#e2e8f0")
            c.create_text(x, center[1] + span + 17, text=f"{value:+g}", font=font)
            c.create_text(center[0] - span - 7, y, text=f"{value:+g}", anchor="e", font=font)
        c.create_text(
            center[0] + span, center[1] + span + 39, text="ΔX / cm", anchor="e", font=font
        )
        c.create_text(
            center[0] - span, center[1] - span - 15, text="ΔY / cm", anchor="w", font=font
        )
        if self.rows[index]["observation"]:
            a, b = xy(state["prior"]), xy(state["posterior"])
            c.create_line(*a, *b, fill=ORANGE, width=3, arrow="last", tags="correction_arrow")
            c.create_oval(
                a[0] - 10,
                a[1] - 10,
                a[0] + 10,
                a[1] + 10,
                outline="#64748b",
                dash=(2, 2),
                tags="prior_marker",
            )
        for key, color, radius in (("truth", BLUE, 11), ("odom", PURPLE, 8), ("fused", ORANGE, 5)):
            x, y = xy(state[key])
            c.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                outline=color,
                width=3,
                tags=f"{key}_marker",
            )
            direction = np.array([np.cos(state[key][2]), -np.sin(state[key][2])])
            c.create_line(
                x,
                y,
                *(np.array([x, y]) + direction * (radius + 18)),
                fill=color,
                width=2,
                arrow="last",
                tags=f"{key}_heading",
            )
