"""Narrated, captioned film from checked lesson 67/68 archives (no physics reruns).

Generate WAV cues first with narrate_navigation_video.ps1. This renderer uses
stored poses without interpolation, labels replay time/speed, and saves an SRT,
an editable timed script, a source manifest and chapter thumbnails with the MP4.
Optional encoding dependency: uv pip install imageio-ffmpeg.
"""

import argparse
import bisect
import hashlib
import json
import math
import subprocess
import time
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from embodied_learning.navigation_geometry import (
    NavigationRig,
    body_points,
    depth_points,
    lidar_points,
    stop_path,
    world_points,
)
from embodied_learning.replay_viewer.body_overlay import draw_body_guides
from embodied_learning.replay_viewer.campus import read_checked
from embodied_learning.replay_viewer.goal_overlay import draw_goal
from embodied_learning.replay_viewer.navigation_studies import LABELS, NavigationStudySource
from embodied_learning.replay_viewer.scene import SceneRenderer

ROOT = Path(__file__).resolve().parents[1]
W, H = 1920, 1080
BG, PANEL, INK, MUTED = "#081420", "#132538", "#ecf4fc", "#a5b8cc"
CYAN, GOLD, RED, GREEN, PURPLE = "#57dbd0", "#f2bf5b", "#ff8076", "#65dfb0", "#c29aff"
METHOD_COLORS = {
    "stop": GOLD,
    "lidar": PURPLE,
    "depth": GREEN,
    "raw": "#f7ae68",
    "rigid": PURPLE,
    "repaired": GREEN,
    "reference": "#6ebfff",
}


@lru_cache(maxsize=32)
def font(size=32, bold=False):
    return ImageFont.truetype(
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", size
    )


def text(im, xy, value, size=32, color=INK, bold=False, anchor=None):
    ImageDraw.Draw(im).text(xy, str(value), font=font(size, bold), fill=color, anchor=anchor)


def wrap(value, max_width, size):
    lines, line = [], ""
    for char in value:
        if char == "\n":
            lines.append(line)
            line = ""
            continue
        if line and font(size).getlength(line + char) > max_width:
            lines.append(line)
            line = char
        else:
            line += char
    if line:
        lines.append(line)
    return lines


@lru_cache(maxsize=64)
def subtitle_lines(value):
    """Balance two lines so a final word cannot hang alone under a full line."""
    if font(40).getlength(value) <= 1720:
        return [value]
    candidates = []
    for split in range(1, len(value)):
        left, right = value[:split], value[split:]
        numbers = "零一二三四五六七八九十百千万点0123456789"
        if right[0] in "，。；：！？" or (left[-1] in numbers and right[0] in numbers):
            continue
        a, b = font(40).getlength(left), font(40).getlength(right)
        if max(a, b) <= 1720:
            penalty = 0 if left[-1] in "，。；：！？" else 500
            candidates.append((abs(a - b) + penalty, left, right))
    if not candidates:
        raise ValueError(f"Subtitle exceeds two lines: {value}")
    _, left, right = min(candidates)
    return [left, right]


def paragraph(im, xy, value, width, size=30, color=MUTED, gap=12):
    x, y = xy
    for line in wrap(value, width, size):
        text(im, (x, y), line, size, color)
        y += size + gap
    return y


def box(im, bounds, color=PANEL, outline=None, radius=20):
    ImageDraw.Draw(im).rounded_rectangle(
        bounds, radius=radius, fill=color, outline=outline, width=2
    )


def arrow(im, a, b, color=GOLD, width=3, both=False):
    d = ImageDraw.Draw(im)
    d.line([a, b], fill=color, width=width)
    for start, end in ((a, b), (b, a)) if both else ((a, b),):
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        pts = [
            end,
            (end[0] - 14 * math.cos(angle - 0.45), end[1] - 14 * math.sin(angle - 0.45)),
            (end[0] - 14 * math.cos(angle + 0.45), end[1] - 14 * math.sin(angle + 0.45)),
        ]
        d.polygon(pts, fill=color)


def star(im, xy, radius=14, color=GOLD):
    x, y = xy
    pts = [
        (
            x
            + (radius if j % 2 == 0 else radius * 0.43) * math.cos(-math.pi / 2 + j * math.pi / 5),
            y
            + (radius if j % 2 == 0 else radius * 0.43) * math.sin(-math.pi / 2 + j * math.pi / 5),
        )
        for j in range(10)
    ]
    ImageDraw.Draw(im).polygon(pts, fill=color)


def srt_time(seconds):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def timeline(story, out, story_path):
    """Use actual synthesized durations, not guessed subtitle reading times."""
    provenance = json.loads((out / "narration/manifest.json").read_text(encoding="utf-8-sig"))
    if provenance["story_sha256"] != hashlib.sha256(story_path.read_bytes()).hexdigest():
        raise ValueError(
            "Story changed after narration: regenerate WAV cues in a fresh output directory"
        )
    for item in provenance["files"]:
        audio_path = out / "narration" / item["name"]
        if hashlib.sha256(audio_path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Narration hash mismatch: {audio_path}")
    fps = story["fps"]
    sr = None
    pcm = []
    scenes = []
    cues = []
    offset = 0
    idx = 0
    for scene in story["scenes"]:
        begin = offset
        for cue in scene["cues"]:
            path = out / "narration" / f"{idx:03}.wav"
            with wave.open(str(path)) as audio:
                if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                    raise ValueError("Narration must be mono signed 16-bit PCM")
                if sr is not None and audio.getframerate() != sr:
                    raise ValueError("Narration sample rates differ")
                sr = audio.getframerate()
                samples = np.frombuffer(audio.readframes(audio.getnframes()), "<i2").copy()
            if len(samples) == 0 or np.max(np.abs(samples.astype(float))) < 100:
                raise ValueError(f"Empty or silent narration: {path}")
            lead, tail = 0.18, 0.40
            duration_frames = math.ceil((lead + len(samples) / sr + tail) * fps)
            duration = duration_frames / fps
            total_samples = round((offset + duration) * sr) - round(offset * sr)
            padded = np.zeros(total_samples, dtype="<i2")
            start = round(lead * sr)
            padded[start : start + len(samples)] = samples
            pcm.append(padded)
            cues.append(
                {
                    "index": idx,
                    "scene": scene["id"],
                    "start": offset + lead,
                    "end": offset + lead + len(samples) / sr + 0.12,
                    "text": cue,
                }
            )
            offset += duration
            idx += 1
        scenes.append({**scene, "start": begin, "end": offset, "duration": offset - begin})
    signal = np.concatenate(pcm)
    with wave.open(str(out / "narration.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sr)
        audio.writeframes(signal.tobytes())
    srt = (
        "\n\n".join(
            f"{c['index'] + 1}\n{srt_time(c['start'])} --> {srt_time(c['end'])}\n"
            + "\n".join(subtitle_lines(c["text"]))
            for c in cues
        )
        + "\n"
    )
    (out / "navigation-67-68.zh-CN.srt").write_text(srt, encoding="utf-8-sig")
    markdown = [
        "# 67—68课展示视频：分镜与配音稿",
        "",
        "中文配音：Windows 本地合成；实验运动来自存档回放。",
        "",
    ]
    for scene in scenes:
        markdown += [
            f"## {srt_time(scene['start'])[:-4]} — {scene['title']}",
            "",
            f"画面模块：`{scene['visual']}`",
            "",
        ]
        markdown += [c["text"] + "\n" for c in cues if c["scene"] == scene["id"]]
    (out / "script.md").write_text("\n".join(markdown), encoding="utf8")
    (out / "timeline.json").write_text(
        json.dumps(
            {"scenes": scenes, "cues": cues, "duration": offset}, ensure_ascii=False, indent=2
        ),
        encoding="utf8",
    )
    chapters = [
        ";FFMETADATA1",
        "title=" + story["title"],
        "comment=Archived simulation replay; Chinese synthetic narration",
    ]
    for scene in scenes:
        chapters += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={round(scene['start'] * 1000)}",
            f"END={round(scene['end'] * 1000)}",
            "title="
            + scene["title"]
            .replace("\\", "\\\\")
            .replace("=", "\\=")
            .replace(";", "\\;")
            .replace("#", "\\#"),
        ]
    (out / "chapters.ffmetadata").write_text("\n".join(chapters) + "\n", encoding="utf8")
    return scenes, cues, offset


class Film:
    def __init__(self, source, out):
        self.source, self.out = source, out
        self.records, self.engines, self.images = {}, {}, {}
        self.static = {}
        self.rig = NavigationRig()
        self.repair_summary = source.summaries["68"]
        self.grids = read_checked(
            source.directories["68"] / "maps.npz", self.repair_summary["maps_sha256"]
        )
        self.benchmark = json.loads(
            (ROOT / "docs/benchmarks/navigation-studies-67-68-v1.json").read_text(encoding="utf8")
        )

    def record(self, case):
        if case not in self.records:
            self.records[case] = self.source.load(case, case[:2])
        return self.records[case]

    def render(self, case, key, frame, size, view="overview", orbit=None):
        cache_key = case, key, view, size, frame, orbit
        if cache_key in self.images:
            return self.images[cache_key]
        engine_key = case, key, view
        if engine_key not in self.engines:
            e = SceneRenderer()
            e.load(self.record(case))
            e.focus_key = key
            e.camera.azimuth = 90 if case.startswith("67") else 135
            e.camera.elevation = -62 if case.startswith("67") else -55
            e.camera.distance = 11.5 if case.startswith("67") else 36
            self.engines[engine_key] = e
        e = self.engines[engine_key]
        if orbit is not None:
            e.camera.azimuth = orbit
        r = self.record(case)
        result = e.render(frame, size, r.method_tracks(key), True, view)
        if view == "robot":
            result = draw_goal(result, r, frame, key)
            result = draw_body_guides(result, r, frame, key)
        if len(self.images) > 18:
            self.images.clear()
        self.images[cache_key] = result
        return result

    @staticmethod
    def frame(record, seconds):
        return max(
            0,
            min(
                len(record.timestamps) - 1,
                int(np.searchsorted(record.timestamps, seconds, side="right") - 1),
            ),
        )

    def chart(self, im, bounds, r, key, frame, region=None, show_estimate=True, diagnostic=False):
        x0, y0, x1, y1 = bounds
        region = region or r.extent
        xmin, xmax, ymin, ymax = region
        scale = min((x1 - x0) / (xmax - xmin), (y1 - y0) / (ymax - ymin))
        pw, ph = (xmax - xmin) * scale, (ymax - ymin) * scale
        px, py = x0 + ((x1 - x0) - pw) / 2, y0 + ((y1 - y0) - ph) / 2
        point = lambda p: (px + (p[0] - xmin) * scale, py + (ymax - p[1]) * scale)
        d = ImageDraw.Draw(im)
        d.rectangle((px, py, px + pw, py + ph), fill="#0a1827", outline="#3a4c61", width=2)
        for x in np.arange(math.ceil(xmin), xmax, 2):
            d.line([point((x, ymin)), point((x, ymax))], fill="#203347")
        for y in np.arange(math.ceil(ymin), ymax, 2):
            d.line([point((xmin, y)), point((xmax, y))], fill="#203347")
        if r.metadata.get("rig"):
            for p in r.primitives:
                lo, hi = (
                    point((max(xmin, p.cx - p.hx), min(ymax, p.cy + p.hy))),
                    point((min(xmax, p.cx + p.hx), max(ymin, p.cy - p.hy))),
                )
                if hi[0] >= lo[0] and hi[1] >= lo[1]:
                    d.rectangle((*lo, *hi), fill="#43566a")
        else:
            occ = r.maps[r.method_info(key)["display_map"]]
            yy, xx = np.nonzero(occ)
            for x, y in zip((xx + 0.5) * r.resolution_m, (yy + 0.5) * r.resolution_m, strict=True):
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    u, v = point((x, y))
                    d.rectangle((u - 1, v - 1, u + 1, v + 1), fill="#53657a")
        n = min(frame + 1, r.lengths[key])
        color = METHOD_COLORS[key]
        truth = r.track_by_key[r.true_key(key)].poses[:n]
        estimate = r.track_by_key[key].poses[:n]
        if n > 1:
            d.line([point(p) for p in truth], fill="#edf5ff" if show_estimate else color, width=5)
            if show_estimate:
                for j in range(0, n - 1, 5):
                    d.line([point(p) for p in estimate[j : min(n, j + 3)]], fill=color, width=4)
        scan = r.scan_world(frame, key)
        for p in scan:
            if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax:
                u, v = point(p)
                d.ellipse((u - 2, v - 2, u + 2, v + 2), fill=CYAN)
        if diagnostic:
            x = np.linspace(1.5, 10.5, 1801)
            t = np.clip((x - 1.5) / 3.4, 0, 1)
            y = 6 - 0.7 * (3 * t * t - 2 * t * t * t)
            for j in range(0, len(x) - 1, 32):
                d.line(
                    [point(p) for p in zip(x[j : j + 16], y[j : j + 16], strict=True)],
                    fill="#65baff",
                    width=4,
                )
        goal = r.goal_info(frame, key)
        if goal:
            star(im, point(goal[1]), 17)
        u, v = point(truth[-1])
        d.ellipse((u - 7, v - 7, u + 7, v + 7), fill=color, outline=INK, width=2)
        text(
            im,
            (px + pw, py + ph + 8),
            f"x: {xmin:g}—{xmax:g} m    y: {ymin:g}—{ymax:g} m",
            20,
            MUTED,
            anchor="rt",
        )

    def body(self, im):
        box(im, (60, 220, 930, 828))
        box(im, (954, 220, 1860, 828))
        text(im, (90, 246), "俯视：距离从哪里开始量？", 34, INK, True)
        d = ImageDraw.Draw(im)
        ox, oy, s = 330, 535, 530
        p = lambda x, y: (ox + x * s, oy - y * s)
        d.rectangle((*p(-0.25, 0.22), *p(0.30, -0.22)), fill="#24566c", outline=CYAN, width=4)
        d.rectangle((*p(0.70, 0.43), *p(0.77, -0.43)), fill="#9a5557")
        d.ellipse((p(0.08, 0)[0] - 10, oy - 10, p(0.08, 0)[0] + 10, oy + 10), fill=CYAN)
        arrow(im, p(0.08, 0.34), p(0.70, 0.34), CYAN, both=True)
        text(im, (460, 322), "雷达读数 62 cm", 28, CYAN, anchor="mm")
        arrow(im, p(0.30, -0.34), p(0.70, -0.34), GOLD, both=True)
        text(im, (590, 745), "车头间隙 40 cm", 28, GOLD, anchor="mm")
        text(im, (315, 485), "车身", 36, INK, True, anchor="mm")
        text(im, (210, 735), "长 55 cm / 宽 44 cm", 25, MUTED, anchor="mm")
        text(im, (366, 572), "雷达", 24, CYAN, anchor="mm")
        text(im, (764, 540), "墙", 30, INK, anchor="mm")
        text(im, (988, 246), "侧视：底盘、支架与相机一起检查", 34, INK, True)
        ox, ground, s = 1220, 726, 600
        pp = lambda x, z: (ox + x * s, ground - z * s)
        for part in self.rig.parts:
            d.rectangle(
                (*pp(part[0], part[5]), *pp(part[1], part[4])),
                fill="#24566c",
                outline=CYAN,
                width=3,
            )
        d.line([pp(-0.4, 0), pp(0.9, 0)], fill=MUTED, width=3)
        d.ellipse((1258, 526, 1278, 546), fill=CYAN)
        arrow(im, (1338, 396), (1510, 353), PURPLE)
        text(im, (1516, 330), "相机高 55 cm", 28, PURPLE)
        arrow(im, (1270, 535), (1510, 535), CYAN)
        text(im, (1516, 514), "雷达高 32 cm", 28, CYAN)
        text(im, (990, 765), "总高 58 cm；侧向再留 6 cm 余量", 28, GOLD)

    def brake(self, im, progress):
        box(im, (60, 220, 1090, 828))
        box(im, (1114, 220, 1860, 828))
        text(im, (90, 247), "黄色：制动过程中车身经过的区域", 32, INK, True)
        d = ImageDraw.Draw(im)
        sx, sy, scale = 260, 540, 650
        conv = lambda p: (sx + p[0] * scale, sy - p[1] * scale)
        path = stop_path(0.8, 0, self.rig)
        footprint = np.array(
            [[-0.25, -0.22, 0], [0.30, -0.22, 0], [0.30, 0.22, 0], [-0.25, 0.22, 0]]
        )
        shown = max(2, int(min(1, progress * 2) * len(path)))
        for pose in path[:shown:2]:
            pts = [conv(p) for p in world_points(footprint, pose)]
            d.polygon(pts, fill="#75602e", outline=GOLD, width=2)
        d.line([conv(p) for p in path[:shown]], fill=INK, width=4)
        d.polygon([conv(p) for p in footprint], outline=CYAN, width=5)
        d.rectangle(
            (sx + 0.70 * scale, 350, sx + 0.76 * scale, 720), fill="#a7595a", outline=RED, width=3
        )
        text(im, (90, 748), "中心还没到墙，前端已经会碰到", 30, RED)
        text(im, (1150, 252), "校准条件", 30, MUTED)
        text(im, (1150, 308), "0.8 m/s", 70, INK, True)
        text(im, (1150, 403), "预留延迟 0.20 s", 32, MUTED)
        text(im, (1150, 450), "制动减速度 0.8 m/s²", 32, MUTED)
        text(im, (1150, 525), "0.16 + 0.40 = 0.56 m", 40, GOLD, True)
        paragraph(
            im,
            (1150, 592),
            "这是连续公式的中心位移。\n还需加车头前伸 30 厘米，\n以及安全余量 6 厘米。",
            645,
            30,
        )
        text(im, (1150, 760), "转弯：同时计算旋转后的车角", 29, CYAN)

    def height(self, im):
        d = ImageDraw.Draw(im)
        for j, (title, span, desc, color) in enumerate(
            [
                ("普通箱体", (0, 1), "穿过雷达平面", GREEN),
                ("14 cm 矮路障", (0, 0.14), "在扫描平面下方", RED),
                ("悬空横杆", (0.42, 0.57), "在扫描平面上方", RED),
            ]
        ):
            x = 60 + j * 610
            box(im, (x, 220, x + 586, 828))
            text(im, (x + 32, 251), title, 36, INK, True)
            ground = 700
            s = 360
            d.line((x + 35, ground, x + 550, ground), fill=MUTED, width=3)
            d.rectangle(
                (x + 65, ground - 0.30 * s, x + 210, ground), fill="#24566c", outline=CYAN, width=3
            )
            d.rectangle((x + 160, ground - 0.58 * s, x + 175, ground - 0.30 * s), fill=CYAN)
            d.rectangle(
                (x + 370, ground - span[1] * s, x + 475, ground - span[0] * s),
                fill="#a8674a",
                outline=GOLD,
                width=3,
            )
            laser_y = ground - 0.32 * s
            end_x = x + (370 if j == 0 else 525)
            d.line((x + 185, laser_y, end_x, laser_y), fill=CYAN, width=4)
            if j == 0:
                d.ellipse((end_x - 7, laser_y - 7, end_x + 7, laser_y + 7), fill=CYAN)
            text(im, (x + 40, 385), "扫描高度 32 cm", 27, CYAN)
            text(im, (x + 30, 753), desc, 33, color, True)

    def compare(self, im, progress, duration):
        r = self.record("67_plaza_low_0")
        sim = progress * r.timestamps[-1]
        f = self.frame(r, sim)
        for j, key in enumerate(("lidar", "depth")):
            x = 60 + j * 910
            box(im, (x, 220, x + 890, 828))
            color = METHOD_COLORS[key]
            text(im, (x + 24, 237), LABELS[key][0], 34, color, True)
            frame = min(f, r.lengths[key] - 1)
            image = self.render("67_plaza_low_0", key, frame, (840, 320))
            im.paste(image, (x + 24, 288))
            self.chart(im, (x + 28, 626, x + 862, 757), r, key, frame, (1, 11, 5.0, 7.5), False)
            ended = f >= r.lengths[key] - 1
            status = ("发生碰撞，末帧保持" if key == "lidar" else "到达终点") if ended else "执行中"
            text(
                im,
                (x + 24, 784),
                f"t={r.timestamps[frame]:.2f}s · {status}",
                27,
                RED if ended and key == "lidar" else color,
            )
        return f"种子0 · 低障碍 · 按存档时间同步回放 {r.timestamps[-1] / duration:.2f}×；图为真实运动，未插值"

    def camera(self, im, progress):
        r = self.record("67_plaza_low_0")
        f = self.frame(r, 3 + progress * 4.5)
        key = "depth"
        box(im, (60, 220, 940, 828))
        box(im, (964, 220, 1860, 828))
        text(im, (86, 240), "第一视角 · 按实际位姿重渲染", 32, INK, True)
        image = self.render("67_plaza_low_0", key, f, (640, 480), "robot").resize(
            (672, 504), Image.Resampling.LANCZOS
        )
        im.paste(image, (160, 300))
        text(im, (992, 240), "车身坐标：当前点与过去观测", 32, INK, True)
        d = ImageDraw.Draw(im)
        ox, oy, scale = 1390, 672, 184
        xy = lambda p: (ox - p[1] * scale, oy - p[0] * scale)
        pose = r.track_by_key[key].poses[f]
        dep = r.depth[key]
        for j in range(max(0, f - 30), f, 3):
            points = body_points(
                world_points(
                    depth_points(dep["values"][j], dep["valid"][j], self.rig)[::8],
                    r.track_by_key[key].poses[j],
                ),
                pose,
            )
            for p in points:
                if -0.5 < p[0] < 1.7 and abs(p[1]) < 1.8:
                    x, y = xy(p)
                    d.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#66778b")
        cloud = depth_points(dep["values"][f], dep["valid"][f], self.rig)
        for p in cloud[::5]:
            if -0.5 < p[0] < 1.7 and abs(p[1]) < 1.8:
                x, y = xy(p)
                d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=PURPLE)
        scan = r.lidar[key]
        cloud = lidar_points(scan["ranges"][f], scan["hits"][f], self.rig)
        for p in cloud:
            if -0.5 < p[0] < 1.7 and abs(p[1]) < 1.8:
                x, y = xy(p)
                d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=CYAN)
        footprint = np.array(
            [[-0.25, -0.22, 0], [0.30, -0.22, 0], [0.30, 0.22, 0], [-0.25, 0.22, 0]]
        )
        for pose in stop_path(*r.telemetry[key]["commands"][f], self.rig)[::3]:
            d.polygon([xy(p) for p in world_points(footprint, pose)], outline=GOLD, width=2)
        d.polygon([xy(p) for p in footprint], outline=INK, width=4)
        arrow(im, (1390, 590), (1390, 544), INK)
        text(im, (1390, 738), "55 × 44 cm 车身", 28, INK, anchor="mm")
        text(im, (1000, 300), "紫：当前深度   青：雷达   灰：过去3秒抽样", 24, MUTED)
        text(im, (1000, 779), "黄框是本帧速度的刹停扫掠（未加余量）", 27, GOLD)
        return f"t={r.timestamps[f]:.2f}s · 深度参与控制，RGB仅重渲染；灰点只使用当前帧之前的数据"

    def results(self, im):
        box(im, (60, 220, 1860, 828))
        headers = ["方法 / 每组27回合", "到达", "碰撞", "停滞", "拒绝"]
        xs = [96, 806, 1065, 1324, 1583]
        for x, h, c in zip(xs, headers, [INK, GREEN, RED, GOLD, PURPLE], strict=True):
            text(im, (x, 261), h, 34, c, True)
        rows = self.source.summaries["67"]["rows"]
        for j, key in enumerate(("stop", "lidar", "depth")):
            y = 363 + j * 125
            text(im, (96, y + 15), LABELS[key][0], 36, METHOD_COLORS[key], True)
            for x, status in zip(
                xs[1:], ["completed", "collision_abort", "stalled", "no_path"], strict=True
            ):
                count = sum(r["method"] == key and r["status"] == status for r in rows)
                text(im, (x + 45, y), str(count), 68, INK, True)
        text(im, (96, 770), "3种布局 × 3种障碍 × 3个种子；本课定位误差为零是受控设定", 30, MUTED)

    def narrow(self, im):
        box(im, (60, 220, 1270, 828))
        box(im, (1294, 220, 1860, 828))
        r = self.record("67_narrow_crate_0")
        text(im, (90, 248), "融合组在起点拒绝；蓝线为事后几何诊断", 33, INK, True)
        self.chart(im, (95, 366, 1235, 687), r, "depth", 0, (1, 11, 4.6, 7.4), False, True)
        text(im, (96, 765), "蓝线没有送入控制器，不计作到达", 32, "#65baff")
        text(im, (1325, 257), "通道边缘剩余", 28, MUTED)
        text(im, (1325, 307), "60 cm", 68, INK, True)
        text(im, (1325, 402), "实际车宽 44 cm", 32, CYAN)
        text(im, (1325, 459), "规划膨胀半径 50 cm", 30, GOLD)
        text(im, (1325, 551), "诊断路径最小间隙", 28, MUTED)
        text(im, (1325, 595), "6.64 cm", 62, GREEN, True)
        paragraph(im, (1325, 704), "下一步：按朝向检查矩形足印，保留安全余量。", 495, 28)

    def repair(self, im):
        d = ImageDraw.Draw(im)
        for j, key in enumerate(("raw", "rigid", "repaired")):
            x = 60 + j * 610
            box(im, (x, 220, x + 586, 828))
            text(im, (x + 24, 242), LABELS[key][0], 33, METHOD_COLORS[key], True)
            # Draw saved endpoint cells, with identical origin and scale in each panel.
            ox, oy, scale = x + 75, 746, 18.0
            for grid, color in [
                (self.grids["projected"], "#7b8a9b"),
                (self.grids[key], METHOD_COLORS[key]),
            ]:
                yy, xx = np.nonzero(grid)
                for u, v in zip((xx + 0.5) * 0.1, (yy + 0.5) * 0.1, strict=True):
                    px, py = ox + u * scale, oy - v * scale
                    d.rectangle((px - 1, py - 1, px + 1, py + 1), fill=color)
            # The plot starts below the card title.
            error = self.repair_summary["quality"][key]["symmetric_mean_m"] * 100
            text(im, (x + 25, 764), f"墙面平均距离 {error:.1f} cm", 31, METHOD_COLORS[key], True)

    def navigation(self, im, progress, duration):
        r = self.record("68_1")
        t = progress * r.timestamps[-1]
        f = self.frame(r, t)
        for j, key in enumerate(("raw", "repaired")):
            x = 60 + j * 910
            box(im, (x, 220, x + 890, 828))
            text(im, (x + 26, 239), LABELS[key][0], 34, METHOD_COLORS[key], True)
            self.chart(im, (x + 24, 302, x + 610, 758), r, key, f)
            rows = [q for q in self.repair_summary["rows"] if q["method"] == key]
            reached = sum(q["reached"] for q in rows)
            text(im, (x + 620, 327), "三轮真实到点", 27, MUTED)
            text(im, (x + 620, 380), f"{reached}/24", 59, METHOD_COLORS[key], True)
            text(im, (x + 620, 516), "当前轮最终", 27, MUTED)
            text(im, (x + 620, 565), f"{r.method_info(key)['result']['reached']}/8", 56, INK, True)
            end = r.lengths[key] - 1
            phase = "已结束，保持末帧" if f > end else "巡检执行中"
            text(im, (x + 26, 784), f"t={r.timestamps[min(f, end)]:.2f}s · {phase}", 27, MUTED)
        return f"种子1 · 回放 {r.timestamps[-1] / duration:.2f}× · 浅色实线=实际轨迹，彩色虚线=估计轨迹；各组用自己的地图"

    def failure(self, im, progress):
        r = self.record("68_1")
        event = next(e for e in r.method_info("repaired")["result"]["events"] if not e["success"])
        f = min(event["frame"], 430 + int(progress * 26))
        goal = np.array([12.0, 12.0])
        true = r.track_by_key[r.true_key("repaired")].poses[f, :2]
        est = r.track_by_key["repaired"].poses[f, :2]
        box(im, (60, 220, 1120, 828))
        box(im, (1144, 220, 1860, 828))
        d = ImageDraw.Draw(im)
        text(im, (90, 246), "第4个点局部放大 · 固定目标做事后复盘", 31, INK, True)
        conv = lambda p: (380 + (p[0] - 12) * 350, 560 - (p[1] - 12) * 350)
        gx, gy = conv(goal)
        radius = 0.45 * 350
        d.ellipse(
            (gx - radius, gy - radius, gx + radius, gy + radius),
            fill="#173d38",
            outline=GREEN,
            width=4,
        )
        star(im, (gx, gy), 20)
        for p, c in [(true, INK), (est, PURPLE)]:
            x, y = conv(p)
            d.line([(gx, gy), (x, y)], fill=c, width=3)
            d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=c)
        text(im, (115, 742), "绿圈：真实验收半径45 cm", 28, GREEN)
        text(im, (570, 694), "白点：实际位置", 28, INK)
        text(im, (570, 736), "紫点：算法估计位置", 28, PURPLE)
        text(im, (1176, 254), "提前宣布到达的那一帧", 30, MUTED)
        text(im, (1176, 329), "实际距离 46.09 cm", 40, RED, True)
        text(im, (1176, 406), "验收门限 45.00 cm", 38, INK)
        text(im, (1176, 509), "最终只记 7 / 8", 57, GOLD, True)
        paragraph(
            im,
            (1176, 622),
            "定位均值更小，不等于每个点都真正到达。门限保持不变，失败保留。",
            626,
            32,
        )
        return (
            f"种子1 · 第4点在53.64s误报到达；局部图显示 t={r.timestamps[f]:.2f}s，右侧为事件最终值"
        )

    def conclusion(self, im):
        for j, (title, main, body, col) in enumerate(
            [
                ("长平行走廊", "几何约束不足", "贴墙很准，也可能不知道沿墙走了多远。", PURPLE),
                ("窄通道", "身体近似过保守", "圆形膨胀可能堵住真实矩形车能走的路。", GOLD),
                ("不同高度障碍", "观测存在盲区", "增加二维雷达射线，也补不出缺失的高度。", CYAN),
            ]
        ):
            x = 60 + j * 610
            box(im, (x, 220, x + 586, 620))
            text(im, (x + 30, 254), title, 35, col, True)
            text(im, (x + 30, 334), main, 40, INK, True)
            paragraph(im, (x + 30, 432), body, 515, 33)
        box(im, (60, 644, 1860, 828), color="#172d3d")
        text(
            im,
            (94, 670),
            "当前边界：仿真几何标定 + 静态避障 + 恒定尺度偏差的离线修图",
            35,
            GOLD,
            True,
        )
        text(
            im,
            (94, 736),
            "下一步：矩形足印规划 → 到点判据与恢复 → 两课统一机器人后组合验收",
            32,
            INK,
        )

    def make(self, scene, seconds):
        static = scene["visual"] in {
            "body",
            "height",
            "results",
            "narrow",
            "repair",
            "conclusion",
            "window",
        }
        if static and scene["id"] in self.static:
            return self.static[scene["id"]].copy()
        im = Image.new("RGB", (W, H), BG)
        p = (seconds - scene["start"]) / scene["duration"]
        visual = scene["visual"]
        text(im, (60, 27), "EMBODIED AI LAB  /  第67—68课", 25, CYAN, True)
        text(im, (1860, 27), scene["chapter"], 25, MUTED, anchor="rt")
        text(im, (60, 86), scene["title"], 48, INK, True)
        text(im, (60, 159), "让观察、身体、地图和任务结果对应起来", 27, MUTED)
        note = "实验存档与几何示意 · 尺寸为仿真设定，未声称完成实车标定"
        if visual == "intro":
            box(im, (60, 220, 660, 828))
            text(im, (100, 285), "观察", 66, CYAN, True)
            text(im, (100, 397), "身体", 66, GOLD, True)
            text(im, (100, 509), "地图", 66, GREEN, True)
            paragraph(im, (100, 667), "三种问题，三条证据链。保留收益，也保留失败。", 500, 34)
            r = self.record("68_1")
            f = self.frame(r, 20 + p * 12)
            im.paste(
                self.render("68_1", "repaired", f, (1160, 608), orbit=135 + round(p * 15, 1)),
                (700, 220),
            )
            note = "MuJoCo按存档实际位姿重渲染；金色=当前目标/规划，青色=记录中的雷达回波"
        elif visual == "body":
            self.body(im)
        elif visual == "brake":
            self.brake(im, p)
        elif visual == "height":
            self.height(im)
        elif visual == "compare":
            note = self.compare(im, p, scene["duration"])
        elif visual == "camera":
            note = self.camera(im, p)
        elif visual == "results":
            self.results(im)
            note = "统计来自81个完整回合；拒绝、停滞与碰撞分别计数，不把零碰撞写成全成功"
        elif visual == "narrow":
            self.narrow(im)
            note = "1801个姿态的独立几何检查；蓝线只解释规划保守性，没有替换机器人实际轨迹"
        elif visual == "repair":
            self.repair(im)
            note = (
                "三图同尺度24×24m；灰色=同扫描按真值落点的评分参照，彩色=候选图；真值不交给修图器"
            )
        elif visual == "navigation":
            note = self.navigation(im, p, scene["duration"])
        elif visual == "failure":
            note = self.failure(im, p)
        elif visual == "conclusion":
            self.conclusion(im)
        elif visual == "window":
            photo = Image.open(ROOT / "docs/img/navigation-study-body-window.png").convert("RGB")
            photo = ImageOps.contain(photo, (1160, 608))
            im.paste(photo, (60, 220))
            box(im, (1250, 220, 1860, 828))
            text(im, (1280, 265), "一键联动", 45, CYAN, True)
            paragraph(
                im,
                (1280, 363),
                "相机 · 三维场景\n雷达 · 规划目标\n轨迹 · 偏差统计\n车身与刹停标定",
                530,
                36,
                gap=22,
            )
            text(im, (1280, 682), "运行仓库入口", 28, MUTED)
            text(im, (1280, 735), "open_navigation_study.cmd", 25, GOLD, True)
            note = "真实桌面窗口截图（此页为静态界面说明）；视频、字幕、剧本和复现入口一起交付"
        else:
            raise ValueError(visual)
        text(im, (60, 850), note, 23, MUTED)
        if static:
            self.static[scene["id"]] = im.copy()
        return im

    def close(self):
        for e in self.engines.values():
            e.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--story", type=Path, default=ROOT / "docs/video/navigation-67-68-story.json")
    p.add_argument("--output", type=Path, default=ROOT / "results/navigation_video_v1")
    p.add_argument("--preview", action="store_true", help="render representative stills only")
    p.add_argument("--ffmpeg", type=Path)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    story = json.loads(args.story.read_text(encoding="utf8"))
    if story["resolution"] != [W, H]:
        raise ValueError("This film layout requires 1920 x 1080")
    scenes, cues, duration = timeline(story, out, args.story)
    source = NavigationStudySource(
        ROOT / "results/observed_navigation_v1", ROOT / "results/map_repair_v1"
    )
    film = Film(source, out)
    fps = story["fps"]
    ends = [s["end"] for s in scenes]
    cue_starts = [c["start"] for c in cues]

    def compose(scene, t):
        im = film.make(scene, t)
        d = ImageDraw.Draw(im)
        box(im, (48, 894, 1872, 1036), color="#162b3e")
        idx = bisect.bisect_right(cue_starts, t) - 1
        if idx >= 0 and t <= cues[idx]["end"]:
            lines = subtitle_lines(cues[idx]["text"])
            if len(lines) > 2:
                raise ValueError("Subtitle exceeds two lines")
            for j, line in enumerate(lines):
                text(
                    im, (960, 920 + j * 50 if len(lines) == 2 else 946), line, 40, INK, anchor="mt"
                )
        d.rectangle((60, 1054, 1860, 1060), fill="#22364b")
        d.rectangle((60, 1054, 60 + 1800 * t / duration, 1060), fill=CYAN)
        text(im, (1860, 864), "中文合成配音 · 实验回放", 19, MUTED, anchor="rt")
        return im

    try:
        for scene in scenes:
            t = scene["start"] + scene["duration"] * 0.60
            compose(scene, t).save(out / f"preview-{scene['id']}.jpg", quality=90)
        print(
            f"previewed {len(scenes)} scenes; narration-backed duration {duration:.2f}s", flush=True
        )
        if args.preview:
            return
        target = out / "navigation-67-68-narrated.mp4"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        if args.ffmpeg:
            exe = str(args.ffmpeg)
        else:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            exe,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-i",
            str(out / "narration.wav"),
            "-f",
            "ffmetadata",
            "-i",
            str(out / "chapters.ffmetadata"),
            "-map_metadata",
            "2",
            "-map_chapters",
            "2",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-af",
            "loudnorm=I=-18:TP=-2:LRA=7",
            "-ar",
            "48000",
            "-metadata:s:a:0",
            "language=zho",
            "-movflags",
            "+faststart",
            "-shortest",
            str(target),
        ]
        started = time.perf_counter()
        count = round(duration * fps)
        with (out / "encode.log").open("w", encoding="utf8") as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
            try:
                for i in range(count):
                    t = i / fps
                    scene = scenes[min(len(scenes) - 1, bisect.bisect_right(ends, t))]
                    proc.stdin.write(compose(scene, t).tobytes())
                    if i % (fps * 10) == 0:
                        print(
                            f"{i}/{count} frames · {scene['id']} · {time.perf_counter() - started:.1f}s wall",
                            flush=True,
                        )
            finally:
                proc.stdin.close()
            if proc.wait():
                raise RuntimeError("Video encoding failed; see encode.log")
        manifest = {
            "title": story["title"],
            "duration_s": duration,
            "fps": fps,
            "frames": count,
            "size": [W, H],
            "voice": story["voice"],
            "voice_rate": story["voice_rate"],
            "replay": "archived poses only; floor sample at replay time; no interpolation",
            "rgb": "MuJoCo presentation rerender, not recorded RGB input",
            "scenes": scenes,
            "sources": {case: r.metadata["hashes"] for case, r in film.records.items()},
            "input_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    args.story,
                    ROOT / "docs/benchmarks/navigation-studies-67-68-v1.json",
                    out / "narration/manifest.json",
                ]
            },
            "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "render_wall_s": time.perf_counter() - started,
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8"
        )
        print(f"Saved {target}; {duration:.2f}s, {count} frames", flush=True)
    finally:
        film.close()


if __name__ == "__main__":
    main()
