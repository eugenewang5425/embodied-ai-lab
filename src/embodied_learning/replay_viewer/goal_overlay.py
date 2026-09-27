"""Project an explicit world goal into the selected camera; handle offscreen goals."""

import math
from functools import lru_cache

import numpy as np
from PIL import ImageDraw, ImageFont


@lru_cache(maxsize=2)
def font(size=24):
    return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)


def project_goal(pose, goal, camera, size):
    forward = np.array([math.cos(pose[2]), math.sin(pose[2])])
    right = np.array([math.sin(pose[2]), -math.cos(pose[2])])
    relative = np.asarray(goal) - pose[:2] - camera.forward_m * forward
    depth, horizontal = float(relative @ forward), float(relative @ right)
    w, h = size
    if depth <= 0.05:
        return None, "目标在后方"
    f = h / (2 * math.tan(math.radians(camera.vertical_fov_deg) / 2))
    x, y = w / 2 + f * horizontal / depth, h / 2 + f * (camera.height_m - 0.05) / depth
    if 20 <= x <= w - 20 and 50 <= y <= h - 55:
        return (x, y), "目标投影（可能被建筑遮挡）"
    if x < 20:
        return None, "目标在左侧视野外"
    if x > w - 20:
        return None, "目标在右侧视野外"
    return None, "目标在前下方（已接近）" if y > h - 55 else "目标在前上方"


def draw_goal(image, record, frame, key):
    info = record.goal_info(frame, key)
    if info is None:
        return image
    image = image.copy()
    draw = ImageDraw.Draw(image)
    name, goal = info
    pose = record.track_by_key[record.true_key(key)].poses[frame]
    location, hint = project_goal(pose, goal, record.camera, image.size)
    estimated = record.track_by_key[key].poses[frame]
    distance = float(np.linalg.norm(estimated[:2] - goal))
    draw.rectangle((0, 0, image.width, 82), fill="#14202e")
    draw.text((12, 5), f"当前目标：{name}", font=font(), fill="#ffe071")
    draw.text((12, 45), f"估计直线距离 {distance:.1f} m · {hint}", font=font(20), fill="white")
    if location is not None:
        x, y = location
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline="#ffe071", width=3)
        draw.line((x - 20, y, x + 20, y), fill="#ffe071", width=2)
        draw.line((x, y - 20, x, y + 20), fill="#ffe071", width=2)
    else:
        x = image.width - 30 if "右侧" in hint else 30
        if "后方" in hint:
            draw.text(
                (image.width / 2 - 70, image.height - 35),
                "↓ 目标在身后",
                font=font(),
                fill="#ffe071",
            )
        elif "前下方" in hint:
            draw.text(
                (image.width / 2 - 80, image.height - 35),
                "↓ 目标在前下方",
                font=font(),
                fill="#ffe071",
            )
        elif "前上方" in hint:
            draw.text((image.width / 2 - 80, 65), "↑ 目标在前上方", font=font(), fill="#ffe071")
        else:
            direction = 1 if "右侧" in hint else -1
            draw.polygon(
                [
                    (x, image.height / 2),
                    (x - direction * 20, image.height / 2 - 14),
                    (x - direction * 20, image.height / 2 + 14),
                ],
                fill="#ffe071",
            )
    return image
