"""Calibrated ground-width guides are HUD annotations, never camera evidence."""

from dataclasses import replace

import numpy as np
from PIL import ImageDraw

from embodied_learning.navigation_geometry import NavigationRig, project

from .goal_overlay import font


def draw_body_guides(image, record, frame, key):
    if "rig" not in record.metadata:
        return image
    image = image.copy()
    rig = NavigationRig(**record.metadata["rig"])
    if record.metadata.get("tracking_round") == 6:
        rig = replace(rig, margin_m=record.method_info(key)["result"]["controller_margin_m"])
    draw = ImageDraw.Draw(image)
    sx, sy = image.width / rig.width, image.height / rig.height
    for side in (-1, 1):
        pts = np.array(
            [[x, side * (rig.half_width_m + rig.margin_m), 0.01] for x in np.linspace(0.5, 3, 80)]
        )
        uv, front = project(pts, rig)
        visible = front & (uv[:, 1] >= 0) & (uv[:, 1] < rig.height)
        pixels = [(float((u + 0.5) * sx), float((v + 0.5) * sy)) for u, v in uv[visible]]
        if len(pixels) > 1:
            draw.line(pixels, fill="#ffc947", width=3)
    # Ground targets project below the image centre. Put explanatory text under
    # the top goal banner, leaving both goal markers and offscreen arrows clear.
    label_font = font(17)
    draw.rectangle((0, 83, image.width, 136), fill="#172333")
    draw.text(
        (10, 86),
        f"黄线：车宽44cm＋两侧各{rig.margin_m * 100:g}cm余量（地面投影）",
        font=label_font,
        fill="#ffd46b",
    )
    draw.text(
        (10, 109),
        "相机下方有盲区；画面无障碍不代表车身能通过",
        font=label_font,
        fill="white",
    )
    return image
