"""Photo-referenced, editable room geometry shared by Blender and MuJoCo.

This is a manually interpreted proxy, not a metric photogrammetric reconstruction.
Only the desk height and short side have user-supplied measurements.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import mujoco
import numpy as np

COLORS = {
    "wall": [0.62, 0.65, 0.64, 1],
    "floor": [0.68, 0.53, 0.33, 1],
    "wood": [0.60, 0.37, 0.16, 1],
    "darkwood": [0.22, 0.12, 0.08, 1],
    "desk": [0.12, 0.10, 0.09, 1],
    "metal": [0.42, 0.45, 0.46, 1],
    "cloth": [0.70, 0.69, 0.65, 1],
    "pink": [0.72, 0.52, 0.54, 1],
    "screen": [0.025, 0.045, 0.06, 1],
    "glass": [0.40, 0.66, 0.74, 1],
    "white": [0.88, 0.88, 0.83, 1],
}


def default_layout():
    """Dimensions in meters; x across room, y toward window, z up."""
    return {
        "schema_version": 1,
        "status": "photo_referenced_proxy_pending_measurement",
        "coordinate_frame": "x across room; y toward window; z up; meters",
        "room": {"width": 3.40, "depth": 3.80, "height": 2.70},
        "desk": {"height": 0.69, "depth": 0.65, "length": 1.40, "y": 0.75},
        "bed": {"length": 2.00, "width": 1.50, "under_clearance": 0.22},
        "robot": {"radius": 0.15, "height": 0.32, "bottom": 0.02, "lidar_z": 0.18},
        "measurements": [
            {"field": "desk.height", "value_m": 0.69, "source": "user"},
            {
                "field": "desk.depth",
                "value_m": 0.65,
                "source": "user; width interpreted as short side, pending confirmation",
            },
        ],
        "assumptions": [
            "All room dimensions, furniture lengths, clearances and positions are estimates.",
            "Wardrobe is on the wall opposite the window; desk and bed head on the left wall.",
            "Door is on the right wall near the wardrobe; closed in this experiment.",
            "Bed underside and chair bases are simplified; hidden geometry is not measured.",
            "Window is opaque to geometric rays; glass optics, fabric and wheel slip are unmodelled.",
        ],
        "photo_groups": {
            "layout": ["IMG_7861.jpeg", "IMG_7862.jpeg", "IMG_7899.jpeg"],
            "wardrobe": ["IMG_7863.jpeg", "IMG_7870.jpeg", "IMG_7883.jpeg"],
            "desk": ["IMG_7882.jpeg", "IMG_7884.jpeg", "IMG_7885.jpeg"],
            "bed": ["IMG_7886.jpeg", "IMG_7893.jpeg", "IMG_7898.jpeg"],
            "wood_chair": ["IMG_7887.jpeg", "IMG_7889.jpeg"],
            "window": ["IMG_7868.jpeg", "IMG_7874.jpeg", "IMG_7897.jpeg"],
        },
    }


def build_geometry(layout):
    """Simple solids retain vertical clearance, with optional visual-only detail."""
    room, desk, bed = (layout[k] for k in ("room", "desk", "bed"))
    w, d, h = (room[k] for k in ("width", "depth", "height"))
    geoms = []

    def box(name, pos, size, material, collision=True, cutaway=False):
        geoms.append(
            {
                "name": name,
                "type": "box",
                "pos": list(pos),
                "size": list(size),
                "material": material,
                "collision": collision,
                "cutaway": cutaway,
            }
        )

    def cyl(name, pos, radius, height, material, collision=True):
        geoms.append(
            {
                "name": name,
                "type": "cylinder",
                "pos": list(pos),
                "size": [radius, height],
                "material": material,
                "collision": collision,
                "cutaway": False,
            }
        )

    box("floor", [w / 2, d / 2, -0.06], [w + 0.2, d + 0.2, 0.12], "floor")
    box("wall_left", [-0.05, d / 2, h / 2], [0.1, d, h], "wall")
    box("wall_front", [w / 2, -0.05, h / 2], [w, 0.1, h], "wall", cutaway=True)
    box("wall_right", [w + 0.05, d / 2, h / 2], [0.1, d, h], "wall", cutaway=True)
    box("window_sill_wall", [w / 2, d + 0.05, 0.40], [w, 0.1, 0.8], "wall")
    box("window_header", [w / 2, d + 0.05, (h + 2.4) / 2], [w, 0.1, h - 2.4], "wall")
    box("window_left_pier", [0.22, d + 0.05, 1.6], [0.44, 0.1, 1.6], "wall")
    box("window_right_pier", [w - 0.35, d + 0.05, 1.6], [0.7, 0.1, 1.6], "wall")
    box("window_glass", [(w - 0.26) / 2, d + 0.05, 1.6], [w - 1.14, 0.025, 1.6], "glass")
    for x in [0.44, (w - 0.26) / 2, w - 0.7]:
        box(f"window_frame_{x:.2f}", [x, d - 0.01, 1.6], [0.055, 0.09, 1.64], "darkwood")
    for z in [0.8, 1.48, 2.4]:
        box(f"window_rail_{z}", [(w - 0.26) / 2, d - 0.01, z], [w - 1.1, 0.1, 0.07], "darkwood")
    box("window_sill", [(w - 0.26) / 2, d - 0.13, 0.78], [w - 1.0, 0.32, 0.07], "wood")
    for side, cx in enumerate([0.60, w - 0.85]):
        for i in range(5):
            cyl(
                f"curtain_{side}_{i}",
                [cx + (i - 2) * 0.055, d - 0.17, 1.65],
                0.035,
                1.50,
                "cloth",
                False,
            )
    # Closed entrance: decoration on the inside of right wall.
    box("door", [w - 0.014, 0.56, 1.02], [0.04, 0.88, 2.04], "darkwood", False, True)
    box("door_handle", [w - 0.055, 0.88, 1.0], [0.08, 0.03, 0.13], "metal", False, True)
    box("wardrobe", [1.18, 0.34, 1.08], [2.30, 0.62, 2.16], "wood")
    for i in range(5):
        x = 0.065 + (i + 0.5) * 0.448
        box(f"wardrobe_panel_{i}", [x, 0.658, 1.12], [0.415, 0.025, 1.89], "wood", False)
        box(f"wardrobe_handle_{i}", [x + 0.13, 0.69, 1.0], [0.025, 0.035, 0.12], "darkwood", False)
    # Desk top at the supplied 69 cm; short side interpreted as 65 cm.
    dep, length, top, y = (desk[k] for k in ("depth", "length", "height", "y"))
    box("desk_top", [dep / 2 + 0.025, y + length / 2, top - 0.018], [dep, length, 0.036], "desk")
    for i, yy in enumerate([y + 0.13, y + length - 0.13]):
        box(f"desk_leg_{i}", [dep / 2, yy, (top - 0.04) / 2], [0.075, 0.085, top - 0.04], "metal")
        box(f"desk_foot_{i}", [dep / 2, yy, 0.035], [dep - 0.06, 0.13, 0.07], "metal")
    box("desk_crossbar", [0.20, y + length / 2, 0.49], [0.055, length - 0.17, 0.08], "metal")
    box("monitor", [0.21, y + length * 0.64, top + 0.31], [0.055, 0.59, 0.35], "screen")
    box("monitor_stand", [0.21, y + length * 0.64, top + 0.07], [0.06, 0.08, 0.14], "metal")
    box("laptop", [0.21, y + 0.25, top + 0.28], [0.05, 0.35, 0.24], "white")
    box("keyboard", [0.48, y + length * 0.60, top + 0.023], [0.15, 0.42, 0.024], "screen", False)
    # Bed length runs across the room, head at the desk wall.
    bx, by = bed["length"], bed["width"]
    yb = d - by / 2 - 0.04
    c = bed["under_clearance"]
    box("bed_frame", [0.04 + bx / 2, yb, c + 0.065], [bx, by, 0.13], "wood")
    box("mattress", [0.04 + bx / 2, yb, c + 0.21], [bx, by, 0.16], "cloth")
    for i, x in enumerate([0.14, bx - 0.06]):
        for j, yy in enumerate([d - by + 0.08, d - 0.15]):
            box(f"bed_leg_{i}_{j}", [x, yy, c / 2], [0.11, 0.11, c], "wood")
    for i in range(8):
        for j in range(6):
            box(
                f"bed_check_{i}_{j}",
                [0.04 + (i + 0.5) * bx / 8, d - 0.04 - by + (j + 0.5) * by / 6, c + 0.295],
                [bx / 8 - 0.005, by / 6 - 0.005, 0.012],
                "pink" if (i + j) % 2 else "cloth",
                False,
            )
    for i in range(2):
        box(f"pillow_{i}", [0.25, d - by + 0.38 + i * 0.65, c + 0.36], [0.38, 0.55, 0.12], "cloth")
    # Office chair: individual feet plus central column; no solid floor footprint.
    cx, cy = 0.98, y + 0.75
    cyl("office_column", [cx, cy, 0.26], 0.055, 0.48, "metal")
    box("office_seat", [cx, cy, 0.47], [0.49, 0.48, 0.09], "cloth")
    box("office_back", [cx + 0.23, cy, 0.77], [0.075, 0.47, 0.53], "cloth")
    box("office_headrest", [cx + 0.23, cy, 1.08], [0.085, 0.34, 0.14], "cloth")
    for i, dy in enumerate([-0.30, 0.30]):
        box(f"office_arm_{i}", [cx, cy + dy, 0.65], [0.40, 0.07, 0.06], "cloth")
    for i, (dx, dy) in enumerate([(-0.24, -0.20), (-0.24, 0.20), (0.24, -0.20), (0.24, 0.20)]):
        box(
            f"office_spoke_{i}",
            [cx + dx / 2, cy + dy / 2, 0.075],
            [abs(dx) + 0.05, 0.045, 0.04],
            "metal",
        )
        cyl(f"office_wheel_{i}", [cx + dx, cy + dy, 0.05], 0.04, 0.09, "screen")
    # Wooden chair against right wall, cabinet near the window.
    cx, cy = w - 0.37, d - 1.1
    box("woodchair_seat", [cx, cy, 0.43], [0.53, 0.56, 0.06], "wood")
    box("woodchair_back", [cx + 0.23, cy, 0.74], [0.065, 0.56, 0.55], "wood")
    for i, xx in enumerate([cx - 0.20, cx + 0.20]):
        for j, yy in enumerate([cy - 0.23, cy + 0.23]):
            box(f"woodchair_leg_{i}_{j}", [xx, yy, 0.21], [0.055, 0.055, 0.42], "wood")
    for i, yy in enumerate([cy - 0.27, cy + 0.27]):
        box(f"woodchair_arm_{i}", [cx, yy, 0.66], [0.53, 0.045, 0.05], "wood")
    box("chair_backpack", [cx + 0.09, cy, 0.75], [0.22, 0.34, 0.45], "screen")
    box("bedside_cabinet", [w - 0.37, d - 0.35, 0.31], [0.54, 0.52, 0.62], "wood")
    box("cat_scratch_base", [w - 0.16, 1.56, 0.04], [0.27, 0.62, 0.08], "pink")
    box("air_conditioner", [w - 0.14, d - 0.70, 2.25], [0.25, 0.78, 0.28], "white")
    box("light", [w / 2, d / 2, h - 0.06], [0.4, 0.4, 0.1], "white", False)
    return geoms


def model_xml(layout, geoms=None):
    geoms = geoms or build_geometry(layout)
    root = Element("mujoco", model="photo_room_proxy")
    SubElement(root, "compiler", angle="radian")
    SubElement(root, "option", timestep="0.01")
    visual = SubElement(root, "visual")
    SubElement(visual, "global", offwidth="1600", offheight="1200")
    asset = SubElement(root, "asset")
    for name, rgba in COLORS.items():
        SubElement(asset, "material", name=name, rgba=" ".join(map(str, rgba)))
    world = SubElement(root, "worldbody")
    SubElement(world, "light", pos="1.7 1.8 4", dir="0 0 -1", diffuse=".8 .8 .8", castshadow="true")
    for g in geoms:
        size = (
            np.array(g["size"]) / 2
            if g["type"] == "box"
            else np.array([g["size"][0], g["size"][1] / 2])
        )
        SubElement(
            world,
            "geom",
            name=g["name"],
            type=g["type"],
            pos=" ".join(map(str, g["pos"])),
            size=" ".join(map(str, size)),
            material=g["material"],
            group="1" if g["cutaway"] else ("0" if g["collision"] else "2"),
            contype="1" if g["collision"] else "0",
            conaffinity="1" if g["collision"] else "0",
        )
    robot = layout["robot"]
    body = SubElement(
        world, "body", name="robot", pos=f"1.7 1.2 {robot['bottom'] + robot['height'] / 2}"
    )
    SubElement(body, "freejoint", name="robot_pose")
    SubElement(
        body,
        "geom",
        name="robot_body",
        type="cylinder",
        size=f"{robot['radius']} {robot['height'] / 2}",
        rgba=".05 .47 .62 1",
        mass="3",
        group="0",
    )
    SubElement(
        body,
        "geom",
        name="heading",
        type="box",
        pos=f"{robot['radius'] * 0.7} 0 {robot['height'] / 2}",
        size=".045 .025 .01",
        rgba="1 .65 .15 1",
        contype="0",
        conaffinity="0",
        group="2",
    )
    SubElement(
        body,
        "camera",
        name="robot_rgbd",
        pos=f"{robot['radius'] + 0.005} 0 {robot['lidar_z'] - robot['bottom'] - robot['height'] / 2}",
        xyaxes="0 -1 0 0 0 1",
        fovy="65",
    )
    return tostring(root, encoding="unicode")


class RoomWorld:
    def __init__(self, layout):
        self.layout = layout
        self.geoms = build_geometry(layout)
        self.model = mujoco.MjModel.from_xml_string(model_xml(layout, self.geoms))
        self.data = mujoco.MjData(self.model)
        self.robot_id = self.model.body("robot").id
        self.ray_groups = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        self.set_pose([1.7, 1.2, 0])

    def set_pose(self, pose):
        r = self.layout["robot"]
        self.data.qpos[:3] = [pose[0], pose[1], r["bottom"] + r["height"] / 2]
        self.data.qpos[3:7] = [math.cos(pose[2] / 2), 0, 0, math.sin(pose[2] / 2)]
        mujoco.mj_forward(self.model, self.data)

    def scan(self, pose, z=None, rays=64, max_range=4.0):
        if z is None:
            z = self.layout["robot"]["lidar_z"]
        angles = pose[2] + np.arange(rays) * 2 * math.pi / rays
        point = np.array([pose[0], pose[1], z], dtype=float)
        dist = np.full(rays, max_range)
        hits = np.zeros(rays, dtype=bool)
        gid = np.zeros(1, dtype=np.int32)
        for i, a in enumerate(angles):
            value = mujoco.mj_ray(
                self.model,
                self.data,
                point,
                np.array([math.cos(a), math.sin(a), 0.0]),
                self.ray_groups,
                True,
                self.robot_id,
                gid,
            )
            if 0 <= value < max_range:
                dist[i] = value
                hits[i] = True
        return dist, hits

    def contact_names(self, pose):
        self.set_pose(pose)
        names = set()
        for c in self.data.contact:
            if c.dist >= -1e-7:
                continue
            a, b = map(int, c.geom)
            if self.model.geom_bodyid[a] == self.robot_id:
                names.add(self.model.geom(b).name)
            elif self.model.geom_bodyid[b] == self.robot_id:
                names.add(self.model.geom(a).name)
        return sorted(names)

    def occupancy(self, z_min, z_max, res=0.04):
        """Diagnostic volume projection; planner must inflate for robot radius."""
        w, d = (self.layout["room"][k] for k in ("width", "depth"))
        n = math.ceil(max(w, d) / res) + 2
        yy, xx = np.mgrid[:n, :n]
        xx = (xx + 0.5) * res
        yy = (yy + 0.5) * res
        grid = (xx >= w) | (yy >= d)
        for g in self.geoms:
            if not g["collision"] or g["name"] == "floor":
                continue
            x, y, z = g["pos"]
            hz = g["size"][2] / 2 if g["type"] == "box" else g["size"][1] / 2
            if z + hz < z_min or z - hz > z_max:
                continue
            if g["type"] == "box":
                dx, dy, _ = g["size"]
                inside = (abs(xx - x) <= dx / 2 + res / 2) & (abs(yy - y) <= dy / 2 + res / 2)
            else:
                inside = (xx - x) ** 2 + (yy - y) ** 2 <= (g["size"][0] + res / 2) ** 2
            grid |= inside
        grid[0, :] = True
        grid[:, 0] = True
        return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/photo_room_v1")
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    layout = (
        json.loads(args.layout.read_text(encoding="utf-8")) if args.layout else default_layout()
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "layout.json").write_text(
        json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "geometry.json").write_text(
        json.dumps(build_geometry(layout), indent=2), encoding="utf-8"
    )
    (out / "scene.xml").write_text(model_xml(layout), encoding="utf-8")
    world = RoomWorld(layout)
    print(f"Built {world.model.ngeom} MuJoCo geoms; metric room calibration pending.")
    if args.view:
        import mujoco.viewer

        mujoco.viewer.launch(world.model, world.data)


if __name__ == "__main__":
    main()
