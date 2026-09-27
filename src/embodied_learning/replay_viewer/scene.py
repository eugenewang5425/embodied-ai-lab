"""MuJoCo presentation renderer; consumes records, never runs an experiment."""

import math
from itertools import pairwise
from types import SimpleNamespace
from xml.etree.ElementTree import Element, SubElement, tostring

import mujoco
import numpy as np
from matplotlib.colors import to_rgba
from PIL import Image


def grid_primitives(record):
    """Fallback extrusion for adapters that supply only an occupancy grid."""
    grid = next(iter(record.maps.values()))
    result = []
    r = record.resolution_m
    for iy, row in enumerate(grid):
        edges = np.diff(np.r_[False, row, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            result.append(
                SimpleNamespace(
                    kind="rect",
                    cx=record.origin[0] + (start + end) * r / 2,
                    cy=record.origin[1] + (iy + 0.5) * r,
                    hx=(end - start) * r / 2,
                    hy=r / 2,
                )
            )
    return result


def build_model(record):
    campus = record.metadata.get("scene_kind") == "campus"
    root = Element("mujoco", model="record_replay_display")
    SubElement(root, "compiler", angle="radian")
    visual = SubElement(root, "visual")
    SubElement(visual, "global", offwidth="1600", offheight="1000")
    if campus:
        SubElement(
            visual, "headlight", ambient=".45 .45 .45", diffuse=".75 .75 .75", specular=".1 .1 .1"
        )
    asset = SubElement(root, "asset")
    SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=".52 .70 .88" if campus else ".10 .14 .20",
        rgb2=".80 .88 .94" if campus else ".22 .28 .36",
        width="256",
        height="1536",
    )
    world = SubElement(root, "worldbody")
    x0, x1, y0, y1 = record.extent
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    SubElement(world, "light", pos=f"{cx} {cy} 12", dir="0 0 -1", diffuse=".8 .8 .8")
    SubElement(
        world,
        "geom",
        type="plane",
        pos=f"{cx} {cy} 0",
        size=f"{(x1 - x0) / 2} {(y1 - y0) / 2} .01",
        rgba=".44 .51 .38 1" if campus else ".25 .31 .39 1",
    )
    for x in np.arange(math.ceil(x0), x1, 1.0) if not campus else []:
        SubElement(
            world,
            "geom",
            type="box",
            pos=f"{x} {cy} .004",
            size=f".006 {(y1 - y0) / 2} .004",
            rgba=".42 .48 .56 1",
        )
    for y in np.arange(math.ceil(y0), y1, 1.0) if not campus else []:
        SubElement(
            world,
            "geom",
            type="box",
            pos=f"{cx} {y} .004",
            size=f"{(x1 - x0) / 2} .006 .004",
            rgba=".42 .48 .56 1",
        )
    if campus:
        for axis in (0, 1):
            for lane in (2.0, 12.0, 22.0):
                pos = [12.0, 12.0, 0.005]
                size = [12.0, 12.0, 0.005]
                pos[axis], size[axis] = lane, 1.75
                SubElement(
                    world,
                    "geom",
                    type="box",
                    pos=" ".join(map(str, pos)),
                    size=" ".join(map(str, size)),
                    rgba=".20 .23 .27 1",
                )
                for along in np.arange(0.8, 24, 1.6):
                    pos = [along, lane, 0.018] if axis == 1 else [lane, along, 0.018]
                    size = [0.35, 0.035, 0.006] if axis == 1 else [0.035, 0.35, 0.006]
                    SubElement(
                        world,
                        "geom",
                        type="box",
                        pos=" ".join(map(str, pos)),
                        size=" ".join(map(str, size)),
                        rgba=".88 .83 .52 1",
                    )
        # Zebra markings are flat paint; the obstacle geometry remains unchanged.
        for offset in np.arange(-1.3, 1.5, 0.45):
            SubElement(
                world,
                "geom",
                type="box",
                pos=f"{12 + offset} 9.9 .025",
                size=".13 .6 .005",
                rgba=".92 .92 .90 1",
            )
    styles = record.metadata.get("scene_style", [])
    for i, p in enumerate(record.primitives or grid_primitives(record)):
        style = styles[i] if i < len(styles) else {}
        height = style.get("height", 0.7 if i < record.wall_count else 0.5)
        common = {
            "pos": f"{p.cx} {p.cy} {height / 2}",
            "rgba": style.get(
                "color", ".85 .88 .91 1" if i < record.wall_count else ".77 .57 .34 1"
            ),
        }
        if p.kind == "circle":
            SubElement(world, "geom", type="cylinder", size=f"{p.r} {height / 2}", **common)
        elif p.kind == "rect":
            SubElement(world, "geom", type="box", size=f"{p.hx} {p.hy} {height / 2}", **common)
        elif p.kind == "segment":
            # The original point-to-segment distance defines a capsule footprint.
            # A middle rectangle plus two cylinders preserves its round ends.
            yaw = math.atan2(p.hy, p.hx)
            SubElement(
                world,
                "geom",
                type="box",
                size=f"{math.hypot(p.hx, p.hy)} {p.r} {height / 2}",
                quat=f"{math.cos(yaw / 2)} 0 0 {math.sin(yaw / 2)}",
                **common,
            )
            for sign in (-1, 1):
                SubElement(
                    world,
                    "geom",
                    type="cylinder",
                    size=f"{p.r} {height / 2}",
                    pos=f"{p.cx + sign * p.hx} {p.cy + sign * p.hy} {height / 2}",
                    rgba=common["rgba"],
                )
        else:
            raise ValueError(f"unsupported primitive: {p.kind}")
        if style.get("kind") == "building":
            # Window decals stay on the surfaces of the same collision footprint.
            for z in np.arange(1.0, height - 0.3, 1.25):
                for delta in np.arange(-p.hx + 0.7, p.hx - 0.3, 1.3):
                    for sign in (-1, 1):
                        SubElement(
                            world,
                            "geom",
                            type="box",
                            pos=f"{p.cx + delta} {p.cy + sign * (p.hy + 0.015)} {z}",
                            size=".35 .02 .30",
                            rgba=".16 .32 .44 1",
                        )
                for delta in np.arange(-p.hy + 0.7, p.hy - 0.3, 1.3):
                    for sign in (-1, 1):
                        SubElement(
                            world,
                            "geom",
                            type="box",
                            pos=f"{p.cx + sign * (p.hx + 0.015)} {p.cy + delta} {z}",
                            size=".02 .35 .30",
                            rgba=".16 .32 .44 1",
                        )
        elif style.get("kind") == "tree":
            SubElement(
                world,
                "geom",
                type="ellipsoid",
                pos=f"{p.cx} {p.cy} 2.5",
                size=".9 .9 1.0",
                rgba=".20 .45 .23 1",
            )
        elif style.get("kind") == "car":
            SubElement(
                world,
                "geom",
                type="box",
                pos=f"{p.cx} {p.cy} {height + 0.18}",
                size=f"{p.hx * 0.85} {p.hy * 0.55} .18",
                rgba=".16 .25 .32 1",
            )
    for i, track in enumerate(record.tracks):
        body = SubElement(world, "body", name=f"actor_{i}", mocap="true", pos=f"{cx} {cy} .18")
        rgba = " ".join(str(x) for x in to_rgba(track.color))
        if i == 0 or track.role == "truth":
            # Optical -z faces robot +x; image up is world +z.
            SubElement(
                body,
                "camera",
                name="robot_front" if i == 0 else f"robot_front_{i}",
                pos=f"{record.camera.forward_m} 0 {record.camera.height_m - 0.18}",
                xyaxes="0 -1 0 0 0 1",
                fovy=str(record.camera.vertical_fov_deg),
            )
            radius = record.metadata.get("robot_radius_m", 0.15)
            SubElement(body, "geom", type="cylinder", size=f"{radius} .11", rgba=rgba)
            SubElement(
                body,
                "geom",
                type="box",
                pos=f"{radius * 0.52} 0 .09",
                size=f"{radius * 0.32} .03 .025",
                rgba="1 .85 .15 1",
            )
            if campus:
                mast_height = record.camera.height_m - 0.29
                SubElement(
                    body,
                    "geom",
                    type="cylinder",
                    size=f".015 {mast_height / 2}",
                    pos=f"{record.camera.forward_m - 0.035} 0 {0.11 + mast_height / 2}",
                    rgba=".18 .20 .23 1",
                )
                SubElement(
                    body,
                    "geom",
                    type="box",
                    size=".025 .025 .02",
                    pos=f"{record.camera.forward_m - 0.025} 0 {record.camera.height_m - 0.18}",
                    rgba=".1 .12 .15 1",
                )
        else:
            SubElement(body, "geom", type="sphere", size=".12", rgba=rgba)
    model = mujoco.MjModel.from_xml_string(tostring(root, encoding="unicode"))
    # MuJoCo scales znear by scene extent: the default erased nearby surfaces
    # in a 24 m campus (0.478 m near plane). Camera clipping is a metric choice.
    model.vis.map.znear = record.camera.near_clip_m / model.stat.extent
    return model


class SceneRenderer:
    def __init__(self):
        self.record = None
        self.renderer = None
        self.size = None
        self.camera = mujoco.MjvCamera()
        self.focus_key = "truth"
        self.show_lidar = True

    def load(self, record):
        self.close()
        self.record = record
        self.model = build_model(record)
        self.data = mujoco.MjData(self.model)
        if self.focus_key not in record.method_keys:
            self.focus_key = record.method_keys[0]
        self.reset_camera()

    def reset_camera(self):
        x0, x1, y0, y1 = self.record.extent
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [(x0 + x1) / 2, (y0 + y1) / 2, 0.15]
        self.camera.distance = 1.7 * max(x1 - x0, y1 - y0)
        self.camera.azimuth = 140
        self.camera.elevation = -55

    def orbit(self, dx, dy):
        self.camera.azimuth += dx * 0.4
        self.camera.elevation = float(np.clip(self.camera.elevation - dy * 0.3, -89, -8))

    def zoom(self, delta):
        self.camera.distance = float(
            np.clip(self.camera.distance * math.exp(-delta * 0.12), 1.0, 80.0)
        )

    def render(self, frame, size, visible, trails=True, view="overview"):
        if view not in ("overview", "robot"):
            raise ValueError(f"unknown view: {view}")
        width, height = max(100, min(1400, int(size[0]))), max(100, min(900, int(size[1])))
        if self.size != (width, height):
            self.close()
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
            self.size = (width, height)
        for i, track in enumerate(self.record.tracks):
            pose = track.poses[frame]
            body = self.model.body(f"actor_{i}")
            mid = int(body.mocapid[0])
            self.data.mocap_pos[mid] = [pose[0], pose[1], 0.18]
            self.data.mocap_quat[mid] = [math.cos(pose[2] / 2), 0, 0, math.sin(pose[2] / 2)]
            start, count = int(body.geomadr[0]), int(body.geomnum[0])
            # Diagnostic markers and trails are annotations, not camera observations.
            self.model.geom_rgba[start : start + count, 3] = float(
                view == "overview" and track.key in visible
            )
        mujoco.mj_forward(self.model, self.data)
        true_key = self.record.true_key(self.focus_key)
        index = list(self.record.track_by_key).index(true_key)
        camera_name = "robot_front" if index == 0 else f"robot_front_{index}"
        self.renderer.update_scene(
            self.data, camera=camera_name if view == "robot" else self.camera
        )
        if trails and view == "overview":
            scene = self.renderer.scene
            ids = np.unique(np.linspace(0, frame, min(130, frame + 1)).astype(int))
            for track in self.record.tracks:
                if track.key not in visible:
                    continue
                for segment, (a, b) in enumerate(pairwise(ids)):
                    if track.role != "truth" and track.key != "truth" and segment % 2:
                        continue
                    if scene.ngeom >= scene.maxgeom:
                        break
                    g = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(
                        g,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3),
                        np.zeros(3),
                        np.eye(3).reshape(-1),
                        np.array(to_rgba(track.color), dtype=np.float32),
                    )
                    mujoco.mjv_connector(
                        g,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        2.0,
                        np.r_[track.poses[a, :2], 0.025],
                        np.r_[track.poses[b, :2], 0.025],
                    )
                    scene.ngeom += 1
        if view == "overview":
            self._annotations(frame)
        return Image.fromarray(self.renderer.render().copy())

    def _point(self, xyz, radius, color, label=""):
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius),
            np.asarray(xyz, float),
            np.eye(3).ravel(),
            np.asarray(color, np.float32),
        )
        geom.label = label
        scene.ngeom += 1

    def _line(self, a, b, color, width=2.0):
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.asarray(color, np.float32),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float)
        )
        scene.ngeom += 1

    def _annotations(self, frame):
        key = self.focus_key
        if self.show_lidar:
            for point in self.record.scan_world(frame, key):
                self._point([*point, 0.35], 0.045, [0, 0.95, 0.92, 1])
        info = self.record.goal_info(frame, key)
        if info is not None:
            name, goal = info
            self._line([*goal, 0.05], [*goal, 2.0], [1, 0.78, 0.12, 1], 5)
            self._point([*goal, 2.0], 0.23, [1, 0.78, 0.12, 1], name.split()[0])
        if key in self.record.plans:
            path = self.record.plans[key][frame]
            path = path[np.isfinite(path).all(axis=1)]
            for a, b in pairwise(path):
                self._line([*a, 0.08], [*b, 0.08], [1, 0.78, 0.12, 1], 3)

    def close(self):
        if self.renderer is not None:
            # MuJoCo 3.11 close() does not bind its owning GL context. Resizing
            # this pane while the other pane is current can delete that pane's
            # framebuffer and leave its images black. Bind the owner first.
            context = self.renderer._gl_context
            if context is not None:
                context.make_current()
            self.renderer.close()
        self.renderer = None
        self.size = None
