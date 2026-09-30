"""Pinned SO101 model, bounded IK and actual MuJoCo position-servo simulation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).parent / "assets/so101"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TABLE_Z = 0.69
HOME = np.array([0.0, -0.5, 0.7, 0.8, 1.58, 1.1])
DT = 0.002
CONTROL_DT = 0.02


@dataclass(frozen=True)
class SceneConfig:
    object_xyz: tuple[float, float, float] = (0.22, 0.025, TABLE_Z + 0.025)
    object_half_size: float = 0.02
    object_mass: float = 0.05
    object_friction: float = 1.0
    target_xy: tuple[float, float] = (0.18, -0.09)
    gripper_torque_limit_nm: float = 0.3

    def __post_init__(self):
        values = [
            *self.object_xyz,
            *self.target_xy,
            self.object_half_size,
            self.object_mass,
            self.object_friction,
            self.gripper_torque_limit_nm,
        ]
        if not np.isfinite(values).all() or min(values[-4:]) <= 0:
            raise ValueError("Scene values must be finite, with positive size/mass/friction")


def verify_assets() -> dict:
    manifest = json.loads((ASSETS / "provenance.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        data = (ASSETS / name).read_bytes()
        if len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != info["sha256"]:
            raise ValueError(f"SO101 asset hash mismatch: {name}")
    return manifest


@lru_cache(maxsize=1)
def mesh_assets() -> dict[str, bytes]:
    return {f"assets/{p.name}": p.read_bytes() for p in (ASSETS / "assets").glob("*.stl")}


def model_xml(config: SceneConfig) -> str:
    root = ET.fromstring((ASSETS / "so101.xml").read_text(encoding="utf-8"))
    root.set("model", "so101_tabletop_physics")
    root.find("option").set("timestep", str(DT))
    root.find("option").set("iterations", "50")
    # Simulation current-limit assumption, not a measured hardware calibration.
    root.find("actuator/position[@name='gripper']").set(
        "forcerange", f"{-config.gripper_torque_limit_nm} {config.gripper_torque_limit_nm}"
    )
    # Name finger meshes so palm/motor contacts cannot count as bilateral finger grip.
    for body_name, prefix in (("gripper", "fixed_jaw"), ("moving_jaw_so101_v1", "moving_jaw")):
        body = root.find(f".//body[@name='{body_name}']")
        for i, geom in enumerate(body.findall("geom[@class='collision_gripper_mesh']")):
            geom.set("name", f"{prefix}_mesh_{i}")
    visual = root.find("visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1000")
    ET.SubElement(visual, "headlight", ambient=".4 .4 .4", diffuse=".7 .7 .7")
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="table_grid",
        type="2d",
        builtin="checker",
        rgb1=".55 .60 .65",
        rgb2=".65 .70 .75",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="table_material",
        texture="table_grid",
        texrepeat="12 12",
        texuniform="true",
    )
    world = root.find("worldbody")
    world.find("body[@name='base']").set("pos", f"0 0 {TABLE_Z}")
    ET.SubElement(world, "light", pos=".3 -.4 1.8", dir="0 0 -1", diffuse=".8 .8 .8")
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        pos=f".15 0 {TABLE_Z - 0.025}",
        size=".45 .4 .025",
        material="table_material",
        friction=".7 .005 .0001",
    )
    ET.SubElement(
        world, "geom", name="floor", type="plane", pos="0 0 0", size="2 2 .01", rgba=".18 .23 .28 1"
    )
    # A thin painted target has no collision; object support is the table.
    ET.SubElement(
        world,
        "geom",
        name="drop_zone",
        type="box",
        pos=f"{config.target_xy[0]} {config.target_xy[1]} {TABLE_Z + 0.0003}",
        size=".045 .045 .0003",
        rgba=".15 .65 .45 1",
        contype="0",
        conaffinity="0",
    )
    obj = ET.SubElement(world, "body", name="object", pos=" ".join(map(str, config.object_xyz)))
    ET.SubElement(obj, "freejoint", name="object_free")
    half = config.object_half_size
    ET.SubElement(
        obj,
        "geom",
        name="object_geom",
        type="box",
        size=f"{half} {half} {half}",
        mass=str(config.object_mass),
        rgba=".9 .3 .12 1",
        condim="6",
        friction=f"{config.object_friction} .005 .0001",
        solref=".01 1",
    )
    ET.SubElement(
        world,
        "camera",
        name="overview",
        pos=".8 -.8 1.25",
        xyaxes=".707 .707 0 -.32 .32 .89",
        fovy="45",
    )
    ET.SubElement(
        world, "camera", name="overhead", pos=".18 0 1.5", xyaxes="1 0 0 0 1 0", fovy="45"
    )
    return ET.tostring(root, encoding="unicode")


class SO101Simulation:
    def __init__(self, config: SceneConfig | None = None):
        config = SceneConfig() if config is None else config
        self.config = config
        self.xml = model_xml(config)
        # VFS also works in Windows directories with Chinese characters.
        self.model = mujoco.MjModel.from_xml_string(self.xml, assets=mesh_assets())
        self.data = mujoco.MjData(self.model)
        self.kinematics = mujoco.MjData(self.model)
        self.site = self.model.site("gripperframe").id
        self.object_body = self.model.body("object").id
        self.object_geom = self.model.geom("object_geom").id
        self.object_qadr = int(self.model.joint("object_free").qposadr[0])
        self.ranges = np.column_stack(
            (
                np.maximum(self.model.jnt_range[:5, 0], self.model.actuator_ctrlrange[:5, 0]),
                np.minimum(self.model.jnt_range[:5, 1], self.model.actuator_ctrlrange[:5, 1]),
            )
        )
        self.reset()

    def reset(self, q: np.ndarray = HOME):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint coordinates")
        if np.any(q < self.model.jnt_range[:6, 0]) or np.any(q > self.model.jnt_range[:6, 1]):
            raise ValueError("Initial joints outside limits")
        mujoco.mj_resetData(self.model, self.data)
        self.last_physics = []
        self.data.qpos[:6] = q
        self.data.ctrl[:] = q
        mujoco.mj_forward(self.model, self.data)

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=float)
        if q.shape != (5,) or not np.isfinite(q).all():
            raise ValueError("FK requires five finite arm angles")
        self.kinematics.qpos[:] = self.data.qpos
        self.kinematics.qpos[:5] = q
        mujoco.mj_forward(self.model, self.kinematics)
        return (
            self.kinematics.site_xpos[self.site].copy(),
            self.kinematics.site_xmat[self.site].reshape(3, 3).copy(),
        )

    def ik(self, xyz, rotation, seed=None) -> dict:
        xyz, rotation = np.asarray(xyz, float), np.asarray(rotation, float)
        if (
            xyz.shape != (3,)
            or rotation.shape != (3, 3)
            or not np.isfinite(xyz).all()
            or not np.isfinite(rotation).all()
        ):
            raise ValueError("IK needs a finite xyz and 3x3 rotation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-6
        ):
            raise ValueError("IK orientation must be a proper rotation")
        lo, hi = self.ranges.T
        seed = self.data.qpos[:5] if seed is None else np.asarray(seed, float)
        if seed.shape != (5,) or not np.isfinite(seed).all():
            raise ValueError("IK seed requires five finite angles")

        def residual(q):
            pos, rot = self.fk(q)
            return np.r_[pos - xyz, 0.08 * Rotation.from_matrix(rotation.T @ rot).as_rotvec()]

        result = least_squares(
            residual,
            np.clip(seed, lo + 1e-7, hi - 1e-7),
            bounds=(lo, hi),
            max_nfev=100,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        pos, rot = self.fk(result.x)
        error = float(np.linalg.norm(pos - xyz))
        angle = float(np.linalg.norm(Rotation.from_matrix(rotation.T @ rot).as_rotvec()))
        return {
            "q": result.x.copy(),
            "position_error_m": error,
            "orientation_error_rad": angle,
            "reachable": bool(error < 0.003 and angle < 0.08),
        }

    def step(self, command):
        command = np.asarray(command, float)
        if command.shape != (6,) or not np.isfinite(command).all():
            raise ValueError("Expected six finite servo targets")
        if np.any(command < self.model.actuator_ctrlrange[:, 0]) or np.any(
            command > self.model.actuator_ctrlrange[:, 1]
        ):
            raise ValueError("Servo request outside limits")
        self.data.ctrl[:] = command
        raw = []
        for _ in range(round(CONTROL_DT / DT)):
            mujoco.mj_step(self.model, self.data)
            forces, unsafe, penetration = self.contact_metrics()
            raw.append(
                (
                    float(self.data.time),
                    self.data.qpos.copy(),
                    self.data.qvel.copy(),
                    forces,
                    unsafe,
                    penetration,
                )
            )
        self.last_physics = raw
        return self.observe()

    def contact_metrics(self):
        normal = np.zeros(3)  # fixed jaw, moving jaw, other object contacts
        unsafe, penetration = 0.0, 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            force_n = max(0.0, force[0])
            penetration = max(penetration, -float(contact.dist))
            if self.object_geom in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == self.object_geom else contact.geom1
                geom_name = self.model.geom(other).name
                channel = (
                    0
                    if geom_name.startswith("fixed_jaw_")
                    else 1
                    if geom_name.startswith("moving_jaw_")
                    else 2
                )
                normal[channel] += force_n
            else:
                bodies = [int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
                # Arm-table or non-adjacent self contacts; ignore base fixture support.
                if any(b > 1 for b in bodies):
                    unsafe = max(unsafe, force_n)
        return normal, unsafe, penetration

    def observe(self):
        mujoco.mj_forward(self.model, self.data)
        normal, unsafe, penetration = self.contact_metrics()
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "tcp": self.data.site_xpos[self.site].copy(),
            "object": self.data.xpos[self.object_body].copy(),
            "jaw_normal_n": normal,
            "time": float(self.data.time),
            "unsafe_contact_n": unsafe,
            "penetration_m": penetration,
        }


def grasp_rotation(xyz, tilt_rad=0.0):
    """Align jaw closure across the arm's radial plane, with explicit pitch freedom."""
    x, y, _ = np.asarray(xyz, float)
    radial = np.hypot(x - 0.0388353, y)
    if radial <= 0.011807:
        raise ValueError("Target too close to shoulder axis for this approach")
    yaw = np.arctan2(y, x - 0.0388353) - np.arcsin(0.011807 / radial)
    down = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    return (
        Rotation.from_euler("z", yaw).as_matrix()
        @ Rotation.from_euler("y", tilt_rad).as_matrix()
        @ down
    )


def camera_geometry(sim: SO101Simulation, name: str, width=640, height=480):
    """OpenCV camera coordinates: right, down, forward; MuJoCo looks along -z."""
    camera = sim.model.camera(name).id
    focal = height / (2 * np.tan(np.deg2rad(sim.model.cam_fovy[camera]) / 2))
    intrinsic = np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])
    rotation = sim.data.cam_xmat[camera].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    return intrinsic, rotation, sim.data.cam_xpos[camera].copy()


def project_point(sim: SO101Simulation, name, xyz, width=640, height=480):
    intrinsic, rotation, origin = camera_geometry(sim, name, width, height)
    point = rotation.T @ (np.asarray(xyz, float) - origin)
    if point[2] <= 0:
        return None
    pixel = intrinsic @ (point / point[2])
    return pixel[:2], float(point[2])
