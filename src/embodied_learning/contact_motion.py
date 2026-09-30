"""Planar dynamic body with genuine MuJoCo contact response and contact metrics."""

from xml.etree.ElementTree import Element, SubElement, tostring

import mujoco
import numpy as np

PHYSICS_DT = 0.002
MASS_KG = 8.0
YAW_INERTIA = MASS_KG * (0.55**2 + 0.44**2) / 12
FORCE_LIMIT_N = 6.4
TORQUE_LIMIT_NM = YAW_INERTIA * 1.8
LIMITS = {"peak_force_n": 40.0, "penetration_m": 0.005, "continuous_contact_s": 0.5}


def contact_allowed(peak_force_n, penetration_m, continuous_s):
    return (
        peak_force_n <= LIMITS["peak_force_n"]
        and penetration_m <= LIMITS["penetration_m"]
        and continuous_s <= LIMITS["continuous_contact_s"] + 1e-12
    )


def build_xml(boxes, rig):
    root = Element("mujoco", model="planar_navigation_contact")
    SubElement(root, "compiler", angle="radian")
    SubElement(
        root,
        "option",
        timestep=str(PHYSICS_DT),
        gravity="0 0 0",
        integrator="implicitfast",
        iterations="60",
        tolerance="1e-10",
    )
    default = SubElement(root, "default")
    SubElement(
        default,
        "geom",
        friction=".25 .005 .0001",
        solref=".006 1",
        solimp=".95 .99 .001",
        condim="3",
        margin="0",
    )
    world = SubElement(root, "worldbody")
    for i, b in enumerate(boxes):
        center, half = (b[[0, 2, 4]] + b[[1, 3, 5]]) / 2, (b[[1, 3, 5]] - b[[0, 2, 4]]) / 2
        SubElement(
            world,
            "geom",
            name=f"obstacle_{i}",
            type="box",
            pos=" ".join(map(str, center)),
            size=" ".join(map(str, half)),
        )
    body = SubElement(world, "body", name="robot")
    for name, kind, axis in (
        ("x", "slide", "1 0 0"),
        ("y", "slide", "0 1 0"),
        ("yaw", "hinge", "0 0 1"),
    ):
        SubElement(body, "joint", name=name, type=kind, axis=axis, damping="0")
    inertias = MASS_KG * np.array([0.44**2 + 0.30**2, 0.55**2 + 0.30**2, 0.55**2 + 0.44**2]) / 12
    SubElement(
        body,
        "inertial",
        pos=".025 0 .15",
        mass=str(MASS_KG),
        diaginertia=" ".join(map(str, inertias)),
    )
    for i, b in enumerate(rig.parts):
        center, half = (b[[0, 2, 4]] + b[[1, 3, 5]]) / 2, (b[[1, 3, 5]] - b[[0, 2, 4]]) / 2
        SubElement(
            body,
            "geom",
            name=f"body_{i}",
            type="box",
            pos=" ".join(map(str, center)),
            size=" ".join(map(str, half)),
            mass="0",
        )
    return tostring(root, encoding="unicode")


class ContactPlant:
    def __init__(self, boxes, rig, pose):
        self.xml = build_xml(boxes, rig)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = pose
        mujoco.mj_forward(self.model, self.data)
        self.continuous_s = self.max_continuous_s = 0.0

    @property
    def pose(self):
        p = self.data.qpos.copy()
        p[2] = np.arctan2(np.sin(p[2]), np.cos(p[2]))
        return p

    @property
    def velocity(self):
        v = self.data.qvel
        a = self.data.qpos[2]
        return np.array([v[0] * np.cos(a) + v[1] * np.sin(a), v[2]])

    @property
    def stopped(self):
        return np.linalg.norm(self.data.qvel[:2]) < 0.01 and abs(self.data.qvel[2]) < 0.01

    def step(self, command):
        states, forces, contacts = [], [], []
        for _ in range(50):
            a = self.data.qpos[2]
            desired = command[0] * np.array([np.cos(a), np.sin(a)])
            force = 80 * (desired - self.data.qvel[:2])
            force *= min(1.0, FORCE_LIMIT_N / max(np.linalg.norm(force), 1e-12))
            torque = np.clip(
                8 * (command[1] - self.data.qvel[2]), -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM
            )
            self.data.qfrc_applied[:] = [*force, torque]
            mujoco.mj_step(self.model, self.data)
            peak, total, penetration, count = 0.0, 0.0, 0.0, 0
            for j in range(self.data.ncon):
                c = self.data.contact[j]
                if (self.model.geom_bodyid[c.geom1] == 0) == (self.model.geom_bodyid[c.geom2] == 0):
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, j, wrench)
                if wrench[0] <= 1e-6:
                    continue
                count += 1
                peak = max(peak, float(wrench[0]))
                total += float(wrench[0])
                penetration = max(penetration, max(0.0, -float(c.dist)))
            self.continuous_s = self.continuous_s + PHYSICS_DT if count else 0.0
            self.max_continuous_s = max(self.max_continuous_s, self.continuous_s)
            states.append(np.r_[self.data.qpos, self.data.qvel])
            forces.append([*force, torque])
            contacts.append([count, peak, total, penetration, self.continuous_s])
        return np.asarray(states), np.asarray(forces), np.asarray(contacts)
