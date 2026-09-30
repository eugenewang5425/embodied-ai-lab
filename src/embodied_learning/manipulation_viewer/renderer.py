"""Render archived states; annotations never enter the experiment observations."""

from itertools import pairwise

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from embodied_learning.so101 import SceneConfig, SO101Simulation, project_point


class ManipulationRenderer:
    def __init__(self):
        self.sim = None
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.lookat[:] = [0.16, 0, 0.79]
        self.camera.distance = 0.8
        self.camera.azimuth = 135
        self.camera.elevation = -30
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:5] = 0

    def load(self, row, arrays):
        config = SceneConfig(**row["config"])
        if self.sim is None or self.sim.config != config:
            self.close()
            self.sim = SO101Simulation(config)
            self.renderer = mujoco.Renderer(self.sim.model, 480, 640)
        self.row, self.arrays = row, arrays

    def set_frame(self, frame):
        self.frame = frame
        self.sim.data.qpos[:] = self.arrays["qpos"][frame]
        self.sim.data.qvel[:] = self.arrays["qvel"][frame]
        self.sim.data.ctrl[:] = self.arrays["commands"][frame]
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def _point(self, point, color, radius=0.004):
        scene = self.renderer.scene
        if scene.ngeom == scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius),
            np.asarray(point),
            np.eye(3).ravel(),
            np.array(color, dtype=np.float32),
        )
        scene.ngeom += 1

    def _line(self, a, b, color):
        scene = self.renderer.scene
        if scene.ngeom == scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3.0, np.asarray(a), np.asarray(b))
        scene.ngeom += 1

    def render(self, view="scene", contacts=True, trails=True, frames=True):
        camera = self.camera if view == "scene" else view
        self.renderer.update_scene(self.sim.data, camera=camera, scene_option=self.option)
        waypoint = self.arrays["waypoint_xyz"][self.frame]
        if view == "scene":
            if frames:
                for origin, rotation in (
                    (np.array([0.0, 0.0, 0.69]), np.eye(3)),
                    (
                        self.sim.data.site_xpos[self.sim.site],
                        self.sim.data.site_xmat[self.sim.site].reshape(3, 3),
                    ),
                ):
                    for i, color in enumerate(
                        ([1, 0.15, 0.15, 1], [0.15, 0.8, 0.25, 1], [0.15, 0.4, 1, 1])
                    ):
                        self._line(origin, origin + 0.03 * rotation[:, i], color)
            self._point(waypoint, [1, 0.75, 0.05, 1], 0.007)
            self._point(self.arrays["tcp_xyz"][self.frame], [0.05, 0.7, 1, 1])
            if trails:
                ids = np.unique(np.linspace(0, self.frame, min(120, self.frame + 1)).astype(int))
                for name, color in (
                    ("tcp_xyz", [0.05, 0.7, 1, 1]),
                    ("object_xyz", [1, 0.3, 0.15, 1]),
                ):
                    for a, b in pairwise(ids):
                        self._line(self.arrays[name][a], self.arrays[name][b], color)
            if contacts:
                for i in range(self.sim.data.ncon):
                    contact = self.sim.data.contact[i]
                    if self.sim.object_geom not in (contact.geom1, contact.geom2):
                        continue
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.sim.model, self.sim.data, i, force)
                    if force[0] > 0.02:
                        self._point(contact.pos, [1, 0.1, 0.8, 1], 0.0025)
        output = Image.fromarray(self.renderer.render().copy())
        if view in ("wrist_cam", "overhead"):
            # Screen cue is a commanded goal, not an object detector output.
            projected = project_point(self.sim, view, waypoint)
            if projected is not None:
                uv, _ = projected
                x, y = np.clip(uv, [18, 18], [622, 462])
                draw = ImageDraw.Draw(output)
                draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline="#facc15", width=3)
                draw.line((x - 14, y, x + 14, y), fill="#facc15", width=2)
                draw.line((x, y - 14, x, y + 14), fill="#facc15", width=2)
        return output

    def close(self):
        if self.renderer is not None:
            context = self.renderer._gl_context
            if context is not None:
                context.make_current()
            self.renderer.close()
        self.renderer, self.sim = None, None
