"""Render an actual saved MuJoCo replay and estimated trajectories as an animated GIF."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from embodied_learning.photo_room import RoomWorld


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", type=Path, default=Path("results/photo_room_v1"))
    parser.add_argument("--validation", default="validation_v3")
    args = parser.parse_args()
    root = args.room
    record = root / args.validation
    layout = json.loads((root / "layout.json").read_text(encoding="utf-8"))
    report = json.loads((record / "summary.json").read_text(encoding="utf-8"))
    seed = report["rows"][0]["seed"]
    world = RoomWorld(layout)
    # Hide fixtures attached to omitted walls/ceiling in this cutaway only.
    for name in ("air_conditioner", "light"):
        world.model.geom_group[world.model.geom(name).id] = 1
    world.model.vis.headlight.ambient[:] = [0.35, 0.35, 0.35]
    with np.load(record / "trajectories.npz") as data:
        truth = data["truth"]
        odom = data["odom"]
        pf = data[f"scan_height_oracle_seed{seed}"]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [1.65, 1.9, 0.55]
    camera.distance = 6.8
    camera.azimuth = 180
    camera.elevation = -48
    options = mujoco.MjvOption()
    options.geomgroup[:] = 1
    options.geomgroup[1] = 0
    frames = []
    with mujoco.Renderer(world.model, height=540, width=720) as renderer:
        for frame in list(range(0, len(truth), 3)) + [len(truth) - 1]:
            world.set_pose(truth[frame])
            renderer.update_scene(world.data, camera=camera, scene_option=options)
            for trajectory, rgba in ((odom, [0.9, 0.15, 0.1, 1.0]), (pf, [0.02, 0.8, 0.7, 1.0])):
                for before, pose in pairwise(trajectory[: frame + 1 : 2]):
                    scene = renderer.scene
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3),
                        np.array([pose[0], pose[1], 0.024]),
                        np.eye(3).reshape(-1),
                        np.array(rgba, dtype=np.float32),
                    )
                    mujoco.mjv_connector(
                        geom,
                        mujoco.mjtGeom.mjGEOM_LINE,
                        3,
                        np.array([before[0], before[1], 0.024]),
                        np.array([pose[0], pose[1], 0.024]),
                    )
                    scene.ngeom += 1
            im = Image.fromarray(renderer.render()).copy()
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, 720, 44), fill=(20, 27, 36))
            draw.text(
                (12, 6),
                f"3D room proxy | reference replay frame {frame + 1}/{len(truth)} | estimated dimensions",
                fill="white",
            )
            draw.text(
                (12, 25),
                "Red: biased odometry    Teal: scan-matched map PF    Body: prescribed true pose",
                fill="white",
            )
            frames.append(im)
    frames[0].save(
        root / "replay.gif", save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    print(f"Saved {len(frames)} rendered frames to {root / 'replay.gif'}")


if __name__ == "__main__":
    main()
