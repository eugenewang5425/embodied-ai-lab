"""Run with Blender --background --python this_file -- --input results/photo_room_v1."""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation", default="validation_v3")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    out = args.input.resolve()
    geoms = json.loads((out / "geometry.json").read_text(encoding="utf-8"))
    layout = json.loads((out / "layout.json").read_text(encoding="utf-8"))
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    colors = {
        "wall": (0.62, 0.65, 0.64, 1),
        "floor": (0.68, 0.53, 0.33, 1),
        "wood": (0.60, 0.37, 0.16, 1),
        "darkwood": (0.22, 0.12, 0.08, 1),
        "desk": (0.12, 0.10, 0.09, 1),
        "metal": (0.42, 0.45, 0.46, 1),
        "cloth": (0.70, 0.69, 0.65, 1),
        "pink": (0.72, 0.52, 0.54, 1),
        "screen": (0.025, 0.045, 0.06, 1),
        "glass": (0.40, 0.66, 0.74, 1),
        "white": (0.88, 0.88, 0.83, 1),
    }
    mats = {}
    for name, color in colors.items():
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = color
        mat.use_nodes = True
        mat.node_tree.nodes.clear()
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.25 if name == "metal" else 0.65
        if name in ("wood", "darkwood", "floor"):
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            tex = nodes.new("ShaderNodeTexNoise")
            tex.inputs["Scale"].default_value = 5
            coord = nodes.new("ShaderNodeTexCoord")
            mapping = nodes.new("ShaderNodeVectorMath")
            mapping.operation = "MULTIPLY"
            mapping.inputs[1].default_value = (25, 2, 3)
            links.new(coord.outputs["Generated"], mapping.inputs[0])
            links.new(mapping.outputs[0], tex.inputs["Vector"])
            ramp = nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color = tuple(c * 0.72 for c in color[:3]) + (1,)
            ramp.color_ramp.elements[1].color = color
            links.new(tex.outputs["Fac"], ramp.inputs[0])
            links.new(ramp.outputs[0], bsdf.inputs["Base Color"])
        mats[name] = mat
    furniture = bpy.data.collections.new("Photo referenced room - estimated geometry")
    bpy.context.scene.collection.children.link(furniture)
    hidden = bpy.data.collections.new("Cutaway walls - enable for enclosed view")
    bpy.context.scene.collection.children.link(hidden)
    hidden.hide_render = True
    hidden.hide_viewport = True
    for g in geoms:
        if g["type"] == "box":
            bpy.ops.mesh.primitive_cube_add(size=1, location=g["pos"])
            obj = bpy.context.object
            obj.dimensions = g["size"]
        else:
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=24, radius=g["size"][0], depth=g["size"][1], location=g["pos"]
            )
            obj = bpy.context.object
        obj.name = g["name"]
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.data.materials.append(mats[g["material"]])
        if g["type"] == "box":
            bevel = obj.modifiers.new("Soft manufactured edges", "BEVEL")
            bevel.width = min(0.012, min(g["size"]) * 0.15)
            bevel.segments = 2
            obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
        obj["collision_geometry"] = g["collision"]
        obj["dimensional_status"] = (
            "desk top height 0.69 m and short side 0.65 m supplied; others estimated"
        )
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        (
            hidden if g["cutaway"] or g["name"] in {"air_conditioner", "light"} else furniture
        ).objects.link(obj)
    # Floorboard seams are render detail, not physical collision obstacles.
    w, d = (layout["room"][k] for k in ("width", "depth"))
    for i in range(1, int(w / 0.18)):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(i * 0.18, d / 2, 0.0005))
        ob = bpy.context.object
        ob.name = f"floor_seam_{i}"
        ob.dimensions = (0.003, d, 0.001)
        ob.data.materials.append(mats["darkwood"])
    robot = layout["robot"]
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48,
        radius=robot["radius"],
        depth=robot["height"],
        location=(2.7, 0.95, robot["bottom"] + robot["height"] / 2),
    )
    bot = bpy.context.object
    bot.name = "Robot diameter 0.30 m height 0.32 m"
    mat = bpy.data.materials.new("Robot teal")
    mat.diffuse_color = (0.03, 0.43, 0.50, 1)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    shader = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (0.03, 0.43, 0.50, 1)
    material_output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(shader.outputs["BSDF"], material_output.inputs["Surface"])
    bot.data.materials.append(mat)
    archive_file = out / args.validation / "trajectories.npz"
    if archive_file.exists():
        with np.load(archive_file) as archive:
            replay = archive["truth"]
        for frame, pose in enumerate(replay, 1):
            bot.location = (pose[0], pose[1], robot["bottom"] + robot["height"] / 2)
            bot.rotation_euler[2] = pose[2]
            bot.keyframe_insert(data_path="location", frame=frame)
            bot.keyframe_insert(data_path="rotation_euler", frame=frame)
        bpy.context.scene.frame_end = len(replay)
        bpy.context.scene.render.fps = 24
        bpy.context.scene.frame_set(1)
    # Saved route, if validation has run, is an actual replay path.
    route_file = out / args.validation / "preview_path.json"
    if route_file.exists():
        route = json.loads(route_file.read_text(encoding="utf-8"))
        curve = bpy.data.curves.new("Verified contact-free reference route", "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = 0.009
        curve.bevel_resolution = 3
        poly = curve.splines.new("POLY")
        poly.points.add(len(route) - 1)
        for p, xyz in zip(poly.points, route):
            p.co = (xyz[0], xyz[1], 0.018, 1)
        obj = bpy.data.objects.new("Reference route - truth controlled", curve)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(mat)

    def aim(obj, where):
        obj.rotation_euler = (Vector(where) - obj.location).to_track_quat("-Z", "Y").to_euler()

    for name, loc, power, size in [
        ("Softbox", (2, 1, 5), 600, 4),
        ("Window light", (1.6, 3.4, 3.7), 450, 2.5),
    ]:
        light = bpy.data.lights.new(name, "AREA")
        light.energy = power
        light.shape = "DISK"
        light.size = size
        obj = bpy.data.objects.new(name, light)
        bpy.context.collection.objects.link(obj)
        obj.location = loc
        aim(obj, (1.5, 2, 0))
    camera = bpy.data.cameras.new("Overview")
    obj = bpy.data.objects.new("Overview", camera)
    bpy.context.collection.objects.link(obj)
    obj.location = (8.5, 0.6, 7.5)
    aim(obj, (1.55, 1.85, 0.85))
    camera.type = "ORTHO"
    camera.ortho_scale = 6.3
    scene = bpy.context.scene
    scene.camera = obj
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1440
    scene.render.resolution_y = 1280
    scene.render.resolution_percentage = 100
    scene.world.color = (0.35, 0.35, 0.35)
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.7
    scene["model_status"] = layout["status"]
    text = bpy.data.texts.new("READ_ME_model_limits")
    text.write(json.dumps(layout, ensure_ascii=False, indent=2))
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
                area.spaces.active.clip_end = 100
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "photo_room.blend"))
    scene.render.filepath = str(out / "overview.png")
    bpy.ops.render.render(write_still=True)
    # Human eye-height inspection view; separate robot camera comes from MuJoCo.
    hidden.hide_render = False
    obj.location = (2.65, 1.0, 1.45)
    aim(obj, (0.8, 2.65, 0.8))
    camera.type = "PERSP"
    camera.lens = 23
    scene.render.filepath = str(out / "interior.png")
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
