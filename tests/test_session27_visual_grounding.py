"""Lesson 27: foundation-model masks assign landmark identities (no torch).

All tests run on SYNTHETIC strip masks with analytic depth (a level pinhole
camera, three vertical strips at distinct depths standing in for cylinder
masks); the real MobileSAM checkpoint and the bench venv are never touched.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from embodied_learning.differential_drive import SENSOR_IN_BODY, compose
from embodied_learning.experiments.pinhole_projection import (
    HEIGHT_PX,
    K_INTRINSIC,
    WIDTH_PX,
    look_at,
    unproject,
)
from embodied_learning.experiments.visual_grounding import (
    GROUP_NAMES,
    MIN_VALID_PX,
    REQUIRED_INPUT_KEYS,
    analyze_pose,
    camera_ground_pose,
    dilate_mask,
    erode_mask,
    load_input,
    mask_centroid,
    pitch_factor,
    pixel_bearing,
    run_experiment,
    shift_mask,
    solve_camera_pose,
    wrap_angle,
)
from embodied_learning.landmark_localization import bearing_reading, inverse_pose

EYE_LEVEL = np.array([0.0, 0.0, 1.4])
TARGET_LEVEL = np.array([1.0, 0.0, 1.4])
STRIP_DEPTH = (2.5, 3.0, 3.5)  # one depth per strip -> landmarks off one line
STRIP_U = (160.0, 320.0, 480.0)
STRIP_HALF_W = 15
STRIP_ROWS = (100, 400)  # rows [100, 400)
CENTROID_V = 249.5  # mean of STRIP_ROWS

FOCAL_PX = K_INTRINSIC[0, 0]


# --------------------------------------------------------------- synthetic scene
def synthetic_arrays(*, narrow_first_mask=0):
    """One level pose, three vertical strip 'cylinders', analytic depth.

    Landmark i sits at the back-projection of its strip centroid, so the
    nearest-neighbour identity of the exact strip mask is i itself.
    narrow_first_mask: drop that many leftmost columns of mask 0 (IoU < 1).
    """
    rot, trans = look_at(EYE_LEVEL, TARGET_LEVEL)
    depth = np.full((1, HEIGHT_PX, WIDTH_PX), 5.0)
    valid = np.ones((1, HEIGHT_PX, WIDTH_PX), dtype=bool)
    label = np.zeros((1, HEIGHT_PX, WIDTH_PX), dtype=np.int8)
    masks = np.zeros((3, HEIGHT_PX, WIDTH_PX), dtype=bool)
    landmark_xy = np.zeros((3, 2))
    r0, r1 = STRIP_ROWS
    for i, (u_c, d) in enumerate(zip(STRIP_U, STRIP_DEPTH)):
        c0, c1 = int(u_c) - STRIP_HALF_W, int(u_c) + STRIP_HALF_W
        depth[0, r0:r1, c0:c1] = d
        label[0, r0:r1, c0:c1] = i + 1
        mask_c0 = c0 + (narrow_first_mask if i == 0 else 0)
        masks[i, r0:r1, mask_c0:c1] = True
        # landmark = back-projection of the TRUE strip-mask centroid: the strip
        # spans cols [c0, c1), so its centroid sits at c0 + half - 0.5 px
        world = unproject(
            np.array([[c0 + STRIP_HALF_W - 0.5, CENTROID_V]]), np.array([d]), rot, trans
        )[0]
        landmark_xy[i] = world[:2]
    return {
        "pose_rotation": rot[None],
        "pose_translation": trans[None],
        "depth_m": depth,
        "valid": valid,
        "landmark_label": label,
        "model_mask": masks,
        "mask_pose": np.zeros(3, dtype=np.int64),
        "landmark_xy": landmark_xy,
    }


def write_synthetic_npz(path):
    data = synthetic_arrays()
    meta = {
        "model": "synthetic strips (test double, no MobileSAM)",
        "intrinsic": K_INTRINSIC.tolist(),
        "canvas_px": [WIDTH_PX, HEIGHT_PX],
        "landmark_xy": data["landmark_xy"].tolist(),
        "landmark_radius_m": 0.06,
        "landmark_height_m": 1.2,
        "pose_eye_m": [EYE_LEVEL.tolist()],
        "pose_target_m": [TARGET_LEVEL.tolist()],
    }
    payload = {**data, "rgb": np.zeros((1, HEIGHT_PX, WIDTH_PX, 3), dtype=np.uint8)}
    payload["landmark_radius_m"] = np.array(0.06)
    payload["landmark_height_m"] = np.array(1.2)
    payload["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **payload)
    return path


# ------------------------------------------------------------------- mask ops
def test_shift_mask_exact():
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:5, 1:4] = True
    right = shift_mask(mask, 3)
    assert right[2:5, 4:7].all() and not right[:, :4].any()  # cols 1:4 -> 4:7
    left = shift_mask(mask, -2)
    assert left[2:5, 0:2].all() and not left[:, 2:].any()  # cols 1:4 -> clipped
    np.testing.assert_array_equal(shift_mask(mask, 0), mask)
    assert shift_mask(mask, 0) is not mask  # copy, not alias
    assert not shift_mask(mask, 8).any()  # fully out of frame
    assert not shift_mask(mask, -8).any()


def test_morph_square_rectangle():
    mask = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=bool)
    mask[100:300, 200:400] = True
    eroded = erode_mask(mask, 2)
    expected = np.zeros_like(mask)
    expected[102:298, 202:398] = True
    np.testing.assert_array_equal(eroded, expected)
    dilated = dilate_mask(mask, 2)
    expected = np.zeros_like(mask)
    expected[98:302, 198:402] = True
    np.testing.assert_array_equal(dilated, expected)
    # square element: opening/closing restore an interior rectangle exactly
    np.testing.assert_array_equal(dilate_mask(erode_mask(mask, 1), 1), mask)
    np.testing.assert_array_equal(erode_mask(dilate_mask(mask, 1), 1), mask)
    # the frame counts as background: dilation stops at the border
    edge = np.zeros_like(mask)
    edge[:5, :5] = True
    grown = dilate_mask(edge, 3)
    assert not grown[:, 8:].any() and not grown[8:, :].any()
    assert grown[0:8, 0:8].all()  # 0:5 grown by 3 px -> clipped to 0:8


def test_morphology_preserves_rectangle_centroid():
    """Mechanism guard: +-k px erosion/dilation of a strip keeps its centroid.

    This is why the erode/dilate ablation groups track truth_mask in the
    official record: the centroid is a first moment, boundary pixels cancel.
    """
    mask = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=bool)
    mask[100:400, 300:340] = True
    valid = np.ones_like(mask)
    depth = np.full(mask.shape, 3.0)
    base = mask_centroid(mask, valid, depth)
    for k in (4, 8):
        for morphed in (erode_mask(mask, k), dilate_mask(mask, k)):
            centroid = mask_centroid(morphed, valid, depth)
            assert centroid is not None
            np.testing.assert_allclose(centroid[:2], base[:2], atol=1e-9)
            assert centroid[2] == base[2]  # same depth sample


def test_mask_centroid_valid_pixel_rules():
    valid = np.ones((HEIGHT_PX, WIDTH_PX), dtype=bool)
    depth = np.full((HEIGHT_PX, WIDTH_PX), 3.0)
    mask = np.zeros_like(valid)
    mask[100:200, 200:210] = True  # 1000 px
    assert mask_centroid(mask, valid, depth) is not None
    # below the minimum valid count -> rejected
    tiny = np.zeros_like(valid)
    tiny[100:103, 200:206] = True  # 18 px < MIN_VALID_PX
    assert len(np.flatnonzero(tiny)) < MIN_VALID_PX
    assert mask_centroid(tiny, valid, depth) is None
    # invalid-depth pixels are excluded from the centroid: an invalid left
    # half pushes the centroid into the valid right half
    valid2 = np.ones_like(valid)
    valid2[:, :300] = False
    two = np.zeros_like(valid)
    two[100:200, 200:400] = True
    centroid = mask_centroid(two, valid2, depth)
    assert centroid is not None
    assert centroid[0] == pytest.approx(349.5)  # mean of valid cols 300..399
    # depth fallback: NaN at the rounded centroid, finite 3x3 neighbourhood
    depth2 = depth.copy()
    depth2[150, 204] = np.nan
    got = mask_centroid(mask, valid, depth2)
    assert got is not None and got[2] == pytest.approx(3.0)  # neighbours win


# ------------------------------------------------------------------ geometry
def test_camera_ground_pose_and_exact_procrustes_roundtrip():
    rot, trans = look_at(EYE_LEVEL, TARGET_LEVEL)
    np.testing.assert_allclose(camera_ground_pose(rot, trans), [0.0, 0.0, 0.0], atol=1e-12)
    rot_p, trans_p = look_at(np.array([2.0, 1.5, 1.4]), np.array([3.0, 3.0, 0.0]))
    forward = rot_p[2]
    expected_heading = float(np.arctan2(forward[1], forward[0]))
    ground = camera_ground_pose(rot_p, trans_p)
    np.testing.assert_allclose(ground[:2], [-rot_p.T @ trans_p][0][:2][:2], atol=1e-12)
    assert ground[2] == pytest.approx(expected_heading)
    # lesson-18 solve reused verbatim: exact readings recover the SENSOR pose,
    # solve_camera_pose undoes the vehicle composition exactly
    sensor = np.array([0.5, -0.3, 0.7])
    landmarks = synthetic_arrays()["landmark_xy"]
    readings = np.array([bearing_reading(lm, sensor) for lm in landmarks])
    np.testing.assert_allclose(solve_camera_pose(readings, landmarks), sensor, atol=1e-9)
    vehicle = compose(sensor, inverse_pose(SENSOR_IN_BODY))  # sanity of the identity used
    np.testing.assert_allclose(compose(vehicle, SENSOR_IN_BODY), sensor, atol=1e-12)


def test_pixel_bearing_shift_law():
    rot, _ = look_at(EYE_LEVEL, TARGET_LEVEL)  # level camera, heading 0
    base = pixel_bearing(320.0, 240.0, rot, 0.0)
    assert base == pytest.approx(0.0, abs=1e-12)
    delta = pixel_bearing(321.0, 240.0, rot, 0.0) - base
    # rightward shift DECREASES the world bearing; level camera: |dB| = 1/f
    assert delta < 0.0
    assert -delta == pytest.approx(1.0 / FOCAL_PX, rel=1e-6)
    assert pitch_factor(240.0, rot) == pytest.approx(1.0)
    # pitched camera: the law rescales by 1/(cos p + (v-cy)/f * sin p); at the
    # principal-point row (v=240, u=320) the finite difference matches it to
    # third order, so the ratio sits at 1 within float noise
    rot_p, _ = look_at(np.array([2.0, 1.5, 1.4]), np.array([3.0, 3.0, 0.0]))
    heading_p = camera_ground_pose(rot_p, np.zeros(3))[2]
    delta_p = pixel_bearing(321.0, 240.0, rot_p, heading_p) - pixel_bearing(
        320.0, 240.0, rot_p, heading_p
    )
    assert -delta_p / (1.0 / FOCAL_PX) / pitch_factor(240.0, rot_p) == pytest.approx(1.0, abs=2e-3)
    assert pitch_factor(240.0, rot_p) > 1.0  # pitched down: bearing swing grows


# ------------------------------------------------------- candidate analysis
def test_analyze_pose_identity_iou_margin():
    data = synthetic_arrays()
    candidates, chosen = analyze_pose(data, 0)
    assert len(candidates) == 3
    assert [c["nn_identity"] for c in chosen] == [0, 1, 2]
    assert all(c["truth_label"] == i + 1 for i, c in enumerate(chosen))
    assert all(c["iou"] == pytest.approx(1.0) for c in chosen)
    assert all(c["min_dist_m"] == pytest.approx(0.0, abs=1e-9) for c in chosen)
    assert all(c["margin_m"] > 0.5 for c in chosen)  # axes are far apart
    # a mask missing 2 of its 30 columns: IoU drops to 28/30, the centroid
    # moves by ~1 px (~4 mm at 2.5 m), identity and margin survive
    data2 = synthetic_arrays(narrow_first_mask=2)
    _, chosen2 = analyze_pose(data2, 0)
    assert chosen2[0]["nn_identity"] == 0
    assert chosen2[0]["iou"] == pytest.approx(28.0 / 30.0)
    assert chosen2[0]["min_dist_m"] < 0.01


def test_forced_mismatch_explodes():
    data = synthetic_arrays()
    landmarks = data["landmark_xy"]
    sensor = np.array([0.3, -0.2, 0.4])
    readings = np.array([bearing_reading(lm, sensor) for lm in landmarks])
    exact = solve_camera_pose(readings, landmarks)
    assert float(np.linalg.norm(exact[:2] - sensor[:2])) < 1e-9
    rolled = np.roll(readings, 1, axis=0)  # cyclic identity shift, same numbers
    wrong = solve_camera_pose(rolled, landmarks)
    pos_err = float(np.linalg.norm(wrong[:2] - sensor[:2]))
    head_err = abs(float(wrap_angle(wrong[2] - sensor[2])))
    assert pos_err > 1.0  # metres, vs <1e-9 for correct identity
    assert head_err > 0.8  # three-cycle signature approaches 120 deg


# ------------------------------------------------------------- input contract
def test_load_input_contract_guards(tmp_path):
    path = write_synthetic_npz(tmp_path / "bench.npz")
    data, meta = load_input(path)
    assert data["pose_rotation"].shape == (1, 3, 3)
    assert meta["canvas_px"] == [WIDTH_PX, HEIGHT_PX]
    with np.load(path, allow_pickle=False) as npz:
        payload = {key: npz[key].copy() for key in npz.files}
    # missing required key
    broken = {k: v for k, v in payload.items() if k != "valid"}
    missing = tmp_path / "missing_key.npz"
    np.savez_compressed(missing, **broken)
    with pytest.raises(ValueError, match="missing keys"):
        load_input(missing)
    # empty model-mask archive
    empty = dict(payload)
    empty["model_mask"] = np.zeros((0, HEIGHT_PX, WIDTH_PX), dtype=bool)
    empty_path = tmp_path / "empty_masks.npz"
    np.savez_compressed(empty_path, **empty)
    with pytest.raises(ValueError, match="no model masks"):
        load_input(empty_path)
    # intrinsic disagrees with the lesson-22 K
    meta_bad = json.loads(payload["meta_json"].item())
    meta_bad["intrinsic"] = (np.asarray(meta_bad["intrinsic"]) * 2.0).tolist()
    bad_k = dict(payload)
    bad_k["meta_json"] = np.array(json.dumps(meta_bad))
    bad_k_path = tmp_path / "bad_intrinsic.npz"
    np.savez_compressed(bad_k_path, **bad_k)
    with pytest.raises(ValueError, match="intrinsic"):
        load_input(bad_k_path)
    # stored pose disagrees with look_at(eye, target)
    meta_pose = json.loads(payload["meta_json"].item())
    meta_pose["pose_eye_m"] = [[9.0, 9.0, 9.0]]
    bad_pose = dict(payload)
    bad_pose["meta_json"] = np.array(json.dumps(meta_pose))
    bad_pose_path = tmp_path / "bad_pose.npz"
    np.savez_compressed(bad_pose_path, **bad_pose)
    with pytest.raises(ValueError, match="look_at"):
        load_input(bad_pose_path)
    assert REQUIRED_INPUT_KEYS <= set(payload)


# --------------------------------------------------------------- end to end
def test_run_experiment_synthetic_contract(tmp_path):
    input_npz = write_synthetic_npz(tmp_path / "bench.npz")
    output = tmp_path / "run"
    summary = run_experiment(input_npz, output, runs=2, seed=0)
    assert summary["experiment"] == "visual_grounding_identity"
    assert summary["runs_per_group"] == 2 and summary["base_seed"] == 0
    assert summary["n_poses"] == 1
    assert set(summary["groups"]) == set(GROUP_NAMES)
    # synthetic strips: identity perfect, IoU perfect, zero surface bias
    assert summary["identity"]["accuracy"] == 1.0
    assert summary["iou"]["mean"] == pytest.approx(1.0)
    assert summary["surface_range_bias_m"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert summary["masks"]["missed_landmarks"] == 0
    # forced mismatch explodes; every identity-correct group stays sub-metre
    mismatch = summary["groups"]["mismatch"]["pos_mean_m"]
    assert mismatch > 1.0
    for name in GROUP_NAMES:
        if name != "mismatch":
            assert summary["groups"][name]["pos_mean_m"] < mismatch / 10.0
    # tamper-evident archive
    digest = hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    assert summary["trajectories_sha256"] == digest
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert archive["err_norm_m"].shape == (len(GROUP_NAMES), 1, 2)
        assert archive["group_names"].tolist() == list(GROUP_NAMES)
    assert (output / "comparison.png").is_file()
    # guards: existing output and too-few runs
    with pytest.raises(FileExistsError):
        run_experiment(input_npz, output, runs=2, seed=0)
    with pytest.raises(ValueError):
        run_experiment(input_npz, tmp_path / "runs1", runs=1, seed=0)


def test_cli_subprocess_end_to_end(tmp_path):
    input_npz = write_synthetic_npz(tmp_path / "bench.npz")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.visual_grounding",
            "--input",
            str(input_npz),
            "--output",
            str(tmp_path / "cli_run"),
            "--runs",
            "2",
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = tmp_path / "cli_run"
    assert (out / "summary.json").exists()
    assert (out / "trajectories.npz").exists()
    assert (out / "comparison.png").exists()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.visual_grounding",
            "--input",
            str(input_npz),
            "--output",
            str(out),
            "--runs",
            "2",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert again.returncode != 0  # new directories are never overwritten


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel(tmp_path):
    import tkinter as tk

    from embodied_learning.grounding_demo import GroundingDemo, load_replays

    input_npz = write_synthetic_npz(tmp_path / "bench.npz")
    output = tmp_path / "recording"
    run_experiment(input_npz, output, runs=2, seed=0)
    data = load_replays(output)  # digest + contract checks
    root = tk.Tk()
    root.withdraw()
    demo = GroundingDemo(root, data)
    root.update()
    assert demo.mode.get() == "overlays"
    assert demo.fig is not None
    assert "模型出了掩码" in demo.stats.cget("text")
    assert len(demo.fig.axes) == 2
    # redraw must not accumulate axes: the lesson-26 residual-axis guard
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 2
    demo.mode.set("groups")
    demo.redraw()
    assert "这一幕在对照" in demo.stats.cget("text")
    assert "掩码质心观测" in demo.stats.cget("text")
    assert len(demo.fig.axes) == 2
    demo.mode.set("mechanism")
    demo.redraw()
    assert "强制循环错配" in demo.stats.cget("text")
    assert len(demo.fig.axes) == 2
    # tampered summary is rejected by the demo loader
    broken = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    broken["identity"]["accuracy"] = 0.5
    (output / "summary.json").write_text(
        json.dumps(broken, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_replays(output)
    demo.close()
