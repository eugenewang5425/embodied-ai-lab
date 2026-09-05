"""Lesson 26: ICP registration of two noisy depth clouds."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from embodied_learning.experiments.pinhole_projection import (
    to_camera,
)
from embodied_learning.experiments.point_cloud_icp import (
    CONVERGED_TRANS_M,
    DEGEN_SIGMA_INDEX,
    EIGEN_CUTOFF,
    EXPERIMENT,
    MAX_CLOUD_POINTS,
    MAX_ITERS,
    OBJECTIVES,
    POSE_DELTA_M,
    POSE_YAW_DEG,
    RADIUS_SIGMA_M,
    SIGMA_VALUES,
    VOXEL_SIZE_M,
    analytic_normals,
    apply_transform,
    build_clouds,
    estimate_normals,
    icp_register,
    initial_guess_translation,
    nearest_indices,
    pose_error,
    rotation_z,
    run_experiment,
    scene_and_poses,
    solve_point_to_plane,
    solve_rigid_point_to_point,
    symmetry_quotient_error,
    voxel_downsample,
    voxel_select,
)
from embodied_learning.icp_demo import IcpDemo, load_replays


def test_pose_b_and_gt_transform():
    scene = scene_and_poses()
    gt_r, gt_t = scene["gt_rotation"], scene["gt_translation"]
    # the GT B->A transform is exactly a yaw about world z plus R_A applied to
    # the world-frame pose delta
    np.testing.assert_allclose(gt_r, rotation_z(-POSE_YAW_DEG), atol=1e-12)
    np.testing.assert_allclose(gt_t, scene["rotation_a"] @ np.asarray(POSE_DELTA_M), atol=1e-12)
    # it maps a B-frame point of a world point onto its A-frame point
    world = np.array([4.2, 3.6, 1.0])
    xa = to_camera(world[None], scene["rotation_a"], scene["translation_a"])[0]
    xb = to_camera(world[None], scene["rotation_b"], scene["translation_b"])[0]
    np.testing.assert_allclose(gt_r @ xb + gt_t, xa, atol=1e-12)


def test_scene_renders_and_pole_visibility():
    scene = scene_and_poses()  # raises if pose A disagrees with the lesson-23 map
    valid_a = scene["rendered_a"]["valid"]
    valid_b = scene["rendered_b"]["valid"]
    assert valid_a.sum() > 200000 and valid_b.sum() > 200000
    # the -25 deg yaw keeps the pole comfortably inside B's field of view
    assert scene["rendered_a"]["pole"].sum() > 1000
    assert scene["rendered_b"]["pole"].sum() > 1000
    depth_b = scene["rendered_b"]["depth_m"]
    assert np.nanmin(depth_b) > 0.5 and np.nanmax(depth_b) < 8.0


def test_voxel_select_uniform_and_deterministic():
    rng = np.random.default_rng(0)
    points = rng.normal(0.0, 1.0, (5000, 3))
    first = voxel_select(points, np.random.default_rng(7))
    again = voxel_select(points, np.random.default_rng(7))
    np.testing.assert_array_equal(first, again)
    assert len(first) <= MAX_CLOUD_POINTS
    # one representative per occupied voxel
    keys = np.floor(points[first] / VOXEL_SIZE_M).astype(np.int64)
    assert len(np.unique(keys, axis=0)) == len(first)
    down = voxel_downsample(points, np.random.default_rng(7))
    np.testing.assert_allclose(down, points[first])


def test_nearest_indices_hand_case_and_bruteforce():
    query = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    reference = np.array([[0.1, 0.0, 0.0], [0.9, 0.1, 0.0], [0.5, 0.6, 0.4], [2.0, 2.0, 2.0]])
    index, distance = nearest_indices(query, reference)
    np.testing.assert_array_equal(index, [0, 1, 2])
    np.testing.assert_allclose(
        distance,
        [0.1, np.hypot(0.1, 0.1), np.hypot(0.1, 0.1)],
        atol=1e-5,
    )
    rng = np.random.default_rng(3)
    q = rng.normal(0.0, 2.0, (300, 3))
    r = rng.normal(0.0, 2.0, (400, 3))
    index, distance = nearest_indices(q, r)
    brute = np.argmin(((q[:, None, :] - r[None, :, :]) ** 2).sum(-1), axis=1)
    np.testing.assert_array_equal(index, brute)
    exact = np.linalg.norm(q - r[brute], axis=1)
    np.testing.assert_allclose(distance, exact, atol=1e-5)


def test_solve_rigid_point_to_point_recovers_known_transform():
    rng = np.random.default_rng(0)
    points = rng.normal(0.0, 1.0, (80, 3))
    true_r = rotation_z(37.0) @ np.array(
        [[0.936, 0.0, 0.352], [0.0, 1.0, 0.0], [-0.352, 0.0, 0.936]]
    )
    true_t = np.array([0.4, -1.2, 0.7])
    moved = apply_transform(true_r, true_t, points)
    r_hat, t_hat = solve_rigid_point_to_point(moved, points)
    np.testing.assert_allclose(r_hat, true_r.T, atol=1e-12)
    np.testing.assert_allclose(t_hat, -true_r.T @ true_t, atol=1e-12)


def test_solve_point_to_plane_hand_cases_and_sliding_guard():
    rng = np.random.default_rng(1)
    plane = np.column_stack(
        [rng.uniform(-2.0, 2.0, 200), rng.uniform(-2.0, 2.0, 200), np.zeros(200)]
    )
    normals = np.tile([0.0, 0.0, 1.0], (200, 1))
    # a pure normal shift is recovered exactly
    r_inc, t_inc, _cond, rank = solve_point_to_plane(
        plane, plane + np.array([0.0, 0.0, 0.3]), normals, normals
    )
    assert rank == 3  # in-plane translations and the yaw are unobservable
    np.testing.assert_allclose(t_inc, [0.0, 0.0, 0.3], atol=1e-9)
    # a tangential shift leaves the increment at zero: the sliding guard
    r_inc, t_inc, _cond, _rank = solve_point_to_plane(
        plane, plane + np.array([0.25, 0.0, 0.0]), normals, normals
    )
    np.testing.assert_allclose(t_inc, [0.0, 0.0, 0.0], atol=1e-9)
    assert np.linalg.norm(r_inc - np.eye(3)) < 1e-9
    # with identical normals on both sides the two-sided solve equals the
    # one-sided solve, and the observable part of the offset is recovered
    moved = plane + np.array([0.05, -0.1, 0.2])
    n_moving = np.tile([0.0, 0.0, 1.0], (200, 1))
    r_two, t_two, _, _ = solve_point_to_plane(moved, plane, normals, n_moving)
    r_one, t_one, _, _ = solve_point_to_plane(moved, plane, normals, normals)
    np.testing.assert_allclose(r_two, r_one, atol=1e-12)
    np.testing.assert_allclose(t_two, t_one, atol=1e-12)
    np.testing.assert_allclose(t_two, [0.0, 0.0, -0.2], atol=1e-9)


def test_icp_exact_recovery_on_synthetic_two_plane_scene():
    rng = np.random.default_rng(2)
    # interior points only: k-NN PCA normals are exact away from the corner
    ground = np.column_stack(
        [rng.uniform(1.0, 4.0, 400), rng.uniform(1.0, 4.0, 400), np.zeros(400)]
    )
    wall = np.column_stack([np.zeros(200), rng.uniform(1.0, 4.0, 200), rng.uniform(0.5, 1.5, 200)])
    cloud = np.vstack([ground, wall])
    gen_r = rotation_z(15.0)
    # choose the generator translation so the GT translation has no component
    # along the two planes' intersection line (the one unobservable direction)
    gen_t = gen_r @ np.array([0.4, 0.0, 0.1])
    moving = apply_transform(gen_r, gen_t, cloud)
    ans_r, ans_t = gen_r.T, -gen_r.T @ gen_t
    assert abs(ans_t[1]) < 1e-12
    normals = estimate_normals(cloud, k=12)  # exact on clean planes
    r0 = rotation_z(5.0) @ ans_r
    t0 = ans_t + np.array([0.1, 0.0, 0.05])
    res = icp_register(
        moving,
        cloud,
        normals,
        moving_normals=normals,
        init_rotation=r0,
        init_translation=t0,
        objective="point_to_plane",
    )
    assert res["converged"] and res["iterations"] < 40
    rot_err, trans_vec, _ = pose_error(res["rotation"], res["translation"], ans_r, ans_t)
    # the observable subspace (rotation + x/z translation) is recovered to
    # machine precision; the y-component is the exact family coordinate
    # (sliding both planes along their intersection line) and can take any
    # value without changing the cost
    assert rot_err < 1e-9
    assert abs(trans_vec[0]) < 1e-9 and abs(trans_vec[2]) < 1e-9


def test_icp_real_scene_sigma_zero_recovers_observable_subspace():
    scene = scene_and_poses()
    gt_r, gt_t = scene["gt_rotation"], scene["gt_translation"]
    clouds = build_clouds(scene, 0.0, 0, 0, 0)
    nf = analytic_normals(
        clouds["fixed"], clouds["fixed_pole"], scene["rotation_a"], scene["translation_a"]
    )
    nm = analytic_normals(
        clouds["moving"], clouds["moving_pole"], scene["rotation_b"], scene["translation_b"]
    )
    res = icp_register(
        clouds["moving"],
        clouds["fixed"],
        nf,
        moving_normals=nm,
        init_rotation=gt_r,
        init_translation=gt_t,
        objective="point_to_plane",
    )
    assert res["converged"] and res["iterations"] <= 60
    quotient = symmetry_quotient_error(
        res["rotation"],
        res["translation"],
        gt_r,
        gt_t,
        scene["rotation_a"],
        scene["translation_a"],
    )
    assert quotient["trans_err_obs_m"] < 0.02


def test_icp_ground_only_slides_and_is_recorded():
    scene = scene_and_poses()
    gt_r, gt_t = scene["gt_rotation"], scene["gt_translation"]
    clouds = build_clouds(scene, 0.15, DEGEN_SIGMA_INDEX, 0, 0, ground_only=True)
    nf = analytic_normals(
        clouds["fixed"], clouds["fixed_pole"], scene["rotation_a"], scene["translation_a"]
    )
    nm = analytic_normals(
        clouds["moving"], clouds["moving_pole"], scene["rotation_b"], scene["translation_b"]
    )
    r0, t0 = initial_guess_translation(scene, (0.1, 0.0, 0.0))
    res = icp_register(
        clouds["moving"],
        clouds["fixed"],
        nf,
        moving_normals=nm,
        init_rotation=r0,
        init_translation=t0,
        objective="point_to_plane",
    )
    # the run "converges" onto a wrong pose: the in-plane part of the initial
    # error is structurally unobservable and stays where it started
    assert res["converged"] and res["reason"] == "tol"
    assert res["last_rank"] == 3
    _, _, trans_err = pose_error(res["rotation"], res["translation"], gt_r, gt_t)
    assert 0.03 < trans_err < 0.30
    assert trans_err >= CONVERGED_TRANS_M


def test_identity_init_lands_far_along_the_symmetry_valley():
    scene = scene_and_poses()
    gt_r, gt_t = scene["gt_rotation"], scene["gt_translation"]
    clouds = build_clouds(scene, 0.02, 0, 0, 0)
    nf = analytic_normals(
        clouds["fixed"], clouds["fixed_pole"], scene["rotation_a"], scene["translation_a"]
    )
    nm = analytic_normals(
        clouds["moving"], clouds["moving_pole"], scene["rotation_b"], scene["translation_b"]
    )
    res = icp_register(
        clouds["moving"],
        clouds["fixed"],
        nf,
        moving_normals=nm,
        objective="point_to_plane",
    )
    quotient = symmetry_quotient_error(
        res["rotation"],
        res["translation"],
        gt_r,
        gt_t,
        scene["rotation_a"],
        scene["translation_a"],
    )
    # no convergence against the 2 cm criterion, and the rotation error keeps
    # the sign of the 25 deg ground-truth yaw (the valley slide never crosses it)
    assert quotient["naive_trans_m"] > 0.3
    assert quotient["naive_rot_deg"] > 5.0


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    output = tmp_path_factory.mktemp("icp") / "result"
    report = run_experiment(output, runs=2, seed=0, light=True)
    return output, report


def test_run_experiment_guards_and_contract(recording):
    output, report = recording
    assert report["experiment"] == EXPERIMENT
    assert report["schema_version"] == 1
    assert report["canvas_px"] == [640, 480]
    assert report["sigma_values_m"] == list(SIGMA_VALUES)
    assert report["pose_b"]["delta_world_m"] == list(POSE_DELTA_M)
    assert report["pose_b"]["yaw_deg"] == POSE_YAW_DEG
    assert report["icp"]["objectives"] == list(OBJECTIVES)
    expected_max_iters = 40 if report.get("light_mode") else MAX_ITERS
    assert report["icp"]["max_iters"] == expected_max_iters
    assert report["icp"]["p2plane_eigen_cutoff"] == EIGEN_CUTOFF
    assert report["radius_study"]["sigma_m"] == RADIUS_SIGMA_M
    # light fixture: the radius grid is the shrunk 2x2 test variant
    assert report["radius_study"]["rotation_perturb_deg"] == [0.0, 10.0]
    assert report["radius_study"]["translation_perturb_m"] == [0.0, 0.1]
    assert report["pixels"]["pole_b"] > 0
    assert (output / "comparison.png").exists()
    with pytest.raises(FileExistsError):
        run_experiment(output, runs=2, seed=0)
    with pytest.raises(ValueError):
        run_experiment(output.parent / "other", runs=1, seed=0)


def test_sigma_sweep_mechanism(recording):
    _, report = recording
    sweep = report["sigma_sweep"]
    obs = np.asarray(sweep["trans_obs_mean_m"])
    iters = np.asarray(sweep["iters_median"])
    # point-to-plane from the truth init stays at the noise floor for the two
    # lower sigmas and beats point-to-point, which the sampling forces drag
    # tens of centimetres away even from the truth
    assert obs[1, 0] < 0.05 and obs[1, 1] < 0.9
    assert obs[1, 0] < obs[0, 0]
    assert iters[1, 0] < iters[0, 0]
    # the observable error grows with sigma (the linear floor regime)
    assert obs[1, 1] > obs[1, 0]


def test_radius_matrix_and_valley_footprint(recording):
    _, report = recording
    radius = report["radius_study"]
    naive = np.asarray(radius["mean_trans_naive_m"])
    spin = np.asarray(radius["mean_spin_deg"])
    # truth init converges in the observable subspace on average
    truth = radius["truth_init"]["point_to_plane"]
    assert truth["mean_trans_obs_m"] < CONVERGED_TRANS_M
    assert truth["converged_fraction"] >= 0.5
    # the identity init cannot converge: the 25 deg yaw error lives in the
    # symmetry valley and the naive error keeps its footprint
    identity = radius["identity_init"]["point_to_plane"]
    assert identity["converged_fraction"] == pytest.approx(0.0)
    assert identity["mean_trans_err_m"] > 1.0
    assert identity["mean_rot_err_deg"] > 5.0
    # the yaw footprint: with no translation perturbation the naive error is
    # dominated by the valley coordinate and grows with the init yaw
    # (light fixture grid: rows are yaw 0 deg and 10 deg)
    assert naive[1, 0] > 10.0 * naive[0, 0]
    assert abs(spin[1, 0]) > abs(spin[0, 0])


def test_degenerate_counterexample_recorded(recording):
    _, report = recording
    degen = report["degenerate"]
    ground = degen["ground_only"]["point_to_plane"]
    assert ground["converged_count"] == report["runs_per_group"]
    assert ground["converged_but_wrong_count"] == report["runs_per_group"]
    assert ground["trans_mean_m"] >= CONVERGED_TRANS_M
    assert ground["trans_std_m"] < 0.2  # the freeze is deterministic
    reference = degen["degen_reference_run"]
    assert reference["converged"] and reference["rank_last"] == 3


def test_seed_determinism(tmp_path):
    first = run_experiment(tmp_path / "first", runs=2, seed=7, light=True)
    second = run_experiment(tmp_path / "second", runs=2, seed=7, light=True)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    digest_first = hashlib.sha256(
        (tmp_path / "first" / "trajectories.npz").read_bytes()
    ).hexdigest()
    digest_second = hashlib.sha256(
        (tmp_path / "second" / "trajectories.npz").read_bytes()
    ).hexdigest()
    assert digest_first == digest_second


def test_npz_arrays_and_reference_bundle(recording):
    output, _ = recording
    with np.load(output / "trajectories.npz", allow_pickle=False) as data:
        gt_r = data["gt_rotation"]
        fixed = data["ref_fixed_cloud"]
        moving = data["ref_moving_cloud"]
        normals = data["ref_fixed_normals"]
        rotations = data["ref_point_to_plane_rotation"]
        translations = data["ref_point_to_plane_translation"]
        mean_dist = data["ref_point_to_plane_mean_dist_m"]
        radius_frac = data["radius_converged_fraction"]
        sigma_values = data["sigma_values"]
    assert fixed.shape == moving.shape == (MAX_CLOUD_POINTS, 3)
    assert normals.shape == fixed.shape
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(gt_r @ gt_r.T, np.eye(3), atol=1e-12)
    assert rotations.shape[0] == len(mean_dist) + 1
    assert rotations.shape[1:] == (3, 3)
    assert translations.shape == (len(mean_dist) + 1, 3)
    assert radius_frac.shape == (1, 2, 2)  # light fixture grid
    np.testing.assert_array_equal(sigma_values, np.asarray(SIGMA_VALUES))
    # per-step transforms actually move the cloud monotonically closer
    assert mean_dist[-1] <= mean_dist[0]


def test_recording_rejects_tampering(tmp_path):
    output = tmp_path / "result"
    report = run_experiment(output, runs=2, seed=9, light=True)
    path = output / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    summary_path = output / "summary.json"
    original = summary_path.read_text(encoding="utf-8")
    data = json.loads(original)
    data["pose_b"]["yaw_deg"] = 42.0
    summary_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        load_replays(output)
    summary_path.write_text(original, encoding="utf-8")
    with np.load(path, allow_pickle=False) as npz:
        arrays = dict(npz)
    arrays["sweep_trans_mean_m"][0, 0] += 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(output)


def test_cli_end_to_end(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.point_cloud_icp",
            "--output",
            str(tmp_path / "cli_run"),
            "--runs",
            "2",
            "--seed",
            "3",
            "--light",
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
            "embodied_learning.experiments.point_cloud_icp",
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
    assert again.returncode != 0


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel(tmp_path):
    import tkinter as tk

    output = tmp_path / "recording"
    run_experiment(output, runs=2, seed=1, light=True)
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = IcpDemo(root, data)
    root.update()
    assert demo.mode.get() == "clouds"
    assert demo.fig is not None
    assert "两台相机" in demo.stats.cget("text")
    demo.mode.set("iters")
    demo.redraw()
    assert "对应-求解交替" in demo.stats.cget("text")
    demo.slider.set(demo._n_steps() - 2)
    demo.mode.set("error")
    demo.redraw()
    assert "退化反例" in demo.stats.cget("text")
    demo.close()
