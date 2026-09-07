"""Lesson 43: occupancy-grid mapping + A* planning + pure-pursuit tracking.

Unit checks pin the protocol contract: the hand-computed Bresenham line and
log-odds grid update, A* known answers (straight/octile/detour/unreachable,
no diagonal squeezing), obstacle inflation and the coverage metric, the ideal
ray sensor geometry (including the behind-the-origin regression), the
lesson-14/21 kinematics and wheel-conversion contract, the hand-computed
pure-pursuit law (with the differential-drive rotation branch and the
corner-clamped lookahead), scenario construction and paired initializations,
the wall-with-gap behavior contrast (blind pushes, planned detours), micro
determinism, the shrunk end-to-end record contract, aggregates, the CLI,
the demo loader's tamper rejections and the real Tk demo.  No full-scale run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

# The isolated_tk child re-imports this module before any pyplot use: forcing
# the Agg backend keeps run_experiment's figure saving from creating (and
# destroying) a first Tcl interpreter, whose leftovers make the demo test's
# real tk.Tk() die with "invalid command name tcl_findLibrary" on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from embodied_learning.differential_drive import integrate_pose
from embodied_learning.experiments.grid_nav import (
    COVERAGE_THRESHOLD,
    DT,
    EXPERIMENT,
    GEOMETRY,
    GROUPS,
    L_FREE,
    L_OCC,
    SNAPSHOTS,
    GridNavConfig,
    Obstacle,
    aggregate_episodes,
    astar,
    bresenham,
    build_scenarios,
    build_walls,
    cast_rays,
    coverage_fraction,
    expected_npz_keys,
    inflate,
    initial_poses,
    line_of_sight,
    make_plan,
    path_profile,
    pure_pursuit,
    simulate_episode,
    smooth_path,
    step_pose,
    true_occupancy,
    update_grid,
    wheels_from_twist,
    world_to_cell,
)
from embodied_learning.grid_nav_demo import load_replays

SMALL_CONFIG = GridNavConfig(
    world_size_m=4.0,
    horizon_steps=700,
    obstacle_scenes=1,
    inits=2,
    start_xy=(0.6, 0.6),
    goal_xy=(3.4, 3.4),
)
TINY_CONFIG = GridNavConfig(
    world_size_m=4.0,
    horizon_steps=250,
    obstacle_scenes=1,
    inits=1,
    start_xy=(0.6, 0.6),
    goal_xy=(3.4, 3.4),
)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("grid_nav_small") / "run"
    report = run_experiment_with(output)
    return output, report


def run_experiment_with(output, config=SMALL_CONFIG):
    from embodied_learning.experiments.grid_nav import run_experiment

    return run_experiment(output, seed=0, config=config, log=None)


# ----------------------------------------------------------------- primitives
def test_bresenham_hand_computed():
    """Classic all-octant Bresenham on hand-worked lines."""
    assert bresenham(2, 3, 2, 6) == [(2, 3), (2, 4), (2, 5), (2, 6)]
    assert bresenham(0, 0, 3, 3) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert bresenham(0, 0, 4, 2) == [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2)]
    # Bresenham is not symmetric under reversal; this is the hand-traced
    # reverse rasterization of the same line (err starts at dx + dy = 2).
    assert bresenham(4, 2, 0, 0) == [(4, 2), (3, 1), (2, 1), (1, 0), (0, 0)]
    assert bresenham(5, 5, 5, 5) == [(5, 5)]


def test_grid_update_ray_known_answer():
    """One hit ray: cells before the endpoint turn free, the endpoint occupied."""
    log_odds = np.zeros((8, 8))
    angles = np.array([0.0])
    ranges, hit = np.array([5.0]), np.array([True])
    update_grid(log_odds, (0.5, 0.5), angles, ranges, hit, res=1.0, n=8)
    assert log_odds[0, 0] == pytest.approx(L_FREE)
    assert log_odds[0, 4] == pytest.approx(L_FREE)
    assert log_odds[0, 5] == pytest.approx(L_OCC)
    assert log_odds[0, 6] == 0.0 and log_odds[1, 0] == 0.0  # nothing else touched
    probabilities = 1.0 / (1.0 + np.exp(-log_odds))
    assert probabilities[0, 0] == pytest.approx(0.4)
    assert probabilities[0, 5] == pytest.approx(0.7)
    # the same scan repeated: log-odds accumulate and clip at -L_MAX
    # (13 * L_FREE = -5.27 < -L_MAX, so the 13th identical scan clips)
    for _ in range(12):
        update_grid(log_odds, (0.5, 0.5), angles, ranges, hit, res=1.0, n=8)
    assert log_odds[0, 0] == pytest.approx(-5.0)
    assert log_odds[0, 5] == pytest.approx(5.0)  # 10 * L_OCC clips on the way


def test_grid_update_no_hit_marks_free():
    """A max-range miss marks the whole traversed line free; coverage follows."""
    log_odds = np.zeros((8, 8))
    update_grid(log_odds, (0.5, 0.5), np.array([0.0]), np.array([2.0]), np.array([False]), 1.0, 8)
    assert log_odds[0, 0] == pytest.approx(L_FREE)
    assert log_odds[0, 2] == pytest.approx(L_FREE)
    assert log_odds[0, 3] == 0.0
    assert coverage_fraction(log_odds) == pytest.approx(3.0 / 64.0)
    hand = np.zeros(10)
    hand[:2] = [math.log(0.7 / 0.3), math.log(0.4 / 0.6)]
    hand[2] = COVERAGE_THRESHOLD  # exactly at the threshold does not count
    assert coverage_fraction(hand.reshape(2, 5)) == pytest.approx(2.0 / 10.0)


def test_astar_known_answers():
    """Straight, octile, detour through a gap, unreachable, and no squeezing."""
    free = np.ones((20, 20), dtype=bool)
    cells, cost = astar(free, (2, 2), (12, 2))
    assert cost == pytest.approx(10.0) and len(cells) == 11
    assert all(ix == 2 for _iy, ix in cells)  # column fixed, rows vary
    _cells, cost = astar(free, (2, 2), (12, 12))
    assert cost == pytest.approx(10.0 * math.sqrt(2.0))
    # a wall with a one-cell gap forces the path through (9, 7)
    walled = free.copy()
    walled[:, 7] = False
    walled[9, 7] = True
    cells, cost = astar(walled, (2, 2), (12, 12))
    assert (9, 7) in cells and cost > 10.0 * math.sqrt(2.0)
    walled[9, 7] = False
    assert astar(walled, (2, 2), (12, 12)) == (None, math.inf)
    # a diagonal move is rejected when one orthogonal neighbour is blocked
    corner = free.copy()
    corner[2, 3] = False
    cells, cost = astar(corner, (3, 3), (2, 2))
    assert cost == pytest.approx(2.0) and len(cells) == 3  # two straight steps
    _cells, cost = astar(free, (3, 3), (2, 2))
    assert cost == pytest.approx(math.sqrt(2.0))


def test_inflate_disk_and_line_of_sight():
    """Inflation is an Euclidean disk of radius 2 cells; LOS follows Bresenham."""
    occupied = np.zeros((10, 10), dtype=bool)
    occupied[5, 5] = True
    disk = inflate(occupied, 2)
    assert disk[5, 5] and disk[5, 3] and disk[3, 5] and disk[6, 6] and disk[5, 7]
    assert not disk[4, 3] and not disk[2, 5] and not disk[7, 7]  # sqrt(8) > 2
    assert disk.sum() == 13  # the Euclidean disk of radius 2 cells
    assert np.array_equal(inflate(occupied, 0), occupied)
    free = np.ones((6, 6), dtype=bool)
    assert line_of_sight(free, (0, 0), (5, 5))
    blocked = free.copy()
    blocked[2, 2] = blocked[3, 3] = False
    assert not line_of_sight(blocked, (0, 0), (5, 5))
    cells = [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert smooth_path(cells, free) == [cells[0], cells[-1]]
    assert smooth_path(cells, blocked) == cells  # no LOS -> unchanged


def test_sensor_rays_known_geometry():
    """Analytic ray casting: hit distance = d - r, behind-origin is no hit."""
    walls = build_walls(10.0, 0.1)
    circle = Obstacle("circle", 5.0, 7.0, r=0.5)
    behind = Obstacle("circle", 5.0, 3.0, r=0.5)
    angles = np.array([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    ranges, hit = cast_rays(5.0, 5.0, angles, (circle, behind), walls, 2.0)
    assert ranges[1] == pytest.approx(1.5) and hit[1]  # surface at 7.0 - 0.5
    assert ranges[0] == pytest.approx(2.0) and not hit[0]  # circle behind: no hit
    assert ranges[2] == pytest.approx(2.0) and not hit[2]  # wall beyond max range
    assert ranges[3] == pytest.approx(1.5) and hit[3]  # the (5,3) circle ahead downward
    close_wall = Obstacle("rect", 0.0, 5.0, hx=0.1, hy=5.0)  # face at x = 0.1
    ranges, hit = cast_rays(0.5, 5.0, np.array([math.pi]), (), (close_wall,), 2.0)
    assert ranges[0] == pytest.approx(0.4) and hit[0]


def test_kinematics_and_wheel_contract():
    """Lesson-14/21 kinematics verbatim; joint scaling preserves curvature."""
    assert wheels_from_twist(0.2, 0.0, 6.0) == pytest.approx([4.0, 4.0])
    wheels = wheels_from_twist(0.0, 2.0, 6.0)
    assert wheels == pytest.approx([-6.0, 6.0])
    wheels = wheels_from_twist(0.2, 5.0, 6.0)  # raw [-11, 19] jointly scaled
    assert float(np.max(np.abs(wheels))) == pytest.approx(6.0)
    v, omega = GEOMETRY.body_velocity(wheels)
    assert v / omega == pytest.approx(0.2 / 5.0)  # curvature preserved
    pose = np.array([1.0, 2.0, 0.5])
    np.testing.assert_array_equal(
        step_pose(pose, wheels, DT),
        integrate_pose(pose, GEOMETRY.body_velocity(wheels), DT),
    )


def test_pure_pursuit_hand_computed():
    """The pursuit law, the spin branch, the stop ring and the corner clamp."""
    config = GridNavConfig()
    straight = np.array([(0.0, 0.0), (4.0, 0.0)])
    profile = path_profile(straight, config.turn_sharp_rad)
    command = pure_pursuit(np.array([0.0, 0.0, 0.0]), straight, profile, 0, config)
    assert command["mode"] == "tracking"
    assert command["v"] == pytest.approx(0.25) and command["omega"] == pytest.approx(0.0)
    assert command["wheels"] == pytest.approx([5.0, 5.0])
    assert command["lookahead_eff"] == pytest.approx(0.5)
    angled = np.array([(0.0, 0.0), (4.0 * math.cos(math.pi / 6), 4.0 * math.sin(math.pi / 6))])
    command = pure_pursuit(
        np.array([0.0, 0.0, 0.0]), angled, path_profile(angled, math.pi / 4), 0, config
    )
    alpha = math.pi / 6
    assert command["alpha"] == pytest.approx(alpha)
    assert command["v"] == pytest.approx(0.25 * math.cos(alpha))
    assert command["omega"] == pytest.approx(2.0 * math.sin(alpha) / 0.5 * 0.25)
    assert command["wheels"] == pytest.approx(
        [
            (command["v"] - command["omega"] * 0.15) / 0.05,
            (command["v"] + command["omega"] * 0.15) / 0.05,
        ]
    )
    behind = np.array([(0.0, 0.0), (-4.0, 0.0)])
    command = pure_pursuit(
        np.array([0.0, 0.0, 0.0]), behind, path_profile(behind, math.pi / 4), 0, config
    )
    assert command["mode"] == "rotating" and command["v"] == 0.0
    assert command["omega"] == pytest.approx(config.omega_max_rad_s)
    assert command["wheels"] == pytest.approx(
        [(0.0 - 1.2 * 0.15) / 0.05, (0.0 + 1.2 * 0.15) / 0.05]
    )
    near = np.array([(0.0, 0.0), (4.0, 0.0)])
    command = pure_pursuit(
        np.array([3.97, 0.0, 0.0]), near, path_profile(near, math.pi / 4), 1, config
    )
    assert command["mode"] == "stopping" and np.all(command["wheels"] == 0.0)
    corner = np.array([(0.0, 0.0), (0.3, 0.0), (0.3, 4.0)])
    command = pure_pursuit(
        np.array([0.0, 0.0, 0.0]), corner, path_profile(corner, math.pi / 4), 0, config
    )
    assert command["lookahead_eff"] == pytest.approx(0.3)  # clamped before the turn
    far_corner = np.array([(0.0, 0.0), (2.0, 0.0), (2.0, 4.0)])
    command = pure_pursuit(
        np.array([0.0, 0.0, 0.0]), far_corner, path_profile(far_corner, math.pi / 4), 0, config
    )
    assert command["lookahead_eff"] == pytest.approx(0.5)
    # the lookahead never targets through an obstacle that is already close
    command = pure_pursuit(
        np.array([0.0, 0.0, 0.0]),
        far_corner,
        path_profile(far_corner, math.pi / 4),
        0,
        config,
        frontal_clear_m=0.3,
    )
    assert command["lookahead_eff"] == pytest.approx(0.3)
    command = pure_pursuit(
        np.array([0.0, 0.0, 0.0]),
        far_corner,
        path_profile(far_corner, math.pi / 4),
        0,
        config,
        frontal_clear_m=0.05,
    )
    assert command["lookahead_eff"] == pytest.approx(config.lookahead_min_m)


def test_scenario_construction_contract():
    """5-8 obstacles per scene, clearances, determinism, paired initializations."""
    config = GridNavConfig()
    first = build_scenarios(config, 0)
    second = build_scenarios(config, 0)
    assert len(first) == 6 and first[0]["kind"] == "empty"
    for scenario in first[1:]:
        assert scenario["kind"] == "obstacles"
        assert 5 <= len(scenario["obstacles"]) <= 8
        for obstacle in scenario["obstacles"]:
            assert obstacle.distance(*config.start_xy) >= 0.5
            assert obstacle.distance(*config.goal_xy) >= 0.5
            extent = obstacle.extent()
            assert 0.35 <= obstacle.cx - extent and obstacle.cx + extent <= 9.65
            assert 0.35 <= obstacle.cy - extent and obstacle.cy + extent <= 9.65
        assert [o.to_dict() for o in scenario["obstacles"]] == [
            o.to_dict() for o in second[scenario["scene"]]["obstacles"]
        ]
    poses = initial_poses(config, 0, 2)
    assert len(poses) == 3
    for pose, same in zip(poses, initial_poses(config, 0, 2), strict=True):
        np.testing.assert_array_equal(pose, same)
        assert abs(pose[0] - config.start_xy[0]) <= config.init_jitter_m + 1e-12
        assert abs(pose[1] - config.start_xy[1]) <= config.init_jitter_m + 1e-12
        assert abs(pose[2]) <= math.pi + 1e-12


def test_behavior_contrast_wall_gap():
    """A blind car pushes into the wall; the planned car detours through the gap."""
    config = GridNavConfig(
        world_size_m=4.0,
        horizon_steps=900,
        obstacle_scenes=1,
        inits=1,
        start_xy=(0.6, 0.6),
        goal_xy=(3.4, 0.8),
    )
    wall = Obstacle("rect", 2.0, 1.35, hx=0.05, hy=1.25)  # spans y in [0.1, 2.6]
    metrics_a, truth_a, _extras = simulate_episode("A", (wall,), np.array([0.6, 0.6, 0.0]), config)
    metrics_b, truth_b, _extras = simulate_episode("B", (wall,), np.array([0.6, 0.6, 0.0]), config)
    assert metrics_a["outcome"] == "timeout"
    assert metrics_a["collisions"] >= 1 and metrics_a["pressed_steps"] > 100
    assert float(np.max(truth_a[:, 0])) < 2.0  # never passes the wall
    assert metrics_b["outcome"] == "arrived"
    assert metrics_b["collisions"] == 0 and metrics_b["replans"] >= 2
    assert float(np.max(truth_b[:, 1])) > 2.65  # detoured through the gap
    # the reference shortest path also uses the gap; the planned path stays sane
    walls = build_walls(4.0, 0.1)
    n, res = config.grid_cells, config.res_m
    true_map = true_occupancy((wall,), walls, n, res)
    passable = ~inflate(true_map, config.inflate_cells)
    _cells, cost = astar(passable, world_to_cell(0.6, 0.6, res, n), world_to_cell(3.4, 0.8, res, n))
    shortest = cost * res
    assert metrics_b["path_length_m"] / shortest < 2.0


def test_micro_run_deterministic(tmp_path):
    """The same tiny config reproduces the archive bitwise."""
    first = run_experiment_with(tmp_path / "one", TINY_CONFIG)
    second = run_experiment_with(tmp_path / "two", TINY_CONFIG)
    assert first["trajectories_sha256"] == second["trajectories_sha256"]
    assert json.dumps(first["aggregates"], sort_keys=True) == json.dumps(
        second["aggregates"], sort_keys=True
    )
    assert first["scenarios"] == second["scenarios"]


def test_small_run_record_contract(small_run, tmp_path):
    """Shrunk end-to-end: files, episodes, pairing, archive keys, no overwrite."""
    from embodied_learning.experiments.grid_nav import run_experiment

    output, report = small_run
    assert report["experiment"] == EXPERIMENT and report["schema_version"] == 1
    protocol = report["protocol"]
    assert protocol["scene_count"] == 2 and protocol["inits"] == 2
    assert len(report["episodes"]) == protocol["scene_count"] * protocol["inits"] * 2
    for scene in range(protocol["scene_count"]):
        for init in range(protocol["inits"]):
            rows = [
                row for row in report["episodes"] if row["scene"] == scene and row["init"] == init
            ]
            by_group = {row["group"]: row for row in rows}
            assert set(by_group) == set(GROUPS)
            assert by_group["A"]["start_pose"] == by_group["B"]["start_pose"]
    for row in report["episodes"]:
        if row["group"] == "B":
            assert 0.0 < row["coverage_final"] <= 1.0
            assert row["replans"] >= 1 and row["fallback_steps"] is not None
        else:
            assert row["coverage_final"] is None and row["replans"] is None
        if row["outcome"] == "arrived":
            assert row["path_ratio"] is not None and row["path_ratio"] > 0.0
        else:
            assert row["outcome"] == "timeout"
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        assert archive["snap_grids"].shape == (SNAPSHOTS, 40, 40)
        assert archive["snap_poses"].shape == (SNAPSHOTS, 3)
        assert archive["snap_cov"].shape == (SNAPSHOTS,)
        assert archive["true_map_0"].shape == (40, 40)
        assert len(archive["replan_steps"]) >= 1
    for name in ("summary.json", "trajectories.npz", "comparison.png", "scenario_maps.png"):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, seed=0, config=SMALL_CONFIG, log=None)


def test_aggregate_known_answers():
    """Aggregate counts, medians and None handling on synthetic episodes."""
    rows = [
        {
            "outcome": "arrived",
            "path_ratio": 1.2,
            "coverage_final": 0.5,
            "arrival_time_s": 20.0,
            "collisions": 0,
            "pressed_steps": 0,
            "min_obstacle_distance_m": 0.30,
        },
        {
            "outcome": "arrived",
            "path_ratio": 1.0,
            "coverage_final": 0.7,
            "arrival_time_s": 30.0,
            "collisions": 2,
            "pressed_steps": 10,
            "min_obstacle_distance_m": 0.16,
        },
        {
            "outcome": "timeout",
            "path_ratio": None,
            "coverage_final": 0.6,
            "arrival_time_s": None,
            "collisions": 1,
            "pressed_steps": 5,
            "min_obstacle_distance_m": 0.155,
        },
    ]
    aggregate = aggregate_episodes(rows)
    assert aggregate["episodes"] == 3 and aggregate["arrivals"] == 2
    assert aggregate["arrival_rate"] == pytest.approx(2.0 / 3.0)
    assert aggregate["collision_events_total"] == 3
    assert aggregate["collision_events_mean"] == pytest.approx(1.0)
    assert aggregate["episodes_with_collision"] == 2
    assert aggregate["pressed_steps_total"] == 15
    assert aggregate["mean_min_clearance_m"] == pytest.approx((0.30 + 0.16 + 0.155) / 3)
    assert aggregate["mean_path_ratio"] == pytest.approx(1.1)
    assert aggregate["median_path_ratio"] == pytest.approx(1.1)
    assert aggregate["mean_coverage"] == pytest.approx(0.6)
    assert aggregate["mean_arrival_time_s"] == pytest.approx(25.0)
    empty = aggregate_episodes([dict(rows[0], path_ratio=None, arrival_time_s=None)])
    assert empty["mean_path_ratio"] is None and empty["mean_arrival_time_s"] is None


def test_true_shortest_empty_scene():
    """Empty world: the grid shortest path matches the straight diagonal."""
    config = GridNavConfig()
    walls = build_walls(10.0, 0.1)
    n, res = config.grid_cells, config.res_m
    true_map = true_occupancy((), walls, n, res)
    passable = ~inflate(true_map, config.inflate_cells)
    assert not passable[0, 0] and passable[10, 10]
    _cells, cost = astar(passable, world_to_cell(1.0, 1.0, res, n), world_to_cell(9.0, 9.0, res, n))
    assert cost * res == pytest.approx(8.0 * math.sqrt(2.0), abs=0.02)
    assert (
        make_plan(np.zeros((n, n)), np.array([1.0, 1.0, 0.0]), np.array([9.0, 9.0]), config)
        is not None
    )


def test_cli_subprocess_end_to_end(tmp_path):
    """CLI micro run: stdout JSON, files on disk, new directory never overwritten."""
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.grid_nav",
            "--output",
            str(out),
            "--seed",
            "0",
            "--world-size",
            "4",
            "--horizon",
            "400",
            "--scenes",
            "1",
            "--inits",
            "1",
            "--start",
            "0.6",
            "0.6",
            "--goal",
            "3.4",
            "3.4",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["aggregates"]["A"]["empty"]["episodes"] == 1
    assert payload["aggregates"]["B"]["obstacle"]["episodes"] == 1
    assert "met" in payload["hypothesis"]
    for name in ("summary.json", "trajectories.npz", "comparison.png", "scenario_maps.png"):
        assert (out / name).is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.grid_nav",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert again.returncode != 0  # new directories are never overwritten


def test_demo_loader_cross_checks_and_rejects_tampering(small_run, tmp_path):
    """The pristine record passes; hash, key-set, aggregate and map tampering fail."""
    output, _report = small_run
    data = load_replays(output)
    assert data["report"]["experiment"] == EXPERIMENT

    # (1) summary arrivals disagree with the archived metric arrays
    work = tmp_path / "tampered_aggregates"
    shutil.copytree(output, work)
    parsed = json.loads((work / "summary.json").read_text(encoding="utf-8"))
    parsed["aggregates"]["B"]["obstacle"]["arrivals"] += 1
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Arrival counts disagree"):
        load_replays(work)

    # (2) archive bytes tampered -> checksum mismatch
    work2 = tmp_path / "tampered_bytes"
    shutil.copytree(output, work2)
    path = work2 / "trajectories.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["snap_cov"] = payload["snap_cov"] + 0.5
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(work2)

    # (3) an array removed with the hash refreshed -> unexpected key set
    del payload["snap_cov"]
    np.savez_compressed(path, **payload)
    parsed = json.loads((work2 / "summary.json").read_text(encoding="utf-8"))
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work2 / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unexpected archive arrays"):
        load_replays(work2)

    # (4) a true-map cell flipped in the archive (hash refreshed)
    work4 = tmp_path / "tampered_map"
    shutil.copytree(output, work4)
    path4 = work4 / "trajectories.npz"
    with np.load(path4, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["true_map_0"] = ~payload["true_map_0"]
    np.savez_compressed(path4, **payload)
    parsed4 = json.loads((work4 / "summary.json").read_text(encoding="utf-8"))
    parsed4["trajectories_sha256"] = hashlib.sha256(path4.read_bytes()).hexdigest()
    (work4 / "summary.json").write_text(
        json.dumps(parsed4, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="True map disagrees"):
        load_replays(work4)


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel(small_run):
    import tkinter as tk

    from embodied_learning.grid_nav_demo import GridNavDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = GridNavDemo(root, data)
    root.update()
    assert demo.mode.get() == "mapping"
    assert len(demo.fig.axes) == 2
    assert "覆盖率" in demo.stats.cget("text")
    demo.slider.set(demo.slider.cget("to"))
    demo.redraw()
    assert len(demo.fig.axes) == 2  # slider redraws must not accumulate axes
    demo.mode.set("tracking")
    demo.redraw()
    assert len(demo.fig.axes) == 2
    assert "重规划" in demo.stats.cget("text")
    demo.mode.set("compare")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    obstacle = report["aggregates"]["B"]["obstacle"]
    assert f"{obstacle['arrivals']}/{obstacle['episodes']}" in panel  # summary numbers
    demo.redraw()
    assert len(demo.fig.axes) == 4  # mode switches must not accumulate axes
    demo.close()
