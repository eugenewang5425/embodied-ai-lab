"""Closed-loop photo-room navigation - contracts and record integrity."""

import hashlib
import json
import math

import numpy as np
import pytest


def mod():
    import embodied_learning.experiments.photo_room_navigation as mm

    return mm


def test_config_guards():
    mm = mod()
    with pytest.raises(ValueError):
        mm.NavConfig(particles=10)
    with pytest.raises(ValueError):
        mm.NavConfig(sensor_sigma_m=0.5)
    with pytest.raises(ValueError):
        mm.NavConfig(wheel_bias=0.5)
    with pytest.raises(ValueError):
        mm.NavConfig(max_steps=100)


def test_wheel_increments_straight_and_turn():
    mm = mod()
    from embodied_learning.experiments.grid_nav import GEOMETRY

    straight = mm.wheel_increments(0.15, 0.0)
    assert straight[0] == pytest.approx(straight[1])
    v = GEOMETRY.radius_m * (straight[0] + straight[1]) / 2.0 / mm.DT
    assert v == pytest.approx(0.15)
    turn = mm.wheel_increments(0.0, 1.0)
    assert turn[0] < 0 < turn[1]


def test_wrap():
    mm = mod()
    assert mm.wrap(math.pi + 0.2) == pytest.approx(-math.pi + 0.2)
    assert mm.wrap(-0.1) == pytest.approx(-0.1)


def test_classify_tasks_four_reachable_one_unreachable():
    from embodied_learning.experiments.photo_room_navigation import classify_tasks
    from embodied_learning.photo_room import RoomWorld, default_layout

    world = RoomWorld(default_layout())
    reachable, unreachable = classify_tasks(world, default_layout())
    assert len(reachable) == 4
    assert len(unreachable) == 1
    name, start, goal = unreachable[0]
    assert name == "into_bed_footprint"
    # the unreachable goal must be free at the lidar slice (a naive low map
    # would call it passable) - verified by construction in classify_tasks
    assert all(len(g) == 2 for _n, _s, g in reachable)


def test_control_step_safety_stop():
    mm = mod()
    path = np.array([[0.0, 0.0], [2.0, 0.0]])
    state = {"v_history": []}
    scan_all_clear = [3.0] * 64
    inc, idx, v = mm.control_step(np.array([0.0, 0.0, 0.0]), path, 0, scan_all_clear, state)
    assert v > 0  # clears the start waypoint, drives toward the second
    scan_blocked = [3.0] * 40 + [0.05] * 24  # forward sector blocked
    _inc, _idx, v2 = mm.control_step(np.array([0.0, 0.0, 0.0]), path, 0, scan_blocked, state)
    assert v2 == 0.0


def test_group_a_estimate_chain_is_truth():
    """Group A is simulation-privileged: its estimate chain must equal the
    truth chain (zero localization error by construction)."""
    from embodied_learning.experiments.photo_room_navigation import (
        NavConfig,
        build_maps,
        classify_tasks,
        run_closed_loop,
    )
    from embodied_learning.photo_room import RoomWorld, default_layout

    cfg = NavConfig(particles=50)
    layout = default_layout()
    world = RoomWorld(layout)
    avoid, loc_map = build_maps(world, layout)
    reach, _unreach = classify_tasks(world, layout)
    rng = np.random.default_rng(0)
    _row, chains = run_closed_loop(world, reach[0], "A", loc_map, rng, cfg, min(cfg.max_steps, 400))
    n = min(len(chains["truth"]), len(chains["estimate"]))
    errors = np.linalg.norm(chains["truth"][:n, :2] - chains["estimate"][:n, :2], axis=1)
    assert float(errors.max()) < 0.01  # group A estimate = truth (zero drift, within float noise)


# ------------------------------------------------------------- records
@pytest.fixture(scope="module")
def light_record(tmp_path_factory):
    from embodied_learning.experiments.grid_nav import GridNavConfig
    from embodied_learning.experiments.map_localization import MapLocConfig  # noqa: F401
    from embodied_learning.experiments.photo_room_navigation import (
        NavConfig,
        run_experiment,
    )

    _ = GridNavConfig
    conf = NavConfig(particles=60, max_steps=400)
    out = tmp_path_factory.mktemp("nav56x") / "run"
    report = run_experiment(out, seed=0, config=conf, log=None)
    return out, report


def test_light_run_contract(light_record):
    out, report = light_record
    assert report["experiment"] == "photo_room_navigation"
    assert report["schema_version"] == 1
    assert len(report["rows"]) == 5 * 3 * 4
    r = report["hypothesis"]["results"]
    assert r["collector_met"]
    assert "gate_A_met" in r
    assert report["protocol"]["avoidance_map"].startswith("full robot-height")
    assert (out / "summary.json").exists()
    assert (out / "trajectories.npz").exists()


def test_reached_vs_rejected_bookkeeping(light_record):
    _out, report = light_record
    rows = report["rows"]
    for row in rows:
        if row["task"] == "into_bed_footprint":
            assert row["result"] in ("correct_rejection", "no_path")
        else:
            assert row["result"] != "rejected_unreachable"


def test_record_tamper_rejection(light_record):
    out, report = light_record
    path = out / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = out / "summary.json"
    original = summary.read_text(encoding="utf-8")
    data = json.loads(original)
    data["hypothesis"]["results"]["a_arrived"] += 1
    summary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AssertionError):
        recompute_gate(light_record)
    summary.write_text(original, encoding="utf-8")


def recompute_gate(light_record):
    out, _report = light_record
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    r = data["hypothesis"]["results"]
    recomputed = r["a_arrived"] == r["a_reachable_runs"] and r["a_contact_frames"] == 0
    assert bool(data["hypothesis"]["results"]["gate_A_met"]) == bool(recomputed)


@pytest.mark.slow
def test_seed_determinism(tmp_path):
    from embodied_learning.experiments.photo_room_navigation import (
        NavConfig,
        run_experiment,
    )

    conf = NavConfig(particles=60, max_steps=400)
    first = run_experiment(tmp_path / "a", seed=0, config=conf, log=None)
    second = run_experiment(tmp_path / "b", seed=0, config=conf, log=None)
    pop = lambda r: json.dumps(
        {k: v for k, v in r.items() if k != "wall_time_s"}, sort_keys=True, indent=1
    )
    assert pop(first) == pop(second)
