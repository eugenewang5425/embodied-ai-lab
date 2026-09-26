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
    name, _start, _goal = unreachable[0]
    assert name == "into_bed_footprint"
    # the unreachable goal must be free at the lidar slice (a naive low map
    # would call it passable) - verified by construction in classify_tasks
    assert all(len(g) == 2 for _n, _s, g in reachable)


def test_control_step_safety_stop():
    mm = mod()
    path = np.array([[0.0, 0.0], [2.0, 0.0]])
    stop_dist = 0.20  # body radius + margin + one control period
    scan_all_clear = [3.0] * 64
    _inc, _idx, v = mm.control_step(np.array([0.0, 0.0, 0.0]), path, 0, scan_all_clear, stop_dist)
    assert v > 0  # clears the start waypoint, drives toward the second
    scan_blocked = [3.0] * 40 + [0.05] * 24  # forward sector blocked
    _inc, _idx, v2 = mm.control_step(np.array([0.0, 0.0, 0.0]), path, 0, scan_blocked, stop_dist)
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
    _avoid, loc_map = build_maps(world, layout)
    reach, _unreach = classify_tasks(world, layout)
    rng = np.random.default_rng(0)
    _row, chains = run_closed_loop(world, reach[0], "A", loc_map, rng, rng, cfg, min(cfg.max_steps, 400))
    n = min(len(chains["truth"]), len(chains["estimate"]))
    errors = np.linalg.norm(chains["truth"][:n, :2] - chains["estimate"][:n, :2], axis=1)
    assert float(errors.max()) < 0.01  # group A estimate = truth (zero drift, within float noise)


# ------------------------------------------------- wiring regressions
def test_pf_propagates_with_measured_not_true_motion(monkeypatch):
    """The PF must be driven by the MEASURED (biased) wheel increment.

    Regression: v1 passed the true executed increment, so the filter never
    saw the wheel bias its odometry-group rival suffered from."""
    import embodied_learning.experiments.photo_room_navigation as mm
    from embodied_learning.experiments.photo_room_navigation import (
        NavConfig,
        build_maps,
        classify_tasks,
        run_closed_loop,
    )
    from embodied_learning.photo_room import RoomWorld, default_layout

    bias = 0.06
    cfg = NavConfig(particles=50, wheel_bias=bias)
    layout = default_layout()
    world = RoomWorld(layout)
    _avoid, loc_map = build_maps(world, layout)
    reach, _unreach = classify_tasks(world, layout)

    seen = []
    real_propagate = mm.propagate_motion

    def spy(pf, ds, dth, rng, sig_t, sig_r):
        seen.append((ds, dth))
        return real_propagate(pf, ds, dth, rng, sig_t, sig_r)

    monkeypatch.setattr(mm, "propagate_motion", spy)

    rng_s = np.random.default_rng(3)
    rng_pf = np.random.default_rng(4)
    _row, chains = run_closed_loop(world, reach[0], "C", loc_map, rng_s, rng_pf, cfg, 30)

    truth = chains["truth"]
    ds_true = np.hypot(np.diff(truth[:, 0]), np.diff(truth[:, 1]))
    n = min(len(seen), len(ds_true))
    assert n > 10  # enough propagation calls to be meaningful
    seen_ds = np.array([s[0] for s in seen[:n]])
    # the value handed to the filter equals the true increment x (1+bias):
    # if the true increment had been passed, the ratio would be 1.0
    ratio = seen_ds / ds_true[:n]
    assert np.allclose(ratio, 1.0 + bias, atol=1e-9), f"ratio {ratio.min()}..{ratio.max()}"


def test_estimate_chain_length_matches_truth():
    """estimate[k] must align with truth[k]: appended once per timestamp.

    Regression: v1 pre-seeded the chain with the start pose AND appended at
    k=0, producing one extra frame (off-by-one against scans/truth)."""
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
    _avoid, loc_map = build_maps(world, layout)
    reach, _unreach = classify_tasks(world, layout)
    _row, chains = run_closed_loop(
        world,
        reach[0],
        "C",
        loc_map,
        np.random.default_rng(5),
        np.random.default_rng(6),
        cfg,
        25,
    )
    assert len(chains["estimate"]) == len(chains["truth"])
    assert len(chains["commands"]) == len(chains["truth"]) - 1


def test_control_step_receives_computed_stop_distance(monkeypatch):
    """The lidar safety-stop threshold handed to control_step must be the
    geometry-derived safe_stop_distance, not a stale module constant.

    Regression: v1 computed safe_stop_distance but control_step still used
    the old default, so a faster robot could command motion into the
    braking-distance danger zone."""
    import embodied_learning.experiments.photo_room_navigation as mm
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
    _avoid, loc_map = build_maps(world, layout)
    reach, _unreach = classify_tasks(world, layout)
    expected = mm.safe_stop_distance(layout)

    seen = []
    real_control = mm.control_step

    def spy(pose, path, wp, scan, stop_distance):
        seen.append(stop_distance)
        return real_control(pose, path, wp, scan, stop_distance)

    monkeypatch.setattr(mm, "control_step", spy)
    run_closed_loop(
        world,
        reach[0],
        "B",
        loc_map,
        np.random.default_rng(7),
        np.random.default_rng(8),
        cfg,
        20,
    )
    assert seen and all(s == expected for s in seen)


def test_unreachable_enters_same_entry():
    """The unreachable task must flow through run_closed_loop itself - the
    planner returns no_path - and reachability is scored only afterwards.

    Regression: v1 pre-classified the task by label and bypassed the
    navigation loop, so the rejection was written, not navigated."""
    import inspect

    from embodied_learning.experiments.photo_room_navigation import (
        NavConfig,
        build_maps,
        classify_tasks,
        run_closed_loop,
    )
    from embodied_learning.photo_room import RoomWorld, default_layout

    # structural: no reachability label parameter on the entry function
    params = inspect.signature(run_closed_loop).parameters
    assert "is_reachable" not in params and "reachable" not in params

    cfg = NavConfig(particles=50)
    layout = default_layout()
    world = RoomWorld(layout)
    _avoid, loc_map = build_maps(world, layout)
    _reach, unreach = classify_tasks(world, layout)
    row, _chains = run_closed_loop(
        world,
        unreach[0],
        "A",
        loc_map,
        np.random.default_rng(9),
        np.random.default_rng(10),
        cfg,
        30,
    )
    assert row["result"] == "no_path"
    assert row["no_path_frame"] is not None
    assert row["contact_frames"] == 0


def test_resample_syncs_weights():
    """After pf.resample() the filter weights must be uniform over the NEW
    particles, and the external recursion must copy that state.

    Regression: v1 kept updating the pre-resample weight vector, so the next
    recursive measurement update multiplied stale weights into the posterior."""
    from embodied_learning.experiments.map_localization import ParticleFilter

    rng = np.random.default_rng(21)
    grid = np.ones((40, 40), dtype=np.uint8)
    grid[10:30, 10:30] = 0  # an obstacle band for a non-degenerate field
    pf = ParticleFilter(grid, 0.1, 60, rng, init_pose=np.array([2.0, 2.0, 0.0]))
    scan = np.full(64, 1.5)  # the filter samples 32 of the 64-beam scan
    pf.measure(scan, None)
    pf.w[0] = 0.9  # deliberately skew weights
    pf.resample()
    assert np.allclose(pf.w, np.full(len(pf.p), 1.0 / len(pf.p)))
    # the caller-side recursion must mirror the filter state after resample
    from embodied_learning.experiments.photo_room_navigation import measure_recursive

    weights = np.full(len(pf.p), 0.9 / len(pf.p))  # stale, as v1 held
    weights = measure_recursive(pf, scan, weights)
    assert abs(weights.sum() - 1.0) < 1e-9
    assert np.all(weights >= 0.0)


def test_rng_streams_separated():
    """Sensor noise and PF sampling must draw from independent streams:
    same sensor seed + different PF seed => identical scans, different
    estimates. Regression for the v1 single shared rng."""
    from embodied_learning.experiments.photo_room_navigation import (
        NavConfig,
        build_maps,
        classify_tasks,
        run_closed_loop,
    )
    from embodied_learning.photo_room import RoomWorld, default_layout

    cfg = NavConfig(particles=60)
    layout = default_layout()
    world = RoomWorld(layout)
    _avoid, loc_map = build_maps(world, layout)
    reach, _unreach = classify_tasks(world, layout)
    task = reach[0]

    def run(pf_seed):
        return run_closed_loop(
            world,
            task,
            "C",
            loc_map,
            np.random.default_rng(11),
            np.random.default_rng(pf_seed),
            cfg,
            25,
        )

    _r1, c1 = run(100)
    _r2, c2 = run(200)
    _r3, c3 = run(100)
    assert np.array_equal(c1["scan_trace"], c2["scan_trace"])  # same sensor stream
    assert not np.array_equal(c1["estimate"], c2["estimate"])  # different PF stream
    assert np.array_equal(c1["estimate"], c3["estimate"])  # PF stream reproducible


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
        {k: v for k, v in r.items() if k not in ("wall_time_s", "resources")},
        sort_keys=True,
        indent=1,
    )
    assert pop(first) == pop(second)
