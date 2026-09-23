"""Lesson 52: relative loop edges - solver contracts and record integrity."""

import hashlib
import json
import math

import numpy as np
import pytest


def cfg():
    from embodied_learning.experiments.relative_loop_edges import RelativeConfig

    return RelativeConfig()


def rl():
    import embodied_learning.experiments.relative_loop_edges as mm

    return mm()


def mod():
    import embodied_learning.experiments.relative_loop_edges as mm

    return mm


def test_config_guards():
    mm = mod()
    with pytest.raises(ValueError):
        mm.RelativeConfig(gn_iterations=0)
    with pytest.raises(ValueError):
        mm.RelativeConfig(sc_lambda=0.01)
    with pytest.raises(ValueError):
        mm.RelativeConfig(mix_delta=20.0)
    with pytest.raises(ValueError):
        mm.RelativeConfig(mix_ratio=1.0)


def test_poisoned_relative_is_rotated_and_shifted():
    mm = mod()
    z = np.array([2.0, 1.0, 0.3])
    zb = mm.poisoned_relative(z, 90.0, 1.0)
    # rotation: (2, 1) -> (-1, 2); shift +1 along x; theta += 90 deg
    np.testing.assert_allclose(zb[:2], [0.0, 2.0], atol=1e-9)
    assert abs(mm.wrap(zb[2] - 0.3 - math.pi / 2)) < 1e-9


def test_true_relative_matches_matcher_convention():
    mm = mod()
    nodes = np.zeros((3, 3))
    node_steps = np.array([1, 3])
    truth = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.2, 1.4, 0.2]])
    z = mm.true_relative(nodes, node_steps, truth)
    # rotate-then-translate from truth[2] to truth[0]
    dth = mm.wrap(truth[0][2] - truth[2][2])
    c, s = math.cos(dth), math.sin(dth)
    expected = np.array(
        [
            truth[0][0] - c * truth[2][0] + s * truth[2][1],
            truth[0][1] - s * truth[2][0] - c * truth[2][1],
            dth,
        ]
    )
    np.testing.assert_allclose(z, expected, atol=1e-12)


def _tiny():
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edges = [
        (0, 1, np.array([1.0, 0.0, 0.0]), 0.05, 0.05),
        (1, 2, np.array([1.0, 0.0, 0.0]), 0.05, 0.05),
    ]
    return nodes, edges


def test_sc_relative_good_edge_closes_and_keeps_high_s():
    mm = mod()
    nodes, edges = _tiny()
    loop = (0, 2, np.array([2.02, 0.05, 0.02]), 0.06, 0.06)
    x, s, s_hist = mm.solve_rel_sc(nodes, edges, loop, cfg())
    assert s > 0.8
    assert abs(x[2][0] - 2.0) < 0.2
    assert s_hist[0] == 1.0


def test_sc_relative_poisoned_edge_drops_switch():
    mm = mod()
    nodes, edges = _tiny()
    z_bad = mm.poisoned_relative(np.array([2.0, 0.0, 0.0]), 90.0, 1.0)
    loop = (0, 2, z_bad, 0.06, 0.06)
    x, s, _hist = mm.solve_rel_sc(nodes, edges, loop, cfg())
    assert s < 0.3
    np.testing.assert_allclose(x[:, 0], nodes[:, 0], atol=0.05)  # chain intact


def test_mm_relative_poisoned_uses_invalid_component():
    mm = mod()
    nodes, edges = _tiny()
    z_bad = mm.poisoned_relative(np.array([2.0, 0.0, 0.0]), 90.0, 1.0)
    loop = (0, 2, z_bad, 0.06, 0.06)
    x, frac = mm.solve_rel_mm(nodes, edges, loop, cfg())
    assert frac < 0.5  # the valid component mostly loses
    np.testing.assert_allclose(x[:, 0], nodes[:, 0], atol=0.05)


def test_mm_relative_good_uses_valid_component():
    mm = mod()
    nodes, edges = _tiny()
    loop = (0, 2, np.array([2.02, 0.05, 0.02]), 0.06, 0.06)
    _x, frac = mm.solve_rel_mm(nodes, edges, loop, cfg())
    assert frac > 0.5


# ------------------------------------------------------------- records
def light_config():
    from embodied_learning.experiments.grid_nav import GridNavConfig
    from embodied_learning.experiments.loop_closures import LoopConfig
    from embodied_learning.experiments.pose_graph import PoseGraphConfig
    from embodied_learning.experiments.relative_loop_edges import RelativeConfig

    return RelativeConfig(
        base=PoseGraphConfig(
            base=LoopConfig(
                base=GridNavConfig(obstacle_scenes=1, inits=1, rays=24, max_range_m=4.0)
            ),
            gn_iterations=3,
        ),
        gn_iterations=3,
    )


@pytest.fixture(scope="module")
def light_stream(tmp_path_factory):
    from embodied_learning.experiments.relative_loop_edges import run_experiment

    out = tmp_path_factory.mktemp("rel52") / "run"
    report = run_experiment(out, seed=0, config=light_config(), log=None)
    return out, report


def test_light_run_contract(light_stream):
    out, report = light_stream
    assert report["experiment"] == "relative_loop_edges_lesson52"
    assert report["schema_version"] == 1
    r = report["hypothesis"]["results"]
    for key in ("collector_met", "fragility_met", "treatment_met", "prevention_met", "mm_met"):
        assert key in r
    assert r["collector_met"] is True
    assert {"N", "REL-LS-G", "REL-LS-B", "SC-G", "SC-B", "MM-G", "MM-B"} <= set(
        report["aggregates"]
    )
    assert report["protocol"]["loop_edge"]["kind"].startswith("relative")
    assert (out / "trajectories.npz").exists()
    assert (out / "summary.json").exists()


def test_record_tamper_rejection(light_stream):
    out, report = light_stream
    path = out / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = out / "summary.json"
    original = summary.read_text(encoding="utf-8")
    data = json.loads(original)
    # flip a verdict flag: the aggregate consistency check must fire
    data["hypothesis"]["results"]["prevention_met"] = not data["hypothesis"]["results"][
        "prevention_met"
    ]
    summary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AssertionError):
        recompute_prevention(light_stream)
    summary.write_text(original, encoding="utf-8")


def recompute_prevention(light_stream):
    out, _report = light_stream
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    n = data["aggregates"]["N"]["mean_position_error_m"]
    scb = data["aggregates"]["SC-B"]["mean_position_error_m"]
    scb_s = data["aggregates"]["SC-B"]["switch_final"]
    recomputed = bool(
        scb is not None and n is not None and scb <= n + 0.5 and scb_s is not None and scb_s <= 0.3
    )
    assert bool(data["hypothesis"]["results"]["prevention_met"]) == bool(recomputed)


@pytest.mark.slow
def test_seed_determinism(light_stream, tmp_path):
    from embodied_learning.experiments.relative_loop_edges import run_experiment

    _out, first = light_stream
    second = run_experiment(tmp_path / "b", seed=0, config=light_config(), log=None)
    pop = lambda r: json.dumps(
        {k: v for k, v in r.items() if k != "wall_time_s"}, sort_keys=True, indent=1
    )
    assert pop(first) == pop(second)
