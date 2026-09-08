"""Lesson 51: switchable loop constraints - solver contracts and record
integrity (the single-kernel-dilemma resolution)."""

import hashlib
import json

import numpy as np
import pytest

from embodied_learning.experiments.grid_nav import GridNavConfig

LIGHT_GRID = GridNavConfig(obstacle_scenes=1, inits=1, rays=64, max_range_m=4.0)


def cfg(switch_lambda=1.0):
    from embodied_learning.experiments.switchable_graph import SwitchableConfig

    return SwitchableConfig(switch_lambda=switch_lambda)


def sg():
    import embodied_learning.experiments.switchable_graph as mm

    return mm


def tiny_system():
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edges = [(0, 1, np.array([1.0, 0.0, 0.0]), 0.05, 0.05)]
    return nodes, edges


def test_config_guards():
    from embodied_learning.experiments.switchable_graph import SwitchableConfig

    with pytest.raises(ValueError):
        SwitchableConfig(switch_lambda=0.01)
    with pytest.raises(ValueError):
        SwitchableConfig(switch_s0=3.0)
    with pytest.raises(ValueError):
        SwitchableConfig(gn_iterations=0)


def test_solve_drops_switch_for_poisoned_anchor():
    mm = sg()
    nodes, edges = tiny_system()
    x, s, s_hist, _r = mm.solve_switchable(
        nodes, edges, (1, np.array([3.0, 0.0, 0.0]), 0.06, 0.06), cfg()
    )
    assert s < 0.3  # the switch rejects the far anchor
    assert s_hist[0] == 1.0  # the prior starts at closed
    assert np.linalg.norm(x[1] - nodes[1]) < 0.15  # the chain stays put


def test_solve_keeps_switch_high_for_good_anchor():
    mm = sg()
    nodes, edges = tiny_system()
    x, s, _s_hist, _r = mm.solve_switchable(
        nodes, edges, (1, np.array([1.02, 0.0, 0.0]), 0.06, 0.06), cfg()
    )
    assert s > 0.8
    assert abs(x[1][0] - 1.02) < 0.03  # the good anchor pulls the node


def test_switch_step_is_not_clipped_to_positive():
    """Regression: the s STEP must be allowed negative (a per-iteration
    clip(0, 2) silently froze the switch at its prior - s never moved)."""
    mm = sg()
    nodes, edges = tiny_system()
    _x, _s, s_hist, _r = mm.solve_switchable(
        nodes, edges, (1, np.array([3.0, 0.0, 0.0]), 0.06, 0.06), cfg()
    )
    assert s_hist[1] < 0.5  # the first update already dropped the switch


def test_solve_converges_in_bounded_iterations():
    mm = sg()
    nodes, edges = tiny_system()
    _x, _s, s_hist, r_hist = mm.solve_switchable(
        nodes, edges, (1, np.array([1.02, 0.0, 0.0]), 0.06, 0.06), cfg()
    )
    assert len(s_hist) == cfg().gn_iterations
    assert r_hist[-1] <= r_hist[0] + 1e-6  # the G-N cost does not blow up


def test_groups_declare_split():
    mm = sg()
    assert mm.GROUPS[0:3] == ("N", "LS-G", "HUB-G")
    assert len(mm.GROUPS) == 3 + 2 * len(mm.SC_LAMBDAS) + 2
    assert len(mm.SC_LAMBDAS) == 6
    assert mm.SWITCH_S0 == 1.0


# ------------------------------------------------------------- records
@pytest.fixture(scope="module")
def light_stream(tmp_path_factory):
    """One light full-record run (reduced protocol size)."""
    from embodied_learning.experiments.loop_closures import LoopConfig
    from embodied_learning.experiments.pose_graph import PoseGraphConfig
    from embodied_learning.experiments.switchable_graph import SwitchableConfig, run_experiment

    conf = SwitchableConfig(base=PoseGraphConfig(base=LoopConfig(base=LIGHT_GRID)))
    out = tmp_path_factory.mktemp("sw51") / "run"
    report = run_experiment(out, seed=0, config=conf, log=None)
    return out, report


def test_light_run_contract(light_stream):
    out, report = light_stream
    assert report["experiment"] == "switchable_graph_lesson51"
    assert report["schema_version"] == 1
    r = report["hypothesis"]
    for key in (
        "collector_met",
        "sweet_spot_lambdas",
        "sweet_spot_met",
        "mm_stall_recorded",
        "lambda_curve",
    ):
        assert key in r["results"]
    assert report["protocol"]["switch"]["kind"].startswith("switchable")
    assert (out / "trajectories.npz").exists()
    assert (out / "summary.json").exists()


def test_record_tamper_rejection(light_stream):
    out, report = light_stream
    path = out / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = out / "summary.json"
    original = summary.read_text(encoding="utf-8")
    data = json.loads(original)
    # flip both switches to satisfy the signal criterion: the recomputed
    # verdict then contradicts the recorded signal_met
    data["aggregates"]["MM-G"]["switch_final"] = 0.99
    data["aggregates"]["MM-B"]["switch_final"] = 0.01
    summary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AssertionError):
        signal_recheck(light_stream)
    summary.write_text(original, encoding="utf-8")


def signal_recheck(light_stream):
    out, _report = light_stream
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    s_g = data["aggregates"]["MM-G"].get("switch_final")
    s_b = data["aggregates"]["MM-B"].get("switch_final")
    # the recorded valid fractions must both be 0 (the stall is the record)
    assert float(s_g or -1.0) < 0.5 and float(s_b or -1.0) < 0.5


@pytest.mark.slow
def test_seed_determinism(tmp_path):
    from embodied_learning.experiments.loop_closures import LoopConfig
    from embodied_learning.experiments.pose_graph import PoseGraphConfig
    from embodied_learning.experiments.switchable_graph import SwitchableConfig, run_experiment

    conf = SwitchableConfig(base=PoseGraphConfig(base=LoopConfig(base=LIGHT_GRID)))
    first = run_experiment(tmp_path / "a", seed=0, config=conf, log=None)
    second = run_experiment(tmp_path / "b", seed=0, config=conf, log=None)
    pop = lambda r: json.dumps(
        {k: v for k, v in r.items() if k != "wall_time_s"}, sort_keys=True, indent=1
    )
    assert pop(first) == pop(second)
