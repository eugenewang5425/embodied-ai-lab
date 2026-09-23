"""Lesson 53: production solver comparison - contracts and record integrity."""

import hashlib
import json

import numpy as np
import pytest


def mod():
    import embodied_learning.experiments.production_solver as mm

    return mm


def test_config_guards():
    from embodied_learning.experiments.production_solver import ProductionConfig

    with pytest.raises(ValueError):
        ProductionConfig(poison_rot_deg=10.0)
    with pytest.raises(ValueError):
        ProductionConfig(poison_shift_m=20.0)


def test_residual_vector_shape_and_first_pose_free():
    mm = mod()
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.1, 0.05]])
    edges = [
        (0, 1, np.array([1.0, 0.0, 0.0]), 0.05, 0.05),
        (1, 2, np.array([1.0, 0.1, 0.05]), 0.05, 0.05),
    ]
    loop = (0, 2, np.array([2.02, 0.1, 0.05]), 0.06, 0.06)
    x_flat = nodes[1:].reshape(-1)
    r = mm.residual_vector(x_flat, 3, edges, loop)
    assert r.shape == (9,)
    # at the chain itself the residuals are the loop mismatch only
    assert np.linalg.norm(r[:6]) < 1e-9
    assert np.linalg.norm(r[6:]) > 0.0


def test_solve_scipy_recovers_shifted_chain():
    mm = mod()
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edges = [
        (0, 1, np.array([1.0, 0.0, 0.0]), 0.05, 0.05),
        (1, 2, np.array([1.0, 0.0, 0.0]), 0.05, 0.05),
    ]
    loop = (0, 2, np.array([2.04, 0.06, 0.0]), 0.06, 0.06)
    x = mm.solve_scipy(nodes, edges, loop, None, 1.0)
    assert abs(x[2][0] - 2.02) < 0.1  # the loop edge pulls the chain


def test_self_huber_bridge_runs_and_returns_chain():
    mm = mod()
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edges = [
        (0, 1, np.array([1.0, 0.0, 0.0]), 0.05, 0.05),
        (1, 2, np.array([1.0, 0.0, 0.0]), 0.05, 0.05),
    ]
    loop = (0, 2, np.array([6.0, 0.0, 0.0]), 0.06, 0.06)  # a far anchor
    x = mm.solve_self_huber(nodes, edges, loop)
    assert x.shape == (3, 3)
    # the kernel resists the far anchor: the chain stays near odometry
    assert x[2][0] < 3.0


# ------------------------------------------------------------- records
@pytest.fixture(scope="module")
def light_stream(tmp_path_factory):
    from embodied_learning.experiments.grid_nav import GridNavConfig
    from embodied_learning.experiments.loop_closures import LoopConfig
    from embodied_learning.experiments.pose_graph import PoseGraphConfig
    from embodied_learning.experiments.production_solver import ProductionConfig, run_experiment

    conf = ProductionConfig(
        base=PoseGraphConfig(
            base=LoopConfig(
                base=GridNavConfig(obstacle_scenes=1, inits=1, rays=64, max_range_m=4.0)
            )
        )
    )
    out = tmp_path_factory.mktemp("prod53") / "run"
    report = run_experiment(out, seed=0, config=conf, log=None)
    return out, report


def test_light_run_contract(light_stream):
    out, report = light_stream
    assert report["experiment"] == "production_solver_lesson53"
    assert report["schema_version"] == 1
    r = report["hypothesis"]["results"]
    for key in ("collector_met", "kernel_fails_too_met", "dilemma_any_met", "sweep_empty_met"):
        assert key in r
    assert len(report["sweep"]) >= 4
    assert (out / "trajectories.npz").exists()
    assert (out / "summary.json").exists()


def test_record_tamper_rejection(light_stream):
    out, report = light_stream
    path = out / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = out / "summary.json"
    original = summary.read_text(encoding="utf-8")
    data = json.loads(original)
    data["hypothesis"]["results"]["kernel_fails_too_met"] = not data["hypothesis"]["results"][
        "kernel_fails_too_met"
    ]
    summary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AssertionError):
        recompute_kernel(light_stream)
    summary.write_text(original, encoding="utf-8")


def recompute_kernel(light_stream):
    out, _report = light_stream
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    n = data["aggregates"]["N"]["mean_position_error_m"]
    hub_b = data["aggregates"]["SCIPY-HUB-B"]["mean_position_error_m"]
    recomputed = bool(hub_b is not None and n is not None and hub_b > n - 0.5)
    assert bool(data["hypothesis"]["results"]["kernel_fails_too_met"]) == bool(recomputed)


@pytest.mark.slow
def test_seed_determinism(tmp_path):
    """Byte-identical determinism of the NON-scipy pipeline stages.

    The scipy runs are too dense to duplicate in a test (a single record
    costs ~30 min of TRF); determinism of the collection, injection and
    the self-Huber bridge is what OUR code owns - scipy is a third-party
    constant for a fixed version.
    """
    from types import SimpleNamespace

    from embodied_learning.experiments.grid_nav import GridNavConfig
    from embodied_learning.experiments.loop_closures import LoopConfig
    from embodied_learning.experiments.pose_graph import PoseGraphConfig
    from embodied_learning.experiments.production_solver import (
        ProductionConfig,
        solve_self_huber,
    )
    from embodied_learning.experiments.relative_loop_edges import (
        build_rel_edges,
        poisoned_relative,
        true_relative,
    )
    from embodied_learning.experiments.robust_graph import (
        build_groups,
        build_scenarios,
        initial_poses,
    )

    conf = ProductionConfig(
        base=PoseGraphConfig(
            base=LoopConfig(
                base=GridNavConfig(obstacle_scenes=1, inits=1, rays=64, max_range_m=4.0)
            )
        )
    )

    def pipeline():
        scenarios = build_scenarios(conf.base.base.config, 0)
        scene1 = next(s for s in scenarios if s["scene"] == 1)
        start = initial_poses(conf.base.base.config, 0, 1)[0]
        stream = collect_patrol(scene1["obstacles"], start, conf.base.base)
        shim = SimpleNamespace(base=conf.base, poison_rot_deg=conf.poison_rot_deg)
        built = build_groups(stream, shim)
        edges = build_rel_edges(built["nodes"], conf)
        z = true_relative(built["nodes"], built["node_steps"], stream["truth"])
        z_bad = (
            0,
            len(built["nodes"]) - 1,
            poisoned_relative(z, conf.poison_rot_deg, conf.poison_shift_m),
            conf.base.sigma_loop_t,
            conf.base.sigma_loop_r,
        )
        x = solve_self_huber(built["nodes"], edges, z_bad)
        return json.dumps(
            {
                "nodes": built["nodes"].tolist(),
                "z": z.tolist(),
                "x": x.tolist(),
            },
            sort_keys=True,
        )

    from embodied_learning.experiments.loop_closures import collect_patrol

    assert pipeline() == pipeline()
