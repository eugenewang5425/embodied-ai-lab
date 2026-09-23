"""Lesson 55: simulated marker retrieval - contracts and record integrity."""

import hashlib
import json
import math

import numpy as np
import pytest


def mod():
    import embodied_learning.experiments.rgbd_loops as mm

    return mm


def test_config_guards():
    from embodied_learning.experiments.rgbd_loops import RgbdConfig

    with pytest.raises(ValueError):
        RgbdConfig(top_k=1)
    with pytest.raises(ValueError):
        RgbdConfig(probe_stride=0)


def test_cast_rays_ids_tracks_nearest_primitive():
    mm = mod()
    from embodied_learning.experiments.grid_nav import Obstacle

    near = Obstacle("circle", 3.0, 1.0, 0.0, 0.0, 0.5)  # id 0
    far = Obstacle("circle", 3.0, 6.0, 0.0, 0.0, 0.5)  # id 1
    prims = (near, far)
    ids = np.array([7, 3])
    ranges, hit, idx = mm.cast_rays_ids(
        3.0, 3.0, np.array([-math.pi / 2, math.pi / 2]), prims, ids, 20.0
    )
    assert hit[0] and hit[1]
    assert idx[0] == 0  # the near circle's marker
    assert idx[1] == 1
    assert ranges[0] == pytest.approx(1.5)
    assert ranges[1] == pytest.approx(2.5)


def test_rgbd_frame_histogram_is_normalised():
    mm = mod()
    from embodied_learning.experiments.grid_nav import Obstacle

    prims = (
        Obstacle("circle", 3.0, 1.0, 0.0, 0.0, 0.5),
        Obstacle("circle", 5.0, 3.0, 0.0, 0.0, 0.5),
    )
    ids = np.array([2, 5])
    rng = np.random.default_rng(0)
    pose = np.array([3.0, 3.0, 0.0])
    h = mm.rgbd_frame(pose, prims, ids, rng)
    assert abs(float(np.linalg.norm(h)) - 1.0) < 1e-6 or h.sum() == 0
    # determinism: same rng seed -> same histogram
    h2 = mm.rgbd_frame(pose, prims, ids, np.random.default_rng(0))
    np.testing.assert_array_equal(h, h2)


def test_cosine_similarity_basics():
    mm = mod()
    assert mm.cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert mm.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert mm.cosine(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 0.0


def test_probe_exclusion_uses_estimated_arc():
    mm = mod()
    est = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
    )
    probes, arc = mm.probes_from_odometry(est, n_frames=4, stride=1)
    assert probes == [3]
    np.testing.assert_array_equal(arc, [0.0, 5.0, 10.0, 15.0, 20.0])


def light_config():
    """Keep record-contract tests nontrivial without replaying the lesson run.

    The formal lesson uses denser rays and more probes.  This configuration
    still exercises odometry-based candidate exclusion, appearance retrieval,
    geometric verification, archive writing, and the reported gates.
    """
    from embodied_learning.experiments.grid_nav import GridNavConfig
    from embodied_learning.experiments.rgbd_loops import RgbdConfig

    return RgbdConfig(
        base=GridNavConfig(rays=24, max_range_m=4.0, obstacle_scenes=1, inits=1),
        probe_stride=16,
    )


# ------------------------------------------------------------- records
@pytest.fixture(scope="module")
def light_stream(tmp_path_factory):
    from embodied_learning.experiments.rgbd_loops import run_experiment

    out = tmp_path_factory.mktemp("rgbd55") / "run"
    report = run_experiment(out, seed=0, config=light_config(), log=None)
    return out, report


def test_light_run_contract(light_stream):
    out, report = light_stream
    assert report["experiment"] == "rgbd_loops_lesson55"
    assert report["schema_version"] == 2
    r = report["hypothesis"]["results"]
    for key in (
        "precision_topk",
        "precision_met",
        "correctness_met",
        "pipeline_success",
        "efficiency_met",
    ):
        assert key in r
    assert report["protocol"]["retrieval"]["oracle_free"] is True
    assert report["protocol"]["retrieval"]["probe_exclusion"] == "estimated odometry arc >= 15 m"
    assert report["protocol"]["appearance"]["confusion"] == 0.08
    assert (out / "trajectories.npz").exists()
    assert (out / "summary.json").exists()


def test_demo_loads_current_record(light_stream):
    from embodied_learning.rgbd_loops_demo import load_replays

    out, report = light_stream
    replay = load_replays(out)
    assert replay["report"]["schema_version"] == report["schema_version"]
    # Appearance retrieval chooses top-k before geometric point-count guards.
    # A retrieved frame may therefore be absent from the verified-row group.
    assert len(replay["topk_frames"]) == report["protocol"]["retrieval"]["top_k"]
    assert report["aggregates"]["RGBD-TOPK"]["n_candidates"] <= len(replay["topk_frames"])


def test_retrieval_beats_uniform_baseline(light_stream):
    """The treatment claim in its minimal form: the top-k accepted rows are
    at least as correct as the uniform-probe accepted rows."""
    _out, report = light_stream
    agg = report["aggregates"]
    assert agg["RGBD-TOPK"]["direction_correctness"] >= agg["GEO-ALL"]["direction_correctness"]
    assert agg["RGBD-TOPK"]["n_candidates"] < agg["GEO-ALL"]["n_candidates"]


def test_record_tamper_rejection(light_stream):
    out, report = light_stream
    path = out / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = out / "summary.json"
    original = summary.read_text(encoding="utf-8")
    data = json.loads(original)
    data["hypothesis"]["results"]["precision_topk"] = 0.99
    summary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AssertionError):
        recompute_precision(light_stream)
    summary.write_text(original, encoding="utf-8")


def recompute_precision(light_stream):
    out, _report = light_stream
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    with np.load(out / "trajectories.npz", allow_pickle=False) as npz:
        topk = set(npz["topk_frames"].tolist())
        frames = npz["probe_frames_all"]
        near = npz["probe_near_start"]
    near_map = dict(zip(frames.tolist(), near.tolist()))
    recomputed = sum(1 for k in topk if near_map.get(k, 0)) / max(1, len(topk))
    assert abs(recomputed - data["hypothesis"]["results"]["precision_topk"]) < 1e-9


def test_seed_determinism(tmp_path):
    from embodied_learning.experiments.rgbd_loops import run_experiment

    conf = light_config()
    first = run_experiment(tmp_path / "a", seed=0, config=conf, log=None)
    second = run_experiment(tmp_path / "b", seed=0, config=conf, log=None)
    pop = lambda r: json.dumps(
        {k: v for k, v in r.items() if k != "wall_time_s"}, sort_keys=True, indent=1
    )
    assert pop(first) == pop(second)
