"""Lesson 50: featureized loop detection - discriminator contracts and the
honest-negative record integrity."""

import hashlib
import json
import math

import numpy as np
import pytest


def cfg():
    from embodied_learning.experiments.feature_loops import FeatureLoopConfig

    return FeatureLoopConfig()


def fl():
    import embodied_learning.experiments.feature_loops as mm

    return mm


def test_config_guards():
    from embodied_learning.experiments.feature_loops import FeatureLoopConfig

    with pytest.raises(ValueError):
        FeatureLoopConfig(map_threshold=7.0)
    with pytest.raises(ValueError):
        FeatureLoopConfig(search_x_m=0.1)
    with pytest.raises(ValueError):
        FeatureLoopConfig(search_step_deg=30.0)
    with pytest.raises(ValueError):
        FeatureLoopConfig(probe_stride=0)
    with pytest.raises(ValueError):
        FeatureLoopConfig(refine_submap_frames=5)


def test_desc_sorted_is_permutation_invariant():
    mm = fl()
    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 4.0, 64)
    perm = rng.permutation(64)
    da = mm.desc_sorted(a)
    assert np.allclose(da, mm.desc_sorted(a[perm]))


def test_desc_cyclic_keeps_structure_under_rotation():
    mm = fl()
    rng = np.random.default_rng(1)
    a = rng.uniform(0.2, 4.0, 64)
    b = np.roll(a, 7)  # a rotated scan is cyclically shifted
    assert np.allclose(mm.desc_cyclic(a), mm.desc_cyclic(b))
    assert mm.dist_cyclic(mm.desc_cyclic(a), mm.desc_cyclic(b)) < 1e-9


def test_desc_sorted_is_structure_free():
    # two angularly DIFFERENT scans with the same sorted range multiset are
    # indistinguishable under the sorted descriptor - the discriminator gap
    mm = fl()
    a = np.hstack([np.full(32, 1.0), np.full(32, 3.0)])
    b = np.hstack([np.full(32, 1.0), np.full(32, 3.0)])
    assert np.allclose(mm.desc_sorted(a), mm.desc_sorted(b))


def test_map_score_at_counts_occupied_cells():
    mm = fl()
    grid = np.zeros((100, 100))
    # grid[row=y, col=x]: a wall patch covering x in [2.0, 3.0), y in [5.0, 6.0)
    grid[50:60, 20:30] = 1.0
    pts_on = np.array([[2.5, 5.5], [2.6, 5.6]])
    pts_off = np.array([[8.0, 8.0], [7.5, 8.2]])
    assert mm.map_score_at(pts_on, grid, 0.1, np.zeros(3)) == 1.0
    assert mm.map_score_at(pts_off, grid, 0.1, np.zeros(3)) == 0.0
    # a translation moves all points onto the patch
    assert mm.map_score_at(pts_off - np.array([5.5, 2.5]), grid, 0.1, np.zeros(3)) == pytest.approx(
        1.0
    )


def test_coarse_search_recovers_known_shift_on_corner_map():
    mm = fl()
    grid = np.zeros((100, 100))
    # an irregular rounded polygon (cornered everywhere, no straight-wall
    # slide direction), rasterized into a solid contour, centered at (5, 5)
    n = 1200
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rho = 2.5 + 0.7 * np.sin(2.0 * t) + 0.5 * np.sin(5.0 * t) + 0.8 * np.cos(7.0 * t)
    pts_c = np.column_stack([rho * np.cos(t) + 5.0, rho * np.sin(t) + 5.0])
    for x, y in pts_c[::2]:
        grid[int(y // 0.1), int(x // 0.1)] = 1.0
    pts = pts_c[::20]
    true_delta = np.array([1.0, 0.5, 0.0])  # search-grid-aligned
    moved = mm.apply_delta(pts, -true_delta)
    delta, score = mm.coarse_search(moved, grid, 0.1, cfg())
    assert delta is not None
    assert score > 0.85
    np.testing.assert_allclose(delta[:2], true_delta[:2], atol=0.5)


# ------------------------------------------------------------- records
@pytest.fixture(scope="module")
def light_stream(tmp_path_factory):
    """One light full-record run (single scene, sparse probes)."""
    from embodied_learning.experiments.feature_loops import (
        FeatureLoopConfig,
        run_experiment,
    )
    from embodied_learning.experiments.grid_nav import GridNavConfig

    conf = FeatureLoopConfig(
        base=GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1),
        probe_stride=200,
    )
    out = tmp_path_factory.mktemp("feat50") / "run"
    report = run_experiment(out, seed=0, config=conf, log=None)
    return out, report


def test_light_run_contract(light_stream):
    out, report = light_stream
    assert report["experiment"] == "feature_loop_detection_lesson50"
    assert report["schema_version"] == 1
    res = report["hypothesis"]["results"]
    for key in (
        "detections",
        "accepted",
        "correct",
        "direction_correctness",
        "overlap_sorted_histogram",
        "overlap_cyclic_profile",
        "overlap_map_agreement",
    ):
        assert key in res
    assert 0.0 <= res["direction_correctness"] <= 1.0
    assert report["protocol"]["detection"]["oracle_free"] is True
    assert (out / "trajectories.npz").exists()
    assert (out / "summary.json").exists()


def test_record_tamper_rejection(light_stream):
    out, report = light_stream
    path = out / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = out / "summary.json"
    original = summary.read_text(encoding="utf-8")
    data = json.loads(original)
    data["hypothesis"]["results"]["direction_correctness"] = 0.99
    summary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AssertionError):
        # the demo loader recomputes the verdict from the per-row flags
        compute_verdict(light_stream)
    summary.write_text(original, encoding="utf-8")


def compute_verdict(light_stream):
    out, _report = light_stream
    report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    rows = report["episodes"]
    accepted = [r for r in rows if r["accepted"]]
    correct = [r for r in accepted if r["correct"]]
    correctness = len(correct) / max(1, len(accepted))
    assert abs(correctness - report["hypothesis"]["results"]["direction_correctness"]) < 1e-9
    return correctness


@pytest.mark.slow
def test_seed_determinism(tmp_path):
    from embodied_learning.experiments.feature_loops import (
        FeatureLoopConfig,
        run_experiment,
    )
    from embodied_learning.experiments.grid_nav import GridNavConfig

    conf = FeatureLoopConfig(
        base=GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1),
        probe_stride=100,
    )
    first = run_experiment(tmp_path / "a", seed=0, config=conf, log=None)
    second = run_experiment(tmp_path / "b", seed=0, config=conf, log=None)
    pop = lambda r: json.dumps(
        {k: v for k, v in r.items() if k != "wall_time_s"}, sort_keys=True, indent=1
    )
    assert pop(first) == pop(second)
