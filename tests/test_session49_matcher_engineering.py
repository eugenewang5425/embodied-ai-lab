"""Lesson 49: matcher engineering - unit contracts and record integrity."""

import hashlib
import json
import math

import numpy as np
import pytest


# ---------------------------------------------------------------- helpers
def l_wall_ref():
    t = np.linspace(0.0, 8.0, 60)
    return np.vstack(
        [
            np.column_stack([t, np.zeros_like(t)]),
            np.column_stack([np.zeros_like(t), t]),
        ]
    )


def noisy_synthetic(config, seed=0, sigma=0.05):
    ref = rounded_square()
    true_delta = np.array([1.2, -0.8, math.radians(12.0)])
    rng = np.random.default_rng(seed)
    cur = inv_apply_delta(ref.copy(), true_delta) + rng.normal(0.0, sigma, ref.shape)
    return ref, cur, true_delta


def matcher():
    from embodied_learning.experiments.matcher_engineering import MatcherConfig

    return MatcherConfig()


def experiment_module():
    import embodied_learning.experiments.matcher_engineering as mm

    return mm


# ------------------------------------------------------------ config guards
def test_config_guards_reject_out_of_range_values():
    from embodied_learning.experiments.matcher_engineering import MatcherConfig

    with pytest.raises(ValueError):
        MatcherConfig(laps=1)
    with pytest.raises(ValueError):
        MatcherConfig(range_noise_sigma_m=0.8)
    with pytest.raises(ValueError):
        MatcherConfig(top_k=1)
    with pytest.raises(ValueError):
        MatcherConfig(arc_points=10)
    with pytest.raises(ValueError):
        MatcherConfig(normal_neighbors=2)
    with pytest.raises(ValueError):
        MatcherConfig(candidate_near_start_m=0.1)


def test_acceptance_bars_are_in_agreeable_bounds():
    # the pre-registered bars: the gate must be tight but not degenerate
    from embodied_learning.experiments.matcher_engineering import ACCEPT_GATE_MEDIAN_NN_M

    assert 0.03 <= ACCEPT_GATE_MEDIAN_NN_M <= 0.30


# ------------------------------------------------------------- resampling
def test_resample_arc_uniform_spacing_and_no_duplicates():
    from embodied_learning.experiments.matcher_engineering import resample_arc

    t = np.linspace(0.0, 8.0, 200)
    contour = np.column_stack([t, np.sin(t)])
    out, n = resample_arc(contour, 100, 0.45)
    assert n == 100
    chords = np.linalg.norm(np.diff(out, axis=0), axis=1)
    # uniform arc-length sampling: chords stay in a narrow band
    assert chords.max() / max(chords.min(), 1e-9) < 3.0
    # no duplicate consecutive samples
    assert np.linalg.norm(np.diff(out, axis=0), axis=1).min() > 1e-6


def test_resample_arc_breaks_contour_at_large_gaps():
    from embodied_learning.experiments.matcher_engineering import resample_arc

    left = np.column_stack([np.linspace(0.0, 2.0, 50), np.zeros(50)])
    right = np.column_stack([np.linspace(4.0, 6.0, 50), np.zeros(50)])
    contour = np.vstack([left, right])  # a 2 m gap in between
    out, n = resample_arc(contour, 80, 0.45)
    # no sample may fall into the gap
    assert not ((out[:, 0] > 2.5) & (out[:, 0] < 3.5)).any()
    assert n >= 8


def test_resample_arc_returns_none_on_tiny_clouds():
    from embodied_learning.experiments.matcher_engineering import resample_arc

    out, n = resample_arc(np.zeros((2, 2)), 40, 0.45)
    assert out is None and n == 0


# ---------------------------------------------------------------- normals
def test_normals_perpendicular_to_wall_and_consistent_sign_irrelevant():
    from embodied_learning.experiments.matcher_engineering import estimate_normals

    t = np.linspace(0.0, 6.0, 80)
    wall = np.column_stack([t, np.zeros_like(t)])
    normals = estimate_normals(wall, 8)
    # dominant component along the wall normal (y)
    assert float(np.abs(normals[:, 1]).mean()) > 0.95
    assert float(np.abs(normals[:, 0]).mean()) < 0.2
    # the solve is invariant to a sign flip of every normal
    ref, cur, _true_d = noisy_synthetic(matcher(), seed=1)
    from embodied_learning.experiments.matcher_engineering import align_pt2plane

    n1 = estimate_normals(ref, 10)
    d1, _p1, _m1 = align_pt2plane(cur, ref, np.zeros(3), n1, matcher())
    d2, _p2, _m2 = align_pt2plane(cur, ref, np.zeros(3), -n1, matcher())
    np.testing.assert_allclose(d1[:2], d2[:2], atol=1e-6)
    np.testing.assert_allclose(d1[2], d2[2], atol=1e-6)


# ------------------------------------------------------------------- ICPs
def test_pt2plane_polish_keeps_anchored_solution_on_noisy_synthetic():
    """After the p2p gate-ladder anchor, the plane polish does not slide:
    the median-NN stays at the noise/sampling floor and the direction is
    recovered (the record measures the B-vs-K aggregate difference)."""
    from embodied_learning.experiments.matcher_engineering import (
        align_pt2plane,
        align_pt2pt,
        coarse_argmax,
        estimate_normals,
    )

    cfg = matcher()
    ref, cur, true_delta = noisy_synthetic(cfg, seed=3)
    seed, _ = coarse_argmax(cur, ref, cfg)
    d_anchor, _res, med_anchor = align_pt2pt(cur, ref, seed, cfg)
    # the point-to-point gate ladder STALLS short of the basin on the noisy
    # smooth contour (the pure p2p failure mode of the lesson)
    assert med_anchor > 0.15
    delta, _pr, med = align_pt2plane(cur, ref, d_anchor, estimate_normals(ref, 8), cfg)
    assert med < 0.15  # the polish reaches the basin
    err = np.linalg.norm(delta[:2] - true_delta[:2])
    assert err < 0.30  # direction recovered


def inv_apply_delta(points, delta):
    """Inverse of apply_delta in the rotate-then-translate convention."""
    c, s = math.cos(delta[2]), math.sin(delta[2])
    rot = np.array([[c, s], [-s, c]])  # R(-theta)
    q = points - delta[:2]
    return q @ rot.T


def rounded_square(n=120, base=6.0):
    """Asymmetric cornered contour: no exact discrete self-symmetry, no
    straight-wall sliding degeneracy in any direction."""
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    rho = base + 0.7 * np.sin(2.0 * t) + 0.5 * np.sin(5.0 * t) + 0.8 * np.cos(7.0 * t)
    return np.column_stack([rho * np.cos(t), rho * np.sin(t)])


def test_pt2plane_recovers_known_transform_without_noise():
    """Pipeline seeding: coarse grid prior then point-to-plane refine.

    A zero-init GN sits in the SLIDING DEGENERACY of the straight walls
    (plane residual ~0 at a wrong x-offset); the coarse grid is what
    selects the corner-aligned basin - exactly why the pipeline starts
    there.  The refined match itself is validated on a rounded-square
    contour (cornered everywhere, no flat sliding direction).
    """
    from embodied_learning.experiments.matcher_engineering import (
        align_pt2plane,
        align_pt2pt,
        coarse_argmax,
        estimate_normals,
    )

    cfg = matcher()
    ref = l_wall_ref()
    true_delta = np.array([0.8, 0.6, math.radians(-8.0)])
    cur = inv_apply_delta(ref.copy(), true_delta)
    zero_init, _p0, _m0 = align_pt2plane(cur, ref, np.zeros(3), estimate_normals(ref, 8), cfg)
    assert abs(zero_init[0] - true_delta[0]) > 0.3  # the sliding local min

    ref = rounded_square()
    true_delta = np.array([0.8, 0.6, math.radians(-8.0)])
    cur = inv_apply_delta(ref.copy(), true_delta)
    coarse_seed, _ = coarse_argmax(cur, ref, cfg)
    d_anchor, _res, _m = align_pt2pt(cur, ref, coarse_seed, cfg)
    delta, plane_res, med = align_pt2plane(cur, ref, d_anchor, estimate_normals(ref, 8), cfg)
    assert med < 0.03  # exact point correspondence: the anchor+polish floor
    assert plane_res < 0.03
    np.testing.assert_allclose(delta, true_delta, atol=0.05)


# ---------------------------------------------------------------- top-k
def test_topk_returns_distinct_non_max_suppressed_candidates():
    from embodied_learning.experiments.matcher_engineering import coarse_topk

    cfg = matcher()
    ref, cur, _ = noisy_synthetic(cfg, seed=5)
    cands, score = coarse_topk(cur, ref, cfg)
    assert len(cands) == cfg.top_k
    assert score > 0.0
    deltas = np.array(cands)
    # suppression radius: no two kept cells within 2 cells of each other
    d = np.abs(deltas[:, None, :] - deltas[None, :, :])
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            assert d[i, j, 0] > 1.0 or d[i, j, 1] > 1.0 or d[i, j, 2] > math.radians(10.0)


# ---------------------------------------------------------------- groups
def test_match_once_dispatch_contracts():
    from embodied_learning.experiments.matcher_engineering import match_once

    cfg = matcher()
    ref, cur, _ = noisy_synthetic(cfg, seed=7)
    for group in ("B", "P", "R", "K"):
        m = match_once(ref, cur, cfg, group)
        assert m["delta"] is not None
        assert math.isfinite(m["median_nn"])
        assert m["median_nn"] >= 0.0
    # K additionally returns its ranked hypothesis list
    k = match_once(ref, cur, cfg, "K")
    assert len(k["hypotheses"]) == cfg.top_k
    # ranking order: the sorted list's first metric is the returned one
    assert k["hypotheses"][0]["median_nn"] == k["median_nn"]


def test_match_pipeline_direction_on_noiseless_synthetic():
    """The pure p2p ladder stalls short of the basin on a smooth closed
    contour (B rejects); the anchored-then-polished pipeline (P/R/K)
    recovers the exact delta - the ablation difference the lesson builds
    on.  The record measures the aggregate outcome on the patrol pool."""
    from embodied_learning.experiments.matcher_engineering import match_once

    cfg = matcher()
    ref = rounded_square()
    true_delta = np.array([0.8, 0.6, math.radians(-8.0)])
    cur = inv_apply_delta(ref.copy(), true_delta)
    b = match_once(ref, cur, cfg, "B")
    assert not b["accepted"]
    assert b["median_nn"] > 0.15
    for group in ("P", "R", "K"):
        m = match_once(ref, cur, cfg, group)
        assert m["accepted"]
        np.testing.assert_allclose(m["delta"], true_delta, atol=0.05)


def test_match_once_returns_none_and_rejects_for_tiny_clouds():
    from embodied_learning.experiments.matcher_engineering import match_once

    cfg = matcher()
    m = match_once(np.zeros((2, 2)), np.zeros((3, 2)), cfg, "K")
    assert m["delta"] is None
    assert not m["accepted"]


def test_correctness_uses_rotate_then_translate_convention():
    from embodied_learning.experiments.matcher_engineering import is_correct, true_delta

    est = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.1, 1.0, 0.0],
            [1.0, 1.0, 0.1],
        ]
    )
    d_true = true_delta(est, 2)
    # t_r - R(dth) t_c with dth = -0.1 rad (the matcher convention, NOT the
    # world-frame difference [0, 0, 0.1])
    expected = np.array([-0.0948374, 0.1048292, -0.1])
    np.testing.assert_allclose(d_true, expected, atol=1e-6)
    near = d_true + np.array([0.02, -0.02, 0.0])
    assert is_correct({"delta": near}, d_true)
    bad = d_true + np.array([1.0, 0.0, 0.0])
    assert not is_correct({"delta": bad}, d_true)
    rot_bad = d_true + np.array([0.0, 0.0, math.radians(6.0)])
    assert not is_correct({"delta": rot_bad}, d_true)


# ------------------------------------------------------------- records
@pytest.fixture(scope="module")
def light_stream(tmp_path_factory):
    """One light full-record run (single scene, capped candidates)."""
    from embodied_learning.experiments.grid_nav import GridNavConfig
    from embodied_learning.experiments.matcher_engineering import (
        MatcherConfig,
        run_experiment,
    )

    cfg = MatcherConfig(
        base=GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1),
        candidate_max=6,
    )
    out = tmp_path_factory.mktemp("matcher49") / "run"
    report = run_experiment(out, seed=0, config=cfg, log=None)
    return out, report


def test_light_run_contract(light_stream):
    out, report = light_stream
    assert report["experiment"] == "matcher_engineering_lesson49"
    assert report["schema_version"] == 1
    assert report["n_candidates_total"] >= 4
    for group in ("B", "P", "R", "K"):
        agg = report["aggregates"][group]
        assert 0 <= agg["acceptance"] <= 1.0
        assert 0 <= agg["direction_correctness"] <= 1.0
        assert agg["n_candidates"] == report["n_candidates_total"]
    assert report["protocol"]["matcher"]["top_k"] == 5
    assert set(report["hypothesis"]["results"]) >= {
        "collector_met",
        "motivation_met",
        "acceptance_met",
    }
    assert (out / "metric_bars.png").exists()
    assert (out / "trajectories.npz").exists()


def test_record_tamper_rejection(light_stream):
    out, report = light_stream
    path = out / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # tamper-and-cross-check: swap the K acceptance in the summary
    summary = out / "summary.json"
    original = summary.read_text(encoding="utf-8")
    data = json.loads(original)
    data["aggregates"]["K"]["acceptance"] = 0.99
    summary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_replays(out)
    summary.write_text(original, encoding="utf-8")


def load_replays(directory):
    import pathlib

    from embodied_learning.experiments.matcher_engineering import (
        DEFAULT_RESULTS,  # noqa: F401  (import symmetry with the demo)
        EXPERIMENT,
        expected_npz_keys,
    )

    directory = pathlib.Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-49 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
    # cross-check: aggregates must be recomputable from the per-candidate rows
    episode_rows = report["episodes"]
    for group in ("B", "P", "R", "K"):
        rows = [r for r in episode_rows if r["group"] == group]
        agg = report["aggregates"][group]
        if agg["n_candidates"] != len(rows):
            raise ValueError("Candidate count mismatch - tampered summary")
        recomputed = sum(1 for r in rows if r["accepted"]) / len(rows)
        if abs(recomputed - agg["acceptance"]) > 1e-9:
            raise ValueError("Acceptance mismatch - tampered summary")
    return report


@pytest.mark.slow
def test_seed_determinism(tmp_path):
    """Two light runs on the same seed replicate the summary bitwise."""
    from embodied_learning.experiments.grid_nav import GridNavConfig
    from embodied_learning.experiments.matcher_engineering import (
        MatcherConfig,
        run_experiment,
    )

    cfg = MatcherConfig(
        base=GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1),
        candidate_max=4,
    )
    first = run_experiment(tmp_path / "a", seed=0, config=cfg, log=None)
    second = run_experiment(tmp_path / "b", seed=0, config=cfg, log=None)
    pop = lambda report: json.dumps(
        {k: v for k, v in report.items() if k != "wall_time_s"},
        sort_keys=True,
        indent=1,
    )
    assert pop(first) == pop(second)
