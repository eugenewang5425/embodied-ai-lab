"""Known-answer tests for the unified navigation scorer (roadmap Phase 0).

Each test pins one scoring boundary with a hand-built sequence whose correct
answer is computable by inspection:
  - turn-then-move odom composition (world-frame deltas after a rotation);
  - yaw wrap across +-pi;
  - no use of a correction that does not exist yet (causality);
  - a locator with no output is invalid, never a fake-valid zero pose;
  - time matching refuses matches beyond tolerance;
  - coverage counts unique truth frames (burst cannot inflate it).
"""

import math

import numpy as np
import pytest


def mod():
    import embodied_learning.experiments.nav_scoring as ns

    return ns


def test_turn_then_move_composition():
    """After a pure in-place rotation, a world-frame +y odom move must land
    the propagated pose at +y - guarding against body/world composition
    errors in the odom-delta propagation."""
    ns = mod()
    # odom (world frame): rotate to +90 deg at t=1, then move 1 m along +y
    odom = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, math.pi / 2],
            [0.0, 1.0, math.pi / 2],
        ]
    )
    ts_odom = np.array([0.0, 1.0, 2.0])
    truth = odom.copy()  # perfect odometry in this synthetic world
    # a single correction at t=0 equal to truth[0]: continuous pose must
    # track truth exactly for all later frames
    cont = ns.continuous_pose(
        np.array([[0.0, 0.0, 0.0]]),
        odom,
        np.array([0.0]),
        ts_odom,
    )
    assert np.allclose(cont[1], truth[1], atol=1e-12)
    assert np.allclose(cont[2], truth[2], atol=1e-12)


def test_yaw_wraps_across_pi():
    """A correction near +pi followed by odom crossing to -pi must produce a
    wrapped continuous yaw near -pi, not an unwrapped ~+pi pileup."""
    ns = mod()
    odom = np.array(
        [
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -3.0],  # crossed the +pi boundary (wrapped)
        ]
    )
    ts = np.array([0.0, 1.0])
    cont = ns.continuous_pose(np.array([[0.0, 0.0, 3.0]]), odom, np.array([0.0]), ts)
    assert abs(cont[1, 2]) <= math.pi + 1e-12
    assert abs(math.atan2(math.sin(cont[1, 2] - (-3.0)), math.cos(cont[1, 2] - (-3.0)))) < 1e-9


def test_no_future_correction_before_first_estimate():
    """Frames timestamped before the first correction are NaN - the scorer
    must not borrow the first (future) estimate for them."""
    ns = mod()
    n = 10
    ts = np.arange(n, dtype=float)
    odom = np.zeros((n, 3))
    odom[:, 0] = np.linspace(0.0, 0.9, n)  # robot drives +x the whole time
    truth = odom.copy()
    # the only correction arrives at t=5, claiming the true pose
    cont = ns.continuous_pose(np.array([[0.5, 0.0, 0.0]]), odom, np.array([5.0]), ts)
    for k in range(5):
        assert np.all(np.isnan(cont[k])), f"frame {k} used a future correction"
    # from t=5 on: correction + odom delta since t=5 = truth (perfect odom)
    assert np.allclose(cont[5], truth[5], atol=1e-12)
    assert np.allclose(cont[9], truth[9], atol=1e-12)


def test_no_output_is_invalid_not_zero():
    """A locator that never output anything yields an all-NaN trajectory and
    zero valid coverage - previously it silently produced a fake-valid zero
    pose for every frame."""
    ns = mod()
    ts = np.arange(5, dtype=float)
    odom = np.zeros((5, 3))
    truth = np.zeros((5, 3))
    truth[:, 0] = np.linspace(0.0, 0.4, 5)
    cont = ns.continuous_pose(np.zeros((0, 3)), odom, np.zeros(0), ts)
    assert np.all(np.isnan(cont))
    metrics = ns.nan_aware_metrics(cont, truth, ts)
    assert metrics["n_valid"] == 0
    assert metrics["valid_coverage"] == 0.0
    assert metrics["mean_m"] is None


def test_aligned_time_tolerance_rejects_far_estimates():
    """An estimate timestamped outside the tolerance (e.g. a wall-clock
    leak) stays unmatched instead of binding to truth index 0."""
    ns = mod()
    ts_truth = np.arange(10, dtype=float)
    truth = np.zeros((10, 3))
    # one good estimate at t=3, one leaked stamp 1000 s late
    est = np.array([[0.0, 0.0, 0.0], [99.0, 99.0, 0.0]])
    ts_est = np.array([3.0, 1000.0])
    metrics, pairs = ns.score_aligned(est, truth, ts_est, ts_truth, max_gap_s=0.5)
    assert metrics["n_matched"] == 1
    assert metrics["n_unmatched_time"] == 1
    assert pairs == [(0, 3)]


def test_aligned_coverage_counts_unique_truth_frames():
    """Two estimates inside the same truth interval must not push coverage
    past its unique-frame share."""
    ns = mod()
    ts_truth = np.arange(10, dtype=float)
    truth = np.zeros((10, 3))
    est = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ts_est = np.array([0.01, 0.02])  # both match truth frame 0
    metrics, _pairs = ns.score_aligned(est, truth, ts_est, ts_truth, max_gap_s=0.5)
    assert metrics["n_matched"] == 2
    assert metrics["n_unique_truth"] == 1
    assert metrics["coverage"] == pytest.approx(1 / 10)


def test_aligned_empty_report_shape():
    """No matches: every metric key exists with a None/zero value - downstream
    consumers can rely on the schema instead of missing-key errors."""
    ns = mod()
    metrics, pairs = ns.score_aligned(
        np.zeros((0, 3)), np.zeros((4, 3)), np.zeros(0), np.arange(4.0)
    )
    assert pairs == []
    for key in ("n_matched", "n_unique_truth", "n_unmatched_time", "coverage"):
        assert key in metrics
    assert metrics["mean_m"] is None and metrics["coverage"] == 0.0


def test_invalid_span_reported_not_hidden():
    """nan_aware_metrics reports how much timeline was invalid - the mean is
    computed over valid frames and the report carries the invalid share, so
    dropping frames cannot shrink error unnoticed."""
    ns = mod()
    n = 20
    ts = np.arange(n, dtype=float)
    truth = np.zeros((n, 3))
    truth[:, 0] = np.linspace(0.0, 1.9, n)
    cont = np.full((n, 3), np.nan)
    cont[10:] = truth[10:]  # valid only in the second half
    metrics = ns.nan_aware_metrics(cont, truth, ts)
    assert metrics["n_valid"] == 10
    assert metrics["valid_coverage"] == pytest.approx(0.5)
    assert metrics["invalid_span_s"] == pytest.approx(10 * (ts[1] - ts[0]))
    assert metrics["mean_m"] == pytest.approx(0.0)  # perfect where valid


def test_v4_amcl_replay_scores_unchanged_under_audit():
    """Re-score the gated A-base run with the audited scorer: the verified
    stamps sit inside tolerance and start at t=0, so the audited causality
    and tolerance rules must leave the recorded numbers unchanged (Phase-0
    check that old AMCL conclusions survive the audit)."""
    from pathlib import Path

    ns = mod()
    run_dir = Path("results/p3_amcl_batch_v2/A-base_s0/r0")
    if not run_dir.exists():
        pytest.skip("gated A-base record not present on this machine")
    data = dict(np.load(Path("results/amcl_bridge_v2_s0/frozen_input.npz")))
    truth, odom, ts = data["truth"], data["odom"], data["timestamps"]
    with np.load(run_dir / "amcl_poses.npz") as z:
        est = np.column_stack([z["x"], z["y"], z["yaw"]])
        stamp = z["stamp"]
    aligned, _ = ns.score_aligned(est, truth, stamp, ts, max_gap_s=0.5)
    assert aligned["n_matched"] == 348  # same pose count as the record
    assert aligned["mean_m"] == pytest.approx(0.0972, abs=1e-3)
    cont = ns.continuous_pose(est, odom, stamp, ts)
    metrics = ns.nan_aware_metrics(cont, truth, ts)
    assert metrics["n_valid"] == metrics["n_frames"]  # first pose at t=0
    assert metrics["mean_m"] == pytest.approx(0.0984, abs=1e-3)
