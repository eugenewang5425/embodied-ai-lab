"""Known-answer tests for the map-separation decomposition (roadmap Phase 1)."""

import json

import numpy as np
import pytest


def make_episode(dir_path, ideal_mean, self_mean, diverge_at=None, n_frames=40):
    """One fake benchmark episode: summary row + trajectories.npz.

    Error curve is constant `ideal_mean` (self_mean for the self chain);
    with diverge_at, the ideal error jumps to 5.0 there and stays - a clean,
    inspectable divergence for onset detection.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    err = np.full(n_frames, ideal_mean)
    if diverge_at is not None:
        err[diverge_at:] = 5.0
    truth = np.zeros((n_frames, 3))
    truth[:, 0] = np.arange(n_frames) * 0.1  # 0.1 m per frame straight +x
    row = {
        "arm": "ISO",
        "world_seed": 0,
        "sensor_seed": 0,
        "status": "ok",
        "est_mean_m": 1.0,
        "pf_true_mean_m": float(err.mean()),
        "pf_sensor_truth_mean_m": 0.4,
        "pf_self_mean_m": self_mean,
        "pf_self_p95_m": self_mean * 1.1,
        "map_alignment_precision": 0.5,
        "map_alignment_recall": 0.5,
    }
    np.savez_compressed(
        dir_path / "trajectories.npz",
        pf_true_err=err,
        pf_self_err=np.full(n_frames, self_mean),
        truth_chain=truth,
    )
    return row


def make_benchmark(root, particles, rows):
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps({"plan": {"particles": particles}, "rows": rows}), encoding="utf-8"
    )


def test_decomposition_and_onset(tmp_path):
    from embodied_learning.experiments.map_separation import analyze

    lo = tmp_path / "p100"
    hi = tmp_path / "p400"
    # episode A: clean, ideal 0.3 / truth-proj 0.4 / self 1.6
    row_a = make_episode(lo / "iso_w0_s0", ideal_mean=0.3, self_mean=1.6)
    make_benchmark(lo, 100, [row_a])
    # episode B: ideal-map divergence from frame 20; arc to frame 20 = 2.0 m
    row_b = make_episode(
        lo / "iso_w1_s0", ideal_mean=0.3, self_mean=1.8, diverge_at=20
    )
    row_b["world_seed"] = 1
    rows_lo = [row_a, row_b]
    (lo / "summary.json").write_text(
        json.dumps({"plan": {"particles": 100}, "rows": rows_lo}), encoding="utf-8"
    )
    # hi mirrors lo with 400 particles (same episode keys)
    row_a400 = make_episode(hi / "iso_w0_s0", ideal_mean=0.3, self_mean=1.6)
    row_b400 = make_episode(
        hi / "iso_w1_s0", ideal_mean=0.3, self_mean=1.8, diverge_at=20
    )
    row_b400["world_seed"] = 1
    make_benchmark(hi, 400, [row_a400, row_b400])

    report = analyze(lo, hi)

    # clean subset = episode A only (B's ideal arm diverges)
    dec = report["decomposition_m"]["p100"]
    assert dec["n_clean_episodes"] == 1
    # representation gap = truth-proj - ideal = 0.4 - 0.3 = +0.1
    assert dec["representation_gap_median_m"] == pytest.approx(0.1)
    # landing gap = self - truth-proj = 1.6 - 0.4 = +1.2
    assert dec["landing_gap_median_m"] == pytest.approx(1.2)

    # per-arm divergence flags: B ideal diverges, its self arm also >1 m
    assert report["particles"]["p100"]["ideal_diverged_episodes"] == ["iso_w1_s0"]
    assert set(report["particles"]["p100"]["self_diverged_episodes"]) == {
        "iso_w0_s0",
        "iso_w1_s0",
    }

    # onset: frame 20, arc 2.0 m (truth moves 0.1 m/frame, straight)
    onset = report["divergence_onsets_ideal_map_p400"]["iso_w1_s0"]
    assert onset["onset_frame"] == 20
    assert onset["onset_arc_m"] == pytest.approx(2.0, abs=0.05)


def test_onset_ignores_brief_spike(tmp_path):
    """A one-frame spike that recovers is NOT divergence onset."""
    from embodied_learning.experiments.map_separation import divergence_onset

    d = tmp_path / "ep"
    d.mkdir()
    n = 30
    err = np.full(n, 0.2)
    err[10] = 5.0  # single-frame spike, recovers immediately
    truth = np.zeros((n, 3))
    np.savez_compressed(d / "trajectories.npz", pf_true_err=err, truth_chain=truth)
    frame, _arc = divergence_onset(d)
    assert frame is None
