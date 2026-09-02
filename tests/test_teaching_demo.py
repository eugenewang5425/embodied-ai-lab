from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_learning.controllers.lqr import design_lqr
from embodied_learning.teaching_demo import (
    CHART_METRICS,
    CURVE_COLORS,
    PlaybackClock,
    TeachingDemo,
    build_replays,
    chart_samples,
    load_recorded_replay,
)


def test_clock_starts_paused_and_quarter_speed_changes_only_wall_time():
    clock = PlaybackClock(200, 0.04)
    assert clock.paused and clock.index == 0
    assert not clock.advance(10)
    clock.toggle()
    assert not clock.advance(0.08)
    assert clock.advance(0.08)
    assert clock.index == 1  # 0.16 real seconds = 0.04 simulation seconds.
    clock.advance(3.84)
    assert clock.index == 25  # Four real seconds = one simulation second.


def test_pause_step_seek_and_end_never_run_past_saved_data():
    clock = PlaybackClock(10, 0.04)
    clock.step()
    assert clock.index == 1 and clock.paused
    clock.seek(-20)
    assert clock.index == 0
    clock.toggle()
    clock.advance(0.2)
    clock.toggle()
    previous = clock.index
    assert not clock.advance(50)
    assert clock.index == previous
    clock.seek(200)
    assert clock.index == 10 and clock.paused
    clock.step()
    assert clock.index == 10
    clock.toggle()
    assert clock.index == 0 and not clock.paused
    clock.advance(100)
    assert clock.index == 10 and clock.paused
    clock.seek(4)
    assert clock.index == 4 and clock.paused and clock.remainder == 0


def test_speed_change_preserves_state_and_applies_new_rate():
    clock = PlaybackClock(200, 0.04)
    clock.seek(5)
    clock.set_speed(0.1)
    assert clock.index == 5 and clock.paused
    clock.toggle()
    clock.advance(0.4)
    assert clock.index == 6
    clock.set_speed(1)
    clock.advance(0.04)
    assert clock.index == 7


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_clock_rejects_invalid_speed(value):
    with pytest.raises(ValueError):
        PlaybackClock(200, 0.04, value)


@pytest.mark.parametrize("value", [-1, np.inf, np.nan])
def test_clock_rejects_invalid_elapsed(value):
    with pytest.raises(ValueError):
        PlaybackClock(200, 0.04).advance(value)


@pytest.fixture
def recording(tmp_path):
    design = design_lqr(control_weight=1)
    initial = design.controller.reference + np.array([0.2, 0.05, 0, 0])
    # Synthetic archive isolates file loading; generated comparison episodes use MuJoCo.
    states = np.tile(initial, (11, 1))
    controls = np.linspace(0.9, 0.0, 10)
    report = {
        "dt_s": design.dt,
        "design": {
            "R": [[1.0]],
            "actuator_gear": design.actuator_gear,
            "reference": design.controller.reference.tolist(),
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    np.savez_compressed(
        tmp_path / "trajectories.npz",
        displaced_lqr_seed0_states=states,
        displaced_lqr_seed0_controls=controls,
    )
    return tmp_path, states, controls


def fingerprints(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()
    }


def test_recording_is_sliced_without_interpolation_or_writes(recording):
    directory, states, controls = recording
    before = fingerprints(directory)
    replay = load_recorded_replay(directory, 0.2)
    np.testing.assert_array_equal(replay.states, states[:6])
    np.testing.assert_array_equal(replay.controls, controls[:5])
    assert replay.r == 1 and replay.original_seconds == pytest.approx(0.4)
    assert replay.gear == 100
    assert fingerprints(directory) == before
    assert len(load_recorded_replay(directory, 8).controls) == 10


def test_comparisons_use_exact_recorded_initial_state_and_keep_recording(recording):
    directory, states, controls = recording
    before = fingerprints(directory)
    replays = build_replays(seconds=0.4, results=directory)
    assert [p.r for p in replays] == [0.1, 1, 10]
    for replay in replays:
        np.testing.assert_array_equal(replay.states[0], states[0])
        assert replay.states.shape == (len(replay.controls) + 1, 4)
    np.testing.assert_array_equal(replays[1].states, states)
    np.testing.assert_array_equal(replays[1].controls, controls)
    assert fingerprints(directory) == before


def test_bad_archive_is_rejected_instead_of_showing_misaligned_controls(recording):
    directory, states, controls = recording
    np.savez_compressed(
        directory / "trajectories.npz",
        displaced_lqr_seed0_states=states[:-1],
        displaced_lqr_seed0_controls=controls,
    )
    with pytest.raises(ValueError, match="Invalid recorded trajectory"):
        load_recorded_replay(directory, 0.4)


@pytest.mark.parametrize("seconds", [0, -1, 121, np.inf, np.nan])
def test_invalid_replay_duration_is_rejected_before_simulation(seconds):
    with pytest.raises(ValueError):
        build_replays(seconds=seconds)


def test_chart_metric_units_and_control_hold_timestamps(recording):
    directory, states, controls = recording
    replay = load_recorded_replay(directory, 0.4)
    times, values = chart_samples(replay, CHART_METRICS[0])
    np.testing.assert_allclose(times, np.arange(11) * 0.04)
    np.testing.assert_array_equal(values, states[:, 0] * 100)
    _, angle = chart_samples(replay, CHART_METRICS[1])
    np.testing.assert_allclose(angle, np.rad2deg(states[:, 1] - replay.reference[1]))
    t_u, u = chart_samples(replay, CHART_METRICS[2])
    np.testing.assert_allclose(t_u[:6], [0, 0.04, 0.04, 0.08, 0.08, 0.12])
    np.testing.assert_array_equal(u, np.repeat(controls, 2))
    assert t_u[-1] == pytest.approx(0.4)


class CanvasRecorder:
    """Drawing-call unit spy, not a desktop/UI automation driver."""

    def __init__(self):
        self.calls = []

    def winfo_width(self):
        return 1000

    def winfo_height(self):
        return 200

    def delete(self, _):
        self.calls.clear()

    def __getattr__(self, method):
        if method.startswith("create_"):
            return lambda *args, **kwargs: self.calls.append((method, args, kwargs))
        raise AttributeError(method)


def test_overlay_draws_three_color_series_and_toggle_keeps_time(recording):
    directory, _, _ = recording
    demo = TeachingDemo.__new__(TeachingDemo)
    demo.replays = build_replays(0.4, directory)
    demo.replay = demo.replays[1]
    demo.clock = PlaybackClock(10, 0.04)
    demo.clock.seek(5)
    demo.chart = CanvasRecorder()
    demo.metric = SimpleNamespace(get=lambda: CHART_METRICS[0])
    demo.overlay = SimpleNamespace(get=lambda: True)
    demo.draw_chart()
    curves = [
        kw
        for method, _, kw in demo.chart.calls
        if method == "create_line" and kw.get("dash") == (3, 4)
    ]
    assert [c["fill"] for c in curves] == list(CURVE_COLORS[:3])
    demo.overlay = SimpleNamespace(get=lambda: False)
    demo.draw_chart()
    curves = [
        kw
        for method, _, kw in demo.chart.calls
        if method == "create_line" and kw.get("dash") == (3, 4)
    ]
    assert len(curves) == 1 and curves[0]["fill"] == CURVE_COLORS[1]
    assert demo.clock.index == 5 and demo.clock.paused
