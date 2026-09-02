import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_learning.experiments.lqr_measurement_noise import run_noise_experiment
from embodied_learning.noise_demo import load_noise_replays, noise_chart_series, noise_status_text
from embodied_learning.teaching_demo import CHART_METRICS, PlaybackClock, TeachingDemo


@pytest.fixture
def noise_recording(tmp_path):
    directory = tmp_path / "noise"
    run_noise_experiment(directory, episodes=1, horizon=10)
    return directory


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def test_loads_real_saved_data_readonly_with_paired_labels(noise_recording):
    before = hashes(noise_recording)
    replays = load_noise_replays(noise_recording, 0.4, 200)
    assert [p.noise_scale for p in replays] == [0, 1, 3]
    assert [p.label for p in replays] == ["噪声 0×", "噪声 1×", "噪声 3×"]
    assert all(p.r == 1 and p.gear == 100 for p in replays)
    with np.load(noise_recording / "trajectories.npz") as archive:
        for index, replay in enumerate(replays):
            np.testing.assert_array_equal(replay.states, archive[f"case{index}_seed200_states"])
            np.testing.assert_array_equal(
                replay.measurements, archive[f"case{index}_seed200_measurements"]
            )
            np.testing.assert_array_equal(replay.controls, archive[f"case{index}_seed200_controls"])
            assert replay.external_forces_n is None and not replay.terminated
    assert hashes(noise_recording) == before


def test_slicing_keeps_state_reading_action_alignment(noise_recording):
    replays = load_noise_replays(noise_recording, seconds=0.12)
    assert all(
        p.states.shape == (4, 4) and p.measurements.shape == (3, 4) and len(p.controls) == 3
        for p in replays
    )
    assert replays[0].original_seconds == pytest.approx(0.4)
    np.testing.assert_array_equal(replays[0].measurements, replays[0].states[:-1])


def test_chart_never_invents_final_reading_and_control_is_held(noise_recording):
    replays = load_noise_replays(noise_recording)
    selected = replays[1]
    truth, reading = noise_chart_series(replays, selected, CHART_METRICS[1], False)
    assert truth.times[-1] == pytest.approx(0.4)
    assert reading.times[-1] == pytest.approx(0.36)
    np.testing.assert_allclose(
        reading.values, np.rad2deg(selected.measurements[:, 1] - selected.reference[1])
    )
    position = noise_chart_series(replays, selected, CHART_METRICS[0], True)
    assert len(position) == 4 and sum(s.measured for s in position) == 1
    assert len({s.color for s in position}) == 4
    np.testing.assert_array_equal(position[-1].values, selected.measurements[:, 0] * 100)
    controls = noise_chart_series(replays, selected, CHART_METRICS[2], True)
    assert len(controls) == 3 and not any(s.measured for s in controls)
    np.testing.assert_array_equal(controls[1].values, np.repeat(selected.controls, 2))
    np.testing.assert_allclose(controls[1].times[:4], [0, 0.04, 0.04, 0.08])


def test_stats_end_frame_has_truth_but_no_stale_reading_or_command(noise_recording):
    replay = load_noise_replays(noise_recording)[1]
    assert "真实 → 读数" in noise_status_text(replay, 0)
    assert "外部推力 = 0 N" in noise_status_text(replay, 0)
    final = noise_status_text(replay, len(replay.controls))
    assert "无下一步读数或动作" in final
    assert "u =" not in final and "→" not in final


def test_selection_distinguishes_three_conditions_with_same_r(noise_recording):
    demo = TeachingDemo.__new__(TeachingDemo)
    demo.replays = load_noise_replays(noise_recording)
    demo.replay = demo.replays[1]
    demo.clock = PlaybackClock(10, 0.04, 0.1)
    demo.clock.seek(5)
    demo.policy = SimpleNamespace(get=lambda: "噪声 3×")
    demo.slider = SimpleNamespace(configure=lambda **kw: None)
    demo.refresh = lambda: None
    demo.change_policy()
    assert demo.replay is demo.replays[2]
    assert demo.clock.index == 0 and demo.clock.paused and demo.clock.speed == 0.1


def test_bad_seed_and_misaligned_measurements_rejected(noise_recording):
    with pytest.raises(ValueError, match="Seed"):
        load_noise_replays(noise_recording, seed=999)
    path = noise_recording / "trajectories.npz"
    with np.load(path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["case1_seed200_measurements"][0, 1] += 0.01
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="Misaligned"):
        load_noise_replays(noise_recording)


@pytest.mark.parametrize("duration", [0, -1, np.nan, np.inf, 121])
def test_bad_duration_rejected_before_reading(tmp_path, duration):
    with pytest.raises(ValueError):
        load_noise_replays(tmp_path, duration)


def test_invalid_metadata_rejected(noise_recording):
    path = noise_recording / "summary.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["dt_s"] = 0
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        load_noise_replays(noise_recording)
