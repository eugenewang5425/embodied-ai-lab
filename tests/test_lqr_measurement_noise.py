from __future__ import annotations

import json

import numpy as np
import pytest

from embodied_learning.controllers.lqr import design_lqr
from embodied_learning.experiments.lqr_measurement_noise import (
    BASE_SENSOR_STD,
    MeasuredFeedback,
    noise_metrics,
    run_noise_experiment,
)
from embodied_learning.experiments.pd_comparison import run_episode


class ConstantController:
    def __init__(self):
        self.seen = []

    def action(self, state):
        self.seen.append(state.copy())
        return np.zeros(1, dtype=np.float32)


def test_noise_changes_measurement_but_never_directly_moves_plant_or_mutates_state():
    initial = design_lqr().controller.reference
    noise = np.full((5, 4), 0.5)
    raw_controller = ConstantController()
    measured = MeasuredFeedback(raw_controller, noise)
    trace = run_episode("lqr", 0, 5, measured, initial_state=initial)
    baseline = run_episode("lqr", 0, 5, ConstantController(), initial_state=initial)
    np.testing.assert_array_equal(trace.observations, baseline.observations)
    np.testing.assert_array_equal(trace.actions, baseline.actions)
    np.testing.assert_allclose(raw_controller.seen[0], initial + noise[0])
    np.testing.assert_array_equal(initial, baseline.initial_observation)
    np.testing.assert_array_equal(noise, np.full((5, 4), 0.5))
    assert not trace.terminated  # Noisy angle exceeds 0.2 rad; true angle does not.


def test_zero_noise_matches_existing_runner_exactly():
    design = design_lqr(control_weight=1)
    wrapper = MeasuredFeedback(design.controller, np.zeros((100, 4)))
    measured = run_episode("lqr", 7, 100, wrapper)
    baseline = run_episode("lqr", 7, 100, design.controller)
    np.testing.assert_array_equal(measured.actions, baseline.actions)
    np.testing.assert_array_equal(measured.observations, baseline.observations)
    np.testing.assert_array_equal(
        wrapper.measurements, np.vstack([baseline.initial_observation, *baseline.observations])[:-1]
    )


def test_action_uses_current_noisy_state_and_retains_original_saturation():
    design = design_lqr(control_weight=1)
    noise = np.zeros((3, 4))
    noise[:, 1] = [0.001, 1, -1]
    feedback = MeasuredFeedback(design.controller, noise)
    initial = design.controller.reference.copy()
    actions = [feedback.action(initial) for _ in range(3)]
    for i in range(3):
        np.testing.assert_array_equal(actions[i], design.controller.action(initial + noise[i]))
    assert abs(actions[1][0]) == 3 and abs(actions[2][0]) == 3
    np.testing.assert_array_equal(initial, design.controller.reference)
    with pytest.raises(ValueError, match="exhausted"):
        feedback.action(initial)


@pytest.mark.parametrize(
    "noise",
    [
        np.empty((0, 4)),
        np.zeros(4),
        np.zeros((2, 3)),
        np.full((2, 4), np.nan),
        np.full((2, 4), np.inf),
    ],
)
def test_invalid_noise_rejected(noise):
    with pytest.raises(ValueError, match="noise"):
        MeasuredFeedback(ConstantController(), noise)


def test_input_schedule_is_owned_and_each_episode_has_fresh_measurements():
    noise = np.ones((2, 4))
    wrapper = MeasuredFeedback(ConstantController(), noise)
    noise[:] = 0
    wrapper.action(np.zeros(4))
    np.testing.assert_array_equal(wrapper.measurements, np.ones((1, 4)))
    fresh = MeasuredFeedback(ConstantController(), noise)
    assert fresh.measurements == []


def test_metrics_align_pre_action_measurements_not_post_action_states():
    design = design_lqr(control_weight=1)
    noise = np.tile(BASE_SENSOR_STD, (50, 1))
    controller = MeasuredFeedback(design.controller, noise)
    trace = run_episode("lqr", 0, 50, controller, initial_state=design.controller.reference)
    measurements = np.asarray(controller.measurements)
    metrics = noise_metrics(trace, measurements, design.controller.reference, 50)
    np.testing.assert_allclose(
        metrics["measurement_error_rms_by_state"], BASE_SENSOR_STD, atol=1e-14
    )
    assert metrics["control_rms"] > 0
    trace.terminated = True  # Termination on the final step is still a failure.
    assert not noise_metrics(trace, measurements, design.controller.reference, 50)[
        "survived_horizon"
    ]


def test_saved_experiment_paired_seed_alignment_repeatability_and_no_overwrite(tmp_path):
    directory = tmp_path / "noise"
    report = run_noise_experiment(directory, episodes=2, horizon=50)
    assert report == json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert (directory / "comparison.png").stat().st_size > 1000
    reference = np.asarray(report["reference"])
    design = design_lqr(control_weight=1)
    with np.load(directory / "trajectories.npz", allow_pickle=False) as archive:
        for i, scale in enumerate((0, 1, 3)):
            for seed in (200, 201):
                key = f"case{i}_seed{seed}"
                states, controls = archive[f"{key}_states"], archive[f"{key}_controls"]
                n = len(controls)
                noise, measurements = archive[f"{key}_noise"], archive[f"{key}_measurements"]
                assert states.shape == (n + 1, 4)
                np.testing.assert_array_equal(states[0], reference)
                np.testing.assert_array_equal(
                    noise, archive[f"seed{seed}_standard_normal"][:n] * BASE_SENSOR_STD * scale
                )
                np.testing.assert_allclose(measurements, states[:-1] + noise, atol=1e-14)
                np.testing.assert_array_equal(
                    controls, [design.controller.action(z)[0] for z in measurements]
                )
        assert not np.array_equal(archive["case1_seed200_noise"], archive["case1_seed201_noise"])
    again = run_noise_experiment(tmp_path / "repeat", episodes=2, horizon=50)
    assert again == report
    before = (directory / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_noise_experiment(directory)
    assert (directory / "summary.json").read_bytes() == before


@pytest.mark.parametrize("kwargs", [{"episodes": 0}, {"first_seed": -1}, {"horizon": 1}])
def test_invalid_experiment_arguments_create_no_output(tmp_path, kwargs):
    target = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_noise_experiment(target, **kwargs)
    assert not target.exists()
