from __future__ import annotations

import hashlib

import numpy as np
import pytest

from embodied_learning.controllers.lqr import design_lqr
from embodied_learning.experiments.lqr_disturbance import (
    random_push,
    recovery_metrics,
    run_disturbance,
)
from embodied_learning.experiments.pd_comparison import EpisodeTrace, run_episode
from embodied_learning.teaching_demo import load_push_replays


class ConstantController:
    def __init__(self, value):
        self.value = value

    def action(self, _):
        return np.array([self.value], dtype=np.float32)


def test_force_is_newtons_separate_from_gear_and_reset_each_step():
    initial = design_lqr().controller.reference
    pushed = run_episode(
        "lqr",
        0,
        2,
        ConstantController(0),
        initial_state=initial,
        external_forces_n=np.array([100.0, 0.0]),
    )
    motor = run_episode("lqr", 0, 1, ConstantController(1), initial_state=initial)
    np.testing.assert_allclose(pushed.observations[0], motor.observations[0], atol=1e-12)
    coast = run_episode("lqr", 0, 1, ConstantController(0), initial_state=pushed.observations[0])
    np.testing.assert_allclose(pushed.observations[1], coast.observations[0], atol=1e-12)
    assert pushed.actions == [0, 0] and pushed.external_forces_n == [100, 0]


def test_zero_force_preserves_baseline_and_new_episode_has_no_residue():
    design = design_lqr()
    kwargs = {"policy": "lqr", "seed": 7, "horizon": 100, "controller": design.controller}
    before = run_episode(**kwargs)
    explicit_zero = run_episode(**kwargs, external_forces_n=np.zeros(100))
    np.testing.assert_array_equal(before.observations, explicit_zero.observations)
    run_episode(**kwargs, external_forces_n=np.full(100, 100.0))
    np.testing.assert_array_equal(before.observations, run_episode(**kwargs).observations)


@pytest.mark.parametrize("forces", [np.zeros(3), np.array([0.0, np.nan]), np.zeros((2, 1))])
def test_invalid_force_schedule_rejected(forces):
    with pytest.raises(ValueError, match="external_forces_n"):
        run_episode("random", 0, 2, external_forces_n=forces)


def test_random_push_is_repeatable_bounded_and_varies_across_seeds():
    pushes = [random_push(seed, 250, 0.04) for seed in range(100, 120)]
    for seed, forces in zip(range(100, 120), pushes, strict=True):
        np.testing.assert_array_equal(forces, random_push(seed, 250, 0.04))
        indices = np.flatnonzero(forces)
        assert len(indices) == 5 and 50 <= indices[0] <= 75
        assert 30 <= abs(forces[indices[0]]) <= 80
        assert len(np.unique(forces[indices])) == 1
    assert not np.array_equal(pushes[0], pushes[1])
    assert any(np.min(p) < 0 for p in pushes) and any(np.max(p) > 0 for p in pushes)


def test_recovery_time_starts_after_push_end_and_requires_full_tail():
    forces = np.zeros(100)
    forces[10:15] = 30
    observations = np.zeros((100, 4))
    observations[:19, 0] = 0.1  # s[20] is the first recovered state.
    trace = EpisodeTrace(
        0, list(observations), [0.0] * 100, [1.0] * 100, False, [], initial_observation=np.zeros(4)
    )
    row = recovery_metrics(trace, np.zeros(4), 100, forces)
    assert row["recovery_after_push_end_s"] == pytest.approx((20 - 15) * 0.04)
    trace.terminated = True
    assert recovery_metrics(trace, np.zeros(4), 100, forces)["recovery_after_push_end_s"] is None
    trace.terminated = False
    trace.observations[:59] = list(np.full((59, 4), 0.1))
    assert recovery_metrics(trace, np.zeros(4), 100, forces)["recovery_after_push_end_s"] is None


def test_saved_experiment_and_readonly_replay_are_paired_and_do_not_overwrite(tmp_path):
    directory = tmp_path / "push"
    report = run_disturbance(directory, episodes=2)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    replays = load_push_replays(directory, 8, seed=100)
    assert [p.r for p in replays] == [0.1, 1, 10]
    with np.load(directory / "trajectories.npz", allow_pickle=False) as data:
        for p, row in zip(replays, report["conditions"], strict=True):
            np.testing.assert_array_equal(p.states, data[f"r{p.r:g}_seed100_states"][:201])
            np.testing.assert_array_equal(p.controls, data[f"r{p.r:g}_seed100_controls"][:200])
            np.testing.assert_array_equal(
                p.external_forces_n, data["seed100_scheduled_force_n"][:200]
            )
            np.testing.assert_array_equal(p.states[0], report["initial_state"])
            assert row["baseline"]["survived_horizon"]
            assert row["recovered"] == 2
    assert hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
    }
    with pytest.raises(FileExistsError):
        run_disturbance(directory)
    with pytest.raises(ValueError, match="not in"):
        load_push_replays(directory, seed=999)
