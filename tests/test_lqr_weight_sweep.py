from __future__ import annotations

import json

import numpy as np
import pytest

from embodied_learning.experiments.lqr_weight_sweep import input_metrics, run_sweep
from embodied_learning.experiments.pd_comparison import EpisodeTrace


def test_input_metrics_distinguish_commands_forces_and_effort():
    trace = EpisodeTrace(0, [np.zeros(4)] * 2, [1.0, -2.0], [1.0, 1.0], False, [], dt=0.04)
    metrics = input_metrics(trace)
    assert metrics["peak_absolute_control"] == 2
    assert metrics["peak_absolute_actuator_force_n"] == 200
    assert metrics["squared_input_integral"] == pytest.approx(0.2)


@pytest.mark.parametrize("weights", [(1,), (1, 1), (0, 1), (-1, 1), (1, np.nan), (1, np.inf)])
def test_invalid_weights_create_no_output(tmp_path, weights):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_sweep(output, weights)
    assert not output.exists()


def test_paired_sweep_writes_reproducible_traces_and_never_overwrites(tmp_path):
    output = tmp_path / "sweep"
    report = run_sweep(output, episodes=2, horizon=200)
    assert report == json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert (output / "comparison.png").stat().st_size > 1000
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        for i in range(3):
            np.testing.assert_array_equal(
                archive[f"case{i}_displaced0_states"][0], report["displaced_initial_state"]
            )
            for seed in range(2):
                states = archive[f"case{i}_seed{seed}_states"]
                actions = archive[f"case{i}_seed{seed}_controls"]
                assert states.shape == (len(actions) + 1, 4)
                assert np.isfinite(states).all() and np.max(abs(actions)) <= 3
                np.testing.assert_array_equal(states[0], archive[f"case0_seed{seed}_states"][0])
    rows = report["conditions"]
    assert all(row["displaced"]["survived_horizon"] for row in rows)
    assert all(row["displaced"]["settled_at_end"] for row in rows)
    # Specific regression for this model/initial state, not a universal LQR claim.
    assert (
        rows[0]["displaced"]["peak_absolute_control"]
        > rows[1]["displaced"]["peak_absolute_control"]
    )
    assert (
        rows[1]["displaced"]["peak_absolute_control"]
        > rows[2]["displaced"]["peak_absolute_control"]
    )
    assert rows[0]["K"] != rows[1]["K"]
    previous = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_sweep(output)
    assert (output / "summary.json").read_bytes() == previous
