"""The replay benchmark gate counts worlds rather than correlated frames."""

from embodied_learning.experiments.navigation_benchmark import BenchmarkPlan, summarize


def _row(arm, world, sensor, *, mean=1.0, p95=2.0):
    return {
        "arm": arm,
        "world_seed": world,
        "sensor_seed": sensor,
        "status": "ok",
        "est_mean_m": 2.0,
        "pf_true_mean_m": 0.5,
        "pf_sensor_truth_mean_m": 0.7,
        "pf_self_mean_m": mean,
        "est_p95_m": 3.0,
        "pf_self_p95_m": p95,
        "map_alignment_precision": 0.8,
        "map_alignment_recall": 0.8,
    }


def test_small_pilot_cannot_pass_deployment_gate():
    plan = BenchmarkPlan(arms=("ISO", "ANISO"), world_seeds=(0,), sensor_seeds=(0,))
    rows = [_row(arm, 0, 0) for arm in plan.arms]
    assert summarize(rows, plan)["selfbuilt_localization_gate_met"] is False


def test_gate_needs_four_complete_worlds_per_arm():
    plan = BenchmarkPlan()
    rows = [
        _row(arm, world, sensor, mean=4.0 if world >= 3 else 1.0)
        for arm in plan.arms
        for world in plan.world_seeds
        for sensor in plan.sensor_seeds
    ]
    assert summarize(rows, plan)["selfbuilt_localization_gate_met"] is False
    for row in rows:
        if row["world_seed"] == 3:
            row["pf_self_mean_m"] = 1.0
    assert summarize(rows, plan)["selfbuilt_localization_gate_met"] is True


def test_partial_fifth_world_does_not_count_as_valid_world():
    plan = BenchmarkPlan()
    rows = [
        _row(arm, world, sensor)
        for arm in plan.arms
        for world in plan.world_seeds
        for sensor in plan.sensor_seeds
        if not (world == 4 and sensor == 2)
    ]
    result = summarize(rows, plan)
    assert all(item["valid_worlds"] == 4 for item in result["by_arm"].values())
    assert result["selfbuilt_localization_gate_met"] is False
