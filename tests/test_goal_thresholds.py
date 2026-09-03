"""Only tolerance changes; verify paired baseline, metrics, records and replay."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from embodied_learning.experiments.goal_reaching import DT, load_recording, simulate
from embodied_learning.experiments.goal_thresholds import (
    VARIANTS,
    compare_baseline,
    digest,
    load_thresholds,
    run_thresholds,
    stopping_metrics,
)
from embodied_learning.goal_control import DEFAULT_CONFIG, goal_command
from embodied_learning.threshold_demo import ThresholdDemo, dwell_seconds, restart_frames


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    path = tmp_path_factory.mktemp("thresholds") / "comparison"
    run_thresholds(path, runs=2)
    return path


@pytest.mark.parametrize("radius", [0.02, 0.01, 0.005])
def test_threshold_changes_only_stopping_and_noiseless_accuracy(radius):
    config = replace(DEFAULT_CONFIG, estimated_stop_radius_m=radius)
    arrays, trial = simulate([1.6, 0.8], "odom", 0, config=config, noise_scale=0)
    assert trial["true_success"]
    assert trial["true_final_distance_m"] <= radius + 1e-12
    assert trial["estimated_final_distance_m"] <= radius
    assert np.all(arrays["commands"][-11:] == 0)
    for frame in range(trial["steps"]):
        np.testing.assert_array_equal(
            arrays["commands"][frame],
            goal_command(arrays["estimated"][frame], trial["goal"], config)["wheels"],
        )
    np.testing.assert_array_equal(
        goal_command([0, 0, 0], [1, 0], config)["wheels"], goal_command([0, 0, 0], [1, 0])["wheels"]
    )


def test_restarts_and_dwell_exclude_terminal_zero():
    modes = np.array([0, 2, 2, 0, 2, 1, 2, 3])
    commands = np.array([[1, 1] if m in (0, 1) else [0, 0] for m in modes])
    arrays = {"modes": modes, "commands": commands}
    metrics = stopping_metrics(arrays)
    assert metrics["settling_attempts"] == 3
    assert metrics["restart_count"] == 2
    assert metrics["moving_duration_s"] == 3 * DT
    assert metrics["first_stop_attempt_s"] == DT
    np.testing.assert_array_equal(restart_frames(arrays), [3, 5])
    assert dwell_seconds(arrays, 1) == 0
    assert dwell_seconds(arrays, 2) == DT
    assert dwell_seconds(arrays, 3) == 0
    assert dwell_seconds(arrays, 7) == DT
    arrays = {"modes": np.array([0, 0, 4]), "commands": np.array([[1, 1], [1, 1], [0, 0]])}
    assert stopping_metrics(arrays)["first_stop_attempt_s"] is None
    assert stopping_metrics(arrays)["settling_attempts"] == 0
    assert dwell_seconds(arrays, 2) == 0


def test_roundtrip_only_one_parameter_and_baseline_exact(recording):
    report, records = load_thresholds(recording)
    assert len(records) == 24 and len(report["pairs"]) == 16 and len(report["rows"]) == 12
    assert compare_baseline(records, recording / "cm2")["exact_array_match_trials"] == 8
    for key, radius, _, _ in VARIANTS:
        child, _ = load_recording(recording / key)
        assert child["controller"] == asdict(
            replace(DEFAULT_CONFIG, estimated_stop_radius_m=radius)
        )
    for identity, (arrays, trial) in records.items():
        if identity[0] != "cm2":
            continue
        reproduced, metric = simulate(trial["goal"], trial["method"], trial["seed"])
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], reproduced[name])
        assert trial["true_success"] == metric["true_success"]
    with pytest.raises(FileExistsError):
        run_thresholds(recording)


@pytest.mark.parametrize("method", ["odom", "fused"])
def test_noise_and_motion_are_identical_until_tolerances_affect_action(recording, method):
    _, records = load_thresholds(recording)
    a = records[("cm2", "near", 0, method)][0]
    first_stop = int(np.flatnonzero(a["modes"] == 2)[0])
    for variant in ("cm1", "cm05"):
        b = records[(variant, "near", 0, method)][0]
        n = min(len(a["encoder_noise"]), len(b["encoder_noise"]))
        np.testing.assert_array_equal(a["encoder_noise"][:n], b["encoder_noise"][:n])
        for field in ("truth", "estimated", "prior", "encoders"):
            np.testing.assert_array_equal(a[field][: first_stop + 1], b[field][: first_stop + 1])
        np.testing.assert_array_equal(a["commands"][:first_stop], b["commands"][:first_stop])
        assert np.any(b["commands"][first_stop] != 0)


def test_timeout_inside_true_target_is_not_success(recording):
    _, records = load_thresholds(recording)
    arrays, trial = records[("cm05", "far", 1, "fused")]
    assert trial["true_final_distance_m"] < 0.03
    assert trial["duration_s"] == 40 and trial["terminal_reason"] == "timeout"
    assert not trial["true_success"] and not trial["false_arrival"]
    np.testing.assert_array_equal(arrays["commands"][-1], [0, 0])


@pytest.mark.parametrize(
    "damage", ["metrics", "checksum", "extra_parameter", "seed", "missing_variant"]
)
def test_loader_rejects_inconsistent_records(recording, tmp_path, damage):
    import shutil

    path = tmp_path / "damaged"
    shutil.copytree(recording, path)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if damage == "metrics":
        summary["rows"][0]["true_success_count"] += 1
    elif damage == "checksum":
        summary["variants"][0]["summary_sha256"] = "bad"
    elif damage == "missing_variant":
        summary["variants"].pop()
    else:
        child_path = path / "cm1" / "summary.json"
        child = json.loads(child_path.read_text(encoding="utf-8"))
        if damage == "extra_parameter":
            child["controller"]["distance_gain"] *= 2
        else:
            child["seed"] += 1
        child_path.write_text(json.dumps(child), encoding="utf-8")
        summary["variants"][1]["summary_sha256"] = digest(child_path)
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        load_thresholds(path)


@pytest.mark.isolated_tk
def test_replay_motion_tolerances_timer_selection_and_timeout(recording):
    import tkinter as tk

    report, records = load_thresholds(recording)
    root = tk.Tk()
    root.withdraw()
    demo = ThresholdDemo(root, report, records)
    demo.canvas.winfo_width = lambda: 700
    demo.canvas.winfo_height = lambda: 550
    try:
        root.update()
        demo.redraw()
        assert demo.clock.paused and demo.clock.speed == 0.25
        initial = demo.canvas.coords("cm2_truth_centre")
        demo.seek(30)
        assert demo.canvas.coords("cm2_truth_centre") != initial
        demo.approach()
        assert demo.zoom.get() and demo.clock.paused and demo.clock.index > 0
        assert demo.canvas.bbox("axis_label")[3] < demo.canvas.bbox("map_caption")[1]
        assert demo.canvas.bbox("axis_label")[1] > demo.canvas.bbox("map_header")[3]
        diameters = []
        for variant, *_ in VARIANTS:
            x1, _, x2, _ = demo.canvas.coords(f"{variant}_tolerance")
            diameters.append(x2 - x1)
        np.testing.assert_allclose(np.array(diameters) / diameters[0], [1, 0.5, 0.25])
        demo.next_restart()
        assert "计时清零" in demo.event.cget("text")
        demo.finish()
        assert demo.clock.index == demo.clock.steps
        frozen = demo.canvas.coords("cm2_truth_centre")
        demo.seek(demo.current("cm2")[1]["steps"])
        assert demo.canvas.coords("cm2_truth_centre") == frozen
        demo.example_button.invoke()
        assert (demo.case, demo.method, demo.run) == ("far", "fused", 1)
        demo.finish()
        assert "超时" in demo.cards["cm05"].cget("text")
        assert "40.00 s" in demo.cards["cm05"].cget("text")
        demo.estimates.set(False)
        demo.redraw()
        assert not demo.canvas.find_withtag("cm05_estimated_centre")
        demo.method_box.current(0)
        demo.select()
        assert demo.method == "odom" and demo.clock.index == 0
        demo.restart()
        demo.toggle()
        demo.clock.advance(0.16)
        assert demo.clock.index == 1
        demo.toggle()
        demo.step()
        assert demo.clock.index == 2
    finally:
        demo.close()
