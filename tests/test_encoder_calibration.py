import hashlib
import json
import shutil

import numpy as np
import pytest

from embodied_learning.calibration_demo import (
    TEACHING_LABELS,
    CalibrationDemo,
    load_replays,
    measurement_columns,
)
from embodied_learning.encoder_calibration import correct_right_encoder, fit_right_correction
from embodied_learning.experiments.encoder_calibration import (
    calibration_measurements,
    fit_methods,
    run_experiment,
)
from embodied_learning.experiments.mobile_odometry import run_experiment as run_odometry
from embodied_learning.odometry import estimate_poses
from embodied_learning.odometry_demo import OdometryDemo


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.02, 1.3])
def test_fit_recovers_unknown_scale_from_separate_signed_measurements(scale):
    rows = calibration_measurements(scale)
    factors = fit_methods(rows)
    assert factors[1] == pytest.approx(1 / scale, abs=1e-13)
    assert factors[2] == pytest.approx(1.01 / scale, abs=1e-13)
    assert [row["steps"] for row in rows] == [50, 100, 150, 75]
    assert rows[-1]["external_distance_m"] < 0


def test_fit_is_origin_least_squares_not_mean_of_ratios():
    # x = radius * measured_angle = [1,2], distances = [1,3].
    assert fit_right_correction([10, 20], [1, 3], 0.1) == pytest.approx(7 / 5)
    assert fit_right_correction([-10, -20], [-1, -3], 0.1) == pytest.approx(7 / 5)
    assert fit_right_correction([1e-100, 2e-100], [1e-101, 3e-101], 0.1) == pytest.approx(7 / 5)


@pytest.mark.parametrize(
    "angles, distances, radius",
    [
        ([], [], 0.05),
        ([0], [0], 0.05),
        ([1], [-1], 0.05),
        ([1], [0], 0.05),
        ([1, 2], [1], 0.05),
        ([[1]], [1], 0.05),
        ([np.nan], [1], 0.05),
        ([1], [np.inf], 0.05),
        ([1], [1], 0),
        ([1], [1], -0.05),
        ([1], [1], np.nan),
    ],
)
def test_invalid_calibration_rejected(angles, distances, radius):
    with pytest.raises(ValueError):
        fit_right_correction(angles, distances, radius)


def test_correction_preserves_zero_points_left_wheel_input_and_causality():
    raw = np.array([[10, 20], [14, 24.08], [10, 20], [6, 15.92]])
    original = raw.copy()
    fixed = correct_right_encoder(raw, 1 / 1.02)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(fixed[:, 0], raw[:, 0])
    np.testing.assert_allclose(fixed[:, 1], [20, 24, 20, 16])
    np.testing.assert_array_equal(correct_right_encoder(raw[:2], 1 / 1.02), fixed[:2])
    np.testing.assert_allclose(estimate_poses(fixed)[-1], [-0.2, 0, 0], atol=1e-14)
    np.testing.assert_array_equal(correct_right_encoder(raw[:1], 0.5), raw[:1])


@pytest.mark.parametrize("factor", [0, -1, np.inf, np.nan])
def test_invalid_correction_rejected(factor):
    with pytest.raises(ValueError):
        correct_right_encoder([[0, 0]], factor)


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    base = tmp_path_factory.mktemp("calibration")
    source, output = base / "source", base / "result"
    run_odometry(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    report = run_experiment(source, output)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    return source, output, report


def test_frozen_holdouts_raw_identity_corrected_errors_and_reference_limit(recording):
    source, output, report = recording
    replays, _ = load_replays(output)
    with np.load(source / "trajectories.npz") as baseline:
        for replay, case in zip(replays, report["cases"]):
            key, arrays = case["key"], replay.arrays
            for name in ("true_poses", "wheels_rad_s", "wheel_angles_rad", "landmark_sensor"):
                np.testing.assert_array_equal(arrays[name], baseline[f"{key}_{name}"])
            for name in ("poses", "encoder_angles_rad", "position_error_m", "heading_error_rad"):
                np.testing.assert_array_equal(
                    arrays[f"raw_{name}"], baseline[f"{key}_right_2pct_{name}"]
                )
                np.testing.assert_allclose(
                    arrays[f"reference_bias_{name}"],
                    baseline[f"{key}_right_1pct_{name}"],
                    atol=1e-12,
                )
            assert arrays["calibrated_position_error_m"].max() < 1e-10
            assert np.abs(arrays["calibrated_heading_error_rad"]).max() < 1e-10
            assert arrays["reference_bias_position_error_m"][-1] > 0.07
            assert all(not a.flags.writeable for a in arrays.values())
    with pytest.raises(FileExistsError):
        run_experiment(source, output)
    assert (
        report["holdout_source"]["trajectories_sha256"]
        == hashlib.sha256((source / "trajectories.npz").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("damage", ["checksum", "fit", "factor", "readings", "pose"])
def test_calibration_replay_rejects_corruption(recording, tmp_path, damage):
    _, original, _ = recording
    output = tmp_path / "damaged"
    shutil.copytree(original, output)
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    cal = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    if damage in ("checksum", "fit"):
        cal["runs"][0]["external_distance_m"] += 0.01
        (output / "calibration.json").write_text(json.dumps(cal), encoding="utf-8")
        if damage == "fit":
            report["calibration_sha256"] = hashlib.sha256(
                (output / "calibration.json").read_bytes()
            ).hexdigest()
    elif damage == "factor":
        report["cases"][0]["estimates"][1]["correction_factor"] = 1
    else:
        path = output / "trajectories.npz"
        with np.load(path) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        name = "encoder_angles_rad" if damage == "readings" else "poses"
        arrays[f"straight_calibrated_{name}"][1, 0] += 1
        np.savez_compressed(path, **arrays)
        report["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        load_replays(output)


@pytest.mark.isolated_tk
def test_calibration_slow_replay_keeps_timestamp_and_draws_all_methods(recording):
    import tkinter as tk

    _, output, _ = recording
    replays, variants = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = OdometryDemo(root, replays, variants=variants, calibration=True)
    demo.canvas.winfo_width, demo.canvas.winfo_height = lambda: 650, lambda: 330
    demo.chart.winfo_width, demo.chart.winfo_height = lambda: 1000, lambda: 125
    try:
        root.update()
        assert demo.clock.paused and demo.clock.speed == 0.25
        assert "第十六课" in root.title()
        assert demo.variant.get() == "独立标定"
        demo.step()
        assert "+0.1632 → +0.1600" in demo.stats.cget("text")
        demo.seek(300)
        assert "位置误差 0.00 cm" in demo.stats.cget("text")
        demo.variant.set("未标定")
        demo.select_variant()
        assert demo.clock.index == 300 and demo.clock.paused
        assert "位置误差 19.40 cm" in demo.stats.cget("text")
        demo.variant.set("测距偏大1%")
        demo.select_variant()
        assert "位置误差 9.69 cm" in demo.stats.cget("text")
        demo.choice.set(replays[1].metadata["label"])
        demo.select_case()
        assert demo.clock.index == 0 and demo.clock.steps == 600
        demo.seek(600)
        assert "位置误差 7.72 cm" in demo.stats.cget("text")
        assert "没有下一步轮速" in demo.stats.cget("text")
    finally:
        demo.close()


def test_visible_measurements_only_change_external_reference():
    rows = calibration_measurements(1.02)
    original = json.dumps(rows)
    exact = measurement_columns(rows, 1.0)
    biased = measurement_columns(rows, 1.01)
    np.testing.assert_array_equal(exact[:, :2], biased[:, :2])
    np.testing.assert_allclose(biased[:, 2], exact[:, 2] * 1.01)
    np.testing.assert_allclose(exact[0], [8.16, 0.408, 0.4])
    assert json.dumps(rows) == original
    with pytest.raises(ValueError):
        measurement_columns(rows, 1.1)


@pytest.mark.isolated_tk
def test_guided_calibration_measure_fit_validate_and_optional_landmark(recording):
    import tkinter as tk

    _, output, _ = recording
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    root = tk.Tk()
    root.withdraw()
    demo = CalibrationDemo(root, output)
    replay = demo.replay
    replay.canvas.winfo_width, replay.canvas.winfo_height = lambda: 650, lambda: 330
    replay.chart.winfo_width, replay.chart.winfo_height = lambda: 1000, lambda: 125
    try:
        root.update()
        assert "看测量" in root.title()
        assert demo.notebook.index(demo.notebook.select()) == 0
        assert demo.notebook.tab(2, "state") == "disabled"
        assert demo.fitted is None and demo.validation_button.instate(["disabled"])
        demo.begin_validation()
        assert demo.notebook.index(demo.notebook.select()) == 0
        assert len(demo.table.get_children()) == 4
        assert "+0.4000" in demo.measurement_detail.get()
        assert "+0.4080" in demo.measurement_detail.get()
        demo.table.selection_set("3")
        demo.describe_measurement()
        assert "-0.6000" in demo.measurement_detail.get()
        assert "-0.6120" in demo.measurement_detail.get()
        demo.notebook.select(1)
        root.update()
        demo.calculate()
        assert demo.fitted == pytest.approx(1 / 1.02)
        assert "0.4000 ÷ 0.4080" in demo.formula.get()
        assert "0.980392157" in demo.formula.get()
        assert demo.notebook.tab(2, "state") == "normal"
        demo.begin_validation()
        root.update()
        assert demo.notebook.index(demo.notebook.select()) == 2
        assert replay.variant.get() == TEACHING_LABELS[1]
        assert not replay.show_landmark.get()
        replay.seek(300)
        assert "位置误差 0.00 cm" in replay.stats.cget("text")
        assert "落图" not in replay.stats.cget("text")
        assert not replay.canvas.find_withtag("landmark")
        assert len(replay.metric_box.cget("values")) == 2
        replay.variant.set(TEACHING_LABELS[0])
        replay.select_variant()
        assert replay.clock.index == 300
        assert "位置误差 19.40 cm" in replay.stats.cget("text")
        poses_before = replay.replay.arrays["raw_poses"].copy()
        replay.show_landmark.set(True)
        replay.toggle_landmark()
        assert len(replay.canvas.find_withtag("landmark")) == 6
        assert "灯杆没动" in replay.stats.cget("text")
        assert len(replay.metric_box.cget("values")) == 3
        replay.metric.set("地标落图误差 / cm")
        replay.redraw()
        replay.show_landmark.set(False)
        replay.toggle_landmark()
        assert replay.metric.get() == "位置误差 / cm"
        assert not replay.canvas.find_withtag("landmark")
        np.testing.assert_array_equal(replay.replay.arrays["raw_poses"], poses_before)
        replay.seek(100)
        replay.toggle()
        demo.notebook.select(0)
        root.update()
        assert replay.clock.paused
        index = replay.clock.index
        demo.toggle_playback()
        assert replay.clock.paused  # Space cannot start an invisible replay.
        demo.reference_multiplier.set(1.01)
        demo.change_instrument()
        assert demo.fitted is None and demo.notebook.tab(2, "state") == "disabled"
        assert demo.validation_button.instate(["disabled"])
        assert "+0.4040" in demo.measurement_detail.get()
        assert "+0.4080" in demo.measurement_detail.get()
        demo.calculate()
        assert demo.fitted == pytest.approx(1.01 / 1.02)
        assert replay.variant.get() == TEACHING_LABELS[2]  # Direct tab entry also uses the fit.
        assert "偏大的尺子" in demo.fit_status.get()
        demo.begin_validation()
        root.update()
        assert replay.clock.index == index and replay.clock.paused
        replay.seek(300)
        assert "位置误差 9.69 cm" in replay.stats.cget("text")
        replay.choice.set(replay.replays[1].metadata["label"])
        replay.select_case()
        assert replay.clock.index == 0 and replay.clock.steps == 600
    finally:
        demo.close()
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
