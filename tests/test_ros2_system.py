"""Portable contracts; real WSL/ROS receipts are verified by the integration runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from embodied_learning.differential_drive import SENSOR_IN_BODY, compose
from embodied_learning.experiments.landmark_fusion import run_experiment, run_runs
from embodied_learning.experiments.ros2_system import prepare, wsl_path
from embodied_learning.landmark_localization import LANDMARKS, observe
from embodied_learning.odometry import heading_error
from embodied_learning.ros2_motion_view import motion_snapshot
from embodied_learning.ros2_stream import StreamingFusion, frame_stamp, stamp_frame
from embodied_learning.ros2_system_demo import RosSystemDemo, load_trace


@pytest.mark.parametrize("frame", [0, 1, 24, 25, 50, 300, 600, 800])
def test_integer_timestamp_roundtrip(frame):
    sec, nanosec = frame_stamp(frame)
    assert stamp_frame(sec, nanosec) == frame
    assert sec > 0


@pytest.mark.parametrize("args", [(0, 0), (1, -1), (1, 1), (1, 1_000_000_000)])
def test_reject_non_grid_time(args):
    with pytest.raises(ValueError):
        stamp_frame(*args)


@pytest.mark.parametrize("key", ["straight", "square", "long"])
@pytest.mark.parametrize("observations_first", [False, True])
def test_stream_matches_batch_in_both_arrival_orders(key, observations_first):
    case = run_runs(key, 2, 0)
    frames = case["observation_frames"]
    stream, received = StreamingFusion(), []
    for frame, angles in enumerate(case["encoders"][0]):
        obs = np.flatnonzero(frames == frame)
        z = case["observations"][0, int(obs[0])] if len(obs) else None
        if z is not None and observations_first:
            received.extend(stream.add_observation(frame, z))
        received.extend(stream.add_encoders(frame, angles))
        if z is not None and not observations_first:
            # No premature position output before the matching observation arrives.
            assert received[-1]["frame"] == frame - 1
            received.extend(stream.add_observation(frame, z))
    assert [row["frame"] for row in received] == list(range(len(case["truth"])))
    np.testing.assert_allclose([r["odom"] for r in received], case["odom"][0], atol=1e-12)
    np.testing.assert_allclose([r["fused"] for r in received], case["fused"][0], atol=1e-12)
    for row in received:
        actual = compose(row["map_to_odom"], row["odom"])
        np.testing.assert_allclose(actual[:2], row["fused"][:2], atol=1e-12)
        assert abs(heading_error(actual[2], row["fused"][2])) < 1e-12
    assert not stream.encoders and not stream.observations


def test_stream_buffers_future_encoders_and_missing_observation():
    stream = StreamingFusion()
    assert stream.add_encoders(1, [0.2, 0.2]) == []
    assert [row["frame"] for row in stream.add_encoders(0, [0, 0])] == [0, 1]
    for frame in range(2, 50):
        stream.add_encoders(frame, [frame * 0.2, frame * 0.2])
    assert stream.add_encoders(50, [10, 10]) == []
    assert stream.add_encoders(51, [10.2, 10.2]) == []
    z = observe([0.5, 0, 0], LANDMARKS, np.random.default_rng(1), 0, 0)
    rows = stream.add_observation(50, z)
    assert [row["frame"] for row in rows] == [50, 51]
    np.testing.assert_allclose(rows[0]["fused"], [0.5, 0, 0], atol=1e-12)


def test_stream_rejects_duplicates_stale_nonfinite_and_unbounded_future():
    stream = StreamingFusion()
    stream.add_encoders(0, [0, 0])
    for frame, angles in (
        (0, [0, 0]),
        (-1, [0, 0]),
        (500, [0, 0]),
        (1.2, [0, 0]),
        (1, [0, np.nan]),
    ):
        with pytest.raises(ValueError):
            stream.add_encoders(frame, angles)
    with pytest.raises(ValueError, match="schedule"):
        stream.add_observation(3, [[1, 0]] * 3)


def test_math_core_import_does_not_load_gui_mujoco_or_ros():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import embodied_learning.ros2_stream,sys; "
                "assert not ({'mujoco','gymnasium','tkinter','matplotlib','rclpy'} & set(sys.modules))"
            ),
        ],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_windows_path_conversion_keeps_unicode_and_spaces():
    assert wsl_path("D:/项目/具身 人工智能") == "/mnt/d/项目/具身 人工智能"
    with pytest.raises(ValueError):
        wsl_path("relative/path")
    with pytest.raises(ValueError):
        wsl_path("//server/share/file")


@pytest.fixture(scope="module")
def source_recording(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ros_source") / "source"
    run_experiment(directory, runs=2, seed=0)
    return directory


def test_export_separates_measurements_from_reference(tmp_path, source_recording):
    output = tmp_path / "prepared"
    manifest = prepare(output, source_recording)
    with np.load(output / "sensor_input.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"encoders", "observations", "observation_frames"}
        assert archive["encoders"].shape == (601, 2)
    assert manifest["steps"] == 600 and len(manifest["observation_frames"]) == 12
    with pytest.raises(FileExistsError):
        prepare(output, source_recording)
    with pytest.raises(ValueError):
        prepare(tmp_path / "invalid", source_recording, run_index=-1)
    assert not (tmp_path / "invalid").exists()


@pytest.fixture
def synthetic_trace(tmp_path):
    """UI-only fixture, labelled synthetic; NOT evidence of ROS communication."""
    import hashlib

    case = run_runs("straight", 2, 0)
    stream, rows = StreamingFusion(), []
    for frame in range(101):
        if frame == 50 or frame == 100:
            stream.add_observation(frame, case["observations"][0, frame // 50 - 1])
        update = stream.add_encoders(frame, case["encoders"][0, frame])[0]
        sec, ns = frame_stamp(frame)
        rows.append(
            {
                "frame": frame,
                "time_s": frame * 0.04,
                "stamp_sec": sec,
                "stamp_nanosec": ns,
                "observation": frame in [50, 100],
                "truth": case["truth"][frame].tolist(),
                "odom": update["odom"].tolist(),
                "fused": update["fused"].tolist(),
                "map_to_odom": update["map_to_odom"].tolist(),
                "map_to_sensor": compose(update["fused"], SENSOR_IN_BODY).tolist(),
                "received_counts": {
                    "encoders": frame + 1,
                    "landmarks": frame // 50,
                    "odom": frame + 1,
                    "fused": frame + 1,
                },
            }
        )
    trace = tmp_path / "ros_trace.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = {
        "experiment": "ros2_message_and_tf_bridge",
        "schema_version": 1,
        "steps": 100,
        "dt_s": 0.04,
        "observation_frames": [50, 100],
        "route": "synthetic_ui_fixture",
        "run_index": 0,
        "process_ids": {"sensor": 1, "localizer": 2, "inspector": 3},
        "reference_max_abs_difference": 0.0,
        "received_counts": rows[-1]["received_counts"],
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
    }
    (tmp_path / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    return tmp_path


def test_trace_rejects_corruption(synthetic_trace):
    report, rows = load_trace(synthetic_trace)
    assert len(rows) == 101 and report["steps"] == 100
    with (synthetic_trace / "ros_trace.jsonl").open("a", encoding="utf-8") as trace:
        trace.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_trace(synthetic_trace)


@pytest.mark.isolated_tk
def test_ros_teaching_window_counts_stamps_tf_and_playback(synthetic_trace):
    import tkinter as tk

    report, rows = load_trace(synthetic_trace)
    root = tk.Tk()
    root.withdraw()
    demo = RosSystemDemo(root, report, rows)
    demo.canvas.winfo_width = lambda: 1100
    try:
        root.update()
        assert demo.clock.paused and demo.clock.speed == 0.25
        demo.before_button.invoke()
        assert demo.clock.index == 49
        demo.next_button.invoke()
        assert demo.clock.index == 50
        assert "观测到齐" in demo.state_text.cget("text")
        table = [
            demo.message_table.item(key)["values"] for key in demo.message_table.get_children()
        ]
        assert table[1][-1] == 1
        assert float(table[1][-2]) == 3.0
        demo.notebook.select(2)
        assert len(demo.tf_table.get_children()) == 4
        demo.step()
        assert demo.clock.index == 51
        assert "只有本帧编码器" in demo.state_text.cget("text")
        demo.toggle()
        demo.clock.advance(0.16)
        assert demo.clock.index == 52
        demo.toggle()
        demo.seek(100)
        assert demo.clock.paused
    finally:
        demo.close()


@pytest.mark.parametrize("key", ["straight", "square", "long"])
def test_motion_display_recovers_same_frame_prior_without_truth_or_future(key):
    case = run_runs(key, 2, 0)
    observed = set(case["observation_frames"])
    rows = [
        {
            "truth": truth.tolist(),
            "odom": case["odom"][0, i].tolist(),
            "fused": case["fused"][0, i].tolist(),
            "observation": i in observed,
        }
        for i, truth in enumerate(case["truth"])
    ]
    frozen = json.dumps(rows)
    for i in observed:
        before, after = motion_snapshot(rows, i, True), motion_snapshot(rows, i)
        np.testing.assert_allclose(before["prior"][:2], case["prior"][0, i, :2], atol=1e-12)
        assert abs(heading_error(before["prior"][2], case["prior"][0, i, 2])) < 1e-12
        np.testing.assert_array_equal(before["truth"], after["truth"])
        np.testing.assert_array_equal(after["fused"], case["fused"][0, i])
        # A changed truth or later measurement cannot change the estimated prior.
        altered = json.loads(frozen)
        altered[i]["truth"] = [999, -999, 2]
        altered = altered[: i + 1]
        np.testing.assert_array_equal(motion_snapshot(altered, i)["prior"], before["prior"])
    assert json.dumps(rows) == frozen


def test_motion_labels_straight_and_in_place_turn_from_recorded_truth():
    case = run_runs("square", 2, 0)
    rows = [
        {
            "truth": truth,
            "odom": case["odom"][0, i],
            "fused": case["fused"][0, i],
            "observation": i in case["observation_frames"],
        }
        for i, truth in enumerate(case["truth"])
    ]
    assert "起点" in motion_snapshot(rows, 0)["action"]
    assert "直行" in motion_snapshot(rows, 100)["action"]
    assert "原地左转" in motion_snapshot(rows, 101)["action"]
    np.testing.assert_array_equal(rows[100]["truth"][:2], rows[101]["truth"][:2])
    assert "不是反馈导航" in motion_snapshot(rows, 600)["action"]


@pytest.mark.isolated_tk
def test_motion_canvas_same_time_correction_is_not_robot_teleport(synthetic_trace):
    import tkinter as tk

    report, rows = load_trace(synthetic_trace)
    frozen = json.dumps(rows)
    root = tk.Tk()
    root.withdraw()
    demo = RosSystemDemo(root, report, rows)
    for canvas, width in ((demo.world_canvas, 640), (demo.zoom_canvas, 450)):
        canvas.winfo_width = lambda width=width: width
        canvas.winfo_height = lambda: 400
    try:
        root.update()
        demo.redraw()
        assert demo.notebook.index(demo.notebook.select()) == 0
        assert demo.turn_button.instate(["disabled"])  # This fixture is a straight route.
        initial_body = demo.world_canvas.coords("true_body")
        assert len(demo.world_canvas.find_withtag("true_wheel")) == 2
        assert len(demo.world_canvas.find_withtag("landmark")) == 3
        assert not demo.world_canvas.find_withtag("observation_ray")
        demo.seek(49)
        assert demo.world_canvas.coords("true_body") != initial_body
        assert demo.compare_button.instate(["disabled"])
        demo.next_button.invoke()
        assert demo.clock.index == 50
        assert len(demo.world_canvas.find_withtag("observation_ray")) == 3
        body = demo.world_canvas.coords("true_body")
        arrow = demo.zoom_canvas.coords("correction_arrow")
        posterior = demo.zoom_canvas.coords("fused_marker")
        demo.compare_button.invoke()
        assert demo.clock.index == 50 and demo.clock.paused and demo.before_correction
        assert demo.world_canvas.coords("true_body") == body
        assert demo.zoom_canvas.coords("correction_arrow") == arrow
        assert demo.zoom_canvas.coords("fused_marker") != posterior
        assert "校正前预测" in demo.event_label.cget("text")
        demo.compare_button.invoke()
        assert demo.zoom_canvas.coords("fused_marker") == posterior
        demo.step()
        assert demo.clock.index == 51 and not demo.before_correction
        assert not demo.zoom_canvas.find_withtag("correction_arrow")
        assert not demo.world_canvas.find_withtag("observation_ray")
        assert json.dumps(rows) == frozen
    finally:
        demo.close()
