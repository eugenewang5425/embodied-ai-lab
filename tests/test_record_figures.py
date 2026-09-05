"""Agg smoke tests: lesson-20/21 figure plates rebuilt from frozen records.

The plates must render headlessly from the formal record directories without
writing anywhere under results/, and every number printed on a plate must be
derivable from the record files these tests read directly.
"""

from __future__ import annotations

import os

os.environ["MPLBACKEND"] = "Agg"

import json
import struct
from pathlib import Path

import pytest

from embodied_learning.experiments import record_figures as rf
from embodied_learning.experiments.goal_reaching import METHODS

ROOT = Path(__file__).resolve().parents[1]
ROS2_RECORD = ROOT / "results" / "ros2_system_2026-09-03_v2"
GOAL_RECORD = ROOT / "results" / "goal_reaching_2026-09-03"
THRESHOLD_RECORD = ROOT / "results" / "goal_thresholds_2026-09-03"

needs_ros2 = pytest.mark.skipif(not ROS2_RECORD.is_dir(), reason="frozen ros2 record missing")
needs_goal = pytest.mark.skipif(not GOAL_RECORD.is_dir(), reason="frozen goal record missing")
needs_threshold = pytest.mark.skipif(
    not THRESHOLD_RECORD.is_dir(), reason="frozen threshold record missing"
)

METHOD_LABELS = {key: label for key, label, _ in METHODS}


def png_pixels(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"
    return struct.unpack(">II", data[16:24])


@needs_ros2
def test_ros2_plate_renders_to_tmp(tmp_path):
    output = tmp_path / "lesson-20-ros2-timeline.png"
    rf.render_ros2(ROS2_RECORD, output)
    assert output.is_file() and output.stat().st_size > 10_000
    assert png_pixels(output)[0] == rf.OUTPUT_WIDTH_PX
    assert not list(tmp_path.glob("*-full.png")), "scratch render file left behind"


@needs_goal
def test_goal_plate_renders_to_tmp(tmp_path):
    output = tmp_path / "lesson-21-goal-outcomes.png"
    rf.render_goal(GOAL_RECORD, output)
    assert output.is_file() and output.stat().st_size > 10_000
    assert png_pixels(output)[0] == rf.OUTPUT_WIDTH_PX


@needs_threshold
def test_threshold_plate_renders_to_tmp(tmp_path):
    output = tmp_path / "lesson-21a-thresholds.png"
    rf.render_threshold(THRESHOLD_RECORD, output)
    assert output.is_file() and output.stat().st_size > 10_000
    assert png_pixels(output)[0] == rf.OUTPUT_WIDTH_PX


@needs_ros2
def test_ros2_panel_numbers_match_summary():
    summary = json.loads((ROS2_RECORD / "summary.json").read_text(encoding="utf-8"))
    counts = summary["received_counts"]
    counts_text = rf.ros2_total_counts_text(summary)
    for key in ("encoders", "landmarks", "odom", "fused"):
        assert str(counts[key]) in counts_text
    assert counts == {"encoders": 601, "landmarks": 12, "odom": 601, "fused": 601}
    assert rf.format_sci(summary["reference_max_abs_difference"]) == "8.88e-16"
    assert rf.format_sci(summary["tf_chain_max_abs_difference"]) == "1.53e-15"
    _, trace, _ = rf.load_ros2_record(ROS2_RECORD)
    assert rf.observation_times(trace) == [2.0 * k for k in range(1, 13)]
    assert trace[49]["received_counts"]["landmarks"] == 0
    assert trace[50]["received_counts"]["landmarks"] == 1
    assert trace[50]["stamp_sec"] == 3


@needs_goal
def test_goal_panel_titles_match_summary():
    summary = json.loads((GOAL_RECORD / "summary.json").read_text(encoding="utf-8"))
    by_case = {case["case"]: case["methods"] for case in summary["comparisons"]}
    assert by_case["near"]["odom"]["true_success_count"] == 16
    assert by_case["near"]["fused"]["true_success_count"] == 20
    assert by_case["far"]["odom"]["true_success_count"] == 5
    assert by_case["far"]["fused"]["true_success_count"] == 11
    for case in summary["comparisons"]:
        for method, stats in case["methods"].items():
            title = rf.goal_panel_title(
                case["label"], METHOD_LABELS[method], stats, summary["runs"]
            )
            assert f"通过 {stats['true_success_count']}/{summary['runs']}" in title
            assert f"误判到达 {stats['false_arrival_count']}/{summary['runs']}" in title
            assert f"{stats['mean_true_final_distance_m'] * 100:.3f} cm" in title


@needs_threshold
def test_threshold_annotations_match_summary():
    summary = json.loads((THRESHOLD_RECORD / "summary.json").read_text(encoding="utf-8"))
    rows = {(row["case"], row["method"], row["variant"]): row for row in summary["rows"]}
    assert rf.threshold_bar_label(rows[("far", "fused", "cm2")]) == "11/9/0"
    assert rf.threshold_bar_label(rows[("far", "fused", "cm1")]) == "13/3/4"
    assert rf.threshold_bar_label(rows[("far", "fused", "cm05")]) == "7/0/13"
    assert rf.threshold_bar_label(rows[("far", "odom", "cm05")]) == "5/15/0"
    trials = {}
    for variant in ("cm2", "cm1", "cm05"):
        child = json.loads(
            (THRESHOLD_RECORD / variant / "summary.json").read_text(encoding="utf-8")
        )
        trials[variant] = next(
            trial
            for trial in child["trials"]
            if (trial["case"], trial["method"], trial["run"]) == ("far", "fused", 1)
        )
    lines = rf.threshold_sample_lines(trials)
    joined = "\n".join(lines)
    for expected in ("3.856", "0.994", "1.923", "2.675"):
        assert expected in joined, f"missing {expected} in sample lines: {joined}"
    assert "超时" in lines[-1] and "通过" in lines[1] and "停偏" in lines[0]
