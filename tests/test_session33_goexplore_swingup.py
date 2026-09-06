"""Lesson 33: Go-Explore archive exploration and BC robustification.

Unit checks pin the cell discretization against known answers, the set_state
return privilege bit for bit (round trip and stitched-slice replay), the
archive member criterion and deterministic cell selection, the capture
criterion against the reused lesson-7 recovery_metrics, and the re-exported
lesson-32 BC loss against a hand-computed answer; shrunk end-to-end checks
cover the record contract, the bitwise teacher-path replay, the honest
phase-2-null path, the CLI, the demo loader's tamper rejections and the real
Tk demo. No full-budget exploration is run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

# The isolated_tk child re-imports this module before any pyplot use: forcing
# the Agg backend keeps run_experiment's figure saving from creating (and
# destroying) a first Tcl interpreter, whose leftovers make the demo test's
# real tk.Tk() die with "invalid command name tcl_findLibrary" on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from embodied_learning.experiments.goexplore_swingup import (
    ANGLE_BINS,
    CART_BINS,
    EXPERIMENT,
    MAX_SEGMENTS,
    SEED_OFFSET_ACT,
    SEGMENT_STEPS,
    Archive,
    balance_engaged,
    bc_loss_and_gradient,
    build_teacher_pairs,
    capture_slice_metrics,
    cell_centers,
    expected_npz_keys,
    exploration_action,
    find_capture_start,
    normalized_settled_error,
    return_fidelity_guard,
    run_experiment,
    run_exploration,
    state_cell,
    train_robustified_bc,
)
from embodied_learning.experiments.ppo_swingup import (
    GaussianPolicy,
    RewardFunction,
    down_start_state,
    normalize_observation,
)
from embodied_learning.experiments.swingup_comparison import (
    SETTLED_TOLERANCES,
    recovery_metrics,
)
from embodied_learning.goexplore_demo import load_replays
from embodied_learning.swingup import design_swingup_lqr, make_swingup_environment

SMALL_CONFIG = {
    "segments": 2600,  # seed-0 phase 1 captures the stable band around segment 2200
    "segment_steps": 150,
    "eval_seed_count": 2,
    "bc_epochs": 5,
    "captures_kept": 2,
}


@pytest.fixture(scope="module")
def design_and_reward():
    design = design_swingup_lqr()
    reward = RewardFunction(design.controller.reference)
    return design, reward


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment (with a capture) shared by record checks."""
    output = tmp_path_factory.mktemp("goexplore_small") / "run"
    report = run_experiment(output, seed=0, **SMALL_CONFIG)
    return output, report


# --------------------------------------------------------------- the archive
def test_state_cell_known_answers():
    """Angle/cart binning: upright, the seams, multi-turn theta, edge clamping."""
    reference = np.zeros(4)
    assert state_cell([0.0, 0.0, 0.0, 0.0], reference) == (6, 3)
    assert state_cell([0.0, np.pi - 1e-9, 0.0, 0.0], reference) == (11, 3)
    assert state_cell([0.0, -np.pi, 0.0, 0.0], reference) == (0, 3)
    # a multi-turn raw theta must land in the same cell as its wrapped alpha
    assert state_cell([0.0, 4.0 * np.pi + 0.1, 0.0, 0.0], reference) == (
        state_cell([0.0, 0.1, 0.0, 0.0], reference)
    )
    assert state_cell([2.4, 0.0, 0.0, 0.0], reference)[1] == CART_BINS - 1  # clamp right
    assert state_cell([-2.4, 0.0, 0.0, 0.0], reference)[1] == 0  # clamp left
    # +pi wraps to -pi, so it belongs to the first bin
    assert state_cell([-2.5, np.pi, 0.0, 0.0], reference) == (0, 0)
    alpha, x = cell_centers((6, 3), reference)
    # bin 6 spans alpha in [0, +30 degrees), so its center sits at +15 degrees
    assert alpha == pytest.approx(np.deg2rad(15.0), abs=1e-12)
    assert x == pytest.approx(0.4, abs=1e-12)
    assert normalized_settled_error([0.0, 0.0, 0.0, 0.0], reference) == pytest.approx(0.0)
    # 0.01 m of cart error counts 0.5 of the 0.02 m tolerance; angle dominates at 0.02 rad
    assert normalized_settled_error([0.01, 0.02, 0.0, 0.0], reference) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        normalized_settled_error([np.nan, 0.0, 0.0, 0.0], reference)


def test_return_fidelity_guard_bitwise():
    """The return privilege is exact: set_state round trip + stitched replay."""
    design = design_swingup_lqr()
    guard = return_fidelity_guard(design.controller.reference, design.dt, master_seed=0)
    assert guard["bitwise_identical_set_state"] is True
    assert guard["bitwise_identical_stitched_replay"] is True
    assert guard["steps"] >= 1


def test_archive_member_criterion_and_selection(design_and_reward):
    """Lexicographic membership (error, steps, serial) and deterministic selection."""
    _design, reward = design_and_reward
    reference = reward.reference
    archive = Archive(reference)

    def member_state(alpha, x=0.0, omega=0.0):
        state = reference.copy()
        state[1] += alpha
        state[0] = x
        state[3] = omega
        return state

    root = down_start_state(reference)
    root_cell = state_cell(root, reference)
    assert archive.add(
        root_cell, root, parent_member=None, seg_states=[root], seg_controls=[], steps=0
    )
    assert archive.cells[root_cell]["steps"] == 0
    # a worse candidate (larger settled error) never replaces the member
    assert not archive.add(
        root_cell,
        root + np.array([0.0, 0.0, 0.0, 5.0]),
        parent_member=None,
        seg_states=[root],
        seg_controls=[],
        steps=0,
    )
    # a closer-to-stable candidate replaces regardless of path length
    better = member_state(np.pi - 1e-6, omega=0.0)
    assert archive.add(
        root_cell, better, parent_member=None, seg_states=[better], seg_controls=[], steps=50
    )
    assert np.array_equal(archive.cells[root_cell]["state"], better)
    assert archive.cells[root_cell]["sel"] == 0  # replacement keeps the visit count

    rng = np.random.default_rng([7, 8100])
    cell = archive.select(rng)
    assert cell == root_cell
    assert archive.cells[cell]["sel"] == 1
    again = Archive(reference)
    again.add(root_cell, root, parent_member=None, seg_states=[root], seg_controls=[], steps=0)
    other_rng = np.random.default_rng([7, 8100])
    assert again.select(other_rng) == cell  # same seed -> same choice
    with pytest.raises(ValueError):
        Archive(reference).select(rng)


def test_member_path_stitching_is_physically_exact(design_and_reward):
    """A stitched two-node chain replays bit for bit from a fresh environment."""
    design, _reward = design_and_reward
    reference, dt = design.controller.reference, design.dt
    archive = Archive(reference)
    root = down_start_state(reference)
    archive.add(
        state_cell(root, reference),
        root,
        parent_member=None,
        seg_states=[root],
        seg_controls=[],
        steps=0,
    )
    rng = np.random.default_rng([3, SEED_OFFSET_ACT])
    env = make_swingup_environment(max_episode_steps=8)
    try:
        env.reset(seed=0)
        env.unwrapped.set_state(root[:2], root[2:])
        env.unwrapped.data.qfrc_applied[0] = 0.0
        states, controls = [root.copy()], []
        for _ in range(6):
            action = np.array([rng.uniform(-3.0, 3.0)], np.float32)
            state, _, terminated, _truncated, _info = env.step(action)
            assert not terminated
            controls.append(float(action[0]))
            states.append(np.asarray(state, dtype=float).copy())
    finally:
        env.close()
    assert archive.add(
        state_cell(states[-1], reference),
        states[-1],
        parent_member=archive.cells[state_cell(root, reference)],
        seg_states=states,
        seg_controls=controls,
        steps=6,
    )
    stitched_states, stitched_controls = archive.full_path(state_cell(states[-1], reference))
    assert len(stitched_states) == len(states) == 7  # root state + the six segment states
    assert len(stitched_controls) == 6
    # replay the stitched actions from the stitched first state in a fresh env
    env = make_swingup_environment(max_episode_steps=8)
    try:
        env.reset(seed=99)
        env.unwrapped.set_state(stitched_states[0][:2], stitched_states[0][2:])
        env.unwrapped.data.qfrc_applied[0] = 0.0
        for step, action in enumerate(stitched_controls):
            state, _, terminated, _truncated, _info = env.step(np.array([action], np.float32))
            assert not terminated
            np.testing.assert_array_equal(np.asarray(state, dtype=float), stitched_states[step + 1])
    finally:
        env.close()
    assert dt == 0.04  # the lesson-7 control period the stitching relies on


# ---------------------------------------------------------- phase-1 machinery
def test_capture_criterion_matches_lesson7_recovery_metrics(design_and_reward):
    """find_capture_start + capture_slice_metrics reuse the lesson-7 criteria."""
    design, _reward = design_and_reward
    reference, dt = design.controller.reference, design.dt
    rng = np.random.default_rng([5, SEED_OFFSET_ACT])
    env = make_swingup_environment(max_episode_steps=200)
    try:
        # a hold segment: reset into the upright band with a small omega, LQR engaged
        start = reference.copy()
        start[1] += 0.05
        start[3] = 0.4
        env.reset(seed=0)
        env.unwrapped.set_state(start[:2], start[2:])
        env.unwrapped.data.qfrc_applied[0] = 0.0
        states, controls = [start.copy()], []
        engaged = balance_engaged(start, False, reference)
        for _ in range(150):
            action, mode = exploration_action(
                states[-1], engaged, design.controller, rng, reference
            )
            engaged = mode == "lqr"
            state, _, terminated, _truncated, _info = env.step(action)
            assert not terminated
            controls.append(float(action[0]))
            states.append(np.asarray(state, dtype=float).copy())
        states_array = np.asarray(states, dtype=float)
    finally:
        env.close()
    cap_start = find_capture_start(states_array, reference, dt)
    assert cap_start is not None
    # cap_start is exactly the first index whose 2 s window stays in tolerance
    error = states_array - reference
    error[:, 1] = (error[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
    in_tolerance = np.all(np.abs(error) <= SETTLED_TOLERANCES, axis=1)
    need = round(2.0 / dt)
    expected = next(
        (
            index - need
            for index in range(len(states_array))
            if index >= need and in_tolerance[index - need : index + 1].all()
        ),
        None,
    )
    assert cap_start == expected
    metrics = capture_slice_metrics(states_array[cap_start:], controls[cap_start:], reference, dt)
    direct = recovery_metrics(
        {
            "states": states_array[cap_start:],
            "controls": np.asarray(controls[cap_start:]),
            "applied_force_n": np.zeros(len(controls) - cap_start),
            "scheduled_force_n": np.zeros(len(controls) - cap_start),
            "end_flags": np.array([False, True]),
            "modes": np.full(len(controls) - cap_start, "cap", dtype="<U3"),
        },
        {"failure_reason": ""},
        reference,
        dt,
    )
    assert metrics["recovered"] is True
    assert metrics["settled_at_s"] == direct["settled_at_s"] == 0.0
    assert np.array_equal(SETTLED_TOLERANCES, [0.02, 0.01, 0.02, 0.02])
    # a hanging-down rollout never enters the settled band
    down_states = np.asarray([down_start_state(reference)] * 60)
    assert find_capture_start(down_states, reference, dt) is None


def test_exploration_smoke_and_seed_determinism(design_and_reward):
    """Tiny archive exploration is deterministic under fixed seeds and grows."""
    design, reward = design_and_reward
    reference, dt = reward.reference, 0.04
    logs = []

    def one():
        return run_exploration(
            reference,
            dt,
            design.controller,
            segments=60,
            segment_steps=40,
            master_seed=0,
            log=logs.append,
        )

    first, second = one(), one()
    np.testing.assert_array_equal(first["coverage_curve"], second["coverage_curve"])
    np.testing.assert_array_equal(first["selection_cells"], second["selection_cells"])
    np.testing.assert_array_equal(first["grids"]["cell_states"], second["grids"]["cell_states"])
    assert first["env_steps"] == second["env_steps"] > 0
    assert first["cells_occupied"] == second["cells_occupied"] > 1
    occupied, total = first["archive"].coverage()
    assert occupied == first["cells_occupied"] and total == ANGLE_BINS * CART_BINS
    assert first["failure_counts"]["cart_safety_boundary"] >= 0
    assert 0.0 <= first["lqr_step_fraction"] <= 1.0
    assert MAX_SEGMENTS * SEGMENT_STEPS == 6000 * 150  # the fixed phase-1 step budget
    with pytest.raises(ValueError):
        run_exploration(reference, dt, design.controller, segments=0)


# ------------------------------------------------------------ shrunk e2e
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert report["guard"]["bitwise_identical_set_state"] is True
    assert report["guard"]["bitwise_identical_stitched_replay"] is True
    assert report["baseline"]["successes"] == report["baseline"]["episodes"] == 2
    assert report["baseline"]["deterministic_identical_repeats"] is True
    phase1 = report["phase1"]
    grid = report["protocol"]["grid"]
    assert grid["cells_total"] == grid["angle_bins"] * grid["cart_bins"]
    assert 0 < phase1["cells_occupied"] <= phase1["cells_total"]
    assert abs(phase1["coverage_final"] - phase1["cells_occupied"] / phase1["cells_total"]) < 1e-12
    assert phase1["capture_count"] >= 1  # the small budget captures the stable band
    assert phase1["kept_captures"] == min(phase1["capture_count"], SMALL_CONFIG["captures_kept"])
    assert phase1["first_capture_step"] is not None
    reference = np.asarray(report["protocol"]["reference_state"], dtype=float)
    dt = float(report["protocol"]["dt_s"])
    for capture in phase1["per_capture"]:
        with np.load(output / "trajectories.npz", allow_pickle=False) as data:
            seg_states = data[f"capture{capture['index']}_seg_states"]
            seg_controls = data[f"capture{capture['index']}_seg_controls"]
        verified = capture_slice_metrics(
            seg_states[int(capture["cap_start"]) :],
            seg_controls[int(capture["cap_start"]) :],
            reference,
            dt,
        )
        assert verified["recovered"] is True
        assert verified["settled_at_s"] == capture["settled_at_s"]
    phase2 = report["phase2"]
    assert phase2["run"] is True
    assert phase2["teacher_pairs"] == sum(
        capture["teacher_steps"] for capture in phase1["per_capture"]
    )
    assert phase2["stochastic"]["episodes"] == 2
    assert phase2["stochastic"]["successes"] in (0, 1, 2)
    four_way = report["four_way_comparison"]
    assert [row["successes"] for row in four_way][:3] == [2, 0, 0]
    assert four_way[1]["episodes"] == 60 and four_way[2]["episodes"] == 60
    assert four_way[2]["episodes_with_upright_arrival"] == 33  # cited lesson-32 arrival norm
    assert four_way[3]["label"].startswith("Go-Explore+BC")
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
    for name in (
        "summary.json",
        "trajectories.npz",
        "archive_map.png",
        "capture_analysis.png",
        "robustification.png",
    ):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, seed=0, **SMALL_CONFIG)


def test_stitched_teacher_path_replays_bitwise(small_run):
    """The phase-2 teacher path is real physics: a middle slice replays exactly."""
    output, _report = small_run
    with np.load(output / "trajectories.npz", allow_pickle=False) as data:
        full_states = data["capture0_full_states"]
        full_controls = data["capture0_full_controls"]
        seg_states = data["capture0_seg_states"]
    assert len(full_controls) + 1 == len(full_states)
    assert np.all(np.abs(full_controls) <= 3.0)
    # the stitched path ends exactly at the capture segment's end
    np.testing.assert_array_equal(full_states[-len(seg_states) :], seg_states)
    middle = len(full_states) // 3
    env = make_swingup_environment(max_episode_steps=len(full_controls))
    try:
        env.reset(seed=7)
        env.unwrapped.set_state(full_states[middle][:2], full_states[middle][2:])
        env.unwrapped.data.qfrc_applied[0] = 0.0
        for offset, action in enumerate(full_controls[middle : middle + 12]):
            state, _, terminated, _truncated, _info = env.step(np.array([action], np.float32))
            assert not terminated
            np.testing.assert_array_equal(
                np.asarray(state, dtype=float), full_states[middle + offset + 1]
            )
    finally:
        env.close()


def test_teacher_pairs_and_bc_training(design_and_reward, small_run):
    """BC pairs follow the lesson-7 alignment; the trainer is deterministic."""
    _design, reward = design_and_reward
    reference = reward.reference
    output, report = small_run
    with np.load(output / "trajectories.npz", allow_pickle=False) as data:
        captures = [
            {
                "full_states": data[f"capture{index}_full_states"],
                "full_controls": data[f"capture{index}_full_controls"],
            }
            for index in range(len(report["phase1"]["per_capture"]))
        ]
    obs, actions = build_teacher_pairs(captures, reference)
    total = sum(len(capture["full_controls"]) for capture in captures)
    assert obs.shape == (total, 5) and actions.shape == (total,)
    np.testing.assert_array_equal(
        obs[0], normalize_observation(captures[0]["full_states"][0], reference)
    )
    assert np.all(np.abs(actions) <= 3.0)

    # the re-exported lesson-32 BC loss: hand-computed answer on a constant policy
    policy = GaussianPolicy(5, (2,), seed=3, log_std_init=0.5)
    for weight in policy.trunk.weights:
        weight[:] = 0.0
    for bias in policy.trunk.biases:
        bias[:] = 0.0
    policy.trunk.biases[-1][:] = 0.5
    demo_obs = np.zeros((3, 5))
    demo_actions = np.array([1.0, -1.0, 0.5])
    mse, grads = bc_loss_and_gradient(policy, demo_obs, demo_actions)
    assert mse == pytest.approx(2.5 / 3.0)  # residuals 0.5-1, 0.5+1, 0 -> mean square
    assert float(grads[-1][0]) == pytest.approx(2.0 * float(np.array([-0.5, 1.5, 0.0]).mean()))
    with pytest.raises(ValueError):
        bc_loss_and_gradient(policy, np.zeros((0, 5)), np.zeros(0))

    def train():
        return train_robustified_bc(
            obs, actions, epochs=8, init_seed=[0, 8300], shuffle_seed=[0, 8400]
        )

    first_policy, first_curve = train()
    second_policy, second_curve = train()
    np.testing.assert_array_equal(first_curve, second_curve)
    for a, b in zip(first_policy.trunk.weights, second_policy.trunk.weights, strict=True):
        assert np.array_equal(a, b)
    assert first_curve[-1] < first_curve[0]  # the teacher is actually learnable
    np.testing.assert_array_equal(
        first_policy.log_std, np.full(1, np.log(1.0))
    )  # the exploration std is never trained
    with pytest.raises(ValueError):
        train_robustified_bc(obs[:0], actions[:0], epochs=1)


def test_phase2_null_path_when_no_capture(tmp_path):
    """Phase 1 without a capture honestly skips phase 2 and still writes a record."""
    output = tmp_path / "null_run"
    report = run_experiment(
        output, seed=0, segments=1, segment_steps=2, eval_seed_count=2, bc_epochs=1, captures_kept=1
    )
    assert report["phase1"]["capture_count"] == 0
    assert report["phase1"]["kept_captures"] == 0
    assert report["phase2"]["run"] is False
    assert report["phase2"]["teacher_pairs"] == 0
    assert "phase 1" in report["phase2"]["verdict"]
    assert report["hypothesis"]["phase1_captured"] is False
    assert report["four_way_comparison"][3]["episodes"] == 0
    assert (output / "capture_analysis.png").is_file()  # the placeholder figure
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        assert "bc_loss_curve" not in archive.files and "eval_states" not in archive.files


def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.goexplore_swingup",
            "--output",
            str(out),
            "--seed",
            "0",
            "--segments",
            "50",
            "--segment-steps",
            "40",
            "--eval-seeds",
            "2",
            "--bc-epochs",
            "2",
            "--captures-kept",
            "2",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["guard"]["bitwise_identical_set_state"] is True
    assert payload["guard"]["bitwise_identical_stitched_replay"] is True
    assert payload["baseline"] == "2/2"
    assert payload["goexplore_bc"] == "0/0"  # no capture within 50 tiny segments
    assert payload["zero_to_one"] is False
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "archive_map.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.goexplore_swingup",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert again.returncode != 0  # new directories are never overwritten


def test_demo_loader_cross_checks_and_rejects_tampering(small_run, tmp_path):
    output, _report = small_run
    data = load_replays(output)  # the pristine record passes every cross-check
    assert data["report"]["experiment"] == EXPERIMENT

    work = tmp_path / "tampered"
    shutil.copytree(output, work)
    summary_text = (work / "summary.json").read_text(encoding="utf-8")

    def rewrite(work_dir, parsed):
        (work_dir / "summary.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # (1) summary successes disagree with the archive
    parsed = json.loads(summary_text)
    parsed["phase2"]["stochastic"]["successes"] += 1
    rewrite(work, parsed)
    with pytest.raises(ValueError, match="BC successes disagree"):
        load_replays(work)
    (work / "summary.json").write_text(summary_text, encoding="utf-8")

    # (2) archive bytes tampered -> checksum mismatch
    path = work / "trajectories.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["coverage_curve"] = payload["coverage_curve"] * 2.0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(work)

    # (3) an array removed with the hash refreshed -> unexpected key set
    del payload["case0_states"]
    np.savez_compressed(path, **payload)
    parsed = json.loads(summary_text)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rewrite(work, parsed)
    with pytest.raises(ValueError, match="Unexpected archive arrays"):
        load_replays(work)

    # (4) the capture's settle time tampered in the summary; restore the pristine
    # archive and its recorded hash first so only the summary differs
    with np.load(output / "trajectories.npz", allow_pickle=False) as pristine:
        payload = {key: pristine[key].copy() for key in pristine.files}
    np.savez_compressed(path, **payload)
    rewrite(work, json.loads(summary_text))
    parsed = json.loads(summary_text)
    parsed["phase1"]["per_capture"][0]["settled_at_s"] = 9.99
    rewrite(work, parsed)
    with pytest.raises(ValueError, match="settle time disagree"):
        load_replays(work)

    # (5) the BC curve end tampered with a fresh hash (summary otherwise pristine)
    parsed = json.loads(summary_text)
    payload["bc_loss_curve"][-1] = payload["bc_loss_curve"][-1] / 2.0
    np.savez_compressed(path, **payload)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rewrite(work, parsed)
    with pytest.raises(ValueError, match="BC curve end"):
        load_replays(work)


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel(small_run):
    import tkinter as tk

    from embodied_learning.goexplore_demo import GoExploreDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = GoExploreDemo(root, data)
    root.update()
    assert demo.mode.get() == "archive"
    assert len(demo.fig.axes) == 3  # heatmap + colorbar + coverage curve
    panel = demo.stats.cget("text")
    phase1 = report["phase1"]
    assert (
        f"{phase1['cells_occupied']}/{phase1['cells_total']}" in panel
    )  # panel numbers come from the summary
    assert "稳定带" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 3  # mode switches must not accumulate axes
    demo.mode.set("capture")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "捕获" in panel and "recovery_metrics" in panel
    assert f"{report['phase1']['per_capture'][0]['tail_s']:.2f} s" in panel
    demo.mode.set("outcome")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "基线" in panel and "DAPG" in panel
    stochastic = report["phase2"]["stochastic"]
    assert f"{stochastic['successes']}/{stochastic['episodes']}" in panel
    assert f"{report['phase2']['teacher_pairs']}" in panel
    demo.close()
