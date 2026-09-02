"""Evaluate two compatible focus-recorded sessions; never copy or change source photos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from monocular_depth.calibration import detect_board, make_board, solve_observations
from monocular_depth.focus import validate_frame_focus
from monocular_depth.io import read_image
from monocular_depth.records import read_json, write_json


def load_observations(directory, label):
    session = read_json(directory / "session.json")
    if not session.get("focus_lock_required"):
        raise ValueError("Missing focus policy")
    board = make_board(session["board"])
    observations = []
    for path in sorted(directory.glob("view_*.png")):
        trace = read_json(path.with_suffix(".json"))
        target = validate_frame_focus(trace["focus_before_frame"], trace["focus_after_frame"])
        frame = read_image(path)
        if frame is None:
            raise ValueError(f"Cannot read {path}")
        size = [frame.shape[1], frame.shape[0]]
        if trace["image_size"] != size or session["actual_image_size"] != size:
            raise ValueError("Resolution mismatch")
        if trace["image_file"] != path.name:
            raise ValueError("Photo provenance mismatch")
        detection = detect_board(frame, board)
        if detection is None:
            raise ValueError(f"Insufficient corners in {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        observations.append({
            "name": f"{label}_{path.name}", "path": str(path.resolve()),
            "obj": detection[0], "img": detection[1], "ids": detection[3].ravel(),
            "target": target, "sha256": digest,
        })
    if not observations:
        raise ValueError("Empty session")
    return session, observations


def evaluate(train, heldout, output):
    train_session, training = load_observations(train, "supplement")
    held_session, held = load_observations(heldout, "trial")
    fields = ("camera_index", "actual_image_size", "requested_image_size", "fourcc_reported", "board")
    if any(train_session[key] != held_session[key] for key in fields):
        raise ValueError("Session camera-mode/board records are incompatible")
    targets = {item["target"] for item in training + held}
    if len(targets) != 1:
        raise ValueError("Mixed focus targets")
    output.mkdir(parents=True, exist_ok=False)
    size = tuple(train_session["actual_image_size"])
    fit = solve_observations([v["obj"] for v in training], [v["img"] for v in training], size)
    matrix, distortion = np.array(fit["camera_matrix"]), np.array(fit["distortion_coefficients"])
    held_errors, held_squared = [], []
    for item in held:
        ok, rotation, translation = cv2.solvePnP(item["obj"], item["img"], matrix, distortion)
        if not ok:
            raise ValueError("Held-out pose estimation failed")
        projected, _ = cv2.projectPoints(item["obj"], rotation, translation, matrix, distortion)
        squared = np.sum((projected.reshape(-1, 2) - item["img"].reshape(-1, 2)) ** 2, axis=1)
        held_squared.extend(squared.tolist())
        held_errors.append({"file": item["name"], "rms_px": float(np.sqrt(squared.mean()))})

    accepted, rejected = [], []
    for item in held + training:
        duplicate = any(
            np.array_equal(item["ids"], old["ids"])
            and np.linalg.norm(item["img"].reshape(-1, 2) - old["img"].reshape(-1, 2), axis=1).mean() < 5
            for old in accepted
        )
        if duplicate:
            rejected.append({"file": item["name"], "reason": "near-duplicate viewpoint"})
        else:
            accepted.append(item)
    combined = solve_observations([v["obj"] for v in accepted], [v["img"] for v in accepted], size)
    combined.update({
        "source": "charuco_images", "board": train_session["board"],
        "accepted_images": [v["name"] for v in accepted], "rejected_images": rejected,
        "source_image_paths": {v["name"]: v["path"] for v in accepted},
        "source_image_sha256": {v["name"]: v["sha256"] for v in accepted},
        "focus_lock_verified_for_accepted_frames": True,
        "fixed_focus_driver_units": next(iter(targets)),
        "capture_session": {
            **{key: train_session[key] for key in fields},
            "focus_lock_required": True,
            "source_sessions": [str(heldout.resolve()), str(train.resolve())],
            "note": "Joint diagnostic from matching recorded modes; identical optics not proven by records alone.",
        },
    })
    write_json(output / "combined-candidate.json", combined)
    report = {
        "training_session": str(train.resolve()), "heldout_session": str(heldout.resolve()),
        "training_views": len(training), "heldout_views": len(held),
        "matching_record_fields": list(fields), "fixed_focus_driver_units": next(iter(targets)),
        "training_rms_px": fit["rms_px"],
        "cross_session_heldout_rms_px": float(np.sqrt(np.mean(held_squared))),
        "heldout_per_view": held_errors,
        "heldout_note": "Held-out corners fit their own board poses, not training intrinsics. Not independent metric validation.",
        "joint_views": combined["valid_views"], "joint_rms_px": combined["rms_px"],
        "joint_quality_pass": combined["quality_pass"], "promoted_to_default": False,
    }
    write_json(output / "pair-audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.train, args.heldout, args.output)
