"""Select calibration views, audit selection bias, and export recoverable copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from monocular_depth.calibration import detect_board, make_board, solve_observations
from monocular_depth.focus import validate_frame_focus
from monocular_depth.io import read_image
from monocular_depth.records import read_json, write_json
from monocular_depth.selection import coverage, heldout_errors, select_views


def run(candidate, output, retain_count=None):
    record = read_json(candidate)
    names = record["accepted_images"]
    board = make_board(record["board"])
    size = tuple(record["image_size"])
    objects, pixels, traces = [], [], []
    for name in names:
        path = Path(record["source_image_paths"][name])
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["source_image_sha256"][name]:
            raise ValueError("Source image changed since candidate")
        trace = read_json(path.with_suffix(".json"))
        target = validate_frame_focus(trace["focus_before_frame"], trace["focus_after_frame"])
        frame = read_image(path)
        if frame is None or (frame.shape[1], frame.shape[0]) != size:
            raise ValueError("Invalid image size")
        if trace["image_file"] != path.name or trace["image_size"] != list(size):
            raise ValueError("Focus record does not match photo")
        if target != record["fixed_focus_driver_units"]:
            raise ValueError("Focus mismatch")
        detection = detect_board(frame, board)
        if detection is None:
            raise ValueError("Missing valid corners")
        objects.append(detection[0])
        pixels.append(detection[1])
        traces.append(trace)
    output.mkdir(parents=True, exist_ok=False)
    chosen, frontier, policy = select_views(
        objects,
        pixels,
        size,
        progress=lambda count, rms: print(f"Frontier: {count} views, RMS {rms:.6f}", flush=True),
    )
    if retain_count is not None:
        matching = [point for point in frontier if len(point["indices"]) == retain_count]
        if not matching:
            raise ValueError("Requested count is not on the geometry-preserving frontier")
        chosen = matching[0]
        policy["requested_retained_count"] = retain_count
        policy["count_choice_note"] = (
            "Explicit trade-off choice; not an independent validation result"
        )
    selected = chosen["indices"]
    policy["selected_hull_image_fraction"] = coverage(pixels, selected, size)
    write_json(
        output / "frontier.json",
        {
            "policy": policy,
            "selected_count": len(selected),
            "candidates": [
                {
                    "views": len(p["indices"]),
                    "rms_px": p["fit"]["rms_px"],
                    "max_view_rms_px": max(p["fit"]["per_view_rms_px"]),
                    "quality_pass": p["fit"]["quality_pass"],
                    "files": [names[i] for i in p["indices"]],
                    "last_removed": names[p["removed_index"]]
                    if p["removed_index"] is not None
                    else None,
                }
                for p in frontier
            ],
        },
    )
    # Conditional CV: selection used these images, so this is NOT unbiased validation.
    conditional_squared, conditional_folds = [], []
    for fold in range(5):
        train = [i for j, i in enumerate(selected) if j % 5 != fold]
        held = [i for j, i in enumerate(selected) if j % 5 == fold]
        fit = solve_observations([objects[i] for i in train], [pixels[i] for i in train], size)
        errors, squared = heldout_errors(objects, pixels, held, fit)
        conditional_squared.extend(squared)
        conditional_folds.append(
            {"fold": fold, "heldout": errors, "camera_matrix": fit["camera_matrix"]}
        )
    # Re-run the complete selector strictly inside each training fold. Evaluate ALL
    # held-out photos, including difficult ones: no held-out residual-based removal.
    nested_squared, nested_folds = [], []
    for fold in range(5):
        train = [i for i in range(len(names)) if i % 5 != fold]
        held = [i for i in range(len(names)) if i % 5 == fold]
        fold_selected, fold_frontier, _ = select_views(
            [objects[i] for i in train],
            [pixels[i] for i in train],
            size,
        )
        if retain_count is not None:
            matching = [p for p in fold_frontier if len(p["indices"]) == retain_count]
            if not matching:
                raise ValueError("Requested count cannot be selected within a training fold")
            fold_selected = matching[0]
        errors, squared = heldout_errors(objects, pixels, held, fold_selected["fit"])
        nested_squared.extend(squared)
        nested_folds.append(
            {
                "fold": fold,
                "selected_train_indices": [train[i] for i in fold_selected["indices"]],
                "train_quality_pass": fold_selected["fit"]["quality_pass"],
                "heldout": errors,
            }
        )
        print(f"Nested fold {fold + 1}/5 complete", flush=True)
    selected_names = [names[i] for i in selected]
    rejected_names = [name for name in names if name not in selected_names]
    result = {**record, **chosen["fit"]}
    result.update(
        {
            "accepted_images": selected_names,
            "rejected_images": [
                {"file": name, "reason": "backward selection; not deleted"}
                for name in rejected_names
            ],
            "source_image_paths": {
                name: record["source_image_paths"][name] for name in selected_names
            },
            "source_image_sha256": {
                name: record["source_image_sha256"][name] for name in selected_names
            },
            "selection_policy": policy,
            "source_candidate": str(candidate.resolve()),
            "validation_status": "Selected-data reprojection gate only; independent metric validation pending",
        }
    )
    write_json(output / "selected-camera.json", result)
    for i in selected:
        name = names[i]
        destination = output / "images" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["source_image_paths"][name], destination)
        write_json(
            destination.with_suffix(".json"),
            {
                **traces[i],
                "image_file": name,
                "original_image_path": record["source_image_paths"][name],
                "original_image_sha256": record["source_image_sha256"][name],
            },
        )
    write_json(
        output / "images" / "session.json",
        {
            **record["capture_session"],
            "selection_policy": policy,
            "note": "Derived selection; untouched original sessions and per-photo provenance retained",
        },
    )
    report = {
        "all_names_in_index_order": names,
        "selected_images": selected_names,
        "excluded_images": rejected_names,
        "selected_count": len(selected),
        "selected_rms_px": result["rms_px"],
        "selected_max_view_rms_px": max(result["per_view_rms_px"]),
        "selected_quality_pass": result["quality_pass"],
        "geometry_policy": policy,
        "conditional_selected_5fold_rms_px": float(np.sqrt(np.mean(conditional_squared))),
        "conditional_note": "Selection before splitting; optimistic selected-set consistency, not independent validation",
        "conditional_folds": conditional_folds,
        "nested_all_views_5fold_rms_px": float(np.sqrt(np.mean(nested_squared))),
        "nested_note": "Re-select within training fold; evaluate every held-out view without residual-based filtering",
        "nested_folds": nested_folds,
        "promoted_to_default": False,
        "metric_accuracy_validated": False,
    }
    write_json(output / "selection-audit.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "selected_count",
                    "selected_rms_px",
                    "selected_max_view_rms_px",
                    "selected_quality_pass",
                    "conditional_selected_5fold_rms_px",
                    "nested_all_views_5fold_rms_px",
                    "selected_images",
                    "excluded_images",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retain-count", type=int)
    args = parser.parse_args()
    run(args.candidate, args.output, args.retain_count)
