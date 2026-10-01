"""Registered final-only cases must not weaken paired comparison completeness."""

import json
from pathlib import Path

from embodied_learning.experiments.navigation_clearance_study import METHODS, registered_cases
from embodied_learning.navigation_study_demo import qualified_navigation_batch
from embodied_learning.replay_viewer.navigation_studies import registered_method_keys


def row_for(case):
    layout, obstacle, seed, cohort, delay, method, profile = case
    return {"layout": layout, "obstacle": obstacle, "seed": seed, "cohort": cohort,
                "condition_key": delay, "method": method, "profile": profile,
                "passed": True, "reached": 1, "physically_stopped": True, "false_arrival": False,
                "announced": cohort != "impossible_negative", "severe_contact": False}


def test_registered_final_only_views_keep_all_three_paired_methods():
    cases = registered_cases()
    protocol = {"cases": cases, "methods": list(METHODS[13])}
    assert registered_method_keys(protocol, row_for(cases[0])) == METHODS[13]
    final_only = next(c for c in cases if c[0] == "street_east")
    assert registered_method_keys(protocol, row_for(final_only)) == ("corridor_depth",)
    assert registered_method_keys(protocol, row_for(cases[-1])) == ("corridor_depth",)


def test_default_demonstration_requires_full_batch_and_real_success(tmp_path):
    cases = registered_cases()
    summary = {"protocol": {"cases": cases}, "rows": [row_for(c) for c in cases]}
    file = Path(tmp_path) / "summary.json"
    file.write_text(json.dumps(summary), encoding="utf8")
    assert qualified_navigation_batch(tmp_path)
    summary["rows"].pop()
    file.write_text(json.dumps(summary), encoding="utf8")
    assert not qualified_navigation_batch(tmp_path)
    summary["rows"] = [row_for(c) for c in cases]
    summary["rows"][2]["physically_stopped"] = False
    file.write_text(json.dumps(summary), encoding="utf8")
    assert not qualified_navigation_batch(tmp_path)
    summary["rows"][2]["physically_stopped"] = True
    summary["rows"][-1]["announced"] = True
    file.write_text(json.dumps(summary), encoding="utf8")
    assert not qualified_navigation_batch(tmp_path)
