"""Preferred route padding must not invalidate a physically clear escape start."""

from dataclasses import replace

import numpy as np

from embodied_learning.clearance_navigation import (
    ClearanceNavigationController,
    GradualClearanceField,
)
from embodied_learning.footprint_planning import FootprintField
from embodied_learning.navigation_geometry import NavigationRig


def test_preferred_padding_relaxes_start_but_actual_body_still_blocks():
    rig = replace(NavigationRig(), margin_m=0.02)
    prior = np.zeros((120, 120), bool)
    start = np.array([1.5, 1.5, 0])
    points = np.array([[1.81, 1.5]])
    assert not FootprintField(prior, points, rig).free(start)
    field = GradualClearanceField(prior, points, rig, start)
    assert field.relax_start and field.free(start)
    assert field.free(np.array([1.3, 1.5, 0]))
    assert not field.free(np.array([1.51, 1.5, 0]))
    occupied = GradualClearanceField(prior, np.array([[1.7, 1.5]]), rig, start)
    assert not occupied.free(start)


def test_clear_start_keeps_preferred_padding_everywhere():
    rig = replace(NavigationRig(), margin_m=0.02)
    prior = np.zeros((120, 120), bool)
    start = np.array([1.5, 1.5, 0])
    points = np.array([[1.9, 1.5]])
    field = GradualClearanceField(prior, points, rig, start)
    assert not field.relax_start
    assert not field.free(np.array([1.59, 1.5, 0]))


def test_route_padding_does_not_change_actual_body_or_stop_predictor():
    rig = replace(NavigationRig(), margin_m=0)
    controller = ClearanceNavigationController(np.empty((0, 6)), rig)
    assert controller.rig.margin_m == controller.predictor.rig.margin_m == 0
    assert controller.planning_rig.margin_m == 0.02
    np.testing.assert_array_equal(controller.rig.parts, rig.parts)


def test_formal_roster_has_matched_comparison_and_independent_stability_cases():
    from collections import Counter

    from embodied_learning.experiments.navigation_clearance_study import registered_cases

    cases = registered_cases()
    assert len(cases) == len(set(cases)) == 87
    final = [c for c in cases if c[5] == "corridor_depth" and c[3] != "impossible_negative"]
    assert len(final) == 60
    assert Counter(c[0] for c in final) == {"street": 20, "street_east": 20, "street_wide": 20}
    assert Counter(c[2] for c in final) == {seed: 12 for seed in range(32, 37)}
    paired = [c for c in cases if c[3] == "paired_holdout"]
    assert Counter(c[5] for c in paired) == {
        "legacy_depth": 12, "physical_depth": 12, "corridor_depth": 12,
    }
    negative = [c for c in cases if c[3] == "impossible_negative"]
    assert len(negative) == 3 and all(c[0] == "blocked" for c in negative)


def test_impossible_control_really_has_smaller_mouth_than_body():
    from embodied_learning.experiments.navigation_clearance_study import map_scene

    prior, _ = map_scene("bypass_crate", "blocked")
    prior[-2, 3], prior[-1, 2] = 5.85, 6.15
    assert prior[-1, 2] - prior[-2, 3] < min(0.55, 0.44)
