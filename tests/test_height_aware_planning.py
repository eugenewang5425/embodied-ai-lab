"""Measured heights must separate base, mast and camera collision decisions."""

from dataclasses import replace

import numpy as np

from embodied_learning.height_aware_planning import HeightAwareField
from embodied_learning.navigation_geometry import NavigationRig, point_clearance, world_points


def test_high_return_inside_xy_base_does_not_falsely_block_the_base():
    prior = np.zeros((120, 120), bool)
    rig = replace(NavigationRig(), margin_m=0.02)
    pose = np.array([5.71145509, 5.15066805, 0.02111237])
    local = np.array([[0.29348886, 0.23914137, 0.50020011]])
    field = HeightAwareField(prior, world_points(local, pose), rig, pose)
    assert field.free(pose) and not field.relax_start
    # Same XY now occupies the real base: height is the only changed input.
    local[0, 2] = 0.14
    blocked = HeightAwareField(prior, world_points(local, pose), rig, pose)
    assert blocked.relax_start
    assert not blocked.free(pose + [0.02, 0.02, 0])


def test_upper_body_and_low_obstacle_are_both_checked():
    rig = replace(NavigationRig(), margin_m=0)
    pose = np.array([2.0, 2.0, 0])
    for point in ([0.17, 0, 0.45], [0.19, 0, 0.55], [0, 0, 0.14]):
        field = HeightAwareField(np.zeros((120, 120), bool),
                                 world_points(np.array([point]), pose), rig, pose)
        assert not field.free(pose)
    above = HeightAwareField(np.zeros((120, 120), bool),
                             world_points(np.array([[0.19, 0, 0.75]]), pose), rig, pose)
    assert above.free(pose)


def test_zero_margin_point_checks_match_independent_3d_clearance():
    rng = np.random.default_rng(14)
    rig = replace(NavigationRig(), margin_m=0)
    prior = np.zeros((120, 120), bool)
    pose = np.array([2.0, 2.0, 0])
    points = np.column_stack((rng.uniform(1.6, 2.4, (40, 2)), rng.uniform(0, 0.8, 40)))
    field = HeightAwareField(prior, points, rig, pose)
    for yaw in np.linspace(-np.pi, np.pi, 45):
        candidate = np.array([2.0, 2.0, yaw])
        local = points.copy()
        local[:, :2] -= candidate[:2]
        assert field.free(candidate) == (point_clearance(local, np.array([[0, 0, yaw]]), rig) > 0)


def test_unknown_height_prior_wall_stays_blocking():
    prior = np.zeros((120, 120), bool)
    prior[20, 22] = True
    pose = np.array([2.0, 2.0, 0])
    field = HeightAwareField(prior, np.empty((0, 3)), NavigationRig(), pose)
    assert not field.free(pose)


def test_round14_reuses_private_lifecycle_and_leaves_baseline_globals_unchanged():
    from embodied_learning.experiments import navigation_clearance_study as baseline
    from embodied_learning.experiments.navigation_height_study import (
        private_engine,
        registered_cases,
    )

    engine = private_engine()
    assert engine is not baseline
    assert baseline.LAYOUT_BOXES["street"] == engine.LAYOUT_BOXES["street"]
    assert "street_west" not in baseline.LAYOUT_BOXES
    cases = registered_cases()
    assert len(cases) == len(set(cases)) == 75
    final = [c for c in cases if c[5] == "height_depth" and c[3] != "impossible_negative"]
    assert len(final) == 60 and {c[2] for c in final} == set(range(37, 42))
    assert len([c for c in cases if c[5] == "corridor_depth"]) == 12
    assert len([c for c in cases if c[3] == "impossible_negative"]) == 3
