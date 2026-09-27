import numpy as np
import pytest

from embodied_learning.experiments.grid_nav import Obstacle, cast_rays
from embodied_learning.experiments.map_repair import (
    compose,
    integrate_scales,
    map_metrics,
    match_scans,
    relative,
    repair,
)


def observation(pose, obstacles, rays=180):
    angles = pose[2] + np.arange(rays) * (2 * np.pi / rays)
    return cast_rays(*pose[:2], angles, obstacles, (), 8.0)


def test_relative_motion_composition_wraps_turns_correctly():
    a, b = np.array([2.0, 3.0, 3.13]), np.array([2.5, 4.0, -3.10])
    assert np.allclose(compose(a, relative(a, b)), b)


def test_inverse_constant_scale_reconstructs_known_motion():
    from embodied_learning.navigation_geometry import advance

    truth, measured = [np.zeros(3)], [np.zeros(3)]
    for v, w in [(1, 0)] * 8 + [(0, 1)] * 12 + [(1, 0)] * 10:
        truth.append(advance(truth[-1], v, w, 0.1))
        measured.append(advance(measured[-1], v * 1.02, w * 1.02, 0.1))
    fixed = integrate_scales(np.array(measured), 1 / 1.02, 1 / 1.02)
    assert np.allclose(fixed, truth, atol=1e-10)


def test_scan_matching_recovers_known_transform_from_geometry_only():
    obstacles = [
        Obstacle("rect", 3, 0, 0.1, 3),
        Obstacle("rect", 0, 3, 3, 0.1),
        Obstacle("rect", -2, 0, 0.1, 3),
        Obstacle("circle", 1, 1, r=0.4),
    ]
    a, b = np.zeros(3), np.array([0.25, 0.08, 0.06])
    ar, ah = observation(a, obstacles)
    br, bh = observation(b, obstacles)
    fitted, quality = match_scans(ar, ah, br, bh, b + [0.03, -0.02, 0.01])
    assert quality["accepted"]
    assert np.linalg.norm(fitted[:2] - b[:2]) < 0.02
    assert abs(fitted[2] - b[2]) < 0.01


def test_parallel_walls_cannot_calibrate_unobserved_along_wall_motion():
    obstacles = [Obstacle("rect", 0, 2, 100, 0.1), Obstacle("rect", 0, -2, 100, 0.1)]
    ar, ah = observation(np.zeros(3), obstacles)
    br, bh = observation(np.array([0.25, 0, 0]), obstacles)
    _, quality = match_scans(ar, ah, br, bh, np.array([0.26, 0, 0]))
    assert not quality["accepted"]


def test_repair_rejects_missing_constraints_instead_of_using_truth():
    odom = np.column_stack((np.linspace(0, 3, 40), np.zeros((40, 2))))
    with pytest.raises(ValueError, match="too few observable"):
        repair(odom, np.full((40, 64), 8.0), np.zeros((40, 64), bool))


def test_map_quality_has_symmetric_coverage_and_known_metric_offset():
    a = np.zeros((30, 30), bool)
    a[5:20, 10] = True
    b = np.roll(a, 3, axis=1)
    result = map_metrics(a, b)
    assert result["precision"] == 0 and result["recall"] == 0
    assert result["symmetric_mean_m"] == pytest.approx(0.3)
