import mujoco
import numpy as np
import pytest

from embodied_learning.experiments.so101_approach_study import roster
from embodied_learning.gripper_calibration import FingerMidpointCalibration
from embodied_learning.so101 import SO101Simulation, grasp_rotation


def test_midpoint_prediction_matches_independent_actual_geometry():
    sim = SO101Simulation()
    calibration = FingerMidpointCalibration(sim)
    rng = np.random.default_rng(73)
    for _ in range(12):
        sim.data.qpos[:5] = rng.uniform(sim.ranges[:, 0] * 0.5, sim.ranges[:, 1] * 0.5)
        sim.data.qpos[5] = rng.uniform(0, 1.1)
        mujoco.mj_forward(sim.model, sim.data)
        rotation = sim.data.site_xmat[sim.site].reshape(3, 3)
        predicted = sim.data.site_xpos[sim.site] + rotation @ calibration.local_midpoint(
            sim.data.qpos[5]
        )
        expected = (
            sim.data.geom_xpos[calibration.fixed] + sim.data.geom_xpos[calibration.moving]
        ) / 2
        np.testing.assert_allclose(predicted, expected, atol=1e-12)


def test_compensation_keeps_height_and_centers_horizontal_opening():
    calibration = FingerMidpointCalibration(SO101Simulation())
    center = np.array([0.22, 0.055, 0.713])
    for opening in (0, 0.3, 1.1):
        target, rotation = calibration.tcp_target(center, opening, -0.45)
        actual = target + rotation @ calibration.local_midpoint(opening)
        np.testing.assert_allclose(actual[:2], center[:2], atol=1e-10)
        assert target[2] == center[2]
        np.testing.assert_allclose(rotation, grasp_rotation(target, -0.45))


def test_calibration_does_not_change_plant_state():
    sim = SO101Simulation()
    before = sim.data.qpos.copy()
    FingerMidpointCalibration(sim).tcp_target([0.22, 0.055, 0.713], 0.5, 0)
    np.testing.assert_array_equal(sim.data.qpos, before)
    with pytest.raises(ValueError):
        FingerMidpointCalibration(sim).local_midpoint(float("nan"))


def test_frozen_roster_has_complete_pairs_and_independent_development():
    cases = roster(False)
    assert len(cases) == 72
    assert len({(p, m) for p, _, _, m in cases}) == 72
    for method, count in (("joint", 27), ("center_cartesian", 27), ("open", 9), ("offset", 9)):
        assert sum(m == method for _, _, _, m in cases) == count
    assert not {xy for _, xy, _, _ in cases} & {xy for _, xy, _, _ in roster(True)}
