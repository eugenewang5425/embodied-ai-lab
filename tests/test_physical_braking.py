"""Independent force-plant references for full-state stopping forecasts."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from embodied_learning.contact_motion import ContactPlant
from embodied_learning.navigation_geometry import NavigationRig, point_clearance, rotation
from embodied_learning.physical_braking import (
    PhysicalStopPredictor,
    measured_body_velocity,
    swept_point_clearance,
)


@pytest.mark.parametrize(
    "velocity", [[0.2, 0, 0], [0.2, 0.04, 0.3], [0, 0.08, -0.4], [-0.15, -0.02, 0.2]]
)
def test_forecast_matches_independent_plant_including_sideways_motion(velocity):
    rig = replace(NavigationRig(), margin_m=0)
    predictor = PhysicalStopPredictor(rig)
    _, commands, expected = predictor.predict(velocity)
    plant = ContactPlant(np.empty((0, 6)), rig, np.zeros(3))
    plant.data.qvel[:] = velocity
    mujoco.mj_forward(plant.model, plant.data)
    actual = [np.r_[plant.data.qpos, plant.data.qvel].copy()]
    for command in commands:
        state, _, contacts = plant.step(command)
        assert not contacts.any()
        actual.extend(state)
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
    assert np.max(np.abs(expected[-1, 3:])) < 1e-4


def test_stationary_forward_speed_does_not_hide_sideways_braking_distance():
    rig = replace(NavigationRig(), margin_m=0)
    poses, _, _ = PhysicalStopPredictor(rig).predict([0, 0.08, 0])
    assert poses[-1, 1] > 0.007


def test_measured_velocity_rotates_without_dropping_lateral_component():
    yaw = 0.6
    world = np.r_[np.array([0.2, 0.04]) @ rotation(yaw).T, 0.3]
    np.testing.assert_allclose(measured_body_velocity(world, [4, 5, yaw]), [0.2, 0.04, 0.3])


def test_pending_intentions_extend_stop_and_inputs_are_read_only():
    rig = replace(NavigationRig(), margin_m=0)
    predictor = PhysicalStopPredictor(rig)
    velocity = np.array([0.2, 0, 0])
    target = np.array([0.2, 0])
    pending = [(target, False, 9), (target, False, 10)]
    a, _, _ = predictor.predict(velocity)
    b, _, _ = predictor.predict(velocity, pending, np.zeros(2), True)
    assert 0.039 < a[-1, 0] < 0.042
    assert b[-1, 0] - a[-1, 0] == pytest.approx(0.04, abs=1e-6)
    np.testing.assert_array_equal(velocity, [0.2, 0, 0])
    np.testing.assert_array_equal(target, [0.2, 0])


def test_predictor_has_no_environment_obstacles_and_rejects_missing_state():
    predictor = PhysicalStopPredictor(NavigationRig())
    assert predictor.model.ngeom == 3
    with pytest.raises(ValueError, match="full finite"):
        predictor.predict([0.2, 0])


@pytest.mark.parametrize("margin", [0, 0.02])
def test_batched_clearance_matches_scalar_body_parts(margin):
    rng = np.random.default_rng(12)
    points = rng.uniform([-1, -1, 0], [1, 1, 0.7], (300, 3))
    poses = rng.uniform([-0.3, -0.3, -1], [0.3, 0.3, 1], (77, 3))
    rig = NavigationRig()
    assert swept_point_clearance(points, poses, rig, margin) == pytest.approx(
        point_clearance(points, poses, rig, margin), abs=1e-12
    )


def test_omitting_state_copies_does_not_change_forecast_or_commands():
    predictor = PhysicalStopPredictor(NavigationRig())
    a, u, _ = predictor.predict([0.2, 0.04, 0.3])
    b, v, _ = predictor.predict([0.2, 0.04, 0.3], record_states=False)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(u, v)
