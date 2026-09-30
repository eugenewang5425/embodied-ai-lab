"""Known answers for the contact plant, mild-contact policy and study design."""

import numpy as np
import pytest

from embodied_learning.contact_motion import (
    FORCE_LIMIT_N,
    TORQUE_LIMIT_NM,
    ContactPlant,
    contact_allowed,
)
from embodied_learning.navigation_geometry import NavigationRig

WALL = np.array([[1, 1.2, -10, 10, 0, 1.0]])


def test_free_motion_acceleration_and_force_budget():
    p = ContactPlant(WALL, NavigationRig(), [0, 0, 0])
    states, forces, contacts = p.step([0.6, 0])
    assert states[-1, 3] == pytest.approx(0.08, abs=1e-10)
    assert states[-1, 0] == pytest.approx(0.00408, abs=1e-10)
    assert np.linalg.norm(forces[:, :2], axis=1).max() <= FORCE_LIMIT_N + 1e-10
    assert abs(forces[:, 2]).max() <= TORQUE_LIMIT_NM
    assert not contacts.any()


def test_contact_blocks_wall_and_persistent_push_is_not_light():
    p = ContactPlant(WALL, NavigationRig(), [0, 0, 0])
    contacts = np.vstack([p.step([0.6, 0])[2] for _ in range(30)])
    assert 0.699 < p.pose[0] < 0.701
    assert contacts[:, 0].max() >= 1
    assert contacts[:, 1].max() > 40
    assert contacts[:, 4].max() > 0.5
    assert not contact_allowed(contacts[:, 1].max(), contacts[:, 3].max(), contacts[:, 4].max())


def test_brief_grazing_can_continue_and_release_from_wall():
    p = ContactPlant(WALL, NavigationRig(), [0.76, 0, 1.54])
    contacts = np.vstack([p.step([0.2, 0.15 if i >= 18 else 0])[2] for i in range(30)])
    assert (contacts[:, 0] > 0).any()
    assert contact_allowed(contacts[:, 1].max(), contacts[:, 3].max(), contacts[:, 4].max())
    assert p.pose[1] > 0.5
    assert p.pose[0] < 0.76
    assert not contacts[-50:, 0].any()


@pytest.mark.parametrize("values", [(40.01, 0, 0), (0, 0.00501, 0), (0, 0, 0.501)])
def test_any_light_contact_limit_can_reject(values):
    assert not contact_allowed(*values)


def test_policy_accepts_boundary_without_requiring_six_cm_clearance():
    assert contact_allowed(40, 0.005, 0.5)


def test_stopped_uses_lateral_speed_too():
    p = ContactPlant(WALL, NavigationRig(), [0, 0, 0])
    p.data.qvel[:] = [0, 0.02, 0]
    assert p.velocity[0] == 0
    assert not p.stopped


def test_registered_cases_are_new_seeds_and_paired_arms():
    from collections import Counter

    from embodied_learning.experiments.contact_navigation import registered_cases

    cases = registered_cases()
    assert len(cases) == 39
    assert Counter(c[5] for c in cases) == {
        "margin_strict": 13,
        "zero_strict": 13,
        "zero_light": 13,
    }
    assert {c[2] for c in cases} == {14, 15, 16}


def test_impossible_mouth_starts_outside_and_can_refuse_without_motion():
    from embodied_learning.experiments.contact_navigation import episode
    from embodied_learning.navigation_robustness import CONDITIONS

    row, data = episode("blocked", "crate", 0, CONDITIONS["delay_2"], "zero_light", max_steps=2)
    assert not row["passed"] and row["status"] == "no_path"
    assert row["contact_events"] == 0
    assert row["travelled_m"] == 0
    assert data["substep_state"].shape == (0, 50, 6)
    assert data["actual_boxes"][-3, 0] == 3
