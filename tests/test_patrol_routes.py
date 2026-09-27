"""Patrol route generator tests (roadmap Phase 1 / P5).

Hand-made room grids give computable answers (straight lengths, wall offsets);
the photo-room integration cross-checks the lesson-64 reachability
classification through the NEW module; a benchmark-scene grid validates the
row/col <-> world orientation convention against true_occupancy.
"""

import math

import numpy as np
import pytest

from embodied_learning.experiments.patrol_routes import (
    REASON_INFLATED,
    REASON_NO_PATH,
    REASON_OCCUPIED,
    REASON_OUTSIDE,
    REASON_START_BLOCKED,
    PatrolMap,
    PatrolSpec,
    build_route,
)


def room_grid(n=20, wall=1):
    """Open square room with solid border walls."""
    g = np.zeros((n, n), dtype=bool)
    g[:wall, :] = g[-wall:, :] = True
    g[:, :wall] = g[:, -wall:] = True
    return g


def test_to_cell_to_world_roundtrip():
    g = room_grid()
    pmap = PatrolMap(g, 0.1, origin=(0.0, 0.0))
    for rc in ((2, 3), (10, 10), (18, 5)):
        x, y = pmap.to_world(*rc)
        assert pmap.to_cell((x, y)) == rc


def test_straight_route_known_length_and_clearance():
    pmap = PatrolMap(room_grid(), 0.1)
    route = build_route(
        pmap, PatrolSpec((0.45, 0.45), (("far", (1.55, 1.55)),), close_loop=False), 0.15
    )
    assert route.feasible and route.envelope_ok
    assert route.length_m == pytest.approx(1.1 * math.sqrt(2), rel=0.05)
    # EDT measures to occupied CELL CENTERS: endpoint clearance ~ 0.42 m
    assert 0.0 < route.min_clearance_m <= 0.5
    assert route.rejected_points == []


def test_point_in_obstacle_rejected():
    g = room_grid()
    g[10, 10] = True
    pmap = PatrolMap(g, 0.1)
    route = build_route(
        pmap, PatrolSpec((0.45, 0.45), (("box", (1.05, 1.05)),), close_loop=False), 0.15
    )
    assert not route.feasible
    assert route.rejected_points == [("box", REASON_OCCUPIED)]


def test_walled_off_point_rejected_no_path():
    g = room_grid()
    g[2:18, 10] = True  # full internal wall, no door
    pmap = PatrolMap(g, 0.1)
    route = build_route(
        pmap, PatrolSpec((0.45, 0.45), (("other_side", (1.45, 1.45)),), close_loop=False), 0.15
    )
    assert not route.feasible
    assert route.rejected_points == [("other_side", REASON_NO_PATH)]


def test_point_blocked_by_envelope_inflation():
    pmap = PatrolMap(room_grid(), 0.1)
    # cell (2,2) is free but only 0.1 m from the walls - a 0.15 m robot
    # cannot occupy it; with radius 0.05 it can
    spec = PatrolSpec((1.0, 1.0), (("corner_hug", (0.25, 0.25)),), close_loop=False)
    # (0.25, 0.25) sits one cell from the walls: free, but inside the 0.15 m
    # robot's inflation - the point-in-obstacle vs envelope distinction
    big = build_route(pmap, spec, 0.15)
    small = build_route(pmap, spec, 0.05)
    assert not big.feasible
    assert big.rejected_points == [("corner_hug", REASON_INFLATED)]
    assert small.feasible


def test_start_blocked_and_outside_grid():
    pmap = PatrolMap(room_grid(), 0.1)
    inside_wall = build_route(
        pmap, PatrolSpec((0.05, 0.05), (("x", (1.0, 1.0)),), close_loop=False), 0.15
    )
    assert inside_wall.rejected_points[0][1] == REASON_START_BLOCKED
    outside = build_route(
        pmap, PatrolSpec((1.0, 1.0), (("x", (5.0, 5.0)),), close_loop=False), 0.15
    )
    assert outside.rejected_points == [("x", REASON_OUTSIDE)]


def test_close_loop_returns_and_grows_length():
    pmap = PatrolMap(room_grid(), 0.1)
    spec_open = PatrolSpec(
        (0.45, 0.45), (("a", (1.5, 0.45)), ("b", (1.5, 1.5))), close_loop=False
    )
    spec_closed = PatrolSpec(
        (0.45, 0.45), (("a", (1.5, 0.45)), ("b", (1.5, 1.5))), close_loop=True
    )
    ro = build_route(pmap, spec_open, 0.15)
    rc = build_route(pmap, spec_closed, 0.15)
    assert ro.feasible and rc.feasible
    assert rc.length_m > ro.length_m  # return leg is added
    assert np.allclose(rc.waypoints_world[0], rc.waypoints_world[-1], atol=1e-9)
    assert rc.legs[-1]["to"] == "__return_to_start__"


def test_protocol_tag_recorded():
    pmap = PatrolMap(room_grid(), 0.1)
    diag = build_route(
        pmap, PatrolSpec((0.45, 0.45), (("p", (1.5, 1.5)),), protocol="diagnostic"), 0.15
    )
    est = build_route(
        pmap, PatrolSpec((0.45, 0.45), (("p", (1.5, 1.5)),), protocol="estimation"), 0.15
    )
    assert diag.protocol == "diagnostic" and est.protocol == "estimation"
    with pytest.raises(ValueError):
        PatrolSpec((0.45, 0.45), (("p", (1.5, 1.5)),), protocol="truth_replay")


def test_photo_room_cross_validates_lesson64():
    """The generator must independently reproduce the lesson-64
    classification on the same prior avoidance grid: 4 reachable tasks
    feasible, the bed-footprint goal rejected, the closed lap feasible."""
    from embodied_learning.experiments.patrol_routes import photo_room_tasks

    pmap, tasks = photo_room_tasks()
    reachable_names = [
        n for n, t in tasks.items() if n not in ("bed_footprint_unreachable", "inspection_lap_closed")
    ]
    assert len(reachable_names) == 4
    for name in reachable_names:
        route = build_route(pmap, tasks[name], 0.15)
        assert route.feasible and route.envelope_ok, name
    bed = build_route(pmap, tasks["bed_footprint_unreachable"], 0.15)
    assert not bed.feasible
    assert bed.rejected_points[0][1] in (REASON_OCCUPIED, REASON_INFLATED)
    lap = build_route(pmap, tasks["inspection_lap_closed"], 0.15)
    assert lap.feasible and lap.envelope_ok
    assert lap.length_m > 3.0  # a real lap, not a degenerate stub


def test_benchmark_scene_grid_orientation():
    """On the ISO benchmark world, generator feasibility must agree with the
    true_occupancy grid the paired pilot actually used (orientation check)."""
    from embodied_learning.experiments.aniso_env import _build_world
    from embodied_learning.experiments.grid_nav import build_walls, true_occupancy
    from embodied_learning.experiments.map_localization import MapLocConfig

    base = MapLocConfig().base
    obstacles, walls = _build_world("ISO", 0, base)
    physical = build_walls(base.world_size_m, base.res_m) if walls is None else walls
    true_map = np.asarray(
        true_occupancy(obstacles, physical, base.grid_cells, base.res_m), dtype=bool
    )
    pmap = PatrolMap(true_map, base.res_m)
    occ = np.argwhere(true_map)
    from embodied_learning.experiments.patrol_routes import _clearance_field

    clearance = _clearance_field(pmap)
    # inflation blocks cells within a 2-cell disk of any obstacle (EDT
    # <= 0.2 m in cell centers); require strictly more clearance so the
    # start/goal pass the envelope check regardless of clutter position
    wide_free = np.argwhere((~true_map) & (clearance >= 0.25))
    assert len(wide_free) >= 2
    start_rc = tuple(wide_free[len(wide_free) // 2])
    goal_rc = tuple(wide_free[0])
    start_xy = pmap.to_world(*start_rc)
    goal_xy = pmap.to_world(*goal_rc)
    route = build_route(
        pmap, PatrolSpec(start_xy, (("g", goal_xy),), close_loop=False), 0.15
    )
    assert route.feasible
    assert route.envelope_ok
    assert route.length_m > 0
    # a goal inside a wall must be rejected with the occupancy reason
    occ_rc = tuple(occ[len(occ) // 2])
    occ_xy = pmap.to_world(*occ_rc)
    blocked = build_route(
        pmap, PatrolSpec(start_xy, (("wall", occ_xy),), close_loop=False), 0.15
    )
    assert not blocked.feasible
    assert blocked.rejected_points == [("wall", REASON_OCCUPIED)]
