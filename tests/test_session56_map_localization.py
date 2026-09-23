"""Lesson 56: mapping-localization separation - contracts and records."""

import hashlib
import json

import numpy as np
import pytest


def mod():
    import embodied_learning.experiments.map_localization as mm

    return mm


def test_config_guards():
    from embodied_learning.experiments.map_localization import MapLocConfig

    with pytest.raises(ValueError):
        MapLocConfig(pf_particles=10)
    with pytest.raises(ValueError):
        MapLocConfig(pf_rays=4)
    with pytest.raises(ValueError):
        MapLocConfig(kidnap_fraction=1.0)


def test_mapping_frame_selection_uses_estimated_arc():
    mm = mod()
    estimated = np.array(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
    )
    assert mm.first_lap_frames(estimated, lap_len_m=10.0) == [0, 1, 2]
    extended = np.array([[float(x), 0.0, 0.0] for x in range(0, 40, 5)])
    assert mm.first_lap_frames(extended) == [0, 1, 2, 3]


def test_marker_map_paints_physical_observation_at_estimated_pose():
    mm = mod()
    observed = np.zeros((3, 12))
    observed[0, 2] = 1.0
    observed[2, 5] = 1.0
    estimated = np.array([[1.1, 2.1, 0.0], [1.1, 2.1, 0.0], [3.1, 4.1, 0.0], [3.1, 4.1, 0.0]])
    marker_map = mm.marker_map_from_observations(observed, estimated, [0, 1, 2], world_size_m=10.0)
    assert marker_map[4, 2, 2] == 1.0
    assert marker_map[8, 6, 5] == 1.0


def test_scan_alignment_tolerates_quantization_but_detects_drift():
    mm = mod()
    reference = np.zeros((20, 20))
    reference[10, 10] = 1.0
    near = np.zeros_like(reference)
    near[10, 11] = 1.0
    far = np.zeros_like(reference)
    far[10, 14] = 1.0
    assert mm.scan_map_alignment(near, reference, 0.15)["precision"] == 1.0
    assert mm.scan_map_alignment(far, reference, 0.15)["precision"] == 0.0


def test_particle_filter_likelihood_field_basics():
    mm = mod()
    grid = np.zeros((100, 100))
    grid[50, :] = 1.0  # a wall across the middle
    rng = np.random.default_rng(0)
    pf = mm.ParticleFilter(grid, 0.1, 50, rng)
    assert pf.lf.shape == (100, 100)
    # the field is 0 on the wall and grows away from it
    assert pf.lf[50, 50] == 0.0
    assert pf.lf[40, 50] == pytest.approx(1.0)
    assert pf.lf[0, 50] == pytest.approx(5.0)


def test_pf_measure_prefers_pose_matching_the_map():
    mm = mod()
    grid = np.zeros((100, 100))
    grid[50, :] = 1.0  # wall at y = 5.0 m
    rng = np.random.default_rng(1)
    pf = mm.ParticleFilter(grid, 0.1, 40, rng)
    pf.p = np.zeros((40, 3))
    pf.p[:20, 0] = np.linspace(1, 9, 20)
    pf.p[:20, 1] = 4.9  # 0.1 m before the wall: observed 0.1 m ranges fit
    pf.p[:20, 2] = 0.0
    pf.p[20:, 0] = np.linspace(1, 9, 20)
    pf.p[20:, 1] = 3.0  # 2 m before the wall: observed 0.1 m ranges mismatch
    pf.p[20:, 2] = 0.0
    observed = np.full(64, 0.1)
    observed[16:48] = 20.0  # south-facing rays see nothing
    pf.measure(observed, None)
    # the wall-adjacent batch dominates the weights
    assert float(pf.w[:20].sum()) > 0.9


def test_pf_propagate_body_moves_along_heading():
    mm = mod()
    grid = np.zeros((200, 200))
    rng = np.random.default_rng(2)
    pf = mm.ParticleFilter(grid, 0.1, 30, rng)
    pf.p[:, 0] = 10.0
    pf.p[:, 1] = 10.0
    pf.p[:, 2] = 0.0  # facing +x
    pf.propagate_body(0.5, 0.0)  # straight ahead 0.5*DT... (v=0.5 m/s)
    assert float(np.mean(pf.p[:, 0])) > 10.0
    assert float(np.mean(pf.p[:, 1])) == pytest.approx(10.0, abs=0.05)


def test_reseed_candidates_rank_by_marker_match():
    mm = mod()
    marker_map = np.zeros((20, 20, 12))
    marker_map[10, 5, 3] = 10.0  # strong marker 3 at cell (row 10, col 5)
    marker_map[2, 2, 4] = 1.0
    obs = np.zeros(12)
    obs[3] = 5.0
    cands = mm.reseed_candidates(marker_map, obs)
    # the strong-marker cell ranks first
    assert cands[0] == (2.75, 5.25)  # (x, y) center of col 5, row 10


def test_kidnap_route_is_free_space_and_continuous():
    mm = mod()
    from embodied_learning.experiments.grid_nav import GridNavConfig
    from embodied_learning.experiments.map_localization import MapLocConfig

    base = GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1)
    matcher = MapLocConfig(base=base).matcher
    obstacles, walls = mm._build_world("ISO", 0, base)
    map_stream = mm_mod_collect(base, matcher, obstacles, walls)
    loc_stream = mm_mod_collect(base, matcher, obstacles, walls)
    route = mm.kidnap_route(map_stream["truth"], loc_stream["truth"], 1700)
    assert route.shape == loc_stream["truth"].shape
    steps = np.linalg.norm(np.diff(route[1700:], axis=0), axis=1)
    assert steps.max() < 0.2


def mm_mod_collect(base, matcher, obstacles, walls):
    from embodied_learning.experiments.aniso_env import PATROL_INSET
    from embodied_learning.experiments.matcher_engineering import collect_patrol_laps

    return collect_patrol_laps(
        obstacles, (3.5, 3.5, 0.0), matcher, walls=walls, patrol=PATROL_INSET
    )


# ------------------------------------------------------------- records
@pytest.fixture(scope="module")
def light_stream(tmp_path_factory):
    from embodied_learning.experiments.grid_nav import GridNavConfig
    from embodied_learning.experiments.map_localization import MapLocConfig, run_experiment

    conf = MapLocConfig(
        base=GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1),
        pf_particles=100,
    )
    out = tmp_path_factory.mktemp("maploc56") / "run"
    report = run_experiment(out, seed=0, config=conf, log=None)
    return out, report


def test_light_run_contract(light_stream):
    out, report = light_stream
    assert report["experiment"] == "map_localization_lesson56"
    assert report["schema_version"] == 5
    r = report["hypothesis"]["results"]
    for key in ("collector_met", "bounded_met", "kidnap_met", "smear_ratio", "map_hit_rate"):
        assert key in r
    assert (out / "trajectories.npz").exists()
    assert (out / "summary.json").exists()
    with np.load(out / "trajectories.npz", allow_pickle=False) as npz:
        curves = npz["kidnap_pf_err"]
        xy = npz["truth_chain"][:, :2]
    # The lesson-54 inset patrol stays around 3.5..7.5 m. A regression to
    # collect_patrol_laps' outer-square default reaches 1..9 m instead.
    assert np.min(xy) > 2.5
    assert np.max(xy) < 8.5
    for i, trial in enumerate(r["kidnap_trials"]):
        assert trial["end_err_m"] == pytest.approx(curves[i, trial["recovery_end_frame"]])


def test_demo_loads_current_record(light_stream):
    from embodied_learning.map_localization_demo import load_replays

    out, report = light_stream
    replay = load_replays(out)
    assert replay["report"]["schema_version"] == report["schema_version"]
    assert len(replay["pf_self_err"]) == len(replay["est_err_curve"])


def test_record_tamper_rejection(light_stream):
    out, report = light_stream
    path = out / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = out / "summary.json"
    original = summary.read_text(encoding="utf-8")
    data = json.loads(original)
    data["hypothesis"]["results"]["errors"]["pf_mean_m"] = 0.01
    summary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AssertionError):
        recompute_errors(light_stream)
    summary.write_text(original, encoding="utf-8")


def recompute_errors(light_stream):
    out, _report = light_stream
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    with np.load(out / "trajectories.npz", allow_pickle=False) as npz:
        pf_err = npz["pf_true_err"]
        est_err = npz["est_err_curve"]
    errs = data["hypothesis"]["results"]["errors"]
    assert abs(float(np.mean(pf_err)) - errs["pf_mean_m"]) < 1e-9
    assert abs(float(np.mean(est_err)) - errs["est_mean_m"]) < 1e-9
    assert errs["pf_mean_m"] < 1.0  # the tampered value is still visible


@pytest.mark.slow
def test_seed_determinism(light_stream, tmp_path):
    from embodied_learning.experiments.grid_nav import GridNavConfig
    from embodied_learning.experiments.map_localization import MapLocConfig, run_experiment

    conf = MapLocConfig(
        base=GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1),
        pf_particles=100,
    )
    _out, first = light_stream
    second = run_experiment(tmp_path / "b", seed=0, config=conf, log=None)
    pop = lambda r: json.dumps(
        {k: v for k, v in r.items() if k != "wall_time_s"}, sort_keys=True, indent=1
    )
    assert pop(first) == pop(second)
