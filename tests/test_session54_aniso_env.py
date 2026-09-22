"""Lesson 54: anisotropic-environment control + the metric erratum."""

import hashlib
import json
import math

import numpy as np
import pytest


def mod():
    import embodied_learning.experiments.aniso_env as mm

    return mm


def test_segment_obstacle_distance_and_ray_support():
    from embodied_learning.experiments.grid_nav import Obstacle, cast_rays

    seg = Obstacle("segment", 3.0, 1.0, 2.0, 0.0, 0.05)
    assert seg.distance(3.0, 1.0) == pytest.approx(-0.05)
    assert seg.distance(3.0, 1.15) == pytest.approx(0.10)
    assert seg.distance(7.0, 1.0) == pytest.approx(1.95)
    ranges, hit = cast_rays(3.0, 3.0, np.array([-math.pi / 2]), (), (seg,), 20.0)
    assert hit[0] and ranges[0] == pytest.approx(2.0)
    # a parallel ray never hits
    ranges2, hit2 = cast_rays(3.0, 3.0, np.array([0.0]), (), (seg,), 20.0)
    assert not hit2[0]


def test_segment_obstacle_validation():
    from embodied_learning.experiments.grid_nav import Obstacle

    with pytest.raises(ValueError):
        Obstacle("segment", 0.0, 0.0, 0.0, 0.0, 0.05)  # zero half-vector
    with pytest.raises(ValueError):
        Obstacle("segment", 0.0, 0.0, 1.0, 0.0, 0.0)  # zero thickness


def test_polygon_walls_close_the_loop():
    mm = mod()
    walls = mm.polygon_walls([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    assert len(walls) == 4
    # the last wall connects back to the first vertex
    ends = [(w.cx - w.hx, w.cy - w.hy) for w in walls]
    starts = [(w.cx + w.hx, w.cy + w.hy) for w in walls]
    assert any(abs(s[0] - 0.0) < 1e-9 and abs(s[1] - 0.0) < 1e-9 for s in starts)
    assert any(abs(e[0] - 0.0) < 1e-9 and abs(e[1] - 0.0) < 1e-9 for e in ends)


def test_aniso_world_keeps_patrol_corridor_clear():
    mm = mod()
    for seed in (0, 1):
        _v, walls, clutter = mm.build_aniso_world(seed)
        assert len(walls) == mm.POLY_VERTICES
        worst = 9.0
        for i in range(4):
            a, b = mm.PATROL_INSET[i], mm.PATROL_INSET[(i + 1) % 4]
            for t in np.linspace(0, 1, 100):
                px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                d = min(
                    min(w.distance(px, py) for w in walls),
                    min((c.distance(px, py) for c in clutter), default=9.0),
                )
                worst = min(worst, d)
        # the recorded freeze bugs (chord cutting the corner, clutter pocket)
        # both manifested as <= 0 clearance on the patrol line
        assert worst > 0.4


def test_implied_pose_composition():
    from embodied_learning.experiments.matcher_engineering import implied_pose

    est_k = np.array([2.0, 1.0, math.pi / 2])
    delta = np.array([1.0, 0.0, -math.pi / 2])
    implied = implied_pose(delta, est_k)
    # rotate the est pose by -90 deg: (2,1) -> (1,-2), then +x by 1 -> (2,-2)
    np.testing.assert_allclose(implied[:2], [2.0, -2.0], atol=1e-9)
    assert implied[2] == pytest.approx(0.0)


def test_is_correct_implied_thresholds():
    from embodied_learning.experiments.matcher_engineering import is_correct_implied

    truth = np.array([1.0, 1.0, 0.3])
    assert is_correct_implied(np.array([1.2, 1.1, 0.32]), truth)
    assert not is_correct_implied(np.array([2.0, 1.0, 0.3]), truth)
    assert not is_correct_implied(np.array([1.0, 1.0, 0.5]), truth)


# ------------------------------------------------------------- records
@pytest.fixture(scope="module")
def light_stream(tmp_path_factory):
    from embodied_learning.experiments.aniso_env import AnisoConfig, run_experiment
    from embodied_learning.experiments.grid_nav import GridNavConfig

    conf = AnisoConfig(base=GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1))
    out = tmp_path_factory.mktemp("aniso54") / "run"
    report = run_experiment(out, seed=0, config=conf, log=None)
    return out, report


def test_light_run_contract(light_stream):
    out, report = light_stream
    assert report["experiment"] == "aniso_env_lesson54"
    assert report["schema_version"] == 1
    r = report["hypothesis"]["results"]
    for key in ("collector_met", "iso_corrected", "aniso_corrected", "iso_legacy", "aniso_legacy"):
        assert key in r
    # both arms must have completed the patrol (world validity gate)
    assert r["collector_met"]
    assert 0.0 <= r["iso_corrected"] <= 1.0
    assert 0.0 <= r["aniso_corrected"] <= 1.0
    assert (out / "summary.json").exists()
    assert (out / "trajectories.npz").exists()


def test_record_tamper_rejection(light_stream):
    out, report = light_stream
    path = out / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    summary = out / "summary.json"
    original = summary.read_text(encoding="utf-8")
    data = json.loads(original)
    data["hypothesis"]["results"]["iso_corrected"] = 0.99
    summary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AssertionError):
        recompute_iso(light_stream)
    summary.write_text(original, encoding="utf-8")


def recompute_iso(light_stream):
    out, _report = light_stream
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    rows = data["episodes"] if "episodes" in data else []
    # the aggregate is recomputable from the npz arrays: sum of correct / accepted
    with np.load(out / "trajectories.npz", allow_pickle=False) as npz:
        acc = npz["ISO_K_accepted"]
        cor = npz["ISO_K_correct"]
    recomputed = float(cor.sum() / max(1, acc.sum()))
    assert abs(recomputed - data["hypothesis"]["results"]["iso_corrected"]) < 1e-9
    assert rows is not None


@pytest.mark.slow
def test_seed_determinism(tmp_path):
    from embodied_learning.experiments.aniso_env import AnisoConfig, run_experiment
    from embodied_learning.experiments.grid_nav import GridNavConfig

    conf = AnisoConfig(base=GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1))
    first = run_experiment(tmp_path / "a", seed=0, config=conf, log=None)
    second = run_experiment(tmp_path / "b", seed=0, config=conf, log=None)
    pop = lambda r: json.dumps(
        {k: v for k, v in r.items() if k != "wall_time_s"}, sort_keys=True, indent=1
    )
    assert pop(first) == pop(second)
