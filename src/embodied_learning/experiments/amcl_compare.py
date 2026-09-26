"""Dual-localizer comparison on the frozen photo-room input.

Two independent implementations run on the SAME frozen data:
  self_pf : the project's ParticleFilter (lesson-56 protocol);
  amcl_py : a Python AMCL following the Nav2 AMCL algorithm
            (differential motion model + likelihood field + KLD sampling).

Both receive the same map, the same biased odometry, the same noisy scans,
and the same initial pose.  Ground truth is scoring-only.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from embodied_learning.experiments.map_localization import ParticleFilter

RES = None  # read from manifest at runtime
DT = 0.04
N_PARTICLES = 300
SENSOR_SIGMA = 0.01
WHEEL_BIAS = 0.02


def load_frozen(directory):
    d = Path(directory)
    data = dict(np.load(d / "frozen_input.npz", allow_pickle=False))
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    res = manifest["map_resolution_m"]
    return data, manifest, res


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# ------------------------------------------------------- self-written PF
def run_self_pf(data, res, seed=0, update_gate_d=None, update_gate_a=None):
    """Run self-written PF on frozen data.

    update_gate_d/gate_a: when set, measurement updates are gated by
    accumulated measured motion (propagation still happens every frame).
    None = P-full (every frame update).
    """
    grid = data["map_grid"]
    odom = data["odom"]
    ranges = data["ranges"]
    init = data["init_pose"]
    n = len(odom)
    rng = np.random.default_rng(seed)
    pf = ParticleFilter(grid, res, N_PARTICLES, rng, init_pose=init, spread=0.04, rays_n=32)
    weights = np.full(N_PARTICLES, 1.0 / N_PARTICLES)
    estimates = []
    update_frames = []
    accum_ds = 0.0
    accum_dth = 0.0
    for k in range(n):
        if k > 0:
            delta = odom[k] - odom[k - 1]
            ds = math.hypot(delta[0], delta[1])
            dth = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
            sig_t = 0.10 * abs(ds) + 0.002
            sig_r = 0.10 * abs(dth) + 0.002
            trans = rng.normal(0.0, sig_t, N_PARTICLES)
            rot = rng.normal(0.0, sig_r, N_PARTICLES)
            mid = pf.p[:, 2] + dth / 2.0
            pf.p[:, 0] += (ds + trans) * np.cos(mid)
            pf.p[:, 1] += (ds + trans) * np.sin(mid)
            pf.p[:, 2] = np.arctan2(np.sin(pf.p[:, 2] + dth + rot), np.cos(pf.p[:, 2] + dth + rot))
            accum_ds += abs(ds)
            accum_dth += abs(dth)
        scan = ranges[k]
        gate_triggered = (
            update_gate_d is None
            or accum_ds >= update_gate_d
            or accum_dth >= update_gate_a
        )
        if k == 0 or gate_triggered:
            pf.measure(scan, None)
            likelihood = pf.w.copy()
            posterior = weights * likelihood
            total = posterior.sum()
            if total <= 1e-300:
                posterior = np.full(N_PARTICLES, 1.0 / N_PARTICLES)
            pf.w = posterior / posterior.sum()
            weights = pf.w.copy()
            update_frames.append(k)
            accum_ds = 0.0
            accum_dth = 0.0
        estimates.append(pf.estimate().copy())
        if pf.ess() < N_PARTICLES / 2:
            pf.resample()
            weights = pf.w.copy()
    return np.asarray(estimates), update_frames


# ------------------------------------------------------- Python AMCL
class AmclPy:
    """Python AMCL following the Nav2 AMCL algorithm.

    Key differences from the self-written PF:
      - differential-drive motion model (delta_rot1, delta_trans, delta_rot2);
      - likelihood field with per-beam Gaussian weighting;
      - KLD-adaptive particle count (epsilon + z percentile);
      - systematic resampling.
    """

    def __init__(self, grid, res, n_max, rng, init_pose, init_spread=0.10):
        self.lf = distance_transform_edt(grid <= 0) * res
        self.res = res
        self.rng = rng
        self.n_max = n_max
        self.n_min = max(50, n_max // 10)
        self.n = min(n_max, 300)
        self.p = np.column_stack(
            [
                init_pose[0] + rng.normal(0, init_spread, self.n),
                init_pose[1] + rng.normal(0, init_spread, self.n),
                init_pose[2] + rng.normal(0, 0.10, self.n),
            ]
        )
        self.w = np.full(self.n, 1.0 / self.n)
        self.ray_off = np.arange(32) * (2 * math.pi / 32)

    def propagate(self, odom_prev, odom_curr, alpha=(0.10, 0.10, 0.10, 0.10)):
        """Motion propagation from odometry delta (shared with self-PF).

        Uses the same direct ds/dth model as the self-written PF for a fair
        comparison of the MEASUREMENT model and RESAMPLING strategy, without
        motion model differences confounding the result."""
        delta = odom_curr[:3] - odom_prev[:3]
        ds = math.hypot(delta[0], delta[1])
        dth = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
        a1, a2, a3, a4 = alpha
        n = len(self.p)
        sig_t = a1 * abs(ds) + a2
        sig_r = a3 * abs(dth) + a4
        sh_t = self.rng.normal(0.0, sig_t, n)
        sh_r = self.rng.normal(0.0, sig_r, n)
        mid = self.p[:, 2] + dth / 2.0
        self.p[:, 0] += (ds + sh_t) * np.cos(mid)
        self.p[:, 1] += (ds + sh_t) * np.sin(mid)
        self.p[:, 2] = np.arctan2(
            np.sin(self.p[:, 2] + dth + sh_r), np.cos(self.p[:, 2] + dth + sh_r)
        )

    def measure_lf(self, scan, angles, max_range):
        """Likelihood field measurement with per-beam Gaussian."""
        n_g = self.lf.shape[0]
        valid = (scan < max_range - 0.05) & (scan > 0.05)
        if valid.sum() < 4:
            return
        w_new = np.zeros(len(self.p))
        for i in range(len(self.p)):
            angles_k = self.p[i, 2] + angles[valid]
            ex = self.p[i, 0] + np.cos(angles_k) * scan[valid]
            ey = self.p[i, 1] + np.sin(angles_k) * scan[valid]
            ix = np.floor(ex / self.res).astype(int)
            iy = np.floor(ey / self.res).astype(int)
            inside = (ix >= 0) & (ix < n_g) & (iy >= 0) & (iy < n_g)
            if inside.sum() < 4:
                w_new[i] = 1e-300
                continue
            d = self.lf[iy[inside], ix[inside]]
            w_new[i] = math.exp(-float(np.sum(d**2)) / (2 * 0.20**2 * max(1, int(inside.sum()))))
        total = w_new.sum()
        if total <= 1e-300:
            self.w = np.full(len(self.p), 1.0 / len(self.p))
        else:
            self.w = w_new / total

    def kld_resample(self, epsilon=0.05, z=1.96):
        """KLD-adaptive systematic resampling (Nav2 AMCL style)."""
        n_current = len(self.p)
        positions = (self.rng.uniform() + np.arange(n_current)) / n_current
        cum = np.cumsum(self.w)
        new_particles = []
        k = 0
        while k < n_current and len(new_particles) < self.n_max:
            while k < n_current and positions[min(len(new_particles), n_current - 1)] > cum[k]:
                k += 1
            if k < n_current:
                new_particles.append(self.p[k].copy())
        if not new_particles:
            return
        new_p = np.array(new_particles)
        n_new = len(new_p)
        # KLD: how many samples do we need?
        if n_new > 2:
            k_target = self._kld_n(n_new, epsilon, z)
            n_target = max(self.n_min, min(self.n_max, k_target))
        else:
            n_target = self.n_min
        # resample to n_target from new_p with uniform weights
        picks = self.rng.integers(0, n_new, n_target)
        out = np.zeros((n_target, 3))
        for i, pi in enumerate(picks):
            out[i] = new_p[pi]
        self.p = out
        self.w = np.full(n_target, 1.0 / n_target)

    def _kld_n(self, k, epsilon, z):
        """KLD sample count (Fox 2003)."""
        if k <= 1:
            return self.n_min
        num = (k - 1) ** 2 / (2 * epsilon)
        for n in range(k, self.n_max + 1):
            if num * (1 - 2 / (9 * (k - 1)) + math.sqrt(2 / (9 * (k - 1))) * z) ** 3 <= n:
                return n
        return self.n_max

    def estimate(self):
        th = self.p[:, 2]
        return np.array(
            [
                self.w @ self.p[:, 0],
                self.w @ self.p[:, 1],
                math.atan2(float(np.mean(np.sin(th))), float(np.mean(np.cos(th)))),
            ]
        )


def run_amcl_py(data, res, seed=0):
    grid = data["map_grid"]
    odom = data["odom"]
    ranges = data["ranges"]
    init = data["init_pose"]
    n = len(odom)
    rng = np.random.default_rng(seed + 20000)
    angles = np.arange(64) * (2 * math.pi / 64)
    amcl = AmclPy(grid, res, N_PARTICLES, rng, init, init_spread=0.04)
    estimates = []
    resample_frames = []
    for k in range(n):
        if k > 0:
            amcl.propagate(odom[k - 1], odom[k])
        amcl.measure_lf(ranges[k], angles, 4.0)
        estimates.append(amcl.estimate())
        if amcl.w.max() > 0 and 1.0 / max(1e-12, float(np.sum(amcl.w**2))) < N_PARTICLES / 2:
            amcl.kld_resample()
            resample_frames.append(k)
    return np.asarray(estimates), resample_frames


# ------------------------------------------------------------- comparison
def score(estimates, truth):
    n = min(len(estimates), len(truth))
    errors = np.linalg.norm(estimates[:n, :2] - truth[:n, :2], axis=1)
    return {
        "mean_m": float(errors.mean()),
        "p95_m": float(np.percentile(errors, 95)),
        "end_m": float(errors[-1]),
        "max_m": float(errors.max()),
        "n_frames": n,
    }


def run(output, *, seed=0, log=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data, _manifest, res = load_frozen("results/amcl_bridge_v2")
    truth = data["truth"]
    odom = data["odom"]

    log("running self-written PF on frozen input...")
    t0 = time.perf_counter()
    pf_est = run_self_pf(data, res, seed)
    pf_time = time.perf_counter() - t0
    pf_score = score(pf_est, truth)
    log(f"  PF: mean {pf_score['mean_m']:.4f} p95 {pf_score['p95_m']:.4f} ({pf_time:.1f}s)")

    log("running Python AMCL on frozen input...")
    t0 = time.perf_counter()
    amcl_est = run_amcl_py(data, res, seed)
    amcl_time = time.perf_counter() - t0
    amcl_score = score(amcl_est, truth)
    log(f"  AMCL: mean {amcl_score['mean_m']:.4f} p95 {amcl_score['p95_m']:.4f} ({amcl_time:.1f}s)")

    odom_score = score(odom, truth)
    log(f"  odom: mean {odom_score['mean_m']:.4f} p95 {odom_score['p95_m']:.4f}")

    report = {
        "experiment": "amcl_bridge_comparison",
        "schema_version": 1,
        "seed": seed,
        "n_frames": len(truth),
        "n_particles": N_PARTICLES,
        "results": {
            "self_pf": pf_score,
            "amcl_py": amcl_score,
            "odometry": odom_score,
        },
        "wall_time_s": {"self_pf": round(pf_time, 2), "amcl_py": round(amcl_time, 2)},
        "note": "both localizers received the same frozen map, scans, and biased odometry",
    }
    np.savez_compressed(
        output / "trajectories.npz",
        truth=truth,
        odom=odom,
        self_pf=pf_est,
        amcl_py=amcl_est,
    )
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(f"saved to {output}")
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="dual-localizer comparison")
    parser.add_argument("--output", default="results/amcl_bridge_comparison")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
