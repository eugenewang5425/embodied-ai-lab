"""Causal pose resets plus encoder propagation, without truth or noise filtering."""

import numpy as np

from embodied_learning.differential_drive import finite_vector
from embodied_learning.odometry import (
    DEFAULT_GEOMETRY,
    encoder_array,
    estimate_poses,
    heading_error,
)


def estimate_with_resets(
    encoder_angles_rad,
    observation_frames,
    observation_poses,
    initial_pose=(0, 0, 0),
    geometry=DEFAULT_GEOMETRY,
):
    """Predict each interval, then replace pose if an observation arrives there.

    Inputs are *corrected measured* cumulative encoders and independently solved
    body poses. Neither ground truth nor future observations enter an update.
    Frame k observes the pose AFTER interval k-1 -> k. Resetting does not reset
    encoder counters, recalibrate their scale, or remove observation noise.
    Return (posterior, prior), with continuous yaw branches for both arrays.
    """
    readings = encoder_array(encoder_angles_rad)
    initial = finite_vector(initial_pose, 3)
    frames = np.asarray(observation_frames)
    observations = np.asarray(observation_poses, dtype=float)
    if (
        frames.ndim != 1
        or (frames.size and not np.issubdtype(frames.dtype, np.integer))
        or np.any(frames < 1)
        or np.any(frames >= len(readings))
        or np.any(np.diff(frames) <= 0)
    ):
        raise ValueError("Observation frames must be increasing integer frames in [1, steps]")
    if observations.shape != (len(frames), 3) or not np.isfinite(observations).all():
        raise ValueError("Expected one finite body pose per observation frame")
    posterior = np.empty((len(readings), 3))
    prior = np.empty_like(posterior)
    posterior[0] = prior[0] = initial
    sample = 0
    for frame in range(1, len(readings)):
        prior[frame] = estimate_poses(
            readings[frame - 1 : frame + 1], posterior[frame - 1], geometry
        )[-1]
        posterior[frame] = prior[frame]
        if sample < len(frames) and frame == frames[sample]:
            posterior[frame, :2] = observations[sample, :2]
            posterior[frame, 2] += heading_error(observations[sample, 2], prior[frame, 2])
            sample += 1
    return posterior, prior
