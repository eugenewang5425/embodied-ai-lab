"""Playback clock, independent of rendering rate and Tk callbacks."""

import time

import numpy as np


class ReplayClock:
    def __init__(self, timestamps):
        self.timestamps = np.asarray(timestamps)
        self.frame = 0
        self.playing = False
        self.speed = 1.0
        self._anchor_wall = 0.0
        self._anchor_sim = float(self.timestamps[0])

    def seek(self, frame, now=None):
        self.frame = int(np.clip(frame, 0, len(self.timestamps) - 1))
        self._anchor_sim = float(self.timestamps[self.frame])
        self._anchor_wall = time.perf_counter() if now is None else now

    def play(self, now=None):
        if self.frame == len(self.timestamps) - 1:
            self.seek(0, now)
        self.seek(self.frame, now)
        self.playing = True

    def pause(self, now=None):
        self.advance(now)
        self.playing = False

    def set_speed(self, speed, now=None):
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("playback speed must be finite and positive")
        self.advance(now)
        self.speed = float(speed)
        self.seek(self.frame, now)

    def advance(self, now=None):
        if self.playing:
            now = time.perf_counter() if now is None else now
            target = self._anchor_sim + max(0.0, now - self._anchor_wall) * self.speed
            self.frame = int(
                np.clip(
                    np.searchsorted(self.timestamps, target, side="right") - 1,
                    0,
                    len(self.timestamps) - 1,
                )
            )
            if self.frame == len(self.timestamps) - 1:
                self.playing = False
        return self.frame
