from __future__ import annotations

import logging
import time


class SceneBloatMonitor:
    def __init__(self, sample_interval: float = 1.0, growth_streak_limit: int = 3):
        self.sample_interval = float(sample_interval)
        self.growth_streak_limit = int(growth_streak_limit)
        self._next_sample_ts = 0.0
        self._prev_primitive_count: int | None = None
        self._growth_streak = 0
        self._warned = False

    def sample(self, overlay) -> None:
        now = time.monotonic()
        if now < self._next_sample_ts:
            return
        self._next_sample_ts = now + self.sample_interval

        try:
            render_list = overlay.get_render_list()
        except Exception as exc:
            logging.info('scene bloat sample failed: %s', exc)
            return

        primitive_count = sum(len(cmds) for _, _, cmds in render_list)
        scene_count = len(render_list)
        prev = self._prev_primitive_count

        if prev is not None and primitive_count > prev:
            self._growth_streak += 1
            if not self._warned and self._growth_streak >= self.growth_streak_limit:
                logging.warning(
                    'overlay bloat suspected scenes=%d primitives=%d prev=%d delta=%+d streak=%d',
                    scene_count,
                    primitive_count,
                    prev,
                    primitive_count - prev,
                    self._growth_streak,
                )
                self._warned = True
        else:
            self._growth_streak = 0
            self._warned = False

        self._prev_primitive_count = primitive_count


_SCENE_BLOAT_MONITOR = SceneBloatMonitor()


def sample_scene_bloat(overlay) -> None:
    _SCENE_BLOAT_MONITOR.sample(overlay)
