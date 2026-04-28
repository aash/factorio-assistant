"""PidService — encapsulates all PID movement controller state and operations.

Provides a single point of snail interaction for PID operations.
Can be driven by events from any source (action handlers, gRPC, CLI, etc.).
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Mapping

from assistant.snail_state import SnailState
from assistant import events
from assistant.event_bus import snail_events, SnailEventBus
from assistant.pid_controller import (
    ensure_pid_cfg_loaded,
    save_pid_cfg,
    _run_pid_episode,
    _auto_tune_move_pid_gen,
    _benchmark_move_pid_gen,
    _evaluate_pid_candidate,
)
from leaf.renderers.coords import map_scene_geometry, screen_to_map_coord


class PidService:
    """PID movement controller service.

    Parameters
    ----------
    snail : Snail
        The game interface (ahk, character_coord, cache, etc.).
    snail_state : SnailState
        Persistent PID configuration and running-task state.
    """

    def __init__(self, snail, snail_state: SnailState) -> None:
        self.snail = snail
        self.state = snail_state

    # ── Configuration ──────────────────────────────────────────────────

    def ensure_cfg_loaded(self) -> None:
        """Load PID config from cache into snail_state.pid_cfg."""
        cfg, loaded = ensure_pid_cfg_loaded(
            self.snail, self.state.pid_cfg, self.state.pid_cfg_loaded
        )
        self.state.pid_cfg = cfg
        self.state.pid_cfg_loaded = loaded

    def save_cfg(self) -> None:
        """Persist current PID config to snail cache."""
        save_pid_cfg(self.snail, self.state.pid_cfg)

    def update_params(self, updates: dict[str, float]) -> None:
        """Apply parameter updates and persist."""
        self.state.pid_cfg.update(updates)
        self.save_cfg()
        logging.info(
            "pid params updated: %s",
            {k: self.state.pid_cfg[k] for k in sorted(self.state.pid_cfg.keys())},
        )

    # ── Movement ───────────────────────────────────────────────────────

    def move_to(self, target_x: int, target_y: int) -> dict:
        """Execute a blocking PID move to the given map coordinate.

        Returns the result dict from the PID episode.
        """
        self.ensure_cfg_loaded()

        if self.snail.character_coord is None:
            logging.info("pid move aborted: character_coord is unknown")
            return {"aborted": True}

        target = (int(target_x), int(target_y))
        logging.info(
            "pid move target=%s current=%s",
            target,
            self.snail.character_coord,
        )

        result = _run_pid_episode(self.snail, target, self.state.pid_cfg)
        logging.info(
            "pid move done: target=%s score=%.3f dist=%.2f delay=%.3f switches=%d timeout=%s",
            target,
            float(result["score"]),
            float(result["final_dist"]),
            float(result["delay_mean"]),
            int(float(result["switches"])),
            bool(result["timeout"]),
        )
        return result

    # ── Background tasks (generator-driven) ────────────────────────────

    def start_benchmark(self, radius: int = 20, targets: int = 5) -> None:
        """Start a non-blocking PID benchmark run."""
        self.ensure_cfg_loaded()

        if self.has_background_task():
            logging.info("benchmark cannot start: another PID task is running")
            return

        params = {"radius": max(1, radius), "targets": max(1, targets)}
        self.state.pid_benchmark_task = _benchmark_move_pid_gen(
            self.snail, params, self.state.pid_cfg
        )
        logging.info("benchmark started (radius=%s targets=%s)", params["radius"], params["targets"])

    def start_tune(
        self, iters: int = 10, radius: int = 100, targets: int = 4, step: float = 0.35
    ) -> None:
        """Start a non-blocking PID auto-tune run."""
        self.ensure_cfg_loaded()

        if self.has_background_task():
            logging.info("auto_tune cannot start: another PID task is running")
            return

        opts = {
            "iters": max(1, iters),
            "radius": max(1, radius),
            "targets": max(1, targets),
            "step": max(0.05, step),
        }
        logging.info(f'begin autotune with {opts}')
        self.state.pid_tune_task = _auto_tune_move_pid_gen(
            self.snail, opts, self.state.pid_cfg, lambda: self.save_cfg()
        )
        logging.info(
            "auto_tune started (iters=%s radius=%s targets=%s step=%s)",
            opts["iters"],
            opts["radius"],
            opts["targets"],
            opts["step"],
        )

    def stop_task(self, task: str = "all") -> None:
        """Stop a running PID background task.

        Parameters
        ----------
        task : str
            ``"tune"``, ``"benchmark"``, or ``"all"`` (default).
        """
        if task in ("tune", "all") and self.state.pid_tune_task is not None:
            self.state.pid_tune_task = None
            logging.info("auto_tune task stopped")
        if task in ("benchmark", "all") and self.state.pid_benchmark_task is not None:
            self.state.pid_benchmark_task = None
            logging.info("benchmark task stopped")

    def has_background_task(self) -> bool:
        """Return True if any PID background task is running."""
        return (
            self.state.pid_tune_task is not None
            or self.state.pid_benchmark_task is not None
        )

    def poll_tick(self) -> None:
        """Advance all running PID background tasks by one step.

        Called each frame (via SNAIL_PID_TICK event).
        """
        if self.state.pid_tune_task is not None:
            try:
                next(self.state.pid_tune_task)
            except StopIteration:
                self.state.pid_tune_task = None
                logging.info("auto_tune task completed")
            except Exception as e:
                self.state.pid_tune_task = None
                logging.info("auto_tune task failed: %s", e)

        if self.state.pid_benchmark_task is not None:
            try:
                next(self.state.pid_benchmark_task)
            except StopIteration:
                self.state.pid_benchmark_task = None
                logging.info("benchmark task completed")
            except Exception as e:
                self.state.pid_benchmark_task = None
                logging.info("benchmark task failed: %s", e)

    # ── Map-coord helpers (bridge from screen to map space) ────────────

    def _screen_to_map_target(
        self, mouse_x: int, mouse_y: int
    ) -> tuple[int, int] | None:
        """Convert a screen pixel position to a map-space coordinate.

        Returns None if map tiles are not available.
        """
        if not self.state.map_tiles:
            return None
        tile_size = self.state.map_tiles[0].shape[0]
        origin_x, origin_y, min_x, min_y, tile_w, tile_h = map_scene_geometry(
            self.state.map_offsets, tile_size
        )
        return screen_to_map_coord(
            mouse_x, mouse_y, origin_x, origin_y, min_x, min_y, tile_w, tile_h
        )

    def move_to_mouse_target(self) -> dict:
        """Read mouse position and move PID to the projected map coord.

        Convenience for the ``move_to_mouse_map_coord`` action.
        """
        self.ensure_cfg_loaded()

        if self.snail.character_coord is None:
            logging.info("pid move aborted: character_coord is unknown")
            return {"aborted": True}

        try:
            mouse = self.snail.ahk.get_mouse_position(coord_mode="Screen")
        except Exception as e:
            logging.info("pid move aborted: failed reading mouse position: %s", e)
            return {"aborted": True}
        if mouse is None:
            logging.info("pid move aborted: mouse position unavailable")
            return {"aborted": True}

        target = self._screen_to_map_target(int(mouse.x), int(mouse.y))
        if target is None:
            logging.info("pid move aborted: map tiles unavailable")
            return {"aborted": True}

        return self.move_to(*target)

    # ── Wire to event bus ──────────────────────────────────────────────

    def subscribe(self) -> None:
        """Subscribe all PID event handlers to SnailEventBus.

        Call once from main() after the service is created.
        """
        SnailEventBus.subscribe(
            events.SNAIL_PID_TICK,
            event_callback=lambda: self.poll_tick(),
        )
        SnailEventBus.subscribe(
            events.SNAIL_PID_MOVE_REQUESTED,
            event_callback=lambda target_x, target_y: self.move_to(target_x, target_y),
        )
        SnailEventBus.subscribe(
            events.SNAIL_PID_PARAMS_UPDATED,
            event_callback=lambda updates: self.update_params(updates),
        )
        SnailEventBus.subscribe(
            events.SNAIL_PID_BENCHMARK_REQUESTED,
            event_callback=lambda radius, targets: self.start_benchmark(
                radius=radius, targets=targets
            ),
        )
        SnailEventBus.subscribe(
            events.SNAIL_PID_TUNE_REQUESTED,
            event_callback=lambda iters, radius, targets, step: self.start_tune(
                iters=iters, radius=radius, targets=targets, step=step
            ),
        )
        SnailEventBus.subscribe(
            events.SNAIL_PID_STOP_REQUESTED,
            event_callback=lambda task: self.stop_task(task=task),
        )
