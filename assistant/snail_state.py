"""SnailState — centralized game state for the factorio-assistant.

Owns map graph data, character tracking state, and PID controller state.
Emits domain-scoped events on mutation so the leaf (presentation) layer
can react without direct coupling.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from assistant.event_bus import snail_events
from assistant import events
from assistant.pid_controller import PID_CFG_DEFAULT
from entity_detector import deduce_frame_offset, deduce_frame_offset_verified
from graphics import Rect, crop_image
from map_graph import MapGraphBuilder, MapCaptureResult


CHARACTER_MARKER_SIZE = 400
CHARACTER_TRACKING_WINDOW_SIZE = 128 + 64
CHARACTER_MARKER_FPS = 60
CHARACTER_MARKER_CONFIDENCE_THRESHOLD = 0.4
CHARACTER_MARKER_SIMPLE_CONFIDENCE_THRESHOLD = 0.4
MAP_CAPTURE_WINDOW_SIZE = 400


@dataclass
class SnailState:
    """Encapsulates all mutable game state for the assisted Factorio session.

    Fields are grouped by domain; each mutation method emits the corresponding
    event on ``SnailEventBus`` or ``SnailEventBus``.
    """

    # ── Map graph ──────────────────────────────────────────────────────
    map_graph_builder: MapGraphBuilder = field(default_factory=MapGraphBuilder)
    map_tiles: list[np.ndarray] = field(default_factory=list)
    map_offsets: list[tuple[int, int]] = field(default_factory=list)
    map_composite: np.ndarray | None = None
    map_composite_pngbytes: bytes | None = None
    map_composite_dirty: bool = True

    # ── Character tracking ──────────────────────────────────────────────
    character_marker_coord: list[int] | None = None
    character_marker_prev_crop: np.ndarray | None = None
    character_tracking_window_size: int = CHARACTER_TRACKING_WINDOW_SIZE
    character_marker_next_update_ts: float = 0.0

    # ── PID controller ─────────────────────────────────────────────────
    pid_cfg: dict = field(default_factory=lambda: dict(PID_CFG_DEFAULT))
    pid_cfg_loaded: bool = False
    pid_tune_task: Generator[None, None, None] | None = None
    pid_benchmark_task: Generator[None, None, None] | None = None
    screenshot_counter: int = 0

    # ── Public helpers ─────────────────────────────────────────────────

    def reload_map_from_storage(self, snail) -> None:
        """Load graph + composite from disk and emit graph_loaded."""
        logging.info("reload map from storage")
        self.map_graph_builder = snail.map_graph_builder
        self.map_graph_builder.load_graph_from_disk()
        self.refresh_map_composite_from_graph()
        node_count = len(self.map_graph_builder.graph.nodes)
        edge_count = len(self.map_graph_builder.graph.edges)
        snail_events.emit(
            events.SNAIL_MAP_GRAPH_GRAPH_LOADED,
            node_count=node_count,
            edge_count=edge_count,
        )

    def refresh_map_composite_from_graph(self) -> None:
        """Rebuild tile/offset/composite caches from the builder's graph."""
        tiles, offsets, composite = self.map_graph_builder.load_composite()
        self.map_tiles = tiles
        self.map_offsets = offsets
        self.map_composite = composite

        png_bytes = self.map_graph_builder.composite_png_bytes
        if png_bytes is not None:
            png_bytes = bytes(png_bytes)
            if not _validate_png_bytes(png_bytes):
                logging.warning("invalid map composite png cache; clearing png bytes")
                png_bytes = None
        self.map_composite_pngbytes = png_bytes

        self.map_composite_dirty = True

        snail_events.emit(
            events.SNAIL_MAP_GRAPH_COMPOSITE_UPDATED,
            tiles=tiles,
            offsets=offsets,
            composite=composite,
            png_bytes=self.map_composite_pngbytes,
        )

    @staticmethod
    def map_coord_to_screen(
        coord: tuple[int, int],
        origin_x: int,
        origin_y: int,
        min_x: int,
        min_y: int,
        tile_w: int,
        tile_h: int,
    ) -> tuple[int, int]:
        dx, dy = coord
        return (
            origin_x + (dx - min_x) + tile_w // 2,
            origin_y + (dy - min_y) + tile_h // 2,
        )

    @staticmethod
    def screen_to_map_coord(
        x: int,
        y: int,
        origin_x: int,
        origin_y: int,
        min_x: int,
        min_y: int,
        tile_w: int,
        tile_h: int,
    ) -> tuple[int, int]:
        dx = int(round((x - origin_x) - (tile_w // 2) + min_x))
        dy = int(round((y - origin_y) - (tile_h // 2) + min_y))
        return dx, dy

    def map_tile_match_window_size(self) -> int:
        if self.map_tiles:
            return int(self.map_tiles[0].shape[0])
        return MAP_CAPTURE_WINDOW_SIZE

    # ── Character tracking ─────────────────────────────────────────────

    def seed_character_marker(self, snail) -> bool:
        """Find initial character position from graph and emit tracking_enabled or seed_failed.

        Parameters
        ----------
        snail : Snail
            The game interface (provides character_coord, track_character_coord,
            wait_next_frame, set_character_coord).

        Returns
        -------
        bool
            True if seeding succeeded.
        """
        logging.info(
            "seeding character marker start: cached_coord=%s tracking=%s",
            snail.character_coord,
            snail.track_character_coord,
        )
        img = snail.wait_next_frame()
        seed_crop = self._capture_map_tile_crop(snail, img)

        found = None
        if snail.character_coord is not None:
            anchor = [int(snail.character_coord[0]), int(snail.character_coord[1])]
            logging.info("seeding character marker: validating cached coord against nearby nodes: anchor=%s", anchor)
            found = self.map_graph_builder.find_best_coord_from_image(
                seed_crop,
                CHARACTER_MARKER_CONFIDENCE_THRESHOLD,
                anchor_coord=anchor,
                nearby_limit=3,
            )
            if found is None:
                logging.info("seeding: nearby validation failed, falling back to full graph search")
                found = self.map_graph_builder.find_best_coord_from_image(
                    seed_crop,
                    CHARACTER_MARKER_CONFIDENCE_THRESHOLD,
                )
            if found is None:
                logging.info("seeding failed: no reliable coord from cache or graph")
                snail_events.emit(events.SNAIL_CHARACTER_SEED_FAILED)
                return False
            self.character_marker_coord = [int(found[0]), int(found[1])]
            logging.info("seeding: accepted coord=%s from graph validation", self.character_marker_coord)
        else:
            logging.info("seeding: cached coord unavailable, starting full graph search")
            found = self.map_graph_builder.find_best_coord_from_image(
                seed_crop,
                CHARACTER_MARKER_CONFIDENCE_THRESHOLD,
            )
            if found is None:
                logging.info("seeding failed: full graph search did not find a reliable seed")
                snail_events.emit(events.SNAIL_CHARACTER_SEED_FAILED)
                return False
            self.character_marker_coord = [int(found[0]), int(found[1])]
            logging.info("seeding: accepted coord=%s from full graph search", self.character_marker_coord)
            snail.set_character_coord(self.character_marker_coord)

        self.character_marker_prev_crop = self._capture_character_crop(snail, img)

        snail_events.emit(
            events.SNAIL_CHARACTER_TRACKING_ENABLED,
            coord=tuple(self.character_marker_coord),
        )
        logging.info("character marker seeded at %s", self.character_marker_coord)
        return True

    def update_marker_from_frame(self, snail, img: np.ndarray) -> bool:
        """Update marker position from frame-delta tracking.

        Returns True if the coordinate actually changed.
        """
        if (
            not snail.track_character_coord
            or self.character_marker_coord is None
            or self.character_marker_prev_crop is None
        ):
            return False

        prev_crop = self.character_marker_prev_crop
        crop = self._capture_character_crop(snail, img)

        simple_offset, simple_confidence = deduce_frame_offset(prev_crop, crop)
        if simple_confidence < CHARACTER_MARKER_SIMPLE_CONFIDENCE_THRESHOLD:
            result = deduce_frame_offset_verified(prev_crop, crop)
            offset = result.offset
            confidence = result.confidence
        else:
            offset = simple_offset
            confidence = simple_confidence

        if confidence >= CHARACTER_MARKER_CONFIDENCE_THRESHOLD:
            self.character_marker_coord = [
                int(round(self.character_marker_coord[0] + float(offset[0]))),
                int(round(self.character_marker_coord[1] + float(offset[1]))),
            ]
            snail.set_character_coord(self.character_marker_coord)
            self.character_marker_prev_crop = crop
            self.character_marker_next_update_ts = time.perf_counter() + (1.0 / CHARACTER_MARKER_FPS)

            snail_events.emit(
                events.SNAIL_CHARACTER_COORD_UPDATED,
                coord=tuple(self.character_marker_coord),
                source="frame",
            )
            return True

        self.character_marker_next_update_ts = time.perf_counter() + (1.0 / CHARACTER_MARKER_FPS)
        return False

    # ── Map capture ────────────────────────────────────────────────────

    def capture_tile(
        self,
        img_crop: np.ndarray,
        builder: MapGraphBuilder,
        anchor_coord: list[int] | None = None,
    ) -> MapCaptureResult:
        """Add a cropped tile image to the map graph and emit node_added.

        Parameters
        ----------
        img_crop : np.ndarray
            The cropped tile image to add (already centered on the window).
        builder : MapGraphBuilder
            The map graph to add the tile to.
        anchor_coord : list[int] | None
            Optional perceived character coordinate used as anchor
            for edge matching.

        Returns
        -------
        MapCaptureResult
            The capture result with node/edge info.
        """
        if anchor_coord is None and self.character_marker_coord is not None:
            anchor_coord = self.character_marker_coord

        self.map_graph_builder = builder
        capture = builder.add_capture(img_crop, anchor_coord=anchor_coord)

        self.refresh_map_composite_from_graph()

        logging.info(
            "map_capture: uid=%s coord=%s time_of_day=%.3f status=%s edges=%d bad=%s add_node_ms=%.2f",
            capture.node.uid,
            capture.node.coord,
            capture.node.time_of_day,
            capture.node.status,
            len(capture.edges),
            capture.bad,
            capture.elapsed_ms,
        )

        snail_events.emit(
            events.SNAIL_MAP_GRAPH_NODE_ADDED,
            node_uid=capture.node.uid,
            coord=tuple(capture.node.coord) if capture.node.coord else None,
            edges_count=len(capture.edges),
        )

        return capture

    # ── PID background tasks ───────────────────────────────────────────

    def poll_pid_tasks(self) -> None:
        """Advance non-blocking PID auto-tune and benchmark generators."""
        if self.pid_tune_task is not None:
            try:
                next(self.pid_tune_task)
            except StopIteration:
                self.pid_tune_task = None
                logging.info("auto_tune_move_pid task completed")
            except Exception as e:
                self.pid_tune_task = None
                logging.info("auto_tune_move_pid task failed: %s", e)

        if self.pid_benchmark_task is not None:
            try:
                next(self.pid_benchmark_task)
            except StopIteration:
                self.pid_benchmark_task = None
                logging.info("benchmark_move_pid task completed")
            except Exception as e:
                self.pid_benchmark_task = None
                logging.info("benchmark_move_pid task failed: %s", e)

    def has_pid_background_task(self) -> bool:
        return self.pid_tune_task is not None or self.pid_benchmark_task is not None

    # ── Internal helpers ───────────────────────────────────────────────

    def _character_crop_rect(self, snail) -> Rect:
        return _center_square_rect(snail, self.character_tracking_window_size)

    def _map_tile_crop_rect(self, snail) -> Rect:
        return _center_square_rect(snail, self.map_tile_match_window_size())

    def _capture_character_crop(self, snail, img: np.ndarray) -> np.ndarray:
        return crop_image(img, self._character_crop_rect(snail))

    def _capture_map_tile_crop(self, snail, img: np.ndarray) -> np.ndarray:
        return crop_image(img, self._map_tile_crop_rect(snail))


# ── Module-level helpers (not stateful) ────────────────────────────────


def _validate_png_bytes(png_bytes: bytes) -> bool:
    try:
        buffer = np.frombuffer(png_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        return decoded is not None
    except Exception:
        return False


def _center_square_rect(snail, size_px: int) -> Rect:
    r = snail.window_rect
    dims = np.array([size_px, size_px])
    cent = r.wh() // 2
    return Rect.from_centdims(*cent, *dims)
