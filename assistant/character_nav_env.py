from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise RuntimeError("gymnasium is required for CharacterNavEnv") from exc

from entity_detector import deduce_frame_offset, deduce_frame_offset_verified
from graphics import Rect, crop_image
from map_graph import MapGraphBuilder


KEY_PRESS_SLEEP_TIME = 0.05
CHARACTER_MARKER_SIZE = 400
CHARACTER_SIMPLE_CONFIDENCE_THRESHOLD = 0.4
CHARACTER_UPDATE_CONFIDENCE_THRESHOLD = 0.4
TARGET_REACH_RADIUS_PX = 16.0
SUCCESS_BONUS = 1000.0
OUT_OF_BOUNDS_PENALTY = -1000.0
TIMEOUT_PENALTY = -250.0

# Observation normalization constants
MAX_OBS_DISTANCE = 500.0  # Max distance to consider for normalization

_ACTION_KEYS: dict[int, tuple[str, ...]] = {
    0: ("w",),
    1: ("s",),
    2: ("a",),
    3: ("d",),
    4: ("w", "a"),
    5: ("w", "d"),
    6: ("s", "a"),
    7: ("s", "d"),
    8: (),  # NOOP
}


def parse_target_spec(spec: str) -> list[tuple[int, int]]:
    targets: list[tuple[int, int]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x_s, y_s = chunk.split(",", 1)
        targets.append((int(x_s.strip()), int(y_s.strip())))
    return targets


def _tap_direction(ahk, keys: Iterable[str], sleep_time: float) -> None:
    down = "".join(f"{{{key} down}}" for key in keys)
    up = "".join(f"{{{key} up}}" for key in reversed(tuple(keys)))
    ahk.send_input(down)
    time.sleep(sleep_time)
    ahk.send_input(up)


@dataclass
class _TrackerState:
    builder: MapGraphBuilder
    coord: tuple[int, int] | None = None
    prev_crop: np.ndarray | None = None

    def _crop(self, snail, img: np.ndarray) -> np.ndarray:
        r = snail.window_rect
        dims = np.array([CHARACTER_MARKER_SIZE, CHARACTER_MARKER_SIZE])
        cent = r.wh() // 2
        rr = Rect.from_centdims(*cent, *dims)
        return crop_image(img, rr)

    def seed(self, snail) -> tuple[int, int]:
        img = snail.wait_next_frame()
        crop = self._crop(snail, img)
        if snail.character_coord is not None:
            anchor = [int(snail.character_coord[0]), int(snail.character_coord[1])]
            found = snail.map_graph_builder.find_best_coord_from_image(
                crop,
                CHARACTER_UPDATE_CONFIDENCE_THRESHOLD,
                anchor_coord=anchor,
                nearby_limit=3,
            )
            if found is None:
                found = snail.map_graph_builder.find_best_coord_from_image(
                    crop,
                    CHARACTER_UPDATE_CONFIDENCE_THRESHOLD,
                )
            if found is None:
                raise RuntimeError("character coord is unavailable")
            self.coord = (int(found[0]), int(found[1]))
        else:
            found = snail.map_graph_builder.find_best_coord_from_image(
                crop,
                CHARACTER_UPDATE_CONFIDENCE_THRESHOLD,
            )
            if found is None:
                raise RuntimeError("character coord is unavailable")
            self.coord = (int(found[0]), int(found[1]))
            snail.set_character_coord(self.coord)
        self.prev_crop = crop
        return self.coord

    def update(self, snail, img: np.ndarray) -> tuple[tuple[int, int], float]:
        if self.coord is None or self.prev_crop is None:
            raise RuntimeError("tracker not seeded")

        crop = self._crop(snail, img)
        simple_offset, simple_confidence = deduce_frame_offset(self.prev_crop, crop)
        if simple_confidence < CHARACTER_SIMPLE_CONFIDENCE_THRESHOLD:
            result = deduce_frame_offset_verified(self.prev_crop, crop)
            offset = result.offset
            confidence = float(result.confidence)
        else:
            offset = simple_offset
            confidence = float(simple_confidence)

        if confidence >= CHARACTER_UPDATE_CONFIDENCE_THRESHOLD:
            self.coord = (
                int(round(self.coord[0] + float(offset[0]))),
                int(round(self.coord[1] + float(offset[1]))),
            )
            self.prev_crop = crop
            snail.set_character_coord(self.coord)
        logging.info(f'{self.coord} {confidence}')
        return self.coord, confidence


class CharacterNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        snail,
        target_spec: str,
        *,
        episode_timeout_sec: float = 180.0,
        target_radius_px: float = TARGET_REACH_RADIUS_PX,
        key_press_sleep_time: float = KEY_PRESS_SLEEP_TIME,
    ):
        super().__init__()
        self.snail = snail
        self.targets = parse_target_spec(target_spec)
        if not self.targets:
            raise ValueError("at least one target coord is required")

        self.episode_timeout_sec = float(episode_timeout_sec)
        self.target_radius_px = float(target_radius_px)
        self.key_press_sleep_time = float(key_press_sleep_time)

        self._graph_builder = snail.map_graph_builder
        self._bounds = self._compute_bounds()
        self._tracker = _TrackerState(self._graph_builder)
        self._target_index = 0
        self._steps = 0
        self._episode_started_ts = 0.0
        self._current_coord: tuple[int, int] | None = None

        self.action_space = spaces.Discrete(9)
        # Normalized observation: [dx_norm, dy_norm, distance_ratio]
        # All values in [-1, 1] or [0, 1] for stable RL training
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

    def _compute_bounds(self) -> tuple[int, int, int, int]:
        graph = self._graph_builder.graph
        if graph.tile_size is None:
            raise RuntimeError("map graph tile size is not available")
        tile_size = graph.tile_size
        assert tile_size is not None
        nodes = [node for node in graph.nodes.values() if node.coord is not None]
        if not nodes:
            raise RuntimeError("map graph has no nodes")

        tile_w = int(tile_size[0])
        tile_h = int(tile_size[1])
        coords = [tuple(int(v) for v in node.coord) for node in nodes if node.coord is not None]
        min_x = min(x for x, _ in coords)
        min_y = min(y for _, y in coords)
        max_x = max(x + tile_w for x, _ in coords)
        max_y = max(y + tile_h for _, y in coords)
        return min_x, min_y, max_x, max_y

    def _within_bounds(self, coord: tuple[int, int]) -> bool:
        min_x, min_y, max_x, max_y = self._bounds
        x, y = coord
        return min_x <= x < max_x and min_y <= y < max_y

    def _current_target(self) -> tuple[int, int]:
        return self.targets[self._target_index]

    def _obs(self) -> np.ndarray:
        if self._current_coord is None or self._finished():
            raise RuntimeError("character position unavailable or no target remaining")
        target = self._current_target()
        dx = float(target[0] - self._current_coord[0])
        dy = float(target[1] - self._current_coord[1])
        distance = float((dx * dx + dy * dy) ** 0.5)
        
        # Normalize to [-1, 1] or [0, 1] for stable neural network training
        dx_norm = np.clip(dx / MAX_OBS_DISTANCE, -1.0, 1.0)
        dy_norm = np.clip(dy / MAX_OBS_DISTANCE, -1.0, 1.0)
        distance_ratio = np.clip(distance / MAX_OBS_DISTANCE, 0.0, 1.0)
        
        return np.array([dx_norm, dy_norm, distance_ratio], dtype=np.float32)

    def _distance_to_target(self, coord: tuple[int, int], target: tuple[int, int]) -> float:
        dx = float(coord[0] - target[0])
        dy = float(coord[1] - target[1])
        return float((dx * dx + dy * dy) ** 0.5)

    def _finished(self) -> bool:
        return self._target_index >= len(self.targets)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "target_spec" in options:
            self.targets = parse_target_spec(str(options["target_spec"]))
        if not self.targets:
            raise ValueError("at least one target coord is required")

        self._graph_builder = self.snail.map_graph_builder
        self._bounds = self._compute_bounds()
        self._tracker = _TrackerState(self._graph_builder)
        self._target_index = 0
        self._steps = 0
        self._episode_started_ts = time.monotonic()
        self._current_coord = self._tracker.seed(self.snail)

        obs = self._obs()
        info = {
            "target": self._current_target(),
            "coord": self._current_coord,
            "target_index": self._target_index,
        }
        return obs, info

    def step(self, action: int):
        if self._finished():
            raise RuntimeError("episode already finished")

        self._steps += 1
        keys = _ACTION_KEYS[int(action)]
        if keys:  # Skip key press for NOOP (action 8)
            _tap_direction(self.snail.ahk, keys, self.key_press_sleep_time)

        img = self.snail.wait_next_frame()
        self._current_coord, confidence = self._tracker.update(self.snail, img)
        target = self._current_target()

        distance = self._distance_to_target(self._current_coord, target)
        # Normalize reward to prevent extreme values from destabilizing training
        # Clip to reasonable range [-10, 10] with -1 per unit distance up to 10 units
        reward = -np.clip(distance / self.target_radius_px, 0.0, 10.0)
        terminated = False
        truncated = False
        reason = None

        if not self._within_bounds(self._current_coord):
            reward += OUT_OF_BOUNDS_PENALTY
            terminated = True
            reason = "out_of_bounds"
        elif distance <= self.target_radius_px:
            reward += SUCCESS_BONUS
            self._target_index += 1
            reason = "target_reached"
            if self._target_index >= len(self.targets):
                terminated = True

        if not terminated and (time.monotonic() - self._episode_started_ts) >= self.episode_timeout_sec:
            reward += TIMEOUT_PENALTY
            truncated = True
            reason = "timeout"

        obs = self._obs()
        info = {
            "coord": self._current_coord,
            "target": None if self._finished() else self._current_target(),
            "target_index": self._target_index,
            "distance": distance,
            "confidence": confidence,
            "reason": reason,
        }
        logging.info(f'{info}')
        return obs, reward, terminated, truncated, info

    def close(self):
        return None
