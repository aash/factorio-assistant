from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import floor
from typing import Iterable

import numpy as np

from .metrics import coord_distance


@dataclass(slots=True)
class IndexedCoord:
    uid: str
    coord: tuple[int, int]
    distance: float


class CoordGridIndex:
    """Simple spatial hash index for node coordinates."""

    def __init__(self, cell_size: int = 128):
        self.cell_size = max(16, int(cell_size))
        self._coords: dict[str, tuple[int, int]] = {}
        self._cells: dict[tuple[int, int], set[str]] = defaultdict(set)

    def _cell_key(self, coord: Iterable[int]) -> tuple[int, int]:
        x, y = [int(v) for v in coord]
        return floor(x / self.cell_size), floor(y / self.cell_size)

    def clear(self) -> None:
        self._coords.clear()
        self._cells.clear()

    def rebuild(self, nodes: Iterable[tuple[str, Iterable[int]]]) -> None:
        self.clear()
        for uid, coord in nodes:
            self.insert(uid, coord)

    def insert(self, uid: str, coord: Iterable[int]) -> None:
        cc = (int(coord[0]), int(coord[1]))
        prev = self._coords.get(uid)
        if prev is not None:
            self._cells[self._cell_key(prev)].discard(uid)
        self._coords[uid] = cc
        self._cells[self._cell_key(cc)].add(uid)

    def remove(self, uid: str) -> None:
        prev = self._coords.pop(uid, None)
        if prev is None:
            return
        self._cells[self._cell_key(prev)].discard(uid)

    def nearest(
        self,
        coord: Iterable[int],
        limit: int,
        radius: float | None = None,
        max_rings: int = 10,
    ) -> list[IndexedCoord]:
        if limit <= 0 or not self._coords:
            return []

        origin = (int(coord[0]), int(coord[1]))
        cx, cy = self._cell_key(origin)

        if radius is not None:
            rings = max(1, int(np.ceil(float(radius) / float(self.cell_size))))
        else:
            rings = max_rings

        seen: set[str] = set()
        candidates: list[IndexedCoord] = []

        for ring in range(rings + 1):
            for gx in range(cx - ring, cx + ring + 1):
                for gy in range(cy - ring, cy + ring + 1):
                    for uid in self._cells.get((gx, gy), ()):  # type: ignore[arg-type]
                        if uid in seen:
                            continue
                        seen.add(uid)
                        node_coord = self._coords[uid]
                        dd = coord_distance(node_coord, origin)
                        if radius is not None and dd > float(radius):
                            continue
                        candidates.append(IndexedCoord(uid=uid, coord=node_coord, distance=dd))
            if radius is None and len(candidates) >= limit:
                break

        if not candidates:
            return []

        candidates.sort(key=lambda item: item.distance)
        return candidates[:limit]


class FeatureIndex:
    """Descriptor index: keeps node descriptors and descriptor->node mapping."""

    def __init__(self):
        self._node_descriptors: dict[str, np.ndarray] = {}
        self._descriptor_to_node: list[str] = []

    @property
    def descriptor_to_node(self) -> list[str]:
        return self._descriptor_to_node

    def clear(self) -> None:
        self._node_descriptors.clear()
        self._descriptor_to_node = []

    def rebuild(self, node_descriptors: dict[str, np.ndarray]) -> None:
        self._node_descriptors = {
            uid: desc
            for uid, desc in node_descriptors.items()
            if desc is not None and len(desc) > 0
        }
        mapping: list[str] = []
        for uid, desc in self._node_descriptors.items():
            mapping.extend([uid] * int(desc.shape[0]))
        self._descriptor_to_node = mapping

    def set_node_descriptors(self, uid: str, descriptors: np.ndarray | None) -> None:
        if descriptors is None or len(descriptors) == 0:
            self._node_descriptors.pop(uid, None)
        else:
            self._node_descriptors[uid] = descriptors
        self.rebuild(self._node_descriptors)

    def remove_node(self, uid: str) -> None:
        self._node_descriptors.pop(uid, None)
        self.rebuild(self._node_descriptors)

    def get(self, uid: str) -> np.ndarray | None:
        return self._node_descriptors.get(uid)
