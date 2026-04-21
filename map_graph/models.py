from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MapNode:
    uid: str
    image_path: str
    coord: Optional[list[int]] = None
    time_of_day: float = 0.0
    timestamp: float = 0.0
    status: str = "ok"


@dataclass
class MapEdge:
    from_uid: str
    to_uid: str
    offset: list[int]
    confidence: float
    method_agree: float
    overlap_ratio: float
    phase_corr_response: float
    accepted: bool = True


@dataclass
class MapGraph:
    tile_size: Optional[list[int]] = None
    last_uid: Optional[str] = None
    nodes: dict[str, MapNode] = field(default_factory=dict)
    edges: list[MapEdge] = field(default_factory=list)
