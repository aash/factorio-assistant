from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from entity_detector import deduce_frame_offset_verified

from .constants import OVERLAP_RATIO_THRESHOLD, MAX_NODE_COORD_DISPARITY_PX, NODE_UID_LENGTH
from .metrics import time_of_day_from_image, overlap_ratio_from_offset, coord_distance
from .models import MapGraph, MapNode, MapEdge
from .store import load_graph, save_graph, save_node_image


@dataclass
class MapCaptureResult:
    node: MapNode
    edges: list[MapEdge]
    bad: bool
    conflict_count: int


class MapGraphBuilder:
    def __init__(self, graph: Optional[MapGraph] = None):
        self.graph = graph if graph is not None else load_graph()

    @property
    def tile_size(self) -> Optional[list[int]]:
        return self.graph.tile_size

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        try:
            import cv2
            return cv2.imread(path)
        except Exception:
            return None

    def _candidate_from(
        self,
        other: MapNode,
        img: np.ndarray,
        require_overlap: bool = True,
    ) -> Optional[tuple[MapEdge, list[float]]]:
        if other.coord is None:
            return None
        other_img = self._load_image(other.image_path)
        if other_img is None:
            return None
        tile_size = self.graph.tile_size
        if tile_size is None:
            return None
        assert tile_size is not None

        result = deduce_frame_offset_verified(other_img, img)
        offset = [float(result.offset[0]), float(result.offset[1])]
        if not np.isfinite(offset[0]) or not np.isfinite(offset[1]):
            return None

        overlap_ratio = overlap_ratio_from_offset(offset, tile_size)
        if require_overlap and overlap_ratio < OVERLAP_RATIO_THRESHOLD:
            return None

        implied_coord = [float(other.coord[0] + offset[0]), float(other.coord[1] + offset[1])]
        edge = MapEdge(
            from_uid=other.uid,
            to_uid="",
            offset=offset,
            confidence=float(result.confidence),
            method_agree=float(result.method_agreement_px),
            overlap_ratio=overlap_ratio,
            phase_corr_response=float(result.phase_corr_response),
            accepted=True,
        )
        return edge, implied_coord

    def add_capture(self, img: np.ndarray, timestamp: Optional[float] = None) -> MapCaptureResult:
        if timestamp is None:
            timestamp = time.time()

        if self.graph.tile_size is None:
            self.graph.tile_size = [int(img.shape[1]), int(img.shape[0])]
        else:
            capture_size = [int(img.shape[1]), int(img.shape[0])]
            if self.graph.tile_size != capture_size:
                raise RuntimeError(
                    f"tile size mismatch: graph={self.graph.tile_size} capture={capture_size}"
                )

        uid = uuid.uuid4().hex[:NODE_UID_LENGTH]
        image_path = save_node_image(uid, img)
        node = MapNode(
            uid=uid,
            image_path=image_path,
            coord=None,
            time_of_day=time_of_day_from_image(img),
            timestamp=float(timestamp),
            status="ok",
        )

        edges: list[MapEdge] = []
        candidates: list[tuple[MapEdge, list[float]]] = []
        primary_edge: Optional[MapEdge] = None
        primary_coord: Optional[list[float]] = None

        if not self.graph.nodes:
            node.coord = [0.0, 0.0]
        else:
            previous = self.graph.nodes.get(self.graph.last_uid) if self.graph.last_uid else None
            if previous is not None:
                candidate = self._candidate_from(previous, img, require_overlap=False)
                if candidate is not None:
                    primary_edge, primary_coord = candidate
                    primary_edge.to_uid = uid
                    node.coord = [float(primary_coord[0]), float(primary_coord[1])]
                    candidates.append((primary_edge, primary_coord))

            if node.coord is not None:
                for other in self.graph.nodes.values():
                    if previous is not None and other.uid == previous.uid:
                        continue
                    candidate = self._candidate_from(other, img)
                    if candidate is None:
                        continue
                    edge, implied_coord = candidate
                    edge.to_uid = uid
                    candidates.append((edge, implied_coord))

        bad = False
        conflict_count = 0
        if primary_edge is not None and primary_coord is not None:
            edges.append(primary_edge)
            for edge, implied_coord in candidates[1:]:
                dist = coord_distance(implied_coord, primary_coord)
                if dist > MAX_NODE_COORD_DISPARITY_PX:
                    edge.accepted = False
                    conflict_count += 1
                    bad = True
                    logging.error(
                        "map node %s conflicts with %s: dist=%.2fpx edge_offset=(%.2f, %.2f) primary_coord=(%.2f, %.2f)",
                        uid,
                        edge.from_uid,
                        dist,
                        edge.offset[0],
                        edge.offset[1],
                        primary_coord[0],
                        primary_coord[1],
                    )
                edges.append(edge)
        elif self.graph.nodes:
            node.status = "orphan"

        if bad:
            node.status = "bad"

        self.graph.nodes[node.uid] = node
        self.graph.last_uid = node.uid
        self.graph.edges.extend(edges)
        save_graph(self.graph)

        return MapCaptureResult(node=node, edges=edges, bad=bad, conflict_count=conflict_count)
