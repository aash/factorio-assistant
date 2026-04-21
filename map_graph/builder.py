from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from graphics import blend_translated

from entity_detector import deduce_frame_offset_verified

from .constants import OVERLAP_RATIO_THRESHOLD, OFFSET_CONFIDENCE_THRESHOLD, MAX_NODE_COORD_DISPARITY_PX, NODE_UID_LENGTH
from .metrics import time_of_day_from_image, overlap_ratio_from_offset, coord_distance
from .models import MapGraph, MapNode, MapEdge
from .store import load_graph, save_graph, save_node_image, load_composite_image, save_composite_image, delete_node_image


@dataclass
class MapCaptureResult:
    node: MapNode
    edges: list[MapEdge]
    bad: bool
    conflict_count: int
    elapsed_ms: float


class MapGraphBuilder:
    def __init__(self, graph: Optional[MapGraph] = None):
        self.graph = graph if graph is not None else load_graph()
        self._image_cache: dict[str, np.ndarray] = {}

    @property
    def tile_size(self) -> Optional[list[int]]:
        return self.graph.tile_size

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        cached = self._image_cache.get(path)
        if cached is not None:
            return cached
        try:
            import cv2
            img = cv2.imread(path)
            if img is not None:
                self._image_cache[path] = img
            return img
        except Exception:
            return None

    def load_composite(self) -> tuple[list[np.ndarray], list[tuple[int, int]], Optional[np.ndarray]]:
        tiles: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []
        for node in self.graph.nodes.values():
            if node.coord is None:
                continue
            img = self._load_image(node.image_path)
            if img is None:
                continue
            tiles.append(img)
            offsets.append((int(node.coord[0]), int(node.coord[1])))

        composite = load_composite_image()
        if composite is None and tiles:
            composite = blend_translated(tiles, offsets)
            save_composite_image(composite)
        return tiles, offsets, composite

    @staticmethod
    def _round_offset(offset: np.ndarray) -> list[int]:
        return [int(np.round(offset[0])), int(np.round(offset[1]))]

    @staticmethod
    def _reverse_edge(edge: MapEdge) -> MapEdge:
        return MapEdge(
            from_uid=edge.to_uid,
            to_uid=edge.from_uid,
            offset=[-int(edge.offset[0]), -int(edge.offset[1])],
            confidence=edge.confidence,
            method_agree=edge.method_agree,
            overlap_ratio=edge.overlap_ratio,
            phase_corr_response=edge.phase_corr_response,
            accepted=edge.accepted,
        )

    def _nearby_nodes(self, coord: list[int]) -> list[MapNode]:
        if self.graph.tile_size is None:
            return []
        w = int(self.graph.tile_size[0])
        max_dist = 0.3 * w
        nearby = []
        for node in self.graph.nodes.values():
            if node.coord is None:
                continue
            dd = coord_distance(node.coord, coord)
            # logging.info(f'node distance {node.uid} {dd} {max_dist}')
            if dd <= max_dist:
                nearby.append(node)
        return nearby

    def _candidate_from(
        self,
        other: MapNode,
        img: np.ndarray,
        require_overlap: bool = True,
    ) -> Optional[tuple[MapEdge, list[int]]]:
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
        raw_offset = np.asarray(result.offset, dtype=float)
        if not np.isfinite(raw_offset[0]) or not np.isfinite(raw_offset[1]):
            return None
        offset = self._round_offset(raw_offset)

        overlap_ratio = overlap_ratio_from_offset(offset, tile_size)
        if require_overlap and overlap_ratio < OVERLAP_RATIO_THRESHOLD:
            return None

        implied_coord = [int(other.coord[0] + offset[0]), int(other.coord[1] + offset[1])]
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
        start = time.perf_counter()
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
        primary_edge: Optional[MapEdge] = None
        primary_coord: Optional[list[int]] = None

        if not self.graph.nodes:
            node.coord = [0, 0]
        else:
            previous = self.graph.nodes.get(self.graph.last_uid) if self.graph.last_uid else None
            if previous is not None and previous.coord is not None:
                candidate = self._candidate_from(previous, img, require_overlap=False)
                if candidate is not None:
                    primary_edge, primary_coord = candidate
                    primary_edge.to_uid = uid
                    node.coord = [
                        int(previous.coord[0] + primary_edge.offset[0]),
                        int(previous.coord[1] + primary_edge.offset[1]),
                    ]

        nearby_nodes: list[MapNode] = []
        if primary_coord is not None:
            nearby_nodes = self._nearby_nodes(primary_coord)
            if previous is not None:
                nearby_nodes = [n for n in nearby_nodes if n.uid != previous.uid]

        bad = False
        conflict_count = 0
        if primary_edge is not None and primary_coord is not None:
            total_edges = 1 + len(nearby_nodes)
            strong_edges = 1 if primary_edge.confidence > OFFSET_CONFIDENCE_THRESHOLD else 0
            accepted_edges: list[MapEdge] = []
            if primary_edge.confidence > OFFSET_CONFIDENCE_THRESHOLD:
                accepted_edges.append(primary_edge)
                accepted_edges.append(self._reverse_edge(primary_edge))
            else:
                logging.error(
                    "map node %s primary offset confidence %.2f <= %.2f primary_coord=(%d, %d)",
                    uid,
                    primary_edge.confidence,
                    OFFSET_CONFIDENCE_THRESHOLD,
                    primary_coord[0],
                    primary_coord[1],
                )
            for other in nearby_nodes:
                candidate = self._candidate_from(other, img, require_overlap=False)
                if candidate is None:
                    conflict_count += 1
                    other_coord = other.coord
                    assert other_coord is not None
                    logging.error(
                        "map node %s conflicts with %s: no offset found primary_coord=(%d, %d) other_coord=(%d, %d)",
                        uid,
                        other.uid,
                        primary_coord[0],
                        primary_coord[1],
                        other_coord[0],
                        other_coord[1],
                    )
                    continue

                edge, implied_coord = candidate
                edge.to_uid = uid
                if edge.confidence <= OFFSET_CONFIDENCE_THRESHOLD:
                    edge.accepted = False
                    conflict_count += 1
                    other_coord = other.coord
                    assert other_coord is not None
                    logging.error(
                        "map node %s rejected by %s: offset confidence %.2f <= %.2f primary_coord=(%d, %d) implied_coord=(%d, %d) neighbor_coord=(%d, %d)",
                        uid,
                        other.uid,
                        edge.confidence,
                        OFFSET_CONFIDENCE_THRESHOLD,
                        primary_coord[0],
                        primary_coord[1],
                        implied_coord[0],
                        implied_coord[1],
                        other_coord[0],
                        other_coord[1],
                    )
                    continue

                dist = coord_distance(implied_coord, primary_coord)
                if dist >= MAX_NODE_COORD_DISPARITY_PX:
                    edge.accepted = False
                    conflict_count += 1
                    other_coord = other.coord
                    assert other_coord is not None
                    logging.error(
                        "map node %s conflicts with %s: dist=%.2fpx edge_offset=(%d, %d) primary_coord=(%d, %d) implied_coord=(%d, %d) neighbor_coord=(%d, %d)",
                        uid,
                        other.uid,
                        dist,
                        edge.offset[0],
                        edge.offset[1],
                        primary_coord[0],
                        primary_coord[1],
                        implied_coord[0],
                        implied_coord[1],
                        other_coord[0],
                        other_coord[1],
                    )
                    continue

                strong_edges += 1
                accepted_edges.append(edge)
                accepted_edges.append(self._reverse_edge(edge))

            if strong_edges * 3 < total_edges:
                bad = True
                logging.error(
                    "map node %s rejected: strong_edges=%d total_edges=%d primary_coord=(%d, %d)",
                    uid,
                    strong_edges,
                    total_edges,
                    primary_coord[0],
                    primary_coord[1],
                )
            else:
                edges.extend(accepted_edges)
        elif self.graph.nodes:
            node.status = "orphan"

        if bad:
            node.status = "bad"

        if bad:
            delete_node_image(uid)
            logging.info("map node %s dropped from graph due to bad capture", uid)
        else:
            self.graph.nodes[node.uid] = node
            self.graph.last_uid = node.uid
            self.graph.edges.extend(edges)
            save_graph(self.graph)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return MapCaptureResult(node=node, edges=edges, bad=bad, conflict_count=conflict_count, elapsed_ms=elapsed_ms)
