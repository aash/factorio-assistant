from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from entity_detector import deduce_frame_offset, deduce_frame_offset_verified
from graphics import blend_translated

from .constants import (
    MAX_NODE_COORD_DISPARITY_PX,
    NEARBY_NODE_DISTANCE_RATIO,
    NODE_UID_LENGTH,
    OFFSET_CONFIDENCE_THRESHOLD,
    OVERLAP_RATIO_THRESHOLD,
)
from .indexes import CoordGridIndex, FeatureIndex
from .metrics import coord_distance, overlap_ratio_from_offset, time_of_day_from_image
from .models import MapEdge, MapGraph, MapNode
from .store import (
    delete_node_image,
    drop_map_graph,
    load_composite_image,
    load_graph,
    save_composite_image,
    save_graph,
    save_node_image,
)


@dataclass(slots=True)
class MapCaptureResult:
    node: MapNode
    edges: list[MapEdge]
    bad: bool
    conflict_count: int
    elapsed_ms: float


@dataclass(slots=True)
class CoordValidationResult:
    ok: bool
    perceived_coord: tuple[float, float]
    inferred_coord: tuple[float, float] | None
    error_px: float | None
    confidence: float
    matched_node_uid: str | None
    reason: str = ""


class MapGraphService:
    def __init__(self, graph: Optional[MapGraph] = None):
        self.graph = graph if graph is not None else load_graph()
        self._deduplicate_graph_edges(persist=True)

        self._image_cache: dict[str, np.ndarray] = {}
        self._composite_cache: np.ndarray | None = None
        self._composite_png_cache: bytes | None = None
        self._composite_dirty = True

        # Tile features (SIFT preferred; ORB fallback) + descriptor->node mapping.
        self._node_features: dict[str, np.ndarray] = {}
        self._feature_index = FeatureIndex()

        self._coord_index = CoordGridIndex(cell_size=128)
        self._rebuild_indexes_and_features()

    @property
    def tile_size(self) -> Optional[list[int]]:
        return self.graph.tile_size

    @property
    def composite_image(self) -> np.ndarray | None:
        self._ensure_composite_cache()
        return self._composite_cache

    @property
    def composite_png_bytes(self) -> bytes | None:
        self._ensure_composite_cache()
        return self._composite_png_cache

    def load_from_disk(self) -> None:
        self.graph = load_graph()
        self._deduplicate_graph_edges(persist=True)
        self._image_cache.clear()
        self._invalidate_composite_cache()
        self._rebuild_indexes_and_features()

    def load_graph_from_disk(self) -> None:
        self.load_from_disk()

    def _invalidate_composite_cache(self) -> None:
        self._composite_cache = None
        self._composite_png_cache = None
        self._composite_dirty = True

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        cached = self._image_cache.get(path)
        if cached is not None:
            return cached
        img = cv2.imread(path)
        if img is not None:
            self._image_cache[path] = img
        return img

    def _feature_extractor(self):
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create()
        # Fallback for builds without SIFT support.
        return cv2.ORB_create(nfeatures=800)

    def _extract_features(self, img: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        extractor = self._feature_extractor()
        _kps, desc = extractor.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            return None
        return desc

    def _compute_node_features(self, node: MapNode) -> np.ndarray | None:
        img = self._load_image(node.image_path)
        if img is None:
            return None
        return self._extract_features(img)

    def _rebuild_indexes_and_features(self) -> None:
        nodes_for_index: list[tuple[str, list[int]]] = []
        feature_map: dict[str, np.ndarray] = {}

        tile_w = int(self.graph.tile_size[0]) if self.graph.tile_size else 128
        self._coord_index = CoordGridIndex(cell_size=max(32, tile_w))

        for uid, node in self.graph.nodes.items():
            if node.coord is not None:
                nodes_for_index.append((uid, node.coord))
            desc = self._compute_node_features(node)
            if desc is not None:
                feature_map[uid] = desc

        self._coord_index.rebuild(nodes_for_index)
        self._node_features = feature_map
        self._feature_index.rebuild(self._node_features)

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

        self._ensure_composite_cache(tiles=tiles, offsets=offsets)
        return tiles, offsets, self._composite_cache

    def _ensure_composite_cache(
        self,
        tiles: list[np.ndarray] | None = None,
        offsets: list[tuple[int, int]] | None = None,
    ) -> None:
        if not self._composite_dirty and self._composite_cache is not None:
            return

        if tiles is None or offsets is None:
            tiles = []
            offsets = []
            for node in self.graph.nodes.values():
                if node.coord is None:
                    continue
                img = self._load_image(node.image_path)
                if img is None:
                    continue
                tiles.append(img)
                offsets.append((int(node.coord[0]), int(node.coord[1])))

        if not tiles:
            self._composite_cache = None
            self._composite_png_cache = None
            self._composite_dirty = False
            return

        composite = load_composite_image()
        if composite is None or self._composite_dirty:
            composite = blend_translated(tiles, offsets)
            save_composite_image(composite)

        ok, png = cv2.imencode('.png', composite)
        self._composite_cache = composite
        self._composite_png_cache = png.tobytes() if ok else None
        self._composite_dirty = False

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

    @staticmethod
    def _copy_edge(edge: MapEdge) -> MapEdge:
        return MapEdge(
            from_uid=edge.from_uid,
            to_uid=edge.to_uid,
            offset=[int(edge.offset[0]), int(edge.offset[1])],
            confidence=float(edge.confidence),
            method_agree=float(edge.method_agree),
            overlap_ratio=float(edge.overlap_ratio),
            phase_corr_response=float(edge.phase_corr_response),
            accepted=bool(edge.accepted),
        )

    def _edge_pair(self, edge: MapEdge) -> tuple[MapEdge, MapEdge]:
        forward = self._copy_edge(edge)
        reverse = self._reverse_edge(forward)
        return forward, reverse

    @staticmethod
    def _edge_key(edge: MapEdge) -> tuple[str, str, int, int]:
        if edge.from_uid <= edge.to_uid:
            return (
                edge.from_uid,
                edge.to_uid,
                int(edge.offset[0]),
                int(edge.offset[1]),
            )
        return (
            edge.to_uid,
            edge.from_uid,
            -int(edge.offset[0]),
            -int(edge.offset[1]),
        )

    def _deduplicate_edge_pairs(
        self,
        edges: list[MapEdge],
    ) -> tuple[list[tuple[MapEdge, MapEdge]], int]:
        seen_pairs: dict[tuple[str, str, int, int], tuple[MapEdge, MapEdge]] = {}
        counts: dict[tuple[str, str, int, int], int] = {}
        duplicates = 0
        for edge in edges:
            key = self._edge_key(edge)
            counts[key] = counts.get(key, 0) + 1
            if key not in seen_pairs:
                forward = edge if edge.from_uid <= edge.to_uid else self._reverse_edge(edge)
                forward = self._copy_edge(forward)
                reverse = self._reverse_edge(forward)
                seen_pairs[key] = (forward, reverse)
            else:
                if counts[key] > 2:
                    duplicates += 1
        return list(seen_pairs.values()), duplicates

    @staticmethod
    def _flatten_edge_pairs(pairs: list[tuple[MapEdge, MapEdge]]) -> list[MapEdge]:
        flattened: list[MapEdge] = []
        for forward, reverse in pairs:
            flattened.append(forward)
            flattened.append(reverse)
        return flattened

    def _deduplicate_graph_edges(self, *, persist: bool) -> None:
        pairs, duplicates = self._deduplicate_edge_pairs(self.graph.edges)
        if duplicates:
            self.graph.edges = self._flatten_edge_pairs(pairs)
            if persist:
                save_graph(self.graph)
            logging.warning(
                "map graph removed %d duplicate edges during initialization", duplicates
            )

    def find_nearest_nodes_to_coord(
        self,
        coord: list[int] | tuple[int, int],
        limit: int = 5,
    ) -> list[MapNode]:
        hits = self._coord_index.nearest(coord, limit=max(1, int(limit)))
        out: list[MapNode] = []
        for hit in hits:
            node = self.graph.nodes.get(hit.uid)
            if node is not None and node.coord is not None:
                out.append(node)
        return out

    def _nearby_nodes(self, coord: list[int]) -> list[MapNode]:
        if self.graph.tile_size is None:
            return []
        max_dist = NEARBY_NODE_DISTANCE_RATIO * int(self.graph.tile_size[0])
        hits = self._coord_index.nearest(coord, limit=32, radius=max_dist)
        out: list[MapNode] = []
        for hit in hits:
            node = self.graph.nodes.get(hit.uid)
            if node is not None and node.coord is not None:
                out.append(node)
        return out

    def _connected_components(self) -> list[set[str]]:
        nodes_with_coords = [uid for uid, node in self.graph.nodes.items() if node.coord is not None]
        if not nodes_with_coords:
            return []

        adjacency: dict[str, set[str]] = {uid: set() for uid in nodes_with_coords}
        for edge in self.graph.edges:
            if edge.from_uid in adjacency and edge.to_uid in adjacency:
                adjacency[edge.from_uid].add(edge.to_uid)
                adjacency[edge.to_uid].add(edge.from_uid)

        visited: set[str] = set()
        components: list[set[str]] = []

        for uid in nodes_with_coords:
            if uid in visited:
                continue
            stack = [uid]
            component: set[str] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(neighbor for neighbor in adjacency.get(current, ()) if neighbor not in visited)
            if component:
                components.append(component)
        return components

    def _log_connectivity_change(
        self,
        before: list[set[str]],
        after: list[set[str]],
        node_uid: str,
    ) -> None:
        before_count = len(before)
        after_count = len(after)
        if after_count <= before_count or after_count <= 1:
            return
        if before_count == 0:
            return
        logging.warning(
            "map graph connectivity degraded after node %s: components %d -> %d (sizes=%s)",
            node_uid,
            before_count,
            after_count,
            [len(component) for component in after],
        )

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

        simple_offset, simple_confidence = deduce_frame_offset(other_img, img)
        if simple_confidence < OFFSET_CONFIDENCE_THRESHOLD:
            result = deduce_frame_offset_verified(other_img, img)
            raw_offset = np.asarray(result.offset, dtype=float)
            if not np.isfinite(raw_offset[0]) or not np.isfinite(raw_offset[1]):
                return None
            offset = self._round_offset(raw_offset)
            confidence = float(result.confidence)
            method_agree = float(result.method_agreement_px)
            phase_corr_response = float(result.phase_corr_response)
        else:
            offset = self._round_offset(np.asarray(simple_offset, dtype=float))
            confidence = float(simple_confidence)
            method_agree = 0.0
            phase_corr_response = 0.0

        overlap_ratio = overlap_ratio_from_offset(offset, tile_size)
        if require_overlap and overlap_ratio < OVERLAP_RATIO_THRESHOLD:
            return None

        implied_coord = [int(other.coord[0] + offset[0]), int(other.coord[1] + offset[1])]
        edge = MapEdge(
            from_uid=other.uid,
            to_uid="",
            offset=offset,
            confidence=confidence,
            method_agree=method_agree,
            overlap_ratio=overlap_ratio,
            phase_corr_response=phase_corr_response,
            accepted=True,
        )
        return edge, implied_coord

    def _feature_match_confidence(self, query_descriptors: np.ndarray | None, node_uid: str) -> float:
        if query_descriptors is None or len(query_descriptors) == 0:
            return 0.0
        target_descriptors = self._feature_index.get(node_uid)
        if target_descriptors is None or len(target_descriptors) == 0:
            return 0.0

        if query_descriptors.dtype != target_descriptors.dtype:
            target_descriptors = target_descriptors.astype(query_descriptors.dtype)

        if query_descriptors.dtype == np.uint8:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        try:
            knn = matcher.knnMatch(query_descriptors, target_descriptors, k=2)
        except cv2.error:
            return 0.0

        good = 0
        total = 0
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            total += 1
            if m.distance < 0.75 * n.distance:
                good += 1

        if total == 0:
            return 0.0
        return float(good / total)

    def validate_image_coord_against_graph(
        self,
        img: np.ndarray,
        perceived_coord: tuple[float, float],
        *,
        max_center_error_px: float = 4.0,
        min_confidence: float = 0.35,
        nearby_limit: int = 6,
    ) -> CoordValidationResult:
        if self.graph.tile_size is None:
            return CoordValidationResult(
                ok=False,
                perceived_coord=perceived_coord,
                inferred_coord=None,
                error_px=None,
                confidence=0.0,
                matched_node_uid=None,
                reason="graph has no tile_size",
            )

        tile_w, tile_h = int(self.graph.tile_size[0]), int(self.graph.tile_size[1])
        perceived_top_left = [
            int(round(float(perceived_coord[0]) - tile_w / 2.0)),
            int(round(float(perceived_coord[1]) - tile_h / 2.0)),
        ]

        candidates = self.find_nearest_nodes_to_coord(perceived_top_left, limit=nearby_limit)
        if not candidates:
            return CoordValidationResult(
                ok=False,
                perceived_coord=perceived_coord,
                inferred_coord=None,
                error_px=None,
                confidence=0.0,
                matched_node_uid=None,
                reason="no nearby nodes",
            )

        query_features = self._extract_features(img)
        best: tuple[MapNode, list[int], float, float] | None = None
        # tuple: (node, implied_top_left, offset_confidence, combined_score)

        for node in candidates:
            candidate = self._candidate_from(node, img, require_overlap=False)
            if candidate is None:
                continue
            edge, implied_top_left = candidate
            feature_conf = self._feature_match_confidence(query_features, node.uid)
            implied_center = (
                float(implied_top_left[0] + tile_w / 2.0),
                float(implied_top_left[1] + tile_h / 2.0),
            )
            error_px = coord_distance(implied_center, perceived_coord)

            combined = float(edge.confidence) + 0.35 * float(feature_conf) - 0.02 * float(error_px)
            if best is None or combined > best[3]:
                best = (node, implied_top_left, float(edge.confidence), combined)

        if best is None:
            return CoordValidationResult(
                ok=False,
                perceived_coord=perceived_coord,
                inferred_coord=None,
                error_px=None,
                confidence=0.0,
                matched_node_uid=None,
                reason="no valid candidate",
            )

        node, implied_top_left, offset_confidence, _combined = best
        inferred_center = (
            float(implied_top_left[0] + tile_w / 2.0),
            float(implied_top_left[1] + tile_h / 2.0),
        )
        error_px = coord_distance(inferred_center, perceived_coord)
        ok = offset_confidence >= float(min_confidence) and error_px <= float(max_center_error_px)

        return CoordValidationResult(
            ok=ok,
            perceived_coord=perceived_coord,
            inferred_coord=inferred_center,
            error_px=float(error_px),
            confidence=float(offset_confidence),
            matched_node_uid=node.uid,
            reason="" if ok else "confidence/error threshold not met",
        )

    def _candidate_from_image(
        self,
        other: MapNode,
        img: np.ndarray,
        confidence_threshold: float,
    ) -> Optional[tuple[list[int], float]]:
        if other.coord is None:
            return None
        other_img = self._load_image(other.image_path)
        if other_img is None:
            return None

        simple_offset, simple_confidence = deduce_frame_offset(other_img, img)
        if simple_confidence < confidence_threshold:
            result = deduce_frame_offset_verified(other_img, img)
            offset = result.offset
            confidence = float(result.confidence)
        else:
            offset = simple_offset
            confidence = float(simple_confidence)

        if confidence < confidence_threshold:
            return None

        implied_coord = [
            int(other.coord[0] + float(offset[0])),
            int(other.coord[1] + float(offset[1])),
        ]
        return implied_coord, confidence

    def find_best_coord_from_image(
        self,
        img: np.ndarray,
        confidence_threshold: float,
        anchor_coord: Optional[list[int]] = None,
        nearby_limit: int = 3,
    ) -> Optional[list[int]]:
        # No global search by design.
        if anchor_coord is not None:
            nodes = self.find_nearest_nodes_to_coord(anchor_coord, limit=nearby_limit)
        else:
            if self.graph.last_uid is None:
                return None
            last = self.graph.nodes.get(self.graph.last_uid)
            nodes = [last] if last is not None else []

        best_coord: Optional[list[int]] = None
        best_confidence = float("-inf")
        for other in nodes:
            if other is None:
                continue
            candidate = self._candidate_from_image(other, img, confidence_threshold)
            if candidate is None:
                continue
            implied_coord, confidence = candidate
            if confidence > best_confidence:
                best_confidence = confidence
                best_coord = implied_coord
        return best_coord

    def add_node(
        self,
        img: np.ndarray,
        coord: tuple[int, int] | list[int],
        *,
        timestamp: Optional[float] = None,
        status: str = "ok",
    ) -> MapNode:
        if timestamp is None:
            timestamp = time.time()

        capture_size = [int(img.shape[1]), int(img.shape[0])]
        if self.graph.tile_size is None:
            self.graph.tile_size = capture_size
        elif self.graph.tile_size != capture_size:
            raise RuntimeError(f"tile size mismatch: graph={self.graph.tile_size} capture={capture_size}")

        uid = uuid.uuid4().hex[:NODE_UID_LENGTH]
        image_path = save_node_image(uid, img)
        node = MapNode(
            uid=uid,
            image_path=image_path,
            coord=[int(coord[0]), int(coord[1])],
            time_of_day=time_of_day_from_image(img),
            timestamp=float(timestamp),
            status=status,
        )
        self.graph.nodes[uid] = node
        self.graph.last_uid = uid
        save_graph(self.graph)

        self._image_cache[node.image_path] = img
        desc = self._extract_features(img)
        if desc is not None:
            self._node_features[uid] = desc
        self._feature_index.set_node_descriptors(uid, desc)
        self._coord_index.insert(uid, node.coord)
        self._invalidate_composite_cache()
        return node

    def remove_node(self, uid: str) -> bool:
        node = self.graph.nodes.pop(uid, None)
        if node is None:
            return False

        self.graph.edges = [edge for edge in self.graph.edges if edge.from_uid != uid and edge.to_uid != uid]
        if self.graph.last_uid == uid:
            self.graph.last_uid = next(iter(self.graph.nodes), None)

        save_graph(self.graph)
        delete_node_image(uid)

        self._coord_index.remove(uid)
        self._feature_index.remove_node(uid)
        self._node_features.pop(uid, None)
        self._image_cache.pop(node.image_path, None)
        self._invalidate_composite_cache()
        return True

    def drop_graph(self) -> None:
        drop_map_graph()
        self.graph = MapGraph()
        self._image_cache.clear()
        self._node_features.clear()
        self._feature_index.clear()
        self._coord_index.clear()
        self._invalidate_composite_cache()

    def add_capture(
        self,
        img: np.ndarray,
        timestamp: Optional[float] = None,
        anchor_coord: Optional[list[int]] = None,
    ) -> MapCaptureResult:
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

        prev_components = self._connected_components()

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

        edges_added: list[MapEdge] = []
        primary_edge: Optional[MapEdge] = None
        primary_coord: Optional[list[int]] = None
        reference_node: Optional[MapNode] = None
        bad = False

        if not self.graph.nodes:
            node.coord = [0, 0]
        else:
            candidate_nodes: list[MapNode] = []
            if anchor_coord is not None:
                candidate_nodes = self._nearby_nodes(anchor_coord)[:2]

            if not candidate_nodes and anchor_coord is None and self.graph.last_uid:
                previous = self.graph.nodes.get(self.graph.last_uid)
                if previous is not None and previous.coord is not None:
                    candidate_nodes = [previous]

            if not candidate_nodes:
                bad = True
                logging.error(
                    "map node %s rejected: no connected candidate nodes anchor=%s graph_nodes=%d",
                    uid,
                    anchor_coord,
                    len(self.graph.nodes),
                )

            if not bad:
                best_score = None
                anchor_top_left = None
                if anchor_coord is not None and self.graph.tile_size is not None:
                    anchor_top_left = [
                        int(anchor_coord[0] - self.graph.tile_size[0] // 2),
                        int(anchor_coord[1] - self.graph.tile_size[1] // 2),
                    ]
                for other in candidate_nodes:
                    candidate = self._candidate_from(other, img, require_overlap=False)
                    if candidate is None:
                        continue

                    edge, implied_coord = candidate
                    score = float(edge.confidence)
                    if anchor_top_left is not None and self.graph.tile_size is not None:
                        score -= coord_distance(implied_coord, anchor_top_left) / float(self.graph.tile_size[0])

                    if best_score is None or score > best_score:
                        best_score = score
                        reference_node = other
                        primary_edge = edge
                        primary_coord = implied_coord

                if primary_edge is not None and reference_node is not None and primary_coord is not None:
                    reference_coord = reference_node.coord
                    assert reference_coord is not None
                    primary_edge.to_uid = uid
                    node.coord = [
                        int(reference_coord[0] + primary_edge.offset[0]),
                        int(reference_coord[1] + primary_edge.offset[1]),
                    ]

        nearby_nodes: list[MapNode] = []
        if primary_coord is not None:
            nearby_nodes = self._nearby_nodes(primary_coord)
            if reference_node is not None:
                nearby_nodes = [n for n in nearby_nodes if n.uid != reference_node.uid]
            nearby_nodes = nearby_nodes[:2]

        conflict_count = 0
        if primary_edge is not None and primary_coord is not None:
            total_edges = 1 + len(nearby_nodes)
            strong_edges = 1 if primary_edge.confidence > OFFSET_CONFIDENCE_THRESHOLD else 0
            candidate_edges: list[MapEdge] = []
            if primary_edge.confidence > OFFSET_CONFIDENCE_THRESHOLD:
                candidate_edges.extend(self._edge_pair(primary_edge))
            for other in nearby_nodes:
                candidate = self._candidate_from(other, img, require_overlap=False)
                if candidate is None:
                    conflict_count += 1
                    continue

                edge, implied_coord = candidate
                edge.to_uid = uid
                if edge.confidence <= OFFSET_CONFIDENCE_THRESHOLD:
                    edge.accepted = False
                    conflict_count += 1
                    continue

                dist = coord_distance(implied_coord, primary_coord)
                if dist >= MAX_NODE_COORD_DISPARITY_PX:
                    edge.accepted = False
                    conflict_count += 1
                    continue

                strong_edges += 1
                candidate_edges.extend(self._edge_pair(edge))

            if strong_edges * 3 < total_edges:
                bad = True
            else:
                edges_added.extend(candidate_edges)
        elif self.graph.nodes:
            node.status = "orphan"
            bad = True

        duplicate_edge_pairs_new = 0
        duplicate_edge_pairs_existing = 0
        if edges_added:
            candidate_pairs, duplicate_edge_pairs_new = self._deduplicate_edge_pairs(edges_added)
            existing_keys = {self._edge_key(edge) for edge in self.graph.edges}
            filtered_pairs: list[tuple[MapEdge, MapEdge]] = []
            for forward, reverse in candidate_pairs:
                key = self._edge_key(forward)
                if key in existing_keys:
                    duplicate_edge_pairs_existing += 1
                    continue
                existing_keys.add(key)
                filtered_pairs.append((forward, reverse))
            edges_added = self._flatten_edge_pairs(filtered_pairs)
            if duplicate_edge_pairs_new or duplicate_edge_pairs_existing:
                logging.warning(
                    "map graph deduped edges for node %s (new_duplicates=%d existing_duplicates=%d)",
                    node.uid,
                    duplicate_edge_pairs_new,
                    duplicate_edge_pairs_existing,
                )

        if bad:
            node.status = "bad"
            delete_node_image(uid)
            self._image_cache.pop(image_path, None)
            edges_added = []
        else:
            self.graph.nodes[node.uid] = node
            self.graph.last_uid = node.uid
            self.graph.edges.extend(edges_added)

            new_components = self._connected_components()
            self._log_connectivity_change(prev_components, new_components, node.uid)

            save_graph(self.graph)

            self._image_cache[node.image_path] = img
            desc = self._extract_features(img)
            if desc is not None:
                self._node_features[node.uid] = desc
            self._feature_index.set_node_descriptors(node.uid, desc)
            if node.coord is not None:
                self._coord_index.insert(node.uid, node.coord)
            self._invalidate_composite_cache()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return MapCaptureResult(node=node, edges=edges_added, bad=bad, conflict_count=conflict_count, elapsed_ms=elapsed_ms)
