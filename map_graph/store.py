from __future__ import annotations

import json
import shutil
from pathlib import Path
from dataclasses import asdict

import cv2
import numpy as np

from .constants import DATA_DIR, NODES_DIR, GRAPH_JSON, COMPOSITE_IMAGE
from .models import MapGraph, MapNode, MapEdge


def ensure_layout() -> None:
    NODES_DIR.mkdir(parents=True, exist_ok=True)


def save_composite_image(img: np.ndarray) -> str:
    ensure_layout()
    if not cv2.imwrite(str(COMPOSITE_IMAGE), img):
        raise RuntimeError(f"failed to save composite image to {COMPOSITE_IMAGE}")
    return COMPOSITE_IMAGE.as_posix()


def load_composite_image() -> np.ndarray | None:
    if not COMPOSITE_IMAGE.exists():
        return None
    return cv2.imread(str(COMPOSITE_IMAGE))


def node_image_path(uid: str) -> Path:
    return NODES_DIR / f"{uid}.png"


def save_node_image(uid: str, img: np.ndarray) -> str:
    ensure_layout()
    path = node_image_path(uid)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"failed to save node image to {path}")
    return path.as_posix()


def delete_node_image(uid: str) -> None:
    path = node_image_path(uid)
    if path.exists():
        path.unlink()


def _node_from_dict(data: dict) -> MapNode:
    coord = data.get("coord")
    if coord is not None:
        coord = [int(coord[0]), int(coord[1])]
    return MapNode(
        uid=data["uid"],
        image_path=data["image_path"],
        coord=coord,
        time_of_day=float(data.get("time_of_day", 0.0)),
        timestamp=float(data.get("timestamp", 0.0)),
        status=data.get("status", "ok"),
    )


def _edge_from_dict(data: dict) -> MapEdge:
    return MapEdge(
        from_uid=data["from_uid"],
        to_uid=data["to_uid"],
        offset=[int(data["offset"][0]), int(data["offset"][1])],
        confidence=float(data.get("confidence", 0.0)),
        method_agree=float(data.get("method_agree", 0.0)),
        overlap_ratio=float(data.get("overlap_ratio", 0.0)),
        phase_corr_response=float(data.get("phase_corr_response", 0.0)),
        accepted=bool(data.get("accepted", True)),
    )


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


def load_graph() -> MapGraph:
    if not GRAPH_JSON.exists():
        return MapGraph()
    with GRAPH_JSON.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    node_keys = list(raw.get("nodes", {}).keys())
    tile_size = raw.get("tile_size")
    if tile_size is not None:
        tile_size = [int(tile_size[0]), int(tile_size[1])]
    graph = MapGraph(tile_size=tile_size, last_uid=raw.get("last_uid") or (node_keys[-1] if node_keys else None))
    graph.nodes = {uid: _node_from_dict(node) for uid, node in raw.get("nodes", {}).items()}
    graph.edges = []
    seen = set()
    for edge_data in raw.get("edges", []):
        edge = _edge_from_dict(edge_data)
        key = tuple(sorted((edge.from_uid, edge.to_uid)))
        if key in seen:
            continue
        seen.add(key)
        graph.edges.append(edge)
        graph.edges.append(_reverse_edge(edge))
    return graph


def save_graph(graph: MapGraph) -> None:
    ensure_layout()
    payload = {
        "tile_size": graph.tile_size,
        "last_uid": graph.last_uid,
        "nodes": {uid: asdict(node) for uid, node in graph.nodes.items()},
        "edges": [asdict(edge) for edge in graph.edges],
    }
    with GRAPH_JSON.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def drop_map_graph() -> None:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
