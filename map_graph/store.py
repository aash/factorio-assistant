from __future__ import annotations

import json
import shutil
from pathlib import Path
from dataclasses import asdict

import cv2
import numpy as np

from .constants import DATA_DIR, NODES_DIR, GRAPH_JSON
from .models import MapGraph, MapNode, MapEdge


def ensure_layout() -> None:
    NODES_DIR.mkdir(parents=True, exist_ok=True)


def node_image_path(uid: str) -> Path:
    return NODES_DIR / f"{uid}.png"


def save_node_image(uid: str, img: np.ndarray) -> str:
    ensure_layout()
    path = node_image_path(uid)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"failed to save node image to {path}")
    return path.as_posix()


def _node_from_dict(data: dict) -> MapNode:
    return MapNode(
        uid=data["uid"],
        image_path=data["image_path"],
        coord=data.get("coord"),
        time_of_day=float(data.get("time_of_day", 0.0)),
        timestamp=float(data.get("timestamp", 0.0)),
        status=data.get("status", "ok"),
    )


def _edge_from_dict(data: dict) -> MapEdge:
    return MapEdge(
        from_uid=data["from_uid"],
        to_uid=data["to_uid"],
        offset=list(data["offset"]),
        confidence=float(data.get("confidence", 0.0)),
        method_agree=float(data.get("method_agree", 0.0)),
        overlap_ratio=float(data.get("overlap_ratio", 0.0)),
        phase_corr_response=float(data.get("phase_corr_response", 0.0)),
        accepted=bool(data.get("accepted", True)),
    )


def load_graph() -> MapGraph:
    if not GRAPH_JSON.exists():
        return MapGraph()
    with GRAPH_JSON.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    node_keys = list(raw.get("nodes", {}).keys())
    graph = MapGraph(tile_size=raw.get("tile_size"), last_uid=raw.get("last_uid") or (node_keys[-1] if node_keys else None))
    graph.nodes = {uid: _node_from_dict(node) for uid, node in raw.get("nodes", {}).items()}
    graph.edges = [_edge_from_dict(edge) for edge in raw.get("edges", [])]
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
