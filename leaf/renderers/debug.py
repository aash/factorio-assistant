"""Debug utilities for verifying overlay scene contents.

These are used for development/debugging to ensure the overlay
renderer maintains the expected primitive count and structure.
"""

from __future__ import annotations

import logging
from typing import Set

from overlay import OverlayClient


def verify_scene_primitives_present(
    ov: OverlayClient,
    scene_name: str,
    expected_ids: set[str],
) -> None:
    """Log a warning if the overlay scene is missing expected primitives."""
    if not expected_ids:
        return
    try:
        layers = ov.get_render_list()
    except Exception as exc:
        logging.debug("scene primitive verification failed to fetch render list (ignored): %s", exc)
        return

    scene_cmds = None
    for _z, name, cmds in layers:
        if name == scene_name:
            scene_cmds = cmds
            break
    if scene_cmds is None:
        logging.warning(
            "scene %s not present in render list while expecting %d primitives",
            scene_name,
            len(expected_ids),
        )
        return

    present_ids: set[str] = set()
    for entry in scene_cmds:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        prim_id, cmd = entry[0], entry[1]
        if not isinstance(cmd, (list, tuple)) or not cmd:
            continue
        present_ids.add(str(prim_id))

    expected_ids_norm = {str(v) for v in expected_ids}
    missing = expected_ids_norm - present_ids
    if missing:
        sample = list(sorted(missing))[:3]
        logging.warning(
            "scene %s missing %d/%d expected primitive IDs (sample=%s)",
            scene_name,
            len(missing),
            len(expected_ids_norm),
            sample,
        )


def assert_map_composite_image_scene_has_single_image(ov: OverlayClient) -> None:
    """Assert that the map_composite_image scene contains exactly one image primitive.

    Raises AssertionError on failure.
    """
    try:
        layers = ov.get_render_list()
    except Exception as exc:
        logging.info("map_composite_image verification failed to fetch render list: %s", exc)
        raise AssertionError(
            "map_composite_image scene verification failed: render list unavailable"
        ) from exc

    scene_cmds = None
    for _z, name, cmds in layers:
        if name == "map_composite_image":
            scene_cmds = cmds
            break

    if scene_cmds is None:
        logging.info("map_composite_image scene missing in render list")
        raise AssertionError("map_composite_image scene missing in render list")

    image_count = 0
    primitive_kinds: list[str] = []
    for entry in scene_cmds:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            primitive_kinds.append("<invalid>")
            continue
        cmd = entry[1]
        if not isinstance(cmd, (list, tuple)) or not cmd:
            primitive_kinds.append("<invalid>")
            continue
        kind = str(cmd[0])
        primitive_kinds.append(kind)
        if kind == "image":
            image_count += 1

    if image_count != 1:
        logging.info(
            "map_composite_image scene expected exactly one image, got image_count=%d "
            "total_primitives=%d kinds=%s",
            image_count,
            len(scene_cmds),
            primitive_kinds,
        )
    assert image_count == 1, (
        f"map_composite_image scene expected exactly one image primitive, got {image_count}"
    )
