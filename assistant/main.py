from assistant.snail_state import SnailState
from overlay.overlay_client import configure_logging
import numpy
import logging
import math
from assistant.actions import action_decorator
import cv2
import collections
import time
from common import exit_hotkey, hotkey_handler, timeout
from common import label_brect
from mapar import Snail
from overlay import overlay, SceneDelta
from assistant import fuzzy_match, fuzzy_match_pi, execute_action, ActionContext, register_actions, get_actions
from assistant.command_palette import command_palette_prompt
from assistant.system_stats import ProcessStatsSampler
from leaf.state import LeafState
from leaf.renderers.hud import draw_hud
from leaf.renderers.map_composite import draw_map_composite, clear_last_node_marker
from leaf.renderers.character_marker import draw_character_marker
from leaf.renderers.map_hover import draw_map_node_mouse_hover
from leaf.renderers.coords import map_scene_geometry
from assistant.event_bus import snail_events, SnailEventBus
from assistant import events
from assistant.scene_bloat import sample_scene_bloat
from assistant.pid_controller import (
    PID_CFG_DEFAULT,
    ensure_pid_cfg_loaded,
    save_pid_cfg,
    _run_pid_episode,
    _auto_tune_move_pid_gen,
    _benchmark_move_pid_gen,
)
import argparse
from collections.abc import Generator
from graphics import crop_image, Rect
from entity_detector import deduce_frame_offset, deduce_frame_offset_verified
from map_graph import MapGraphBuilder

HISTORY_MAX = 10
HISTORY_LINE_H = 22
HISTORY_MARGIN = 10
HISTORY_BG_ALPHA = 160
MAP_ANCHOR_X = 40
MAP_ANCHOR_Y = 140
MAP_CAPTURE_WINDOW_SIZE = 400
CHARACTER_MARKER_SIZE = MAP_CAPTURE_WINDOW_SIZE
CHARACTER_TRACKING_WINDOW_SIZE = 128 + 64
CHARACTER_MARKER_FPS = 60
CHARACTER_MARKER_CONFIDENCE_THRESHOLD = 0.4
CHARACTER_MARKER_SIMPLE_CONFIDENCE_THRESHOLD = 0.4

_PID_CFG_DEFAULT = dict(PID_CFG_DEFAULT)
_pid_cfg = dict(_PID_CFG_DEFAULT)
_pid_cfg_loaded = False
_pid_tune_task: Generator[None, None, None] | None = None
_pid_benchmark_task: Generator[None, None, None] | None = None


def draw_history(ov, input_queue, screen_rect):
    x0, y0, w, h = screen_rect
    line_h = HISTORY_LINE_H
    margin = HISTORY_MARGIN
    num = len(input_queue)
    if num == 0:
        _set_scene_visible(ov, 'history', False)
        return
    box_h = num * line_h + margin * 2
    box_y = y0 + h - box_h
    _set_scene_visible(ov, 'history', True)
    with ov.scene('history') as s:
        s.rect(x0 + margin, box_y, w - 2 * margin, box_h,
               pen_color=None, brush_color=(0, 0, 0, HISTORY_BG_ALPHA))
        for i, msg in enumerate(reversed(list(input_queue))):
            ty = box_y + margin + i * line_h + line_h - 2
            s.text(x0 + margin * 2, ty, msg,
                   color=(220, 220, 220, 255), font="JetBrainsMono NFM", size=10)


@action_decorator(name="take_window_screenshot", desc="Takes screenshot of window's full client area and saves it in the root directory", hotkey="^5")
def take_window_screenshot(ctx: ActionContext):
    """Emit a window screenshot request."""
    snail_events.emit(events.SNAIL_SCREENSHOT_WINDOW)


@action_decorator(name="take_non_ui_screenshot", desc="Takes screenshot of non UI detected area and saves it in the root directory")
def take_screenshot(ctx: ActionContext):
    """Emit a non-UI screenshot request."""
    snail_events.emit(events.SNAIL_SCREENSHOT_NON_UI)


_snail_state: SnailState | None = None

_map_tiles = []
_map_offsets = []
_map_composite = None
_map_composite_pngbytes = None
_map_graph_builder = None
_map_prev_crop = None
_map_cum_offset = None
_map_composite_dirty = True
_show_map_overlay = True
_show_map_node_hover = False
_character_marker_coord = None
_character_marker_prev_crop = None
_character_tracking_window_size = CHARACTER_TRACKING_WINDOW_SIZE
_character_marker_next_update_ts = 0.0
_map_node_hover_next_update_ts = 0.0
_show_ui_brect_marks = False
_history_queue = None
_show_history_widget = False
_last_node_marker_active = False
_last_node_marker_uid: str | None = None
_map_composite_scene_origin: tuple[int, int] | None = None
_character_coord_validate_next_ts = 0.0
_map_edge_scene_verify_next_ts = 0.0

_scene_visibility = {
    'history': False,
    'input': False,
    'map_composite_image': True,
    'map_composite': True,
    'map_node_mouse_hover': False,
    'map_character_marker': False,
    'ui_brect_marks': False,
}


def _ui_brect_label(rect: Rect, window: Rect) -> str:
    labels = label_brect(rect, window)
    if not labels:
        return 'ui'
    return ','.join(sorted(str(lbl) for lbl in labels))


def _draw_ui_brect_marks(ov, snail):
    r = snail.window_rect
    _set_scene_visible(ov, 'ui_brect_marks', True)
    with ov.scene('ui_brect_marks') as s:
        for uir in getattr(snail, 'ui_brects', []):
            abs_rect = uir.moved(r.x0, r.y0)
            x, y, w, h = map(int, abs_rect.xywh())
            s.rect(x, y, w, h, pen_color=(0, 255, 0, 255), pen_width=1)
            s.text(x + 4, y + 20, _ui_brect_label(abs_rect, r),
                   color=(0, 255, 0, 255), font="JetBrainsMono NFM", size=10)


def _map_coord_to_screen(coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h):
    dx, dy = coord
    return origin_x + (dx - min_x) + tile_w // 2, origin_y + (dy - min_y) + tile_h // 2


def _screen_to_map_coord(x: int, y: int, origin_x: int, origin_y: int, min_x: int, min_y: int, tile_w: int, tile_h: int) -> tuple[int, int]:
    dx = int(round((x - origin_x) - (tile_w // 2) + min_x))
    dy = int(round((y - origin_y) - (tile_h // 2) + min_y))
    return dx, dy


def _canonical_edge_key(edge) -> tuple[str, str]:
    from_uid = str(edge.from_uid)
    to_uid = str(edge.to_uid)
    if from_uid <= to_uid:
        return from_uid, to_uid
    return to_uid, from_uid


def _sorted_line_coords(x1: int, y1: int, x2: int, y2: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if (x1, y1) <= (x2, y2):
        return (x1, y1), (x2, y2)
    return (x2, y2), (x1, y1)


def _ensure_pid_cfg_loaded(snail):
    global _pid_cfg_loaded, _pid_cfg
    _pid_cfg, _pid_cfg_loaded = ensure_pid_cfg_loaded(snail, _pid_cfg, _pid_cfg_loaded)


def _save_pid_cfg(snail):
    save_pid_cfg(snail, _pid_cfg)
    # snail.cache.to_yaml(snail.CACHE_FILE)


def _mark_map_composite_dirty():
    global _map_composite_dirty
    _map_composite_dirty = True


def _set_scene_visible(ov, name: str, visible: bool):
    # current = _scene_visibility.get(name)
    # if current is visible:
    #     return False
    _scene_visibility[name] = visible
    if visible:
        ov.show_scene(name)
    else:
        ov.hide_scene(name)
    return True


def _hide_map_scenes(ov):
    global _map_composite_scene_origin
    _set_scene_visible(ov, 'map_composite_image', False)
    _set_scene_visible(ov, 'map_composite', False)
    _set_scene_visible(ov, 'map_node_mouse_hover', False)
    _set_scene_visible(ov, 'map_character_marker', False)
    _clear_last_node_marker(ov)
    _map_composite_scene_origin = None


def _refresh_map_composite_from_graph(builder: MapGraphBuilder):
    global _map_tiles, _map_offsets, _map_composite, _map_composite_pngbytes
    tiles, offsets, composite = builder.load_composite()
    _map_tiles = tiles
    _map_offsets = offsets
    _map_composite = composite

    png_bytes = builder.composite_png_bytes
    if png_bytes is not None:
        png_bytes = bytes(png_bytes)
        if not _validate_png_bytes(png_bytes):
            logging.warning('invalid map composite png cache; clearing png bytes')
            png_bytes = None
    _map_composite_pngbytes = png_bytes


def _validate_png_bytes(png_bytes: bytes) -> bool:
    try:
        buffer = numpy.frombuffer(png_bytes, dtype=numpy.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        return decoded is not None
    except Exception:
        return False


def _map_scene_geometry():
    global _map_offsets, _map_tiles
    r = Rect(0, 0, 1920, 1080 * 2)  # ctx.snail.window_rect
    min_x = min(dx for dx, _ in _map_offsets) if _map_offsets else 0
    min_y = min(dy for _, dy in _map_offsets) if _map_offsets else 0
    origin_x = r.x0 + r.w // 2 + MAP_ANCHOR_X + min_x
    origin_y = r.y0 + r.h // 2 + MAP_ANCHOR_Y + min_y
    tile_w = _map_tiles[0].shape[1] if _map_tiles else 0
    tile_h = _map_tiles[0].shape[0] if _map_tiles else 0
    return origin_x, origin_y, min_x, min_y, tile_w, tile_h


def _center_square_rect(snail, size_px: int) -> Rect:
    r = snail.window_rect
    dims = numpy.array([size_px, size_px])
    cent = r.wh() // 2
    return Rect.from_centdims(*cent, *dims)


def _map_tile_match_window_size() -> int:
    if _map_tiles:
        return int(_map_tiles[0].shape[0])
    return MAP_CAPTURE_WINDOW_SIZE


def _character_crop_rect(snail):
    return _center_square_rect(snail, _character_tracking_window_size)


def _map_tile_crop_rect(snail):
    return _center_square_rect(snail, _map_tile_match_window_size())


def _capture_character_crop(snail, img):
    return crop_image(img, _character_crop_rect(snail))


def _capture_map_tile_crop(snail, img):
    return crop_image(img, _map_tile_crop_rect(snail))


def _seed_character_marker(ctx):
    global _character_marker_coord, _character_marker_prev_crop, _character_marker_next_update_ts

    logging.info('seeding character marker start: cached_coord=%s tracking=%s', ctx.snail.character_coord, ctx.snail.track_character_coord)
    img = ctx.snail.wait_next_frame()

    # Seeding must use map-tile-sized crop to match graph node image dimensions.
    seed_crop = _capture_map_tile_crop(ctx.snail, img)

    if ctx.snail.character_coord is not None:
        anchor = [int(ctx.snail.character_coord[0]), int(ctx.snail.character_coord[1])]
        logging.info('seeding character marker: validating cached coord against nearby graph nodes: anchor=%s', anchor)
        found = ctx.snail.map_graph_builder.find_best_coord_from_image(
            seed_crop,
            CHARACTER_MARKER_CONFIDENCE_THRESHOLD,
            anchor_coord=anchor,
            nearby_limit=3,
        )
        if found is None:
            logging.info('seeding character marker: nearby validation failed, falling back to full graph search')
            found = ctx.snail.map_graph_builder.find_best_coord_from_image(
                seed_crop,
                CHARACTER_MARKER_CONFIDENCE_THRESHOLD,
            )
        if found is None:
            logging.info('seeding character marker failed: no reliable coord found from cache or graph')
            return False
        _character_marker_coord = [int(found[0]), int(found[1])]
        logging.info('seeding character marker: accepted coord=%s from graph validation', _character_marker_coord)
    else:
        logging.info('seeding character marker: cached coord unavailable, starting full graph search')
        found = ctx.snail.map_graph_builder.find_best_coord_from_image(
            seed_crop,
            CHARACTER_MARKER_CONFIDENCE_THRESHOLD,
        )
        if found is None:
            logging.info('seeding character marker failed: full graph search did not find a reliable seed')
            return False
        _character_marker_coord = [int(found[0]), int(found[1])]
        logging.info('seeding character marker: accepted coord=%s from full graph search', _character_marker_coord)
        ctx.snail.set_character_coord(_character_marker_coord)

    # Tracking uses smaller crop for faster phase correlation.
    _character_marker_prev_crop = _capture_character_crop(ctx.snail, img)
    _draw_character_marker(ctx, force=True)
    logging.info('character marker seeded at %s', _character_marker_coord)
    return True


@action_decorator(name="toggle_character_tracker", desc="Toggle character coordinate tracking", hotkey="^!t")
def toggle_character_tracker(ctx: ActionContext):
    ctx.snail.set_track_character_coord(not ctx.snail.track_character_coord)
    if ctx.snail.track_character_coord:
        if not _seed_character_marker(ctx):
            ctx.snail.set_track_character_coord(False)
    else:
        _set_scene_visible(ctx.overlay, 'map_character_marker', False)
        logging.info('character tracking disabled')


def _update_character_marker_from_frame(snail, img):
    global _character_marker_coord, _character_marker_prev_crop, _character_marker_next_update_ts

    if not snail.track_character_coord or _character_marker_coord is None or _character_marker_prev_crop is None:
        return False
    prev_crop = _character_marker_prev_crop
    assert prev_crop is not None

    crop = _capture_character_crop(snail, img)
    simple_offset, simple_confidence = deduce_frame_offset(prev_crop, crop)
    if simple_confidence < CHARACTER_MARKER_SIMPLE_CONFIDENCE_THRESHOLD:
        result = deduce_frame_offset_verified(prev_crop, crop)
        offset = result.offset
        confidence = result.confidence
    else:
        offset = simple_offset
        confidence = simple_confidence
    updated = False
    if confidence >= CHARACTER_MARKER_CONFIDENCE_THRESHOLD:
        _character_marker_coord = [
            int(round(_character_marker_coord[0] + offset[0])),
            int(round(_character_marker_coord[1] + offset[1])),
        ]
        snail.set_character_coord(_character_marker_coord)
        _character_marker_prev_crop = crop
        updated = True
    _character_marker_next_update_ts = time.perf_counter() + (1.0 / CHARACTER_MARKER_FPS)
    return updated


def _verify_scene_primitives_present(ov, scene_name: str, expected_ids: set[str]) -> None:
    if not expected_ids:
        return
    try:
        layers = ov.get_render_list()
    except Exception as exc:
        logging.debug('scene primitive verification failed to fetch render list (ignored): %s', exc)
        return

    scene_cmds = None
    for _z, name, cmds in layers:
        if name == scene_name:
            scene_cmds = cmds
            break
    if scene_cmds is None:
        logging.warning('scene %s not present in render list while expecting %d primitives', scene_name, len(expected_ids))
        return

    present_ids: set[str] = set()
    for entry in scene_cmds:
        # Render list for scene APIs is [prim_id, cmd]. Ignore malformed entries.
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
            'scene %s missing %d/%d expected primitive IDs (sample=%s)',
            scene_name,
            len(missing),
            len(expected_ids_norm),
            sample,
        )


def _assert_map_composite_image_scene_has_single_image(ov) -> None:
    try:
        layers = ov.get_render_list()
    except Exception as exc:
        logging.info('map_composite_image verification failed to fetch render list: %s', exc)
        raise AssertionError('map_composite_image scene verification failed: render list unavailable') from exc

    scene_cmds = None
    for _z, name, cmds in layers:
        if name == 'map_composite_image':
            scene_cmds = cmds
            break

    if scene_cmds is None:
        logging.info('map_composite_image scene missing in render list')
        raise AssertionError('map_composite_image scene missing in render list')

    image_count = 0
    primitive_kinds: list[str] = []
    for entry in scene_cmds:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            primitive_kinds.append('<invalid>')
            continue
        cmd = entry[1]
        if not isinstance(cmd, (list, tuple)) or not cmd:
            primitive_kinds.append('<invalid>')
            continue
        kind = str(cmd[0])
        primitive_kinds.append(kind)
        if kind == 'image':
            image_count += 1

    if image_count != 1:
        logging.info(
            'map_composite_image scene expected exactly one image, got image_count=%d total_primitives=%d kinds=%s',
            image_count,
            len(scene_cmds),
            primitive_kinds,
        )
    assert image_count == 1, f'map_composite_image scene expected exactly one image primitive, got {image_count}'


def _draw_map_composite(ctx: ActionContext):
    global _map_composite_pngbytes, _map_composite_scene_origin, _map_edge_scene_verify_next_ts
    if _map_composite is None:
        _set_scene_visible(ctx.overlay, 'map_composite_image', False)
        _set_scene_visible(ctx.overlay, 'map_composite', False)
        return
    assert _map_composite is not None

    origin_x, origin_y, min_x, min_y, tile_w, tile_h = _map_scene_geometry()
    _map_shape = _map_composite.shape
    # _, png = cv2.imencode('.png', display)
    edge_color = (0, 255, 0, 120)

    # new_origin = (origin_x, origin_y)
    # if _map_composite_scene_origin != new_origin:
    #     try:
    #         ctx.overlay.destroy_scene('map_composite_image')
    #         ctx.overlay.destroy_scene('map_composite')
    #         ctx.overlay.destroy_scene('map_last_node_marker')
    #         ctx.overlay.destroy_scene('map_character_marker')
    #         with ctx.overlay.scene('map_composite') as s:
    #             s.line(0,0, 1, 1, (0,0,0,0), 1)

    #     except Exception as exc:
    #         logging.debug('map composite scene reset failed (ignored): %s', exc)
    #     _map_composite_scene_origin = new_origin
    #     logging.debug('map composite scene reset due to map composite origin drift')

    # get screen coords of last added node
    last_node_screen = None
    last_node_uid: str | None = None
    if _map_graph_builder is not None:
        graph = _map_graph_builder.graph
        if graph.last_uid is not None:
            last = graph.nodes.get(graph.last_uid)
            if last is not None and last.coord is not None:
                last_node_screen = _map_coord_to_screen(last.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)
                last_node_uid = graph.last_uid

    ctx.overlay.set_scene_z('map_composite_image', 0)
    ctx.overlay.set_scene_z('map_composite', 10)
    ctx.overlay.set_scene_z('map_character_marker', 998)

    with ctx.overlay.scene_delta('map_composite_image') as s_img:
        pass
        if _map_composite_pngbytes is not None:
            s_img.image(origin_x, origin_y, _map_shape[1], _map_shape[0], png_bytes=_map_composite_pngbytes)

    expected_prim_ids: set[str] = set()
    with ctx.overlay.scene_delta('map_composite') as s:
        if _map_graph_builder is not None:
            edge_pairs: set[tuple[str, str]] = {
                _canonical_edge_key(edge)
                for edge in _map_graph_builder.graph.edges
                if edge.accepted
            }
            # logging.info(f'draw edges {len(edge_pairs)}')
            for from_uid, to_uid in sorted(edge_pairs):
                from_node = _map_graph_builder.graph.nodes.get(from_uid)
                to_node = _map_graph_builder.graph.nodes.get(to_uid)
                if from_node is None or to_node is None:
                    continue
                if from_node.coord is None or to_node.coord is None:
                    continue
                x1, y1 = _map_coord_to_screen(from_node.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)
                x2, y2 = _map_coord_to_screen(to_node.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)
                (nx1, ny1), (nx2, ny2) = _sorted_line_coords(x1, y1, x2, y2)
                prim_id = s.line(nx1, ny1, nx2, ny2, color=edge_color, width=1)
                expected_prim_ids.add(prim_id)
            # assert len(expected_prim_ids) == len(edge_pairs), f'{len(expected_prim_ids)},{len(edge_pairs)}'

        # logging.info(f'{expected_edge_prim_ids}')
        for dx, dy in _map_offsets:
            cx, cy = _map_coord_to_screen((dx, dy), origin_x, origin_y, min_x, min_y, tile_w, tile_h)
            (lx1, ly1), (lx2, ly2) = _sorted_line_coords(cx - 7, cy, cx + 7, cy)
            s.line(lx1, ly1, lx2, ly2, color=(255, 0, 0, 180), width=3)
            (lyx1, lyy1), (lyx2, lyy2) = _sorted_line_coords(cx, cy - 7, cx, cy + 7)
            s.line(lyx1, lyy1, lyx2, lyy2, color=(255, 0, 0, 180), width=3)
            s.text(cx - 12 + 10, cy + 4 - 10, f'{dx},{dy}',
                   color=(255, 230, 120, 255), font='JetBrainsMono NFM', size=9)

    # now_verify = time.perf_counter()
    # if now_verify >= _map_edge_scene_verify_next_ts:
    #     _assert_map_composite_image_scene_has_single_image(ctx.overlay)
    #     _map_edge_scene_verify_next_ts = now_verify + 1.0

    if last_node_screen is not None and last_node_uid is not None:
        if last_node_uid != _last_node_marker_uid:
            _draw_last_node_marker(ctx, last_node_screen, last_node_uid)
    else:
        _clear_last_node_marker(ctx.overlay)


def _draw_last_node_marker(ctx: ActionContext, screen_coord: tuple[int, int], node_uid: str):
    global _last_node_marker_active, _last_node_marker_uid
    if _last_node_marker_active and _last_node_marker_uid != node_uid:
        logging.warning(
            'last node marker scene was still active before redraw; clearing existing marker (prev=%s new=%s)',
            _last_node_marker_uid,
            node_uid,
        )
        try:
            ctx.overlay.destroy_scene('map_last_node_marker')
        except Exception as exc:
            logging.warning('failed to destroy existing last node marker scene: %s', exc)
    elif _last_node_marker_active and _last_node_marker_uid == node_uid:
        return

    with ctx.overlay.scene('map_last_node_marker') as s:
        s.set_z(999)
        cx, cy = screen_coord
        s.ellipse(cx - 12, cy - 12, 24, 24,
                  pen_color=(255, 220, 0, 255), pen_width=2,
                  brush_color=(0, 0, 0, 0))
        s.ellipse(cx - 4, cy - 4, 8, 8,
                  pen_color=(255, 220, 0, 255), pen_width=1,
                  brush_color=(255, 220, 0, 96))
    _last_node_marker_active = True
    _last_node_marker_uid = node_uid


def _clear_last_node_marker(ov):
    global _last_node_marker_active, _last_node_marker_uid
    if _last_node_marker_active:
        try:
            ov.destroy_scene('map_last_node_marker')
        except Exception as exc:
            logging.warning('failed to destroy last node marker scene: %s', exc)
        _last_node_marker_active = False
        _last_node_marker_uid = None


def _draw_map_node_mouse_hover(ctx, force=False):
    global _map_node_hover_next_update_ts
    if not _show_map_node_hover:
        _set_scene_visible(ctx.overlay, 'map_node_mouse_hover', False)
        return
    now = time.perf_counter()
    if not force and now < _map_node_hover_next_update_ts:
        return
    _map_node_hover_next_update_ts = now + (1.0 / 15.0)

    hovered = _get_hovered_map_node(ctx)
    if hovered is None:
        _set_scene_visible(ctx.overlay, 'map_node_mouse_hover', False)
        return

    origin_x, origin_y, min_x, min_y, tile_w, tile_h = _map_scene_geometry()
    cx, cy = _map_coord_to_screen(hovered.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)
    _set_scene_visible(ctx.overlay, 'map_node_mouse_hover', True)
    with ctx.overlay.scene('map_node_mouse_hover') as s:
        s.ellipse(cx - 14, cy - 14, 28, 28,
                  pen_color=(0, 255, 0, 255), pen_width=2,
                  brush_color=(0, 60, 0, 96))


def _draw_character_marker(ctx, force=False):
    global _character_marker_next_update_ts
    if not ctx.snail.track_character_coord or _character_marker_coord is None:
        _set_scene_visible(ctx.overlay, 'map_character_marker', False)
        return
    now = time.perf_counter()
    if not force and now < _character_marker_next_update_ts:
        return
    _character_marker_next_update_ts = now + (1.0 / CHARACTER_MARKER_FPS)

    origin_x, origin_y, min_x, min_y, tile_w, tile_h = _map_scene_geometry()
    cx, cy = _map_coord_to_screen(_character_marker_coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)
    _set_scene_visible(ctx.overlay, 'map_character_marker', True)
    with ctx.overlay.scene('map_character_marker') as s:
        s.set_z(998)
        ccx, ccy = _character_marker_coord
        s.text(cx+20,cy, f'{ccx},{ccy}', color=(220, 220, 220, 255), font="JetBrainsMono NFM", size=8)
        s.ellipse(cx - 10, cy - 10, 20, 20,
                  pen_color=(0, 180, 255, 255), pen_width=2,
                  brush_color=(0, 180, 255, 48))
        s.ellipse(cx - 5, cy - 5, 10, 10,
                  pen_color=(255, 255, 255, 255), pen_width=1,
                  brush_color=(255, 255, 255, 96))


def _validate_character_coord_against_graph(ctx: ActionContext, img=None) -> None:
    global _character_coord_validate_next_ts, _character_marker_coord
    now = time.perf_counter()
    if now < _character_coord_validate_next_ts:
        return
    _character_coord_validate_next_ts = now + 1.0

    builder = _map_graph_builder or ctx.snail.map_graph_builder
    if builder is None:
        return

    coord = ctx.snail.character_coord or _character_marker_coord
    if coord is None:
        return

    tile_crop = None
    if img is not None:
        try:
            tile_crop = _capture_map_tile_crop(ctx.snail, img)
            validation = builder.validate_image_coord_against_graph(tile_crop, (float(coord[0]), float(coord[1])))
            if validation.ok and validation.inferred_coord is not None:
                corrected = [
                    int(round(validation.inferred_coord[0])),
                    int(round(validation.inferred_coord[1])),
                ]
                drift = math.hypot(float(corrected[0]) - float(coord[0]), float(corrected[1]) - float(coord[1]))
                if drift > 2.0 and (_character_marker_coord != corrected or ctx.snail.character_coord != corrected):
                    _character_marker_coord = corrected
                    ctx.snail.set_character_coord(corrected)
                return
        except Exception as exc:
            logging.debug('char coord graph validation failed (ignored): %s', exc)

    if tile_crop is not None:
        found = builder.find_best_coord_from_image(
            tile_crop,
            CHARACTER_MARKER_CONFIDENCE_THRESHOLD,
            anchor_coord=[int(coord[0]), int(coord[1])],
            nearby_limit=3,
        )
        if found is None:
            found = builder.find_best_coord_from_image(
                tile_crop,
                CHARACTER_MARKER_CONFIDENCE_THRESHOLD,
            )
        if found is not None:
            corrected = [int(found[0]), int(found[1])]
            drift = math.hypot(float(corrected[0]) - float(coord[0]), float(corrected[1]) - float(coord[1]))
            if drift > 2.0 and (_character_marker_coord != corrected or ctx.snail.character_coord != corrected):
                _character_marker_coord = corrected
                ctx.snail.set_character_coord(corrected)
                logging.info('char coord recalculated from graph: %s -> %s (drift=%.2f)', coord, corrected, drift)
            return

    nearest = builder.find_nearest_nodes_to_coord(coord, limit=1)
    if not nearest:
        logging.warning('char coord %s has no nearby map graph nodes and graph recalc failed', coord)
        return

    node = nearest[0]
    if node.coord is None:
        return

    dx = coord[0] - int(node.coord[0])
    dy = coord[1] - int(node.coord[1])
    dist = math.hypot(dx, dy)
    tile_size = builder.tile_size or [256, 256]
    tolerance = max(int(tile_size[0]), 128)
    if dist > tolerance:
        logging.warning(
            'char coord %s is %.1f px away from nearest map node %s (%s) (tolerance=%d)',
            coord,
            dist,
            node.uid,
            node.coord,
            tolerance,
        )


def _get_hovered_map_node(ctx):
    builder = _map_graph_builder
    if builder is None:
        return None
    graph = builder.graph
    if graph is None:
        return None
    try:
        mouse_pos = ctx.snail.ahk.get_mouse_position(coord_mode='Screen')
    except Exception:
        return None
    if mouse_pos is None:
        return None

    r = Rect(0, 0, 1920, 1080 * 2)  # ctx.snail.window_rect
    min_x = min(dx for dx, _ in _map_offsets) if _map_offsets else 0
    min_y = min(dy for _, dy in _map_offsets) if _map_offsets else 0
    origin_x = r.x0 + r.w // 2 + MAP_ANCHOR_X + min_x
    origin_y = r.y0 + r.h // 2 + MAP_ANCHOR_Y + min_y
    tile_w = _map_tiles[0].shape[1] if _map_tiles else 0
    tile_h = _map_tiles[0].shape[0] if _map_tiles else 0

    mouse_x, mouse_y = int(mouse_pos.x), int(mouse_pos.y)
    hovered_node = None
    hovered_dist = 18.0
    for node in graph.nodes.values():
        if node.coord is None:
            continue
        node_x, node_y = _map_coord_to_screen(node.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)
        dist = ((mouse_x - node_x) ** 2 + (mouse_y - node_y) ** 2) ** 0.5
        if dist <= hovered_dist:
            hovered_dist = dist
            hovered_node = node
    return hovered_node


def _has_pid_background_task() -> bool:
    return _pid_tune_task is not None or _pid_benchmark_task is not None


def _poll_pid_tasks():
    global _pid_tune_task, _pid_benchmark_task

    if _pid_tune_task is not None:
        try:
            next(_pid_tune_task)
        except StopIteration:
            _pid_tune_task = None
            logging.info('auto_tune_move_pid task completed')
        except Exception as e:
            _pid_tune_task = None
            logging.info('auto_tune_move_pid task failed: %s', e)

    if _pid_benchmark_task is not None:
        try:
            next(_pid_benchmark_task)
        except StopIteration:
            _pid_benchmark_task = None
            logging.info('benchmark_move_pid task completed')
        except Exception as e:
            _pid_benchmark_task = None
            logging.info('benchmark_move_pid task failed: %s', e)


@action_decorator(name="move_to_mouse_map_coord", desc="Move character toward mouse-projected map coord using PID", hotkey="^m")
def move_to_mouse_map_coord(ctx: ActionContext):
    _ensure_pid_cfg_loaded(ctx.snail)

    if ctx.snail.character_coord is None:
        logging.info('pid move aborted: character_coord is unknown')
        return
    if not _map_tiles:
        logging.info('pid move aborted: map tiles are empty')
        return

    try:
        mouse = ctx.snail.ahk.get_mouse_position(coord_mode='Screen')
    except Exception as e:
        logging.info('pid move aborted: failed reading mouse position: %s', e)
        return
    if mouse is None:
        logging.info('pid move aborted: mouse position unavailable')
        return

    origin_x, origin_y, min_x, min_y, tile_w, tile_h = _map_scene_geometry()
    target = _screen_to_map_coord(int(mouse.x), int(mouse.y), origin_x, origin_y, min_x, min_y, tile_w, tile_h)
    logging.info('pid move target: screen=(%s,%s) -> map=%s current=%s', int(mouse.x), int(mouse.y), target, ctx.snail.character_coord)

    result = _run_pid_episode(ctx, target, _pid_cfg)
    logging.info(
        'pid move done: target=%s score=%.3f dist=%.2f delay=%.3f switches=%d timeout=%s',
        target,
        float(result['score']),
        float(result['final_dist']),
        float(result['delay_mean']),
        int(float(result['switches'])),
        bool(result['timeout']),
    )


@action_decorator(name="tune_move_pid", desc="Tune PID params: kp,ki,kd,dt,tolerance,deadzone,max_time", hotkey="^+m")
def tune_move_pid(ctx: ActionContext):
    _ensure_pid_cfg_loaded(ctx.snail)

    if not ctx.args:
        logging.info(
            'pid params: kp=%.4f ki=%.4f kd=%.4f dt=%.3f tol=%.2f deadzone=%.3f max_time=%.2f',
            _pid_cfg['kp'], _pid_cfg['ki'], _pid_cfg['kd'], _pid_cfg['dt'],
            _pid_cfg['tolerance'], _pid_cfg['deadzone'], _pid_cfg['max_time'],
        )
        logging.info('usage: tune_move_pid kp=0.8,ki=0.0,kd=0.1,dt=0.05,tolerance=2,deadzone=0.15,max_time=8')
        return

    updates: dict[str, float] = {}
    for raw in ctx.args:
        token = raw.strip()
        if not token:
            continue
        if '=' not in token:
            logging.info('ignored token (expected key=value): %s', token)
            continue
        key, value = token.split('=', 1)
        key = key.strip().lower()
        value = value.strip()
        if key not in _pid_cfg:
            logging.info('unknown pid key: %s', key)
            continue
        try:
            updates[key] = float(value)
        except ValueError:
            logging.info('invalid float for %s: %s', key, value)

    if not updates:
        logging.info('no valid pid updates provided')
        return

    _pid_cfg.update(updates)
    _save_pid_cfg(ctx.snail)
    logging.info('pid params updated: %s', {k: _pid_cfg[k] for k in sorted(_pid_cfg.keys())})


@action_decorator(name="benchmark_move_pid", desc="Run PID benchmark episodes and log trajectory/delay metrics (non-blocking)")
def benchmark_move_pid(ctx: ActionContext):
    global _pid_benchmark_task
    _ensure_pid_cfg_loaded(ctx.snail)

    if _has_pid_background_task():
        logging.info('benchmark_move_pid cannot start: another PID task is running')
        return

    params = {
        'radius': 20,
        'targets': 5,
    }
    for raw in ctx.args:
        token = raw.strip()
        if '=' not in token:
            continue
        k, v = token.split('=', 1)
        k = k.strip().lower()
        v = v.strip()
        if k in params:
            try:
                params[k] = int(float(v))
            except ValueError:
                logging.info('benchmark_move_pid invalid %s=%s', k, v)

    _pid_benchmark_task = _benchmark_move_pid_gen(ctx, params, _pid_cfg)
    logging.info('benchmark_move_pid started (radius=%s targets=%s)', params['radius'], params['targets'])


@action_decorator(name="stop_benchmark_move_pid", desc="Stop running non-blocking PID benchmark")
def stop_benchmark_move_pid(ctx: ActionContext):
    del ctx
    global _pid_benchmark_task
    if _pid_benchmark_task is None:
        logging.info('benchmark_move_pid is not running')
        return
    _pid_benchmark_task = None
    logging.info('benchmark_move_pid stopped')


@action_decorator(name="auto_tune_move_pid", desc="Auto-tune PID by minimizing trajectory+delay score (non-blocking)")
def auto_tune_move_pid(ctx: ActionContext):
    global _pid_tune_task
    _ensure_pid_cfg_loaded(ctx.snail)

    if _has_pid_background_task():
        logging.info('auto_tune_move_pid cannot start: another PID task is running')
        return

    opts = {
        'iters': 10,
        'radius': 100,
        'targets': 4,
        'step': 0.35,
    }
    for raw in ctx.args:
        token = raw.strip()
        if '=' not in token:
            continue
        k, v = token.split('=', 1)
        k = k.strip().lower()
        v = v.strip()
        if k not in opts:
            continue
        try:
            opts[k] = float(v) if k == 'step' else int(float(v))
        except ValueError:
            logging.info('auto_tune_move_pid invalid %s=%s', k, v)

    _pid_tune_task = _auto_tune_move_pid_gen(ctx, opts, _pid_cfg, lambda: _save_pid_cfg(ctx.snail))
    logging.info('auto_tune_move_pid started (iters=%s radius=%s targets=%s step=%s)', opts['iters'], opts['radius'], opts['targets'], opts['step'])


@action_decorator(name="stop_auto_tune_move_pid", desc="Stop running non-blocking PID auto-tuning")
def stop_auto_tune_move_pid(ctx: ActionContext):
    del ctx
    global _pid_tune_task
    if _pid_tune_task is None:
        logging.info('auto_tune_move_pid is not running')
        return
    _pid_tune_task = None
    logging.info('auto_tune_move_pid stopped')


@action_decorator(name="take_center_screenshot", desc="Takes 100x100 screenshot around center of window", hotkey="^6")
def take_center_screenshot(ctx: ActionContext):
    """Emit a centered square screenshot request.

    Passes the optional pixel-size argument as event payload.
    """
    global _snail_state
    assert _snail_state is not None, "snail_state not initialized"

    w = 100
    if ctx.args:
        try:
            w = int(ctx.args[0])
        except (ValueError, IndexError):
            logging.info("invalid size argument, using default 100")

    _snail_state.screenshot_counter += 1
    snail_events.emit(
        events.SNAIL_SCREENSHOT_CENTER,
        size=w,
        counter=_snail_state.screenshot_counter,
    )


@action_decorator(name="map_capture", desc="Capture tile and blend into map composite. Arg: tile size px", hotkey="^7")
def map_capture(ctx: ActionContext):
    global _map_graph_builder

    img = ctx.snail.wait_next_frame()
    w = MAP_CAPTURE_WINDOW_SIZE
    if len(ctx.args) > 0:
        try:
            w = int(ctx.args[0])
        except Exception as e:
            logging.info(f'invalid argument passed {e}')

    rr = _center_square_rect(ctx.snail, w)
    crop = crop_image(img, rr)

    if _map_graph_builder is None:
        _map_graph_builder = MapGraphBuilder()

    anchor_coord = _character_marker_coord
    if anchor_coord is None and ctx.snail.character_coord is not None:
        anchor_coord = [int(ctx.snail.character_coord[0]), int(ctx.snail.character_coord[1])]
    capture = _map_graph_builder.add_capture(crop, anchor_coord=anchor_coord)

    # Rebuild from graph so _map_composite and _map_composite_pngbytes are always synchronized.
    _refresh_map_composite_from_graph(_map_graph_builder)

    logging.info(
        'map_capture: uid=%s coord=%s time_of_day=%.3f status=%s edges=%d bad=%s add_node_ms=%.2f',
        capture.node.uid,
        capture.node.coord,
        capture.node.time_of_day,
        capture.node.status,
        len(capture.edges),
        capture.bad,
        capture.elapsed_ms,
    )

    # _draw_map_composite(ctx)


@action_decorator(name="map_clear", desc="Clear map composite", hotkey="^!7")
def map_clear(ctx: ActionContext):
    global _map_tiles, _map_offsets, _map_composite
    _map_tiles = []
    _map_offsets = []
    _map_composite = None
    ctx.overlay.destroy_scene('map_composite_image')
    ctx.overlay.destroy_scene('map_composite')


@action_decorator(name="drop_map_graph", desc="Delete persisted map graph and node images")
def drop_map_graph_action(ctx: ActionContext):
    global _map_graph_builder, _map_composite_dirty
    builder = _map_graph_builder if _map_graph_builder is not None else ctx.snail.map_graph_builder
    builder.drop_graph()
    _map_graph_builder = builder
    ctx.snail.map_graph_builder = builder
    ctx.overlay.destroy_scene('map_composite')
    ctx.overlay.destroy_scene('map_composite_image')
    _refresh_map_composite_from_graph(builder)
    _map_composite_dirty = False
    _hide_map_scenes(ctx.overlay)
    logging.info('map graph dropped')


@action_decorator(name="toggle_map_overlay", desc="Toggle map composite and marker drawing", hotkey="^!m")
def toggle_map_overlay(ctx: ActionContext):
    global _show_map_overlay, _map_composite_dirty, _map_node_hover_next_update_ts, _character_marker_next_update_ts
    _show_map_overlay = not _show_map_overlay
    ctx.overlay.show_scene('map_composite_image', _show_map_overlay)
    ctx.overlay.show_scene('map_composite', _show_map_overlay)
    if _show_map_overlay:
        _map_composite_dirty = True
        _map_node_hover_next_update_ts = 0.0
        _character_marker_next_update_ts = 0.0
        _draw_map_composite(ctx)
        _draw_map_node_mouse_hover(ctx, force=True)
        _draw_character_marker(ctx, force=True)
    else:
        _hide_map_scenes(ctx.overlay)
    logging.info('map overlay %s', 'enabled' if _show_map_overlay else 'disabled')


@action_decorator(name="delete_hovered_map_node", desc="Delete the hovered map graph node", hotkey="^!d")
def delete_hovered_map_node(ctx: ActionContext):
    global _map_graph_builder
    hovered_node = _get_hovered_map_node(ctx)
    if hovered_node is None:
        logging.info('no hovered map node to delete')
        return

    uid = hovered_node.uid
    builder = _map_graph_builder
    if builder is None:
        return
    if not builder.remove_node(uid):
        logging.info('failed to delete hovered map node %s', uid)
        return

    _refresh_map_composite_from_graph(builder)
    if _show_map_overlay:
        _draw_map_composite(ctx)
        _draw_map_node_mouse_hover(ctx, force=True)
        _draw_character_marker(ctx, force=True)
    logging.info('deleted hovered map node %s', uid)


@action_decorator(name="toggle_map_node_hover", desc="Toggle map node hover highlight")
def toggle_map_node_hover(ctx: ActionContext):
    global _show_map_node_hover, _map_node_hover_next_update_ts
    _show_map_node_hover = not _show_map_node_hover
    _map_node_hover_next_update_ts = 0.0
    if _show_map_overlay:
        _draw_map_node_mouse_hover(ctx, force=True)
    else:
        _set_scene_visible(ctx.overlay, 'map_node_mouse_hover', False)
    logging.info('map node hover %s', 'enabled' if _show_map_node_hover else 'disabled')


@action_decorator(name="toggle_ui_brect_marks", desc="Toggle UI bounding box labels")
def toggle_ui_brect_marks(ctx: ActionContext):
    global _show_ui_brect_marks
    _show_ui_brect_marks = not _show_ui_brect_marks
    if _show_ui_brect_marks:
        _draw_ui_brect_marks(ctx.overlay, ctx.snail)
    else:
        _set_scene_visible(ctx.overlay, 'ui_brect_marks', False)
    logging.info('ui brect marks %s', 'enabled' if _show_ui_brect_marks else 'disabled')


def _refresh_history_widget(ov, input_queue, screen_rect):
    if _show_history_widget:
        draw_history(ov, input_queue, screen_rect)
    else:
        _set_scene_visible(ov, 'history', False)


@action_decorator(name="toggle_history", desc="Toggle command history widget", hotkey="^!h")
def toggle_history(ctx: ActionContext):
    global _show_history_widget
    if _history_queue is None:
        return
    _show_history_widget = not _show_history_widget
    _refresh_history_widget(ctx.overlay, _history_queue, ctx.snail.window_rect.xywh())
    logging.info('history widget %s', 'enabled' if _show_history_widget else 'disabled')


def _reload_map_from_storage(snail, ov):
    logging.info('reload map from storage')
    global _map_graph_builder, _map_tiles, _map_offsets, _map_composite, _map_composite_dirty
    _map_graph_builder = snail.map_graph_builder
    _map_graph_builder.load_graph_from_disk()
    _refresh_map_composite_from_graph(_map_graph_builder)
    _map_composite_dirty = True
    logging.info(f'reload map from storage {_show_map_overlay}')
    if _show_map_overlay:
        _draw_map_composite(ActionContext(snail=snail, overlay=ov, args=[]))
        _draw_map_node_mouse_hover(ActionContext(snail=snail, overlay=ov, args=[]), force=True)
        _draw_character_marker(ActionContext(snail=snail, overlay=ov, args=[]), force=True)
        _map_composite_dirty = False
    else:
        _hide_map_scenes(ov)


def main():
    # logging.basicConfig(level=logging.DEBUG)
    global _map_composite_dirty

    parser = argparse.ArgumentParser()
    parser.add_argument('-v,--version', help='show version')
    args = parser.parse_args()  # noqa: F841

    with overlay(force_socket_for_image=True) as ov, \
            Snail() as snail, \
            exit_hotkey(ahk=snail.ahk) as cmd_get, \
            hotkey_handler(ahk=snail.ahk, key='^p', cmd='input_prompt') as input_cmd_get, \
            timeout(1000) as is_not_timeout:

        input_queue = collections.deque(maxlen=HISTORY_MAX)
        global _history_queue
        _history_queue = input_queue
        r = snail.window_rect
        leaf_state = LeafState()
        snail_state = SnailState()
        global _snail_state
        _snail_state = snail_state
        register_actions(snail, ov)
        _reload_map_from_storage(snail, ov)
        if snail.track_character_coord:
            if not _seed_character_marker(ActionContext(snail=snail, overlay=ov, args=[])):
                snail.set_track_character_coord(False)

        # Wire screenshot event handlers to snail services
        SnailEventBus.subscribe(
            events.SNAIL_SCREENSHOT_WINDOW,
            event_callback=lambda: snail.save_window_screenshot(),
        )
        SnailEventBus.subscribe(
            events.SNAIL_SCREENSHOT_NON_UI,
            event_callback=lambda: snail.save_non_ui_screenshot(),
        )
        SnailEventBus.subscribe(
            events.SNAIL_SCREENSHOT_CENTER,
            event_callback=lambda size, counter: snail.save_center_screenshot(
                size=size, counter=counter
            ),
        )

        t0 = time.monotonic()
        tfps = collections.deque([0] * 60, maxlen=60)
        UNITS_PER_SECOND = 1000
        stats_sampler = ProcessStatsSampler()
        t0fps = time.perf_counter()
        _character_marker_next_update_ts = 0.0

        while is_not_timeout():
            img = snail.wait_next_frame()
            # img1 = cv2.resize(img, None, fx=0.25, fy=0.25)

            # f, b = cv2.imencode('.png', img1)
            # h, w, _ = img1.shape

            # with ov.scene('frame') as ss:

            #     ss.image(5, 40, w, h, png_bytes=memoryview(b))  # ty:ignore[invalid-argument-type]
            dtfps = int((time.perf_counter() - t0fps) * UNITS_PER_SECOND)
            t0fps = time.perf_counter()
            tfps.appendleft(dtfps)
            stats = stats_sampler.sample()
            assistant_stats = stats.get('assistant')
            overlay_stats = stats.get('overlay')
            assistant_cpu = float(stats.get("assistant_cpu", 0.0))
            overlay_cpu = float(stats.get("overlay_cpu", 0.0))
            assistant_mem = int(stats.get("assistant_mem", (assistant_stats.memory_bytes if assistant_stats else 0)))
            overlay_mem = int(stats.get("overlay_mem", (overlay_stats.memory_bytes if overlay_stats else 0)))
            now_ts = time.monotonic()

            draw_hud(ov, now_ts - t0, tfps,
                     assistant_cpu, overlay_cpu,
                     assistant_mem, overlay_mem)

            loop_ctx = ActionContext(snail=snail, overlay=ov, args=[])
            if _show_map_overlay:
                draw_map_composite(ov, leaf_state,
                    _map_tiles, _map_offsets,
                    _map_composite, _map_composite_pngbytes,
                    _map_graph_builder,
                    _last_node_marker_active, _last_node_marker_uid)
                _map_composite_dirty = False
                draw_map_node_mouse_hover(ov, leaf_state,
                    snail.ahk, _map_graph_builder,
                    _map_offsets, _map_tiles, force=True)
                draw_character_marker(ov, leaf_state,
                    snail.track_character_coord, _character_marker_coord,
                    _map_tiles, _map_offsets, force=True)

            if _show_map_overlay and _show_map_node_hover and time.perf_counter() >= _map_node_hover_next_update_ts:
                draw_map_node_mouse_hover(ov, leaf_state,
                    snail.ahk, _map_graph_builder,
                    _map_offsets, _map_tiles)

            if snail.track_character_coord and time.perf_counter() >= _character_marker_next_update_ts:
                _update_character_marker_from_frame(snail, img)
                _character_marker_next_update_ts = time.perf_counter() + (1.0 / CHARACTER_MARKER_FPS)

            # Progress non-blocking PID tasks on each frame.
            _poll_pid_tasks()

            if _show_map_overlay and snail.track_character_coord and time.perf_counter() >= _character_marker_next_update_ts:
                leaf_state.character_marker_next_update_ts = _character_marker_next_update_ts
                draw_character_marker(ov, leaf_state,
                    snail.track_character_coord, _character_marker_coord,
                    _map_tiles, _map_offsets)

            # if snail.track_character_coord:
            #     _validate_character_coord_against_graph(loop_ctx, img)

            # sample_scene_bloat(ov)

            cmd = cmd_get()
            if cmd == 'exit':
                break
            icmd = input_cmd_get()
            if icmd == 'input_prompt':
                _set_scene_visible(ov, 'input', True)
                try:
                    submitted, query, selected_idx_final, results = command_palette_prompt(
                        ov,
                        r.xywh(),
                        get_actions,
                        fuzzy_match_pi,
                    )
                finally:
                    _set_scene_visible(ov, 'input', False)
                if submitted and results:
                    action = results[min(selected_idx_final, len(results) - 1)]
                    input_queue.append(action["name"])
                    query_arg = query.lstrip(action["name"]).strip()
                    args = [query_arg] if query_arg else []
                    ctx = ActionContext(snail=snail, overlay=ov, args=args)
                    execute_action(action["name"], ctx)
                    _refresh_history_widget(ov, input_queue, r.xywh())
            
