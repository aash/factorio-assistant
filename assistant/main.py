import numpy
import logging
from assistant.actions import action_decorator
import cv2
import collections
import time
from common import exit_hotkey, hotkey_handler, timeout
from common import label_brect
from mapar import Snail
from overlay import overlay
from assistant import fuzzy_match, execute_action, ActionContext, register_actions, get_actions
from assistant.command_palette import command_palette_prompt
from assistant.system_stats import ProcessStatsSampler
from assistant.scene_bloat import sample_scene_bloat
import argparse
from graphics import crop_image, Rect, blend_translated
from entity_detector import deduce_frame_offset, deduce_frame_offset_verified
from map_graph import drop_map_graph, MapGraphBuilder
from map_graph.store import save_composite_image, save_graph, delete_node_image

HISTORY_MAX = 10
HISTORY_LINE_H = 22
HISTORY_MARGIN = 10
HISTORY_BG_ALPHA = 160
MAP_ANCHOR_X = 40
MAP_ANCHOR_Y = 140
CHARACTER_MARKER_SIZE = 400
CHARACTER_MARKER_FPS = 30
CHARACTER_MARKER_CONFIDENCE_THRESHOLD = 0.4
CHARACTER_MARKER_SIMPLE_CONFIDENCE_THRESHOLD = 0.4


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


@action_decorator(name="clear", desc="Clears overlay", hotkey='^!c')
def clear(ctx: ActionContext):
    ov = ctx.overlay
    _hide_map_scenes(ov)

@action_decorator(name="take_window_screenshot", desc="Takes screenshot of window's full client area and saves it in the root directory", hotkey="^5")
def take_window_screenshot(ctx: ActionContext):
    img = ctx.snail.wait_next_frame()
    cv2.imwrite('screen.png', img)

@action_decorator(name="take_non_ui_screenshot", desc="Takes screenshot of non UI detected area and saves it in the root directory")
def take_screenshot(ctx: ActionContext):
    img = ctx.snail.wait_next_frame()
    non_ui_img = crop_image(img, ctx.snail.non_ui_rect)
    cv2.imwrite('screen_non_ui.png', non_ui_img)

_screenshot_counter = 0

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
_character_marker_next_update_ts = 0.0
_map_node_hover_next_update_ts = 0.0
_show_ui_brect_marks = False
_history_queue = None
_show_history_widget = False

_scene_visibility = {
    'history': False,
    'input': False,
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
    _set_scene_visible(ov, 'map_composite', False)
    _set_scene_visible(ov, 'map_node_mouse_hover', False)
    _set_scene_visible(ov, 'map_character_marker', False)


def _refresh_map_composite_from_graph(builder):
    global _map_tiles, _map_offsets, _map_composite, _map_composite_pngbytes
    tiles = []
    offsets = []
    for node in builder.graph.nodes.values():
        if node.coord is None:
            continue
        img = cv2.imread(node.image_path)
        if img is None:
            continue
        tiles.append(img)
        offsets.append((int(node.coord[0]), int(node.coord[1])))

    _map_tiles = tiles
    _map_offsets = offsets
    if tiles:
        _map_composite = blend_translated(tiles, offsets)
        logging.info('refreshed map composite, blend complete')
        save_composite_image(_map_composite)
        _, png = cv2.imencode('.png', _map_composite)
        _map_composite_pngbytes = png.tobytes()
    else:
        _map_composite = None
        _map_composite_pngbytes = None


def _map_scene_geometry():
    r = Rect(0, 0, 1920, 1080 * 2)  # ctx.snail.window_rect
    min_x = min(dx for dx, _ in _map_offsets) if _map_offsets else 0
    min_y = min(dy for _, dy in _map_offsets) if _map_offsets else 0
    origin_x = r.x0 + r.w // 2 + MAP_ANCHOR_X + min_x
    origin_y = r.y0 + r.h // 2 + MAP_ANCHOR_Y + min_y
    tile_w = _map_tiles[0].shape[1] if _map_tiles else 0
    tile_h = _map_tiles[0].shape[0] if _map_tiles else 0
    return origin_x, origin_y, min_x, min_y, tile_w, tile_h


def _character_crop_rect(snail):
    r = snail.window_rect
    dims = numpy.array([CHARACTER_MARKER_SIZE, CHARACTER_MARKER_SIZE])
    cent = r.wh() // 2
    return Rect.from_centdims(*cent, *dims)


def _capture_character_crop(snail, img):
    return crop_image(img, _character_crop_rect(snail))


def _seed_character_marker(ctx):
    global _character_marker_coord, _character_marker_prev_crop, _character_marker_next_update_ts

    logging.info('seeding character marker start: cached_coord=%s tracking=%s', ctx.snail.character_coord, ctx.snail.track_character_coord)
    img = ctx.snail.wait_next_frame()
    crop = _capture_character_crop(ctx.snail, img)

    if ctx.snail.character_coord is not None:
        anchor = [int(ctx.snail.character_coord[0]), int(ctx.snail.character_coord[1])]
        logging.info('seeding character marker: validating cached coord against nearby graph nodes: anchor=%s', anchor)
        found = ctx.snail.map_graph_builder.find_best_coord_from_image(
            crop,
            CHARACTER_MARKER_CONFIDENCE_THRESHOLD,
            anchor_coord=anchor,
            nearby_limit=3,
        )
        if found is None:
            logging.info('seeding character marker: nearby validation failed, falling back to full graph search')
            found = ctx.snail.map_graph_builder.find_best_coord_from_image(
                crop,
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
            crop,
            CHARACTER_MARKER_CONFIDENCE_THRESHOLD,
        )
        if found is None:
            logging.info('seeding character marker failed: full graph search did not find a reliable seed')
            return False
        _character_marker_coord = [int(found[0]), int(found[1])]
        logging.info('seeding character marker: accepted coord=%s from full graph search', _character_marker_coord)
        ctx.snail.set_character_coord(_character_marker_coord)

    _character_marker_prev_crop = crop
    _character_marker_next_update_ts = time.perf_counter() + (1.0 / CHARACTER_MARKER_FPS)
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


def _draw_map_composite(ctx):
    global _map_composite_pngbytes
    if _map_composite is None:
        _set_scene_visible(ctx.overlay, 'map_composite', False)
        return
    assert _map_composite is not None

    origin_x, origin_y, min_x, min_y, tile_w, tile_h = _map_scene_geometry()
    _map_shape = _map_composite.shape
    # _, png = cv2.imencode('.png', display)
    edge_color = (0, 255, 0, 120)

    # get screen coords of last added node
    last_node = None
    if _map_graph_builder is not None:
        graph = _map_graph_builder.graph
        if graph.last_uid is not None:
            last = graph.nodes.get(graph.last_uid)
            if last is not None and last.coord is not None:
                last_node = _map_coord_to_screen(last.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)

    # _set_scene_visible(ctx.overlay, 'map_composite', True)
    with ctx.overlay.scene_delta('map_composite') as s:
        if _map_composite_pngbytes is not None:
            s.image(origin_x, origin_y, _map_shape[1], _map_shape[0], png_bytes=_map_composite_pngbytes)
        if _map_graph_builder is not None:
            for edge in _map_graph_builder.graph.edges:
                if not edge.accepted:
                    continue
                from_node = _map_graph_builder.graph.nodes.get(edge.from_uid)
                to_node = _map_graph_builder.graph.nodes.get(edge.to_uid)
                if from_node is None or to_node is None:
                    continue
                if from_node.coord is None or to_node.coord is None:
                    continue
                x1, y1 = _map_coord_to_screen(from_node.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)
                x2, y2 = _map_coord_to_screen(to_node.coord, origin_x, origin_y, min_x, min_y, tile_w, tile_h)
                s.line(x1, y1, x2, y2, color=edge_color, width=1)
        for dx, dy in _map_offsets:
            cx, cy = _map_coord_to_screen((dx, dy), origin_x, origin_y, min_x, min_y, tile_w, tile_h)
            s.line(cx - 7, cy, cx + 7, cy, color=(255, 0, 0, 180), width=3)
            s.line(cx, cy - 7, cx, cy + 7, color=(255, 0, 0, 180), width=3)
            s.text(cx - 12 + 10, cy + 4 - 10, f'{dx},{dy}',
                   color=(255, 230, 120, 128), font='JetBrainsMono NFM', size=7)
        if last_node is not None:
            cx, cy = last_node
            s.ellipse(cx - 12, cy - 12, 24, 24,
                      pen_color=(255, 220, 0, 255), pen_width=2,
                      brush_color=(0, 0, 0, 0))
            s.ellipse(cx - 4, cy - 4, 8, 8,
                      pen_color=(255, 220, 0, 255), pen_width=1,
                      brush_color=(255, 220, 0, 96))


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
        ccx, ccy = _character_marker_coord
        s.text(cx+20,cy, f'{ccx},{ccy}', color=(220, 220, 220, 255), font="JetBrainsMono NFM", size=8)
        s.ellipse(cx - 10, cy - 10, 20, 20,
                  pen_color=(0, 180, 255, 255), pen_width=2,
                  brush_color=(0, 180, 255, 48))
        s.ellipse(cx - 5, cy - 5, 10, 10,
                  pen_color=(255, 255, 255, 255), pen_width=1,
                  brush_color=(255, 255, 255, 96))


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


@action_decorator(name="take_center_screenshot", desc="Takes 100x100 screenshot around center of window", hotkey="^6")
def take_center_screenshot(ctx: ActionContext):
    global _screenshot_counter
    img = ctx.snail.wait_next_frame()
    r = ctx.snail.window_rect
    logging.info(f'action arguments: {ctx.args}')
    w = 100
    if len(ctx.args) > 0:
        try:
            w = int(ctx.args[0])
        except Exception as e:
            logging.info(f'invalid argument passed {e}')
    dims = numpy.array([w, w])
    cent = r.wh() // 2
    rr = Rect.from_centdims(*cent, *dims)
    crop = crop_image(img, rr)
    logging.info(f'crop rect: {rr}')

    logging.info(f'{r} {img.shape}')
    logging.info(f'{crop.shape}')
    _screenshot_counter += 1
    filename = f"scrn_{_screenshot_counter:04d}.png"
    cv2.imwrite(filename, crop)


@action_decorator(name="map_capture", desc="Capture tile and blend into map composite. Arg: tile size px", hotkey="^7")
def map_capture(ctx: ActionContext):
    global _map_tiles, _map_offsets, _map_composite, _map_graph_builder, _map_composite_pngbytes

    img = ctx.snail.wait_next_frame()
    r = ctx.snail.window_rect
    w = 400
    if len(ctx.args) > 0:
        try:
            w = int(ctx.args[0])
        except Exception as e:
            logging.info(f'invalid argument passed {e}')

    dims = numpy.array([w, w])
    cent = r.wh() // 2
    rr = Rect.from_centdims(*cent, *dims)
    crop = crop_image(img, rr)

    if _map_graph_builder is None:
        _map_graph_builder = MapGraphBuilder()

    anchor_coord = _character_marker_coord
    if anchor_coord is None and ctx.snail.character_coord is not None:
        anchor_coord = [int(ctx.snail.character_coord[0]), int(ctx.snail.character_coord[1])]
    capture = _map_graph_builder.add_capture(crop, anchor_coord=anchor_coord)

    if capture.node.status == 'ok' and capture.node.coord is not None:
        _map_tiles.append(crop)
        _map_offsets.append((int(round(capture.node.coord[0])), int(round(capture.node.coord[1]))))
        _map_composite = blend_translated(_map_tiles, _map_offsets)
        logging.info('map_capture, blended fully')
        save_composite_image(_map_composite)
        _, png = cv2.imencode('.png', _map_composite)
        _map_composite_pngbytes = png.tobytes()

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

    _draw_map_composite(ctx)


@action_decorator(name="map_clear", desc="Clear map composite", hotkey="^!7")
def map_clear(ctx: ActionContext):
    global _map_tiles, _map_offsets, _map_composite
    _map_tiles = []
    _map_offsets = []
    _map_composite = None
    ctx.overlay.destroy_scene('map_composite')


@action_decorator(name="drop_map_graph", desc="Delete persisted map graph and node images")
def drop_map_graph_action(ctx: ActionContext):
    global _map_tiles, _map_offsets, _map_composite, _map_graph_builder, _map_composite_dirty
    drop_map_graph()
    _map_tiles = []
    _map_offsets = []
    _map_composite = None
    _map_composite_dirty = False
    ctx.snail.map_graph_builder = type(ctx.snail.map_graph_builder)()
    _map_graph_builder = ctx.snail.map_graph_builder
    _hide_map_scenes(ctx.overlay)
    logging.info('map graph dropped')


@action_decorator(name="toggle_map_overlay", desc="Toggle map composite and marker drawing", hotkey="^!m")
def toggle_map_overlay(ctx: ActionContext):
    global _show_map_overlay, _map_composite_dirty, _map_node_hover_next_update_ts, _character_marker_next_update_ts
    _show_map_overlay = not _show_map_overlay
    ctx.overlay.show_scene('map_composite', _show_map_overlay)
    # ctx.overlay.show_scene('map_composite', _show_map_overlay and ctx.snail.track_character_coord)
    if _show_map_overlay:
        _map_composite_dirty = True
        _map_node_hover_next_update_ts = 0.0
        _character_marker_next_update_ts = 0.0
        _draw_map_composite(ctx)
        _draw_map_node_mouse_hover(ctx, force=True)
        _draw_character_marker(ctx, force=True)
    logging.info('map overlay %s', 'enabled' if _show_map_overlay else 'disabled')


@action_decorator(name="delete_hovered_map_node", desc="Delete the hovered map graph node", hotkey="^!d")
def delete_hovered_map_node(ctx: ActionContext):
    global _map_graph_builder, _map_tiles, _map_offsets, _map_composite, _map_composite_dirty
    hovered_node = _get_hovered_map_node(ctx)
    if hovered_node is None:
        logging.info('no hovered map node to delete')
        return

    uid = hovered_node.uid
    builder = _map_graph_builder
    if builder is None:
        return
    graph = builder.graph
    graph.nodes.pop(uid, None)
    graph.edges = [edge for edge in graph.edges if edge.from_uid != uid and edge.to_uid != uid]
    if graph.last_uid == uid:
        graph.last_uid = next(iter(graph.nodes), None)
    save_graph(graph)
    delete_node_image(uid)
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
    global _map_composite_dirty

    parser = argparse.ArgumentParser()
    parser.add_argument('-v,--version', help='show version')
    args = parser.parse_args()  # noqa: F841

    with overlay(force_socket_for_image=False, dirty_tracking=False) as ov, \
            Snail() as snail, \
            exit_hotkey(ahk=snail.ahk) as cmd_get, \
            hotkey_handler(ahk=snail.ahk, key='^p', cmd='input_prompt') as input_cmd_get, \
            timeout(1000) as is_not_timeout:

        input_queue = collections.deque(maxlen=HISTORY_MAX)
        global _history_queue
        _history_queue = input_queue
        r = snail.window_rect
        register_actions(snail, ov)
        _reload_map_from_storage(snail, ov)
        if snail.track_character_coord:
            if not _seed_character_marker(ActionContext(snail=snail, overlay=ov, args=[])):
                snail.set_track_character_coord(False)


        t0 = time.monotonic()
        tfps = collections.deque([0] * 60, maxlen=60)
        UNITS_PER_SECOND = 1000
        stats_sampler = ProcessStatsSampler()
        t0fps = time.perf_counter()

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
            with ov.scene('hud') as hud:
                # hud.destroy()
                t = time.monotonic() - t0
                fps = len(tfps) * UNITS_PER_SECOND / sum(tfps)
                hud.rect(26, 54, 340, 76, pen_color=None, brush_color=(0, 0, 0, 80))
                hud.text(40, 80, f'{t:6.3f}  FPS {fps:06.1f}', (0, 255, 0, 255), "JetBrainsMono NFM", 10)
                hud.text(
                    40,
                    98,
                    f'as cpu {stats.get("assistant_cpu", 0.0):05.1f}% mem {((assistant_stats.memory_bytes if assistant_stats else 0) / (1024 * 1024)):06.1f}MB',
                    (0, 255, 0, 255),
                    "JetBrainsMono NFM",
                    10,
                )
                hud.text(
                    40,
                    116,
                    f'ov cpu {stats.get("overlay_cpu", 0.0):05.1f}% mem {((overlay_stats.memory_bytes if overlay_stats else 0) / (1024 * 1024)):06.1f}MB',
                    (0, 255, 0, 255),
                    "JetBrainsMono NFM",
                    10,
                )

            loop_ctx = ActionContext(snail=snail, overlay=ov, args=[])
            if _show_map_overlay:
                #  and _map_composite_dirty:
                _draw_map_composite(loop_ctx)
                _map_composite_dirty = False
                _draw_map_node_mouse_hover(loop_ctx, force=True)
                _draw_character_marker(loop_ctx, force=True)

            if _show_map_overlay and _show_map_node_hover and time.perf_counter() >= _map_node_hover_next_update_ts:
                _draw_map_node_mouse_hover(loop_ctx)

            if snail.track_character_coord:
                _update_character_marker_from_frame(snail, img)

            if _show_map_overlay and snail.track_character_coord and time.perf_counter() >= _character_marker_next_update_ts:
                _draw_character_marker(loop_ctx)

            sample_scene_bloat(ov)

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
                        fuzzy_match,
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
            
