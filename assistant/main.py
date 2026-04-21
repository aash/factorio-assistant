import numpy
import logging
from assistant.actions import action_decorator
import cv2
import collections
import time
from queue import Empty
from common import exit_hotkey, hotkey_handler, timeout
from common import label_brect
from mapar import Snail
from overlay import overlay
from assistant import input_hook
from assistant import key_capture_window
from assistant import fuzzy_match, execute_action, ActionContext, register_actions, get_actions
import argparse
from graphics import crop_image, Rect, blend_translated
from map_graph import MapGraphBuilder, drop_map_graph
from map_graph.store import save_composite_image

HISTORY_MAX = 10
HISTORY_LINE_H = 22
HISTORY_MARGIN = 10
HISTORY_BG_ALPHA = 160
INPUT_BOX_H = 28
RESULT_LINE_H = 36
RESULT_MARGIN = 8
MAX_VISIBLE_RESULTS = 6
MAP_ANCHOR_X = 40
MAP_ANCHOR_Y = 140


def draw_history(ov, input_queue, screen_rect):
    x0, y0, w, h = screen_rect
    line_h = HISTORY_LINE_H
    margin = HISTORY_MARGIN
    num = len(input_queue)
    if num == 0:
        ov.destroy_scene('history')
        return
    box_h = num * line_h + margin * 2
    box_y = y0 + h - box_h
    with ov.scene('history') as s:
        s.rect(x0 + margin, box_y, w - 2 * margin, box_h,
               pen_color=None, brush_color=(0, 0, 0, HISTORY_BG_ALPHA))
        for i, msg in enumerate(reversed(list(input_queue))):
            ty = box_y + margin + i * line_h + line_h - 2
            s.text(x0 + margin * 2, ty, msg,
                   color=(220, 220, 220, 255), font="JetBrainsMono NFM", size=10)


def draw_command_palette(ov, query, results, selected_idx, screen_rect):
    x0, y0, w, h = screen_rect
    pad = 20
    box_x = x0 + pad
    box_y = y0 + pad
    box_w = w - 2 * pad
    num_results = min(len(results), MAX_VISIBLE_RESULTS)
    results_h = num_results * RESULT_LINE_H if num_results else 0
    total_h = INPUT_BOX_H + RESULT_MARGIN + results_h + RESULT_MARGIN if num_results else INPUT_BOX_H + RESULT_MARGIN

    with ov.scene('input') as s:
        s.rect(box_x, box_y, box_w, total_h,
                pen_color=(100, 150, 255, 220), pen_width=2,
                brush_color=(20, 20, 40, 220))
        cursor = query + "|"
        s.text(box_x + 10, box_y + INPUT_BOX_H - 6, cursor,
               color=(255, 255, 255, 255), font="JetBrainsMono NFM", size=10)

        for i, action in enumerate(results[:MAX_VISIBLE_RESULTS]):
            ry = box_y + INPUT_BOX_H + RESULT_MARGIN + i * RESULT_LINE_H
            is_sel = (i == selected_idx)
            if is_sel:
                s.rect(box_x + 4, ry, box_w - 8, RESULT_LINE_H,
                        pen_color=None, brush_color=(60, 90, 160, 160))
            
            name = action.get("name", "")
            desc = action.get("desc", "")
            hotkey = action.get("hotkey")
            
            parts = [name]
            if desc:
                parts.append(desc)
            if hotkey:
                parts.append(hotkey)
            
            line = " | ".join(parts)
            text_color = (255, 255, 255, 255) if is_sel else (180, 180, 180, 200)
            s.text(box_x + 14, ry + 16 + 8, line,
                   color=text_color, font="JetBrainsMono NFM", size=10)


@action_decorator(name="clear", desc="Clears overlay", hotkey='^!c')
def clear(ctx: ActionContext):
    ov = ctx.overlay
    ov.destroy_scene('map_composite')

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
_map_graph_builder = None
_map_prev_crop = None
_map_cum_offset = None
_show_ui_brect_marks = False
_history_queue = None
_show_history_widget = False


def _ui_brect_label(rect: Rect, window: Rect) -> str:
    labels = label_brect(rect, window)
    if not labels:
        return 'ui'
    return ','.join(sorted(str(lbl) for lbl in labels))


def _draw_ui_brect_marks(ov, snail):
    r = snail.window_rect
    with ov.scene('ui_brect_marks') as s:
        for uir in getattr(snail, 'ui_brects', []):
            abs_rect = uir.moved(r.x0, r.y0)
            x, y, w, h = map(int, abs_rect.xywh())
            s.rect(x, y, w, h, pen_color=(0, 255, 0, 255), pen_width=1)
            s.text(x + 4, y + 20, _ui_brect_label(abs_rect, r),
                   color=(0, 255, 0, 255), font="JetBrainsMono NFM", size=10)


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
    global _map_tiles, _map_offsets, _map_composite, _map_graph_builder

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

    capture = _map_graph_builder.add_capture(crop)

    if capture.node.status == 'ok' and capture.node.coord is not None:
        _map_tiles.append(crop)
        _map_offsets.append((int(round(capture.node.coord[0])), int(round(capture.node.coord[1]))))
        _map_composite = blend_translated(_map_tiles, _map_offsets)
        save_composite_image(_map_composite)

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
    global _map_tiles, _map_offsets, _map_composite, _map_graph_builder
    drop_map_graph()
    _map_tiles = []
    _map_offsets = []
    _map_composite = None
    _map_graph_builder = None
    ctx.overlay.destroy_scene('map_composite')
    logging.info('map graph dropped')


@action_decorator(name="toggle_ui_brect_marks", desc="Toggle UI bounding box labels")
def toggle_ui_brect_marks(ctx: ActionContext):
    global _show_ui_brect_marks
    _show_ui_brect_marks = not _show_ui_brect_marks
    if _show_ui_brect_marks:
        _draw_ui_brect_marks(ctx.overlay, ctx.snail)
    else:
        ctx.overlay.destroy_scene('ui_brect_marks')
    logging.info('ui brect marks %s', 'enabled' if _show_ui_brect_marks else 'disabled')


def _refresh_history_widget(ov, input_queue, screen_rect):
    if _show_history_widget:
        draw_history(ov, input_queue, screen_rect)
    else:
        ov.destroy_scene('history')


@action_decorator(name="toggle_history", desc="Toggle command history widget", hotkey="^!h")
def toggle_history(ctx: ActionContext):
    global _show_history_widget
    if _history_queue is None:
        return
    _show_history_widget = not _show_history_widget
    _refresh_history_widget(ctx.overlay, _history_queue, ctx.snail.window_rect.xywh())
    logging.info('history widget %s', 'enabled' if _show_history_widget else 'disabled')


def _draw_map_composite(ctx):
    if _map_composite is None:
        ctx.overlay.destroy_scene('map_composite')
        return
    r = Rect(0, 0, 1920, 1080*2)# ctx.snail.window_rect
    min_x = min(dx for dx, _ in _map_offsets) if _map_offsets else 0
    min_y = min(dy for _, dy in _map_offsets) if _map_offsets else 0
    origin_x = r.x0 + r.w // 2 + MAP_ANCHOR_X + min_x
    origin_y = r.y0 + r.h // 2 + MAP_ANCHOR_Y + min_y

    display = _map_composite
    _, png = cv2.imencode('.png', display)
    tile_w = _map_tiles[0].shape[1] if _map_tiles else 0
    tile_h = _map_tiles[0].shape[0] if _map_tiles else 0
    edge_color = (0, 255, 0, 120)

    def to_screen(coord):
        dx, dy = coord
        return origin_x + (dx - min_x) + tile_w // 2, origin_y + (dy - min_y) + tile_h // 2

    with ctx.overlay.scene('map_composite') as s:
        s.image(origin_x, origin_y, display.shape[1], display.shape[0], png_bytes=memoryview(png))
        if _map_graph_builder is not None:
            seen_edges = set()
            for edge in _map_graph_builder.graph.edges:
                if not edge.accepted:
                    continue
                edge_key = tuple(sorted((edge.from_uid, edge.to_uid)))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                from_node = _map_graph_builder.graph.nodes.get(edge.from_uid)
                to_node = _map_graph_builder.graph.nodes.get(edge.to_uid)
                if from_node is None or to_node is None:
                    continue
                if from_node.coord is None or to_node.coord is None:
                    continue
                x1, y1 = to_screen(from_node.coord)
                x2, y2 = to_screen(to_node.coord)
                s.line(x1, y1, x2, y2, color=edge_color, width=1)
        for dx, dy in _map_offsets:
            cx = origin_x + (dx - min_x) + tile_w // 2
            cy = origin_y + (dy - min_y) + tile_h // 2
            s.line(cx - 7, cy, cx + 7, cy, color=(255, 0, 0, 180), width=3)
            s.line(cx, cy - 7, cx, cy + 7, color=(255, 0, 0, 180), width=3)
            s.text(cx - 12+10, cy + 4-10, f'{dx},{dy}',
                   color=(255, 230, 120, 128), font='JetBrainsMono NFM', size=7)


def _reload_map_from_storage(snail, ov):
    global _map_graph_builder, _map_tiles, _map_offsets, _map_composite
    _map_graph_builder = MapGraphBuilder()
    _map_tiles, _map_offsets, _map_composite = _map_graph_builder.load_composite()
    _draw_map_composite(ActionContext(snail=snail, overlay=ov, args=[]))


def main():

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
        register_actions(snail, ov)
        _reload_map_from_storage(snail, ov)


        t0 = time.monotonic()
        tfps = collections.deque([0] * 60, maxlen=60)
        UNITS_PER_SECOND = 1000

        while is_not_timeout():
            t0fps = time.perf_counter()
            img = snail.wait_next_frame()
            img1 = cv2.resize(img, None, fx=0.25, fy=0.25)

            f, b = cv2.imencode('.png', img1)
            h, w, _ = img1.shape

            with ov.scene('frame') as ss:

                ss.image(5, 40, w, h, png_bytes=memoryview(b))  # ty:ignore[invalid-argument-type]
            dtfps = int((time.perf_counter() - t0fps) * UNITS_PER_SECOND)
            tfps.appendleft(dtfps)
            with ov.scene('hud') as hud:
                t = time.monotonic() - t0
                fps = len(tfps) * UNITS_PER_SECOND / sum(tfps)
                ft = f'{t:6.3f}, FPS = {fps:06.1f}'
                hud.text(40, 80, ft, (0, 255, 0, 255), "JetBrainsMono NFM", 10)

            cmd = cmd_get()
            if cmd == 'exit':
                break
            icmd = input_cmd_get()
            if icmd == 'input_prompt':
                query = ""
                selected_idx = 0
                submitted = False
                with key_capture_window() as (app, cap_win), \
                        input_hook() as key_queue:
                    results = fuzzy_match(query, get_actions())
                    draw_command_palette(ov, query, results, selected_idx, r.xywh())
                    done = False
                    selected_idx_final = 0
                    while not done:
                        while not submitted:
                            try:
                                event = key_queue.get_nowait()
                            except Empty:
                                break
                            etype = event["type"]
                            value = event["value"]
                            if etype == "char" and value != '':
                                query += value
                                selected_idx = 0
                            elif etype == "up":
                                if value == 'Backspace':
                                    query = query[:-1]
                                    selected_idx = 0
                                elif value == 'Enter':
                                    submitted = True
                                    done = True
                                    selected_idx_final = selected_idx
                                elif value == 'Escape':
                                    query = ""
                                    done = True
                                elif value == 'Up':
                                    if selected_idx > 0:
                                        selected_idx -= 1
                                elif value == 'Down':
                                    if selected_idx < len(results) - 1:
                                        selected_idx += 1
                                elif value == 'Right':
                                    query = results[min(selected_idx_final, len(results) - 1)]['name']
                                elif value == 'Space':
                                    query += ' '

                            results = fuzzy_match(query, get_actions())
                            draw_command_palette(ov, query, results, selected_idx, r.xywh())
                        app.processEvents()
                        time.sleep(0.010)
                ov.destroy_scene('input')
                if submitted and results:
                    action = results[min(selected_idx_final, len(results) - 1)]
                    input_queue.append(action["name"])
                    query_arg = query.lstrip(action["name"]).strip()
                    args = [query_arg] if query_arg else []
                    ctx = ActionContext(snail=snail, overlay=ov, args=args)
                    execute_action(action["name"], ctx)
                    _refresh_history_widget(ov, input_queue, r.xywh())
