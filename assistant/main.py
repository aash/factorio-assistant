import logging
from assistant.actions import action_decorator
import cv2
import collections
import time
from queue import Empty
from common import exit_hotkey, hotkey_handler, timeout
from mapar import Snail
from overlay import overlay
from assistant import input_hook
from assistant import key_capture_window
from assistant import fuzzy_match, execute_action, ActionContext, register_actions, get_actions
import argparse
from graphics import crop_image

HISTORY_MAX = 10
HISTORY_LINE_H = 22
HISTORY_MARGIN = 10
HISTORY_BG_ALPHA = 160
INPUT_BOX_H = 28
RESULT_LINE_H = 22
RESULT_MARGIN = 8
MAX_VISIBLE_RESULTS = 8


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
            text_color = (255, 255, 255, 255) if is_sel else (180, 180, 180, 200)
            s.text(box_x + 14, ry + RESULT_LINE_H - 5, action["name"],
                   color=text_color, font="JetBrainsMono NFM", size=10)


@action_decorator(name="take_window_screenshot", desc="Takes screenshot of window's full client area and saves it in the root directory")
def take_window_screenshot(ctx: ActionContext):
    img = ctx.snail.wait_next_frame()
    cv2.imwrite('screen.png', img)

@action_decorator(name="take_non_ui_screenshot", desc="Takes screenshot of non UI detected area and saves it in the root directory")
def take_screenshot(ctx: ActionContext):
    img = ctx.snail.wait_next_frame()
    non_ui_img = crop_image(img, ctx.snail.non_ui_rect)
    cv2.imwrite('screen_non_ui.png', non_ui_img)


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
        r = snail.window_rect
        register_actions(snail, ov)

        with ov.scene('tst') as s:
            s.rect(*r.xywh(), pen_color=(0, 255, 0, 255), pen_width=2)
            for uir in snail.ui_brects:
                s.rect(*uir.moved(r.x0, r.y0).xywh(), pen_color=(0, 255, 0, 255), pen_width=1)

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

                ss.image(r.x0, r.y0, w, h, png_bytes=memoryview(b))  # ty:ignore[invalid-argument-type]
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
                            results = fuzzy_match(query, get_actions())
                            draw_command_palette(ov, query, results, selected_idx, r.xywh())
                        app.processEvents()
                        time.sleep(0.016)
                ov.destroy_scene('input')
                if submitted and results:
                    action = results[min(selected_idx_final, len(results) - 1)]
                    input_queue.append(action["name"])
                    query_arg = query.lstrip(action["name"]).strip()
                    args = [query_arg] if query_arg else []
                    ctx = ActionContext(snail=snail, overlay=ov, args=args)
                    execute_action(action["name"], ctx)
                    draw_history(ov, input_queue, r.xywh())
