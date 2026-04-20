import cv2
import collections
import time
from queue import Empty
from common import exit_hotkey, hotkey_handler, timeout
from mapar import Snail
from overlay import overlay
from assistant import input_hook
from assistant import key_capture_window
import argparse

HISTORY_MAX = 10
HISTORY_LINE_H = 22
HISTORY_MARGIN = 10
HISTORY_BG_ALPHA = 160

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

def draw_input(ov, text, screen_rect):
    x0, y0, w, h = screen_rect
    box_x = x0 + 20
    box_y = y0 + 20
    box_w = w - 40
    box_h = 30
    with ov.scene('input') as s:
        s.rect(box_x, box_y, box_w, box_h,
               pen_color=(100, 150, 255, 200), pen_width=2,
               brush_color=(20, 20, 40, 200))
        s.text(box_x + 10, box_y + box_h - 7, text,
               color=(255, 255, 255, 255), font="JetBrainsMono NFM", size=14)

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

                ss.image(r.x0, r.y0, w, h, png_bytes=memoryview(b))
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
                input_string = ""
                submitted = False
                with key_capture_window() as (app, cap_win), \
                        input_hook() as key_queue:
                    draw_input(ov, input_string + "|", r.xywh())
                    done = False
                    while not done:
                        while True:
                            try:
                                event = key_queue.get_nowait()
                            except Empty:
                                break
                            etype = event["type"]
                            value = event["value"]
                            if etype == "char":
                                input_string += value
                            elif etype == "down":
                                if value == 'Backspace':
                                    input_string = input_string[:-1]
                                elif value == 'Enter':
                                    submitted = True
                                    done = True
                                elif value == 'Escape':
                                    input_string = ""
                                    done = True
                        draw_input(ov, input_string + "|", r.xywh())
                        app.processEvents()
                        time.sleep(0.016)
                ov.destroy_scene('input')
                if submitted and input_string:
                    input_queue.append(input_string)
                    draw_history(ov, input_queue, r.xywh())
