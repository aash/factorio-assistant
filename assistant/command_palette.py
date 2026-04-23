from __future__ import annotations

import time
from queue import Empty

from assistant import input_hook, key_capture_window
from assistant.scene_bloat import sample_scene_bloat


INPUT_BOX_H = 28
RESULT_LINE_H = 36
RESULT_MARGIN = 8
MAX_VISIBLE_RESULTS = 6


def draw_command_palette(ov, query, results, selected_idx, screen_rect, start_idx=0):
    x0, y0, w, h = screen_rect
    pad = 20
    box_x = x0 + pad
    box_y = y0 + pad
    box_w = w - 2 * pad
    visible_results = results[start_idx:start_idx + MAX_VISIBLE_RESULTS]
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

        for i, action in enumerate(visible_results):
            ry = box_y + INPUT_BOX_H + RESULT_MARGIN + i * RESULT_LINE_H
            is_sel = (i + start_idx == selected_idx)
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


def command_palette_prompt(ov, screen_rect, get_actions, fuzzy_match):
    query = ""
    selected_idx = 0
    start_idx = 0
    submitted = False
    results = fuzzy_match(query, get_actions())
    draw_command_palette(ov, query, results, selected_idx, screen_rect, start_idx)

    with key_capture_window() as (app, cap_win), input_hook() as key_queue:
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
                    start_idx = 0
                elif etype == "up":
                    if value == 'Backspace':
                        query = query[:-1]
                        selected_idx = 0
                        start_idx = 0
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
                            if selected_idx < start_idx:
                                start_idx = selected_idx
                    elif value == 'Down':
                        if selected_idx < len(results) - 1:
                            selected_idx += 1
                            if selected_idx >= start_idx + MAX_VISIBLE_RESULTS:
                                start_idx = selected_idx - MAX_VISIBLE_RESULTS + 1
                    elif value == 'Right':
                        if results:
                            query = results[min(selected_idx, len(results) - 1)]['name']
                    elif value == 'Space':
                        query += ' '

                results = fuzzy_match(query, get_actions())
                if selected_idx >= len(results):
                    selected_idx = max(0, len(results) - 1)
                if selected_idx < start_idx:
                    start_idx = selected_idx
                if selected_idx >= start_idx + MAX_VISIBLE_RESULTS:
                    start_idx = max(0, selected_idx - MAX_VISIBLE_RESULTS + 1)
                draw_command_palette(ov, query, results, selected_idx, screen_rect, start_idx)
            sample_scene_bloat(ov)
            app.processEvents()
            time.sleep(0.010)

    return submitted, query, selected_idx_final, results
