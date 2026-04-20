from __future__ import annotations

ACTIONS = [
    {"name": "draw large text", "desc": "Draw large text on overlay"},
    {"name": "draw ellipse", "desc": "Draw an ellipse shape"},
    {"name": "draw rectangle", "desc": "Draw a rectangle shape"},
    {"name": "clear", "desc": "Clears overlay"},
]


def fuzzy_match(query: str, candidates: list[dict], limit: int = 10) -> list[dict]:
    if not query:
        return candidates[:limit]
    q = query.lower()
    scored = []
    for c in candidates:
        name = c["name"].lower()
        if q in name:
            score = name.index(q)
        else:
            score = len(name) + _levenshtein(q, name)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:limit]]


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def execute_action(name: str, ov, screen_rect):
    x0, y0, w, h = screen_rect
    cx = x0 + w // 2
    cy = y0 + h // 2
    if name == "draw large text":
        with ov.scene("action_text") as s:
            s.text(cx - 200, cy, "LARGE TEXT",
                    color=(255, 200, 50, 255), font="JetBrainsMono NFM", size=36, bold=True)
    elif name == "draw ellipse":
        with ov.scene("action_ellipse") as s:
            s.ellipse(cx - 120, cy - 80, 240, 160,
                       pen_color=(0, 200, 255, 220), pen_width=2,
                       brush_color=(0, 100, 200, 60))
    elif name == "draw rectangle":
        with ov.scene("action_rect") as s:
            s.rect(cx - 120, cy - 80, 240, 160,
                    pen_color=(255, 100, 50, 220), pen_width=2,
                    brush_color=(200, 50, 20, 60))
    elif name == "clear":
        ov.destroy_scene('action_rect')
        ov.destroy_scene('action_ellipse')
        ov.destroy_scene('action_text')
            