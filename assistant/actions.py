from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any

_ACTIONS: dict[str, dict] = {}


@dataclass
class ActionContext:
    snail: Any
    overlay: Any
    args: list[str] = field(default_factory=list)


def action_decorator(
    name: str,
    desc: str = "",
    hotkey: str | None = None,
):
    if not name.isidentifier():
        raise ValueError(f"action name must be valid python identifier, got {name!r}")

    def decorator(func: Callable[[ActionContext], None]) -> Callable[[ActionContext], None]:
        _ACTIONS[name] = {
            "func": func,
            "desc": desc,
            "hotkey": hotkey,
            "name": name,
        }
        return func
    return decorator


def get_actions() -> list[dict]:
    return [
        {"name": info["name"], "desc": info["desc"], "hotkey": info.get("hotkey")}
        for info in _ACTIONS.values()
    ]


def get_action(name: str) -> dict | None:
    return _ACTIONS.get(name)


def register_actions(snail, ov):
    for info in _ACTIONS.values():
        hk = info.get("hotkey")
        if hk:
            func = info["func"]

            def make_callback(f):
                def callback():
                    ctx = ActionContext(snail=snail, overlay=ov, args=[])
                    f(ctx)
                callback.__name__ = f.__name__
                return callback

            snail.ahk.add_hotkey(hk, make_callback(func))


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


def execute_action(name: str, ctx: ActionContext):
    info = _ACTIONS.get(name)
    if info:
        ctx.args = _parse_args(name, ctx.args)
        info["func"](ctx)


def _parse_args(name: str, args: list[str]) -> list[str]:
    parsed = []
    for arg in args:
        if arg.startswith(" "):
            parsed.extend(arg[3:].split(","))
        elif "," in arg:
            parsed.extend(arg.split(","))
        else:
            parsed.append(arg)
    return parsed

            