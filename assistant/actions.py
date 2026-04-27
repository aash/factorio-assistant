from __future__ import annotations
from overlay import OverlayClient
from mapar import Snail
from dataclasses import dataclass, field
import re
from typing import Callable

_ACTIONS: dict[str, dict] = {}
_LAST_ARGS: dict[str, list[str]] = {}


@dataclass
class ActionContext:
    snail: Snail
    overlay: OverlayClient
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
            name = info["name"]

            def make_callback(f, n):
                def callback():
                    args = _LAST_ARGS.get(n, [])
                    ctx = ActionContext(snail=snail, overlay=ov, args=args)
                    f(ctx)
                callback.__name__ = f.__name__
                return callback

            snail.ahk.add_hotkey(hk, make_callback(func, name))


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


def fuzzy_match_pi(query: str, candidates: list[dict], limit: int = 10) -> list[dict]:
    """Pi-style fuzzy filter/scoring for command candidates.

    Mirrors the fuzzy scoring logic used by Pi TUI:
    - subsequence matching (characters must appear in order)
    - rewards consecutive and word-boundary matches
    - penalizes gaps and later positions
    - supports whitespace-tokenized query (all tokens must match)
    """
    trimmed = query.strip()
    if not trimmed:
        return candidates[:limit]

    tokens = [t for t in trimmed.split() if t]
    scored: list[tuple[float, dict]] = []

    for candidate in candidates:
        text = " ".join(
            str(candidate.get(k, "")) for k in ("name", "desc", "hotkey")
        )

        total_score = 0.0
        all_match = True
        for token in tokens:
            matches, score = _pi_fuzzy_match(token, text)
            if not matches:
                all_match = False
                break
            total_score += score

        if all_match:
            scored.append((total_score, candidate))

    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:limit]]


def _pi_fuzzy_match(query: str, text: str) -> tuple[bool, float]:
    query_lower = query.lower()
    text_lower = text.lower()

    def _match_query(normalized_query: str) -> tuple[bool, float]:
        if not normalized_query:
            return True, 0.0
        if len(normalized_query) > len(text_lower):
            return False, 0.0

        query_index = 0
        score = 0.0
        last_match_index = -1
        consecutive_matches = 0

        for i, ch in enumerate(text_lower):
            if query_index >= len(normalized_query):
                break
            if ch != normalized_query[query_index]:
                continue

            is_word_boundary = i == 0 or bool(re.match(r"[\s\-_./:]", text_lower[i - 1]))

            if last_match_index == i - 1:
                consecutive_matches += 1
                score -= consecutive_matches * 5
            else:
                consecutive_matches = 0
                if last_match_index >= 0:
                    score += (i - last_match_index - 1) * 2

            if is_word_boundary:
                score -= 10

            score += i * 0.1
            last_match_index = i
            query_index += 1

        if query_index < len(normalized_query):
            return False, 0.0

        return True, score

    primary_matches, primary_score = _match_query(query_lower)
    if primary_matches:
        return primary_matches, primary_score

    alpha_numeric_match = re.match(r"^(?P<letters>[a-z]+)(?P<digits>[0-9]+)$", query_lower)
    numeric_alpha_match = re.match(r"^(?P<digits>[0-9]+)(?P<letters>[a-z]+)$", query_lower)

    swapped_query = ""
    if alpha_numeric_match:
        swapped_query = (
            f"{alpha_numeric_match.group('digits')}{alpha_numeric_match.group('letters')}"
        )
    elif numeric_alpha_match:
        swapped_query = (
            f"{numeric_alpha_match.group('letters')}{numeric_alpha_match.group('digits')}"
        )

    if not swapped_query:
        return primary_matches, primary_score

    swapped_matches, swapped_score = _match_query(swapped_query)
    if not swapped_matches:
        return primary_matches, primary_score

    return True, swapped_score + 5


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
        if ctx.args:
            _LAST_ARGS[name] = ctx.args
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

            