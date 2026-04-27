from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Generator, Mapping, MutableMapping
from typing import Callable

from assistant.actions import ActionContext


PID_CFG_DEFAULT: dict[str, float] = {
    'kp': 0.7,
    'ki': 0.0,
    'kd': 0.12,
    'dt': 0.05,
    'tolerance': 2.0,
    'deadzone': 0.15,
    'max_time': 8.0,
}


def ensure_pid_cfg_loaded(snail, pid_cfg: MutableMapping[str, float], pid_cfg_loaded: bool) -> tuple[dict[str, float], bool]:
    if pid_cfg_loaded:
        return dict(pid_cfg), True

    merged = dict(PID_CFG_DEFAULT)
    cached = snail.cache.get('pid_nav')
    if isinstance(cached, Mapping):
        for k in merged:
            if k in cached:
                try:
                    merged[k] = float(cached[k])
                except Exception:
                    pass

    return merged, True


def save_pid_cfg(snail, pid_cfg: Mapping[str, float]) -> None:
    snail.cache.pid_nav = {
        'kp': float(pid_cfg['kp']),
        'ki': float(pid_cfg['ki']),
        'kd': float(pid_cfg['kd']),
        'dt': float(pid_cfg['dt']),
        'tolerance': float(pid_cfg['tolerance']),
        'deadzone': float(pid_cfg['deadzone']),
        'max_time': float(pid_cfg['max_time']),
    }


def _set_movement_keys(snail, desired: set[str], pressed: set[str]):
    for k in list(pressed - desired):
        snail.ahk.send_input(f'{{{k} up}}')
    for k in list(desired - pressed):
        snail.ahk.send_input(f'{{{k} down}}')
    pressed.clear()
    pressed.update(desired)


def _pid_desired_keys(ux: float, uy: float, deadzone: float) -> set[str]:
    desired: set[str] = set()
    if ux > deadzone:
        desired.add('d')
    elif ux < -deadzone:
        desired.add('a')
    if uy > deadzone:
        desired.add('s')
    elif uy < -deadzone:
        desired.add('w')
    return desired


def _run_pid_episode(ctx: ActionContext, target: tuple[int, int], cfg: Mapping[str, float]) -> dict[str, float | int | bool]:
    kp = float(cfg['kp'])
    ki = float(cfg['ki'])
    kd = float(cfg['kd'])
    dt = max(0.01, float(cfg['dt']))
    tol = max(0.0, float(cfg['tolerance']))
    deadzone = max(0.0, float(cfg['deadzone']))
    max_time = max(dt, float(cfg['max_time']))

    ix = 0.0
    iy = 0.0
    prev_ex = 0.0
    prev_ey = 0.0
    pressed: set[str] = set()

    start = time.perf_counter()
    steps = 0
    switches = 0
    iae = 0.0
    jerk = 0.0
    path_len = 0.0
    timeout = False

    prev_current: tuple[int, int] | None = None
    prev_move: tuple[float, float] | None = None
    last_cmd_change_ts: float | None = None
    delay_samples: list[float] = []

    try:
        while True:
            now = time.perf_counter()
            current = ctx.snail.character_coord
            if current is None:
                break
            cur = (int(current[0]), int(current[1]))

            ex = float(target[0] - cur[0])
            ey = float(target[1] - cur[1])
            dist = math.hypot(ex, ey)
            iae += dist * dt
            if dist <= tol:
                break

            ix += ex * dt
            iy += ey * dt
            dx = (ex - prev_ex) / dt
            dy = (ey - prev_ey) / dt
            prev_ex = ex
            prev_ey = ey

            ux = kp * ex + ki * ix + kd * dx
            uy = kp * ey + ki * iy + kd * dy

            desired = _pid_desired_keys(ux, uy, deadzone)
            if desired != pressed:
                switches += 1
                last_cmd_change_ts = now
            _set_movement_keys(ctx.snail, desired, pressed)

            if prev_current is not None:
                mv = (float(cur[0] - prev_current[0]), float(cur[1] - prev_current[1]))
                mv_norm = math.hypot(mv[0], mv[1])
                path_len += mv_norm
                if prev_move is not None:
                    jerk += math.hypot(mv[0] - prev_move[0], mv[1] - prev_move[1])
                prev_move = mv
                if mv_norm > 0.0 and last_cmd_change_ts is not None:
                    delay_samples.append(max(0.0, now - last_cmd_change_ts))
                    last_cmd_change_ts = None
            prev_current = cur

            steps += 1
            if now - start >= max_time:
                timeout = True
                break
            time.sleep(dt)
    finally:
        _set_movement_keys(ctx.snail, set(), pressed)

    final = ctx.snail.character_coord
    final_dist = math.hypot(float(target[0] - int(final[0])), float(target[1] - int(final[1]))) if final is not None else 1e9
    delay_mean = (sum(delay_samples) / len(delay_samples)) if delay_samples else max_time

    score = (
        1.00 * iae
        + 0.35 * jerk
        + 0.12 * switches
        + 1.80 * max(0.0, final_dist - tol)
        + 0.80 * delay_mean
        + (25.0 if timeout else 0.0)
    )

    return {
        'score': score,
        'iae': iae,
        'jerk': jerk,
        'switches': float(switches),
        'delay_mean': delay_mean,
        'final_dist': final_dist,
        'steps': float(steps),
        'timeout': timeout,
        'path_len': path_len,
    }


def _build_relative_targets(origin: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    ox, oy = int(origin[0]), int(origin[1])
    r = max(1, int(radius))
    return [
        (ox + r, oy),
        (ox + r, oy + r),
        (ox, oy + r),
        (ox - r, oy + r),
        (ox - r, oy),
        (ox - r, oy - r),
        (ox, oy - r),
        (ox + r, oy - r),
        (ox, oy),
    ]


def _evaluate_pid_candidate(ctx: ActionContext, cfg: Mapping[str, float], radius: int, target_count: int) -> dict[str, float | int | bool]:
    if ctx.snail.character_coord is None:
        return {'score': 1e9, 'failed': True}

    origin = (int(ctx.snail.character_coord[0]), int(ctx.snail.character_coord[1]))
    targets = _build_relative_targets(origin, radius)[:max(1, int(target_count))]
    episodes: list[dict[str, float | int | bool]] = []
    total = 0.0
    for target in targets:
        ep = _run_pid_episode(ctx, target, cfg)
        episodes.append(ep)
        total += float(ep['score'])

    mean_score = total / max(1, len(episodes))
    return {
        'score': mean_score,
        'episodes': float(len(episodes)),
        'failed': False,
    }


def _run_pid_episode_gen(ctx: ActionContext, target: tuple[int, int], cfg: Mapping[str, float]) -> Generator[None, None, dict[str, float | int | bool]]:
    kp = float(cfg['kp'])
    ki = float(cfg['ki'])
    kd = float(cfg['kd'])
    dt = max(0.01, float(cfg['dt']))
    tol = max(0.0, float(cfg['tolerance']))
    deadzone = max(0.0, float(cfg['deadzone']))
    max_time = max(dt, float(cfg['max_time']))

    ix = iy = 0.0
    prev_ex = prev_ey = 0.0
    pressed: set[str] = set()

    start = time.perf_counter()
    next_step_ts = start
    steps = 0
    switches = 0
    iae = 0.0
    jerk = 0.0
    path_len = 0.0
    timeout = False

    prev_current: tuple[int, int] | None = None
    prev_move: tuple[float, float] | None = None
    last_cmd_change_ts: float | None = None
    delay_samples: list[float] = []

    try:
        while True:
            now = time.perf_counter()
            if now < next_step_ts:
                yield
                continue
            next_step_ts = now + dt

            current = ctx.snail.character_coord
            if current is None:
                break
            cur = (int(current[0]), int(current[1]))

            ex = float(target[0] - cur[0])
            ey = float(target[1] - cur[1])
            dist = math.hypot(ex, ey)
            iae += dist * dt
            if dist <= tol:
                break

            ix += ex * dt
            iy += ey * dt
            dx = (ex - prev_ex) / dt
            dy = (ey - prev_ey) / dt
            prev_ex = ex
            prev_ey = ey

            ux = kp * ex + ki * ix + kd * dx
            uy = kp * ey + ki * iy + kd * dy

            desired = _pid_desired_keys(ux, uy, deadzone)
            if desired != pressed:
                switches += 1
                last_cmd_change_ts = now
            _set_movement_keys(ctx.snail, desired, pressed)

            if prev_current is not None:
                mv = (float(cur[0] - prev_current[0]), float(cur[1] - prev_current[1]))
                mv_norm = math.hypot(mv[0], mv[1])
                path_len += mv_norm
                if prev_move is not None:
                    jerk += math.hypot(mv[0] - prev_move[0], mv[1] - prev_move[1])
                prev_move = mv
                if mv_norm > 0.0 and last_cmd_change_ts is not None:
                    delay_samples.append(max(0.0, now - last_cmd_change_ts))
                    last_cmd_change_ts = None
            prev_current = cur

            steps += 1
            if now - start >= max_time:
                timeout = True
                break
            yield
    finally:
        _set_movement_keys(ctx.snail, set(), pressed)

    final = ctx.snail.character_coord
    final_dist = math.hypot(float(target[0] - int(final[0])), float(target[1] - int(final[1]))) if final is not None else 1e9
    delay_mean = (sum(delay_samples) / len(delay_samples)) if delay_samples else max_time

    score = (
        1.00 * iae
        + 0.35 * jerk
        + 0.12 * switches
        + 1.80 * max(0.0, final_dist - tol)
        + 0.80 * delay_mean
        + (25.0 if timeout else 0.0)
    )

    return {
        'score': score,
        'iae': iae,
        'jerk': jerk,
        'switches': float(switches),
        'delay_mean': delay_mean,
        'final_dist': final_dist,
        'steps': float(steps),
        'timeout': timeout,
        'path_len': path_len,
    }


def _evaluate_pid_candidate_gen(ctx: ActionContext, cfg: Mapping[str, float], radius: int, target_count: int) -> Generator[None, None, dict[str, float | int | bool]]:
    if ctx.snail.character_coord is None:
        return {'score': 1e9, 'failed': True}

    origin = (int(ctx.snail.character_coord[0]), int(ctx.snail.character_coord[1]))
    targets = _build_relative_targets(origin, radius)[:max(1, int(target_count))]
    total = 0.0
    for target in targets:
        ep = yield from _run_pid_episode_gen(ctx, target, cfg)
        total += float(ep['score'])
        yield

    mean_score = total / max(1, len(targets))
    return {
        'score': mean_score,
        'episodes': float(len(targets)),
        'failed': False,
    }


def _auto_tune_move_pid_gen(
    ctx: ActionContext,
    opts: Mapping[str, int | float],
    pid_cfg: MutableMapping[str, float],
    on_save: Callable[[], None],
) -> Generator[None, None, None]:
    base = {k: float(pid_cfg[k]) for k in pid_cfg.keys()}
    best_cfg = dict(base)
    best_eval = yield from _evaluate_pid_candidate_gen(ctx, best_cfg, radius=int(opts['radius']), target_count=int(opts['targets']))
    best_score = float(best_eval['score'])
    logging.info('auto_tune_move_pid baseline score=%.3f cfg=%s', best_score, {k: round(best_cfg[k], 4) for k in sorted(best_cfg.keys())})

    step = max(0.05, float(opts['step']))
    iters = max(1, int(opts['iters']))

    for i in range(iters):
        cand = dict(best_cfg)
        cand['kp'] = max(0.0, cand['kp'] * (1.0 + random.uniform(-step, step)))
        cand['ki'] = max(0.0, cand['ki'] * (1.0 + random.uniform(-step, step)))
        cand['kd'] = max(0.0, cand['kd'] * (1.0 + random.uniform(-step, step)))
        cand['deadzone'] = min(2.0, max(0.01, cand['deadzone'] * (1.0 + random.uniform(-step, step))))
        cand['tolerance'] = min(20.0, max(0.5, cand['tolerance'] * (1.0 + random.uniform(-step, step))))
        cand['dt'] = min(0.20, max(0.02, cand['dt'] * (1.0 + random.uniform(-0.2 * step, 0.2 * step))))

        ev = yield from _evaluate_pid_candidate_gen(ctx, cand, radius=int(opts['radius']), target_count=int(opts['targets']))
        score = float(ev['score'])
        logging.info('auto_tune iter=%d/%d score=%.3f cand=%s', i + 1, iters, score, {k: round(cand[k], 4) for k in sorted(cand.keys())})
        if score < best_score:
            best_score = score
            best_cfg = cand
            logging.info('auto_tune new best score=%.3f', best_score)
        yield

    pid_cfg.update(best_cfg)
    on_save()

    ctx.snail.cache.pid_nav_last_autotune = {
        'best_score': float(best_score),
        'iters': iters,
        'radius': int(opts['radius']),
        'targets': int(opts['targets']),
        'step': step,
        'best_cfg': {k: float(best_cfg[k]) for k in sorted(best_cfg.keys())},
        'ts': time.time(),
    }
    ctx.snail.cache.to_yaml(ctx.snail.CACHE_FILE)

    logging.info('auto_tune_move_pid done: best_score=%.3f cfg=%s', best_score, {k: round(best_cfg[k], 4) for k in sorted(best_cfg.keys())})


def _benchmark_move_pid_gen(
    ctx: ActionContext,
    params: Mapping[str, int],
    pid_cfg: Mapping[str, float],
) -> Generator[None, None, None]:
    result = yield from _evaluate_pid_candidate_gen(
        ctx,
        pid_cfg,
        radius=int(params['radius']),
        target_count=int(params['targets']),
    )
    if bool(result.get('failed')):
        logging.info('benchmark_move_pid failed: character_coord unavailable')
        return

    ctx.snail.cache.pid_nav_last_benchmark = {
        'params': {k: float(pid_cfg[k]) for k in sorted(pid_cfg.keys())},
        'score': float(result['score']),
        'radius': int(params['radius']),
        'targets': int(params['targets']),
        'ts': time.time(),
    }
    ctx.snail.cache.to_yaml(ctx.snail.CACHE_FILE)

    logging.info(
        'benchmark_move_pid score=%.3f radius=%d targets=%d params=%s',
        float(result['score']),
        int(params['radius']),
        int(params['targets']),
        {k: round(float(pid_cfg[k]), 4) for k in sorted(pid_cfg.keys())},
    )
