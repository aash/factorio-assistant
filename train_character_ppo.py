from __future__ import annotations
import os
import sys
import logging

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mapar import Snail

from assistant.character_nav_env import CharacterNavEnv

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Convert milliseconds to seconds
        record.relativeCreatedInSeconds = record.relativeCreated / 1000.0
        record.levelnameChar = record.levelname[0]
        # Call the original format method
        return super().format(record)

def configure_logging():
    if not os.path.exists('logs'):
        os.mkdir('logs')
    formatter = CustomFormatter('%(levelnameChar)s %(relativeCreatedInSeconds)6.2f %(name)s %(message)s')


    file_handler = logging.FileHandler("logs/train_character_ppo.log", encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)


    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            file_handler,
            console_handler
        ],
    )



def make_env(snail, target_spec: str, episode_timeout_sec: float, target_radius_px: float, key_press_sleep_time: float):
    def _factory():
        return Monitor(
            CharacterNavEnv(
                snail=snail,
                target_spec=target_spec,
                episode_timeout_sec=episode_timeout_sec,
                target_radius_px=target_radius_px,
                key_press_sleep_time=key_press_sleep_time,
            )
        )

    return _factory


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, help="Target list in x0,y0;x1,y1; format")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-out", default="character_ppo.zip")
    parser.add_argument("--episode-timeout-sec", type=float, default=180.0)
    parser.add_argument("--target-radius-px", type=float, default=16.0)
    parser.add_argument("--key-press-sleep-time", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with Snail() as snail:
        vec_env = DummyVecEnv([
            make_env(
                snail,
                args.targets,
                args.episode_timeout_sec,
                args.target_radius_px,
                args.key_press_sleep_time,
            )
        ])

        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
        )
        model.learn(total_timesteps=args.timesteps)
        model.save(args.model_out)


if __name__ == "__main__":
    main()
