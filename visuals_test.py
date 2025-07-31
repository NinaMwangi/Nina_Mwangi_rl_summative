import os
import sys
import argparse
import imageio
import numpy as np

# Ensure project root is on the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import AngazaEnv
from implementation.rendering import GridRenderer

# Optional: only needed for --mode model
try:
    from stable_baselines3.common.base_class import BaseAlgorithm
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False

import pygame
from pygame import surfarray


def capture_frame(screen):
    """Capture the current pygame screen as an (H, W, 3) uint8 array."""
    frame = surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))  # Convert (W,H,3) to (H,W,3)
    return frame


def run_random(env, renderer, steps, fps, save_gif, gif_path):
    obs, _ = env.reset()
    frames = []

    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        renderer.render(env.agent_pos, env.threat_pos, env.safe_zone, fps=fps)

        if save_gif:
            frames.append(capture_frame(renderer.screen))

        if done:
            obs, _ = env.reset()

    if save_gif and frames:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Saved GIF to {gif_path}")


def run_model(env, renderer, steps, fps, save_gif, gif_path, model_path, deterministic=True):
    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is not installed or not importable. "
            "Install it and try again, or run with --mode random."
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Lazy import to avoid crashing in random mode
    from stable_baselines3 import DQN, PPO, A2C
    # REINFORCE is not in SB3; if you implement your own, adapt here.

    # Try each algo loader until one works
    algo_candidates = [DQN, PPO, A2C]
    model = None
    for Algo in algo_candidates:
        try:
            model = Algo.load(model_path)
            print(f"Loaded model using {Algo.__name__}")
            break
        except Exception:
            continue

    if model is None:
        # Fallback to BaseAlgorithm.load if direct fails
        try:
            model = BaseAlgorithm.load(model_path)
            print("[✔] Loaded model using BaseAlgorithm.load()")
        except Exception as e:
            raise RuntimeError(
                f"Could not load the model with known SB3 algorithms. Error: {e}"
            )

    obs, _ = env.reset()
    frames = []

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        renderer.render(env.agent_pos, env.threat_pos, env.safe_zone, fps=fps)

        if save_gif:
            frames.append(capture_frame(renderer.screen))

        if done:
            obs, _ = env.reset()

    if save_gif and frames:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Saved GIF to {gif_path}")


def main():
    parser = argparse.ArgumentParser(description="SafeHavenEnv Visual Test")
    parser.add_argument("--mode", choices=["random", "model"], default="random",
                        help="Run random rollout or a trained model.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained SB3 model (required for --mode model).")
    parser.add_argument("--steps", type=int, default=30, help="Number of steps to simulate.")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for rendering and GIF.")
    parser.add_argument("--gif_path", type=str, default="simulation.gif", help="Output GIF file path.")
    parser.add_argument("--no_gif", action="store_true", help="Disable GIF saving.")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic actions for model mode.")

    args = parser.parse_args()

    env = AngazaEnv()
    renderer = GridRenderer(grid_size=env.grid_size, title=f"Angaza Simulation - {args.mode.title()} Mode")

    try:
        if args.mode == "random":
            run_random(
                env, renderer,
                steps=args.steps,
                fps=args.fps,
                save_gif=not args.no_gif,
                gif_path=args.gif_path
            )
        else:  # model mode
            if args.model_path is None:
                raise ValueError("You must provide --model_path when running with --mode model.")
            run_model(
                env, renderer,
                steps=args.steps,
                fps=args.fps,
                save_gif=not args.no_gif,
                gif_path=args.gif_path,
                model_path=args.model_path,
                deterministic=args.deterministic
            )
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
