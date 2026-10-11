"""Train and record a complete, reproducible dashboard run."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .env import GridReachingEnv
from .train import evaluate, policy_text, train


def render_frame(env, trail, label, total_reward):
    """Render observations as RGB pixels; keep the environment logic separate."""
    image = Image.new("RGB", (640, 736), "#101b25")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    draw.text((32, 22), label, font=font, fill="#e8f1f2")
    draw.text((32, 54), f"Step {env.steps:03d}   /   Return {total_reward:+.0f}",
              font=small, fill="#95aeb9")
    cell = 576 / env.size

    def box(state, inset=3):
        row, col = divmod(state, env.size)
        x, y = 32 + col * cell, 100 + row * cell
        return (x + inset, y + inset, x + cell - inset, y + cell - inset)

    for state in range(env.observation_space.n):
        draw.rounded_rectangle(box(state), radius=5, fill="#1e303e")
    for state in set(trail):
        draw.rounded_rectangle(box(state), radius=5, fill="#28585a")
    centers = []
    for state in trail:
        row, col = divmod(state, env.size)
        centers.append((32 + (col + 0.5) * cell, 100 + (row + 0.5) * cell))
    if len(centers) > 1:
        draw.line(centers, fill="#55aaa2", width=3)
    draw.rounded_rectangle(box(env.goal, 7), radius=7, fill="#eabd6b")
    draw.ellipse(box(env.state, 12), fill="#80e4d1", outline="#ecfff9", width=3)
    status = "Goal reached" if env.state == env.goal else (
        "Time limit reached" if env.done else "Moving toward the gold target"
    )
    draw.text((32, 698), status, font=small, fill="#e8f1f2")
    return np.asarray(image)


def record_episode(env, q, start, output, seed=10000, fps=6):
    rng = np.random.default_rng(seed)
    state, _ = env.reset(seed=seed, options={"start_state": start})
    trail, total_reward = [state], 0.0
    label = "Random policy" if q is None else "Learned policy"
    frame = render_frame(env, trail, label, total_reward)
    Image.fromarray(frame).save(output.with_suffix(".jpg"))
    # H.264 + yuv420p plays in browsers; faststart permits playback before download.
    with imageio.get_writer(str(output), fps=fps, codec="libx264",
                            pixelformat="yuv420p", macro_block_size=16,
                            output_params=["-movflags", "+faststart"]) as writer:
        writer.append_data(frame)
        while True:
            action = int(rng.integers(4)) if q is None else int(q[state].argmax())
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            trail.append(state)
            frame = render_frame(env, trail, label, total_reward)
            writer.append_data(frame)
            if terminated or truncated:
                break
        for _ in range(fps):
            writer.append_data(frame)
    return {"policy": "random" if q is None else "learned", "start": start,
            "steps": env.steps, "return": total_reward, "success": terminated,
            "video": output.name, "poster": output.with_suffix(".jpg").name,
            "duration": (env.steps + 1 + fps) / fps}


def create_run(root, episodes=5000, seed=0):
    if not 1 <= episodes <= 100000 or not 0 <= seed < 2**32:
        raise ValueError("Use 1–100000 episodes and a seed between 0 and 2**32-1")
    created = datetime.now(timezone.utc)
    run_id = created.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    directory = Path(root) / run_id
    directory.mkdir(parents=True)
    env = GridReachingEnv()
    q, history = train(env, episodes=episodes, seed=seed)
    np.savez(directory / "policy.npz", q=q, size=env.size, max_steps=env.max_steps)
    (directory / "policy.txt").write_text(policy_text(q, env.size) + "\n")
    with (directory / "training.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "epsilon", "return", "steps", "success"])
        writer.writerows(history)
    recordings = []
    for start in (0, 22, 45, 90):
        for name, table in (("learned", q), ("random", None)):
            recordings.append(record_episode(
                env, table, start, directory / f"{name}-{start}.mp4", seed=10000 + start
            ))
    # Average over up to 100 episodes per point to make progress readable.
    window = min(100, episodes)
    curve = []
    for offset in range(0, len(history), window):
        batch = history[offset : offset + window]
        curve.append({"episode": batch[-1][0],
                      "return": float(np.mean([row[2] for row in batch])),
                      "success_rate": float(np.mean([row[4] for row in batch]))})
    result = {"id": run_id, "created_at": created.isoformat(),
              "episodes": episodes, "seed": seed, "size": env.size,
              "max_steps": env.max_steps, "alpha": 0.1, "gamma": 0.95,
              "evaluation_seed": 10000, "curve_window": window,
              "learned": evaluate(env, q), "random": evaluate(env),
              "curve": curve, "recordings": recordings}
    # Publish the manifest last: incomplete runs never appear in the dashboard.
    (directory / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    env.close()
    return result
