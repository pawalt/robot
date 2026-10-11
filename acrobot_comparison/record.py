"""Record fixed, preselected evaluation starts rather than best-looking episodes."""

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .env import DT, measurements
from .train import evaluate, network


def frame(state, step, label):
    image = Image.new("RGB", (640, 640), "#101b25")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    _, height, upright = measurements(np.asarray([state]))
    draw.text((28, 20), label, font=font, fill="#edf4f5")
    draw.text((28, 52), f"{step * DT:.2f}s   Height {height[0]:.2f}   {'BALANCED' if upright[0] else 'NOT BALANCED'}", font=small, fill="#80e4d1" if upright[0] else "#eabd6b")
    center, length = np.array([320., 355.]), 116
    t1, t2 = state[:2]
    elbow = center + length * np.array([np.sin(t1), np.cos(t1)])
    tip = elbow + length * np.array([np.sin(t1 + t2), np.cos(t1 + t2)])
    draw.line((320, 105, 320, 605), fill="#293946", width=1)
    draw.line((220, 123, 420, 123), fill="#eabd6b", width=2)
    draw.text((28, 606), "Elbow torque only · target: both links straight up", font=small, fill="#9aafbd")
    draw.line((*center, *elbow), fill="#80e4d1", width=12)
    draw.line((*elbow, *tip), fill="#b5cce3", width=12)
    for point, color in ((center, "#edf4f5"), (elbow, "#eabd6b"), (tip, "#80e4d1")):
        x, y = point
        draw.ellipse((x-8, y-8, x+8, y+8), fill=color)
    return np.asarray(image)


def record_checkpoints(directory, config):
    recordings = []
    for update in sorted(set([0, config["updates"] // 2 // 10 * 10, config["updates"]])):
        checkpoint = directory / f"policy-{update}.pt"
        if not checkpoint.exists():
            continue
        policy = network(3)
        policy.load_state_dict(torch.load(checkpoint, weights_only=True))
        for mode in ("balance", "swingup"):
            metrics, trace = evaluate(policy, count=32, mode=mode, horizon=config["horizon"], capture=True)
            name = f"{mode}-{update}.mp4"
            label = f"{config['method'].upper()} · seed {config['seed']} · {mode}"
            Image.fromarray(frame(trace[0], 0, label)).save(directory / name.replace(".mp4", ".jpg"))
            with imageio.get_writer(str(directory / name), fps=20, codec="libx264", pixelformat="yuv420p", output_params=["-movflags", "+faststart"]) as writer:
                for step, state in enumerate(trace):
                    writer.append_data(frame(state, step, label))
            recordings.append({"mode": mode, "update": update, "steps": update * config["groups"] * config["group_size"] * config["horizon"], "video": name, "poster": name.replace(".mp4", ".jpg"), "action_selection": "argmax", "evaluation_seed": 10000, "episode_index": 0})
    return recordings
