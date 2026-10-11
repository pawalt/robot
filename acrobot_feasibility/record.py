"""Record preselected initial state zero for each shown controller and scale."""
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .study import simulate


def frame(state, time, torque, label):
    image = Image.new("RGB", (640, 640), "#101b25")
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), label, fill="#edf4f5", font=ImageFont.load_default(size=21))
    draw.text((25, 55), f"Time {time:.2f}s   Torque {torque:+.3f} N m", fill="#80e4d1", font=ImageFont.load_default(size=18))
    center = np.array([320., 360.])
    elbow = center + 116 * np.array([np.sin(state[0]), np.cos(state[0])])
    tip = elbow + 116 * np.array([np.sin(state[0] + state[1]), np.cos(state[0] + state[1])])
    draw.line((320, 110, 320, 610), fill="#293946", width=1)
    draw.line((230, 128, 410, 128), fill="#eabd6b", width=2)
    draw.line((*center, *elbow), fill="#80e4d1", width=12)
    draw.line((*elbow, *tip), fill="#b5cce3", width=12)
    for point in (center, elbow, tip):
        x, y = point
        draw.ellipse((x-7,y-7,x+7,y+7), fill="#eabd6b")
    return np.asarray(image)


def record(directory):
    configs = [{"controller": c, "scale": scale, "limit": 1.0}
               for scale in (0.05, 1.0) for c in ("continuous", "discrete3", "discrete9", "passive")]
    _, trace = simulate(0.05, configs, count=1, capture=True)
    recordings = []
    for index, config in enumerate(configs):
        name = f"{config['controller']}-{int(config['scale'] * 100)}"
        label = f"{config['controller']} · disturbance {config['scale']}x"
        initial = frame(trace[0]["states"][index], 0, 0, label)
        Image.fromarray(initial).save(directory / f"{name}.jpg")
        with imageio.get_writer(str(directory / f"{name}.mp4"), fps=20, codec="libx264", pixelformat="yuv420p", output_params=["-movflags", "+faststart"]) as writer:
            for point in trace:
                writer.append_data(frame(point["states"][index], point["time"], point["torques"][index], label))
        recordings.append({**config, "dt": 0.05, "video": name + ".mp4", "poster": name + ".jpg", "episode_index": 0})
    return recordings
