"""Record the first four held-out starts, without selecting for performance."""
import imageio.v2 as imageio
from PIL import Image

from acrobot_feasibility.record import frame
from .controller import DT, rollout


def record(directory, config):
    results, (states, actions) = rollout([config], seed=42000, count=4,
                                        seconds=60, physics_dt=0.005, capture=True)
    recordings = []
    for index in range(4):
        name = f"swingup-{index}"
        label = f"Energy + LQR | held-out start {index + 1}"
        Image.fromarray(frame(states[0, index], 0, actions[0, index], label)).save(directory / f"{name}.jpg")
        with imageio.get_writer(str(directory / f"{name}.mp4"), fps=20, codec="libx264",
                               pixelformat="yuv420p", output_params=["-movflags", "+faststart"]) as writer:
            for step in range(len(states)):
                writer.append_data(frame(states[step, index], step * DT, actions[step, index], label))
        recordings.append({"video": f"{name}.mp4", "poster": f"{name}.jpg",
                           "episode_index": index, "seed": 42000,
                           "longest_hold": results[0]["per_episode_hold"][index]})
    return recordings
