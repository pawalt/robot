"""Record preselected first audit episodes of final policies on Modal."""
import json
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from acrobot_feasibility.record import frame
from .sweep import load_actor
from .train import evaluate
from .curriculum import test_suffix


def record(directory, source):
    torch.set_num_threads(1)
    result = json.loads((directory / 'results.json').read_text())
    assert result['status'] == 'complete'
    actor = load_actor(result['arm'], directory / 'final-retained.pt')
    data = np.load(source / 'controller-0.npz')
    recordings = []
    for mode in ['balance', 'suffix-2', 'bottom']:
        if mode == 'suffix-2':
            row, (xs, us) = test_suffix(actor, data['states'], 2., seed=96000 + result['seed'], count=64, capture=True)
        else:
            row, (xs, us) = evaluate(actor, data['capture_matrix'], seed=96000 + result['seed'], count=64,
                                     mode=mode, seconds=45, capture=True)
        audit_mode = 'suffix_2s' if mode == 'suffix-2' else mode
        expected = next(m for m in result['final_audit'] if m['mode'] == audit_mode)
        assert np.allclose(row['per_episode_final_hold'], expected['per_episode_final_hold'], atol=.011)
        label = f"{result['arm']} | final neural policy | {mode}"
        Image.fromarray(frame(xs[0], 0, us[0], label)).save(directory / f'{mode}.jpg')
        with imageio.get_writer(str(directory / f'{mode}.mp4'), fps=20, codec='libx264',
                                pixelformat='yuv420p', output_params=['-movflags', '+faststart']) as writer:
            for t, x in enumerate(xs):
                writer.append_data(frame(x, t * .05, us[t], label))
        recordings.append(dict(mode=mode, checkpoint='final-retained', episode_index=0,
                               video=f'{mode}.mp4', poster=f'{mode}.jpg',
                               final_hold_seconds=row['per_episode_final_hold'][0]))
    result['recordings'] = recordings
    (directory / 'results.json').write_text(json.dumps(result, indent=2))
    return result
