"""Independent replay, robustness checks, and recordings for optimized paths."""
import json
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from acrobot_feasibility.record import frame
from .trajectory import replay, DT


def publish(directory):
    searches=[json.loads(p.read_text()) for p in sorted(directory.glob('search-*.json'))]
    capture_file=directory/'capture.json'
    region=json.loads(capture_file.read_text()) if capture_file.exists() else None
    paths=[]
    for search in searches:
        if not search['success']: continue
        with np.load(directory/search['file']) as data:
            states,actions,gains=(data[key] for key in ('states','actions','gains'))
            limit=float(data['limit'])
        nominal,trace=replay(states,actions,gains,limit,physics_dt=.005,capture=True,capture_region=region)
        fine,_=replay(states,actions,gains,limit,physics_dt=.0025,capture_region=region)
        robustness=[]
        for scale in (.001,.01,.05):
            initials=np.random.default_rng(51000).uniform(-scale,scale,(64,4))
            row,_=replay(states,actions,gains,limit,physics_dt=.005,initials=initials,capture_region=region)
            robustness.append({'scale':scale,**row})
        controller_file=f'controller-{search["seed"]}.npz'
        extra=({'capture_matrix':np.asarray(region['lyapunov_matrix']),
                'capture_level':region['empirical_level']} if region else {})
        np.savez(directory/controller_file,states=states,actions=actions,gains=gains,limit=limit,**extra)
        name=f'swingup-{search["seed"]}'
        xs,us=trace
        label=f'Trajectory feedback | search {search["seed"]} | limit {limit:g}'
        Image.fromarray(frame(xs[0,0],0,us[0,0],label)).save(directory/f'{name}.jpg')
        with imageio.get_writer(str(directory/f'{name}.mp4'),fps=20,codec='libx264',
                               pixelformat='yuv420p',output_params=['-movflags','+faststart']) as writer:
            for t in range(len(xs)):
                writer.append_data(frame(xs[t,0],t*DT,us[t,0],label))
        max_speeds=np.max(np.abs(xs[:,:,2:]),axis=(0,1)).tolist()
        paths.append(dict(seed=search['seed'],file=controller_file,limit=limit,
                          swingup_seconds=len(actions)*DT,nominal=nominal,fine=fine,
                          robustness=robustness,video=name+'.mp4',poster=name+'.jpg',
                          max_joint_speeds=max_speeds,peak_torque=float(np.max(np.abs(us)))))
        print('Validated path',search['seed'],'fine',fine['final_10s_rate'],
              'perturbed',[(r['scale'],r['final_10s_rate']) for r in robustness],flush=True)
    result=dict(id=directory.name,kind='trajectory_optimization',paths=paths,searches=searches,
                capture_region=region,recordings=[dict(video=p['video'],poster=p['poster']) for p in paths])
    (directory/'results.json').write_text(json.dumps(result,indent=2))
    return result
