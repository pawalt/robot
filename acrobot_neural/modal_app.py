"""All neural training, evaluation, and recording run on Modal."""
import modal
from modal_resources import PROJECT,project_image,volume
app=modal.App('robot-acrobot-neural')
image=project_image(extras=['acrobot','feasibility'])
for package in ['acrobot_neural','acrobot_swingup','acrobot_feasibility','acrobot_comparison']:
    image=image.add_local_dir(PROJECT/package,f'/root/{package}',ignore=['__pycache__','artifacts'])

@app.function(image=image,cpu=2,memory=4096,timeout=7200,volumes={'/data':volume})
def train_job(updates: int=60):
    from datetime import datetime,timezone
    from uuid import uuid4
    from pathlib import Path
    from .train import run,SOURCE
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory=Path('/data/neural')/run_id;directory.mkdir(parents=True)
    print('Neural run',run_id,flush=True)
    return run(directory,Path('/data/swingup')/SOURCE,volume.commit,updates)

@app.local_entrypoint()
def train(updates: int=60):
    import json
    result=train_job.remote(updates)
    output=PROJECT/'acrobot_neural'/'artifacts'/result['id'];output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(result,indent=2))
    print('Saved',result['id'])

@app.function(image=image,cpu=2,memory=4096,timeout=3600,volumes={'/data':volume})
def audit_job(run_id: str):
    import re
    from pathlib import Path
    from .audit import audit
    from .train import SOURCE
    if not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):raise ValueError('Invalid run ID')
    volume.reload()
    result=audit(Path('/data/neural')/run_id,Path('/data/swingup')/SOURCE)
    volume.commit()
    return result

@app.local_entrypoint()
def evaluate(run_id: str):
    import json
    result=audit_job.remote(run_id)
    output=PROJECT/'acrobot_neural'/'artifacts'/run_id;output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(result,indent=2))
    print('Audited',run_id,'selected',result['selected_stage'])

@app.function(image=image,cpu=2,memory=4096,timeout=7200,volumes={'/data':volume})
def consolidate_job(parent: str,updates: int=60):
    import re
    from datetime import datetime,timezone
    from uuid import uuid4
    from pathlib import Path
    from .train import consolidate,SOURCE
    if not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',parent):raise ValueError('Invalid run ID')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory=Path('/data/neural')/run_id;directory.mkdir(parents=True)
    print('Consolidation run',run_id,flush=True)
    return consolidate(directory,Path('/data/swingup')/SOURCE,Path('/data/neural')/parent,volume.commit,updates)

@app.local_entrypoint()
def consolidate(parent: str,updates: int=60):
    import json
    result=consolidate_job.remote(parent,updates)
    output=PROJECT/'acrobot_neural'/'artifacts'/result['id'];output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(result,indent=2))
    print('Saved',result['id'])

@app.function(image=image,cpu=8,memory=8192,timeout=600,volumes={'/data':volume})
def benchmark_job():
    import json
    from pathlib import Path
    from .compute import benchmark
    rows=benchmark()
    directory=Path('/data/neural-compute');directory.mkdir(exist_ok=True)
    from datetime import datetime,timezone
    name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json'
    (directory/name).write_text(json.dumps(rows,indent=2));volume.commit()
    return rows

@app.local_entrypoint()
def benchmark():
    import json
    rows=benchmark_job.remote()
    output=PROJECT/'acrobot_neural'/'artifacts'/'compute';output.mkdir(parents=True,exist_ok=True)
    (output/'benchmark.json').write_text(json.dumps(rows,indent=2))

@app.function(image=image,cpu=4,memory=8192,timeout=7200,volumes={'/data':volume})
def curriculum_job(updates: int=40,count: int=256,threads: int=4):
    from datetime import datetime,timezone
    from uuid import uuid4
    from pathlib import Path
    from .curriculum import run
    from .train import SOURCE
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory=Path('/data/neural')/run_id;directory.mkdir(parents=True)
    print('Curriculum run',run_id,flush=True)
    return run(directory,Path('/data/swingup')/SOURCE,Path('/data/neural/20261011T162331-f3d100d1'),volume.commit,updates,count,threads)

@app.local_entrypoint()
def curriculum(updates: int=40,count: int=256,threads: int=4):
    import json
    result=curriculum_job.remote(updates,count,threads)
    output=PROJECT/'acrobot_neural'/'artifacts'/result['id'];output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(result,indent=2))
    print('Saved',result['id'])

@app.function(image=image,cpu=2,memory=4096,timeout=7200,max_containers=10,volumes={'/data':volume})
def sweep_job(run_id: str,arm: str,seed: int,bc_steps: int=4000,ppo_updates: int=10):
    from pathlib import Path
    from .sweep import run,ARMS
    from .train import SOURCE
    import re
    if arm not in ARMS or not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):raise ValueError('Invalid sweep')
    directory=Path('/data/neural-sweeps')/run_id/f'{arm}-{seed}';directory.mkdir(parents=True,exist_ok=True)
    return run(directory,Path('/data/swingup')/SOURCE,Path('/data/neural/20261011T162331-f3d100d1'),arm,seed,volume.commit,bc_steps,ppo_updates)

@app.local_entrypoint()
def sweep(seeds: int=3,bc_steps: int=4000,ppo_updates: int=10):
    import json
    from datetime import datetime,timezone
    from uuid import uuid4
    from .sweep import ARMS
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    print('SWEEP',run_id,len(ARMS)*seeds,'jobs',flush=True)
    output=PROJECT/'acrobot_neural'/'artifacts'/run_id;output.mkdir(parents=True,exist_ok=True)
    results=[]
    tasks=[(run_id,arm,seed,bc_steps,ppo_updates) for seed in range(seeds) for arm in ARMS]
    for result in sweep_job.starmap(tasks,order_outputs=False,return_exceptions=True):
        if isinstance(result,Exception):
            print('FAILED JOB',repr(result),flush=True);results.append({'error':repr(result)})
        else:
            results.append(result);(output/f"{result['arm']}-{result['seed']}.json").write_text(json.dumps(result,indent=2))
        (output/'sweep.json').write_text(json.dumps({'id':run_id,'results':results},indent=2))
    print('SWEEP COMPLETE',run_id,flush=True)


@app.function(image=image,cpu=2,memory=4096,timeout=1800,max_containers=10,volumes={'/data':volume})
def sweep_record_job(run_id: str,arm: str):
    import re
    from pathlib import Path
    from .sweep import ARMS
    from .sweep_record import record
    from .train import SOURCE
    if arm not in ARMS or not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):raise ValueError('Invalid sweep')
    volume.reload()
    result=record(Path('/data/neural-sweeps')/run_id/f'{arm}-0',Path('/data/swingup')/SOURCE)
    volume.commit()
    return result


@app.local_entrypoint()
def sweep_record(run_id: str,arms: str=''):
    import json
    from .sweep import ARMS
    output=PROJECT/'acrobot_neural'/'artifacts'/run_id
    selected=arms.split(',') if arms else ARMS
    if any(arm not in ARMS for arm in selected):raise ValueError('Invalid arm')
    for result in sweep_record_job.starmap([(run_id,arm) for arm in selected],order_outputs=False):
        (output/f"{result['arm']}-{result['seed']}.json").write_text(json.dumps(result,indent=2))
        print('RECORDED',result['arm'],flush=True)


@app.function(image=image,cpu=2,memory=4096,timeout=7200,max_containers=18,volumes={'/data':volume})
def imitation_job(run_id: str,arm: str,seed: int,rounds: int=24,steps: int=1000,loss_kind: str='mse'):
    import re
    from pathlib import Path
    from .imitation import run,ARMS
    from .train import SOURCE
    if arm not in ARMS or not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):raise ValueError('Invalid imitation run')
    if loss_kind not in ['mse','actuator']:raise ValueError('Invalid loss')
    volume.reload()
    directory=Path('/data/neural-imitation')/run_id/f'{arm}-{seed}';directory.mkdir(parents=True,exist_ok=True)
    return run(directory,Path('/data/swingup')/SOURCE,
               Path('/data/neural-sweeps/20261011T165814-20ab07d2/rehearsal-0'),arm,seed,volume.commit,rounds,steps,loss_kind)


@app.local_entrypoint()
def imitation(rounds: int=24,steps: int=1000,seeds: int=3,loss_kind: str='mse'):
    import json
    from datetime import datetime,timezone
    from uuid import uuid4
    from .imitation import ARMS
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    print('IMITATION',run_id,flush=True)
    output=PROJECT/'acrobot_neural'/'artifacts'/run_id;output.mkdir(parents=True,exist_ok=True)
    results=[]
    for result in imitation_job.starmap([(run_id,a,s,rounds,steps,loss_kind) for a in ARMS for s in range(seeds)],order_outputs=False,return_exceptions=True):
        if isinstance(result,Exception):
            print('FAILED',repr(result),flush=True);results.append(dict(error=repr(result)))
        else:
            results.append(result);(output/f"{result['arm']}-{result['seed']}.json").write_text(json.dumps(result,indent=2))
        (output/'imitation.json').write_text(json.dumps(dict(id=run_id,results=results),indent=2))
    print('ALL IMITATION JOBS RETURNED',run_id,flush=True)


@app.function(image=image,cpu=2,memory=4096,timeout=1800,max_containers=18,volumes={'/data':volume})
def imitation_record_job(run_id: str,arm: str,seed: int,confirm: bool=False):
    import re
    from pathlib import Path
    from .imitation import ARMS
    from .imitation_record import record
    from .train import SOURCE
    if arm not in ARMS or not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):raise ValueError('Invalid run')
    volume.reload()
    result=record(Path('/data/neural-imitation')/run_id/f'{arm}-{seed}',Path('/data/swingup')/SOURCE,confirm=confirm)
    volume.commit();return result


@app.local_entrypoint()
def imitation_record(run_id: str,seeds: int=3,arms: str='',confirm: bool=False):
    import json
    from .imitation import ARMS
    output=PROJECT/'acrobot_neural'/'artifacts'/run_id
    selected=arms.split(',') if arms else ARMS
    if any(arm not in ARMS for arm in selected):raise ValueError('Invalid arm')
    for result in imitation_record_job.starmap([(run_id,a,s,confirm) for a in selected for s in range(seeds)],order_outputs=False):
        (output/f"{result['arm']}-{result['seed']}.json").write_text(json.dumps(result,indent=2))
        print('RECORDED',result['arm'],result['seed'],flush=True)


@app.function(image=image,cpu=2,memory=4096,timeout=7200,max_containers=18,volumes={'/data':volume})
def imitation_extend_job(run_id: str,seed: int,trim: bool,rounds: int=48):
    import re,json
    from pathlib import Path
    from .imitation import run
    from .train import SOURCE
    if not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):raise ValueError('Invalid run')
    volume.reload()
    parent=Path('/data/neural-imitation/20261011T172359-d6f517e1/corrective-0')
    original=json.loads((parent/'results.json').read_text())
    assert original['status']=='complete'
    directory=Path('/data/neural-imitation')/run_id/f'corrective-{seed}';directory.mkdir(parents=True,exist_ok=True)
    return run(directory,Path('/data/swingup')/SOURCE,parent,'corrective',seed,volume.commit,
               rounds=rounds,checkpoint_name=original['selected_stage'],initial_back=6.,trim_corrections=trim)


@app.local_entrypoint()
def imitation_extend(rounds: int=48):
    import json
    from datetime import datetime,timezone
    from uuid import uuid4
    tasks=[]
    for trim in [False,True]:
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
        print('EXTENSION',run_id,'trim',trim,flush=True)
        output=PROJECT/'acrobot_neural'/'artifacts'/run_id;output.mkdir(parents=True,exist_ok=True)
        tasks.extend((run_id,seed,trim,rounds) for seed in range(3))
    # Each result identifies its parent run through a wrapper return.
    calls=[(run_id,imitation_extend_job.spawn(run_id,seed,trim,rounds)) for run_id,seed,trim,rounds in tasks]
    for run_id,call in calls:
        result=call.get()
        output=PROJECT/'acrobot_neural'/'artifacts'/run_id
        (output/f"corrective-{result['seed']}.json").write_text(json.dumps(result,indent=2))
        print('EXTENSION COMPLETE',run_id,result['seed'],flush=True)


@app.function(image=image,cpu=2,memory=4096,timeout=1200,volumes={'/data':volume})
def imitation_diagnose_job(run_id: str):
    import re,json
    from pathlib import Path
    from .imitation_diagnostics import diagnose
    from .train import SOURCE
    if not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):raise ValueError('Invalid run')
    volume.reload()
    result=diagnose(Path('/data/neural-imitation')/run_id,Path('/data/swingup')/SOURCE)
    output=Path('/data/neural-imitation-diagnostics');output.mkdir(exist_ok=True)
    (output/f'{run_id}.json').write_text(json.dumps(result,indent=2));volume.commit();return result


@app.local_entrypoint()
def imitation_diagnose(run_id: str):
    import json
    result=imitation_diagnose_job.remote(run_id)
    (PROJECT/'acrobot_neural'/'artifacts'/run_id/'diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


@app.function(image=image,cpu=4,memory=8192,timeout=7200,max_containers=6,volumes={'/data':volume})
def imitation_ppo_job(seed: int,updates: int=40):
    from datetime import datetime,timezone
    from uuid import uuid4
    from pathlib import Path
    from .curriculum import run
    from .audit import audit
    from .train import SOURCE
    volume.reload()
    parent=Path('/data/neural-imitation/20261011T173237-8df2d72c/corrective-0')
    assert (parent/'round-18.pt').is_file()
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory=Path('/data/neural')/run_id;directory.mkdir(parents=True)
    print('IMITATION PPO',run_id,'seed',seed,flush=True)
    run(directory,Path('/data/swingup')/SOURCE,parent,volume.commit,updates=updates,
        seed=seed,start_back=8.,initial_checkpoint='round-18',audit_seed=176000+seed)
    result=audit(directory,Path('/data/swingup')/SOURCE);volume.commit();return result


@app.local_entrypoint()
def imitation_ppo(updates: int=40):
    import json
    for result in imitation_ppo_job.map(range(3),kwargs={'updates':updates},order_outputs=False):
        output=PROJECT/'acrobot_neural'/'artifacts'/result['id'];output.mkdir(parents=True,exist_ok=True)
        (output/'results.json').write_text(json.dumps(result,indent=2))
        print('PPO FOLLOW-UP COMPLETE',result['id'],flush=True)


@app.function(image=image,cpu=2,memory=4096,timeout=1200,max_containers=6,volumes={'/data':volume})
def imitation_ppo_retention_job(run_id: str):
    import re
    from pathlib import Path
    from .ppo_retention import audit_earlier
    from .train import SOURCE
    if not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):raise ValueError('Invalid run')
    volume.reload();result=audit_earlier(Path('/data/neural')/run_id,Path('/data/swingup')/SOURCE)
    volume.commit();return result


@app.local_entrypoint()
def imitation_ppo_retention(run_ids: str):
    import json
    for result in imitation_ppo_retention_job.map(run_ids.split(','),order_outputs=False):
        (PROJECT/'acrobot_neural'/'artifacts'/result['id']/'results.json').write_text(json.dumps(result,indent=2))
        print('RETENTION AUDITED',result['id'],flush=True)


@app.function(image=image,cpu=2,memory=4096,timeout=1800,volumes={'/data':volume})
def reward_calibration_job():
    from pathlib import Path
    from .reward_sweep import calibrate
    from .train import SOURCE
    volume.reload()
    return calibrate(Path('/data/swingup')/SOURCE,Path('/data/neural/20261011T174348-20a2d60f'))


@app.function(image=image,cpu=4,memory=8192,timeout=7200,max_containers=12,volumes={'/data':volume})
def reward_job(kind: str,seed: int,calibration: dict,updates: int=40):
    from datetime import datetime,timezone
    from uuid import uuid4
    from pathlib import Path
    from .reward_sweep import run,finish
    from .train import SOURCE
    volume.reload()
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory=Path('/data/neural')/run_id;directory.mkdir(parents=True)
    print('REWARD RUN',kind,seed,run_id,flush=True)
    source=Path('/data/swingup')/SOURCE
    run(directory,source,Path('/data/neural/20261011T174348-20a2d60f'),volume.commit,
        updates=updates,seed=seed,audit_seed=221000+seed,reward_kind=kind,calibration=calibration)
    result=finish(directory,source);volume.commit();return result


@app.local_entrypoint()
def reward_sweep(updates: int=40):
    import json
    calibration=reward_calibration_job.remote();print('CALIBRATION',calibration,flush=True)
    calls=[reward_job.spawn(kind,seed,calibration,updates) for kind in ['baseline','braking','capture'] for seed in range(3)]
    results=[]
    for call in calls:
        result=call.get();results.append(result)
        output=PROJECT/'acrobot_neural'/'artifacts'/result['id'];output.mkdir(parents=True,exist_ok=True)
        (output/'results.json').write_text(json.dumps(result,indent=2))
        print('REWARD COMPLETE',result['reward_kind'],result['config']['seed'],result['id'],flush=True)
    (PROJECT/'acrobot_neural'/'artifacts'/'reward-summary.json').write_text(json.dumps(results,indent=2))


@app.function(image=image,cpu=2,memory=4096,timeout=1800,max_containers=12,volumes={'/data':volume})
def reward_final_record_job(run_id: str):
    import re
    from pathlib import Path
    from .reward_sweep import record_final
    from .train import SOURCE
    if not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):raise ValueError('Invalid run')
    volume.reload();result=record_final(Path('/data/neural')/run_id,Path('/data/swingup')/SOURCE)
    volume.commit();return result


@app.local_entrypoint()
def reward_record():
    import json
    summary=PROJECT/'acrobot_neural'/'artifacts'/'reward-summary.json'
    runs=json.loads(summary.read_text());results=[]
    for result in reward_final_record_job.map([r['id'] for r in runs],order_outputs=False):
        results.append(result)
        (PROJECT/'acrobot_neural'/'artifacts'/result['id']/'results.json').write_text(json.dumps(result,indent=2))
    summary.write_text(json.dumps(results,indent=2))


@app.function(image=image,cpu=2,memory=4096,timeout=3600,max_containers=24,volumes={'/data':volume})
def radial_job(trajectory: int,sigma: float,noise: float,seed: int=0):
    from pathlib import Path
    from datetime import datetime,timezone
    from uuid import uuid4
    from .radial import run
    from .train import SOURCE
    volume.reload();run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory=Path('/data/neural-radial')/run_id;directory.mkdir(parents=True)
    return run(directory,Path('/data/swingup')/SOURCE,trajectory,sigma,noise,seed,volume.commit)


@app.local_entrypoint()
def radial_sweep():
    import json
    tasks=[(trajectory,sigma,noise) for trajectory in range(3) for sigma in [.003,.01,.03,.1] for noise in [.001,.01]]
    results=[]
    for result in radial_job.starmap(tasks,order_outputs=False):
        results.append(result)
        output=PROJECT/'acrobot_neural'/'artifacts'/result['id'];output.mkdir(parents=True,exist_ok=True)
        (output/'results.json').write_text(json.dumps(result,indent=2))
    (PROJECT/'acrobot_neural'/'artifacts'/'radial-summary.json').write_text(json.dumps(results,indent=2))


@app.function(image=image,cpu=2,memory=4096,timeout=3600,max_containers=6,volumes={'/data':volume})
def radial_audit_job(run_id: str):
    import re
    from pathlib import Path
    from .radial import audit_export
    from .train import SOURCE
    if not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):raise ValueError('Invalid run')
    volume.reload();return audit_export(Path('/data/neural-radial')/run_id,Path('/data/swingup')/SOURCE,volume.commit)


@app.local_entrypoint()
def radial_audit():
    import json
    summary=PROJECT/'acrobot_neural'/'artifacts'/'radial-summary.json'
    runs=json.loads(summary.read_text());chosen=[r for r in runs if r['config']['sigma']==.03 and r['config']['noise']==.01]
    assert len(chosen)==3
    for result in radial_audit_job.map([r['id'] for r in chosen],order_outputs=False):
        (PROJECT/'acrobot_neural'/'artifacts'/result['id']/'results.json').write_text(json.dumps(result,indent=2))
        runs=[result if r['id']==result['id'] else r for r in runs]
    summary.write_text(json.dumps(runs,indent=2))


@app.function(image=image,cpu=2,memory=8192,timeout=3600,max_containers=6,volumes={'/data':volume})
def robust_collect_job(trajectory: int):
    from pathlib import Path
    from datetime import datetime,timezone
    from uuid import uuid4
    from .radial_robust import collect,PARENTS
    from .train import SOURCE
    volume.reload();run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory=Path('/data/neural-robust')/run_id;directory.mkdir(parents=True)
    return collect(directory,Path('/data/swingup')/SOURCE,Path('/data/neural-radial')/PARENTS[trajectory],trajectory,volume.commit)


@app.function(image=image,cpu=2,memory=8192,timeout=3600,max_containers=18,volumes={'/data':volume})
def robust_train_job(trajectory: int,collection: str,units: int,sigma: float):
    from pathlib import Path
    from datetime import datetime,timezone
    from uuid import uuid4
    from .radial_robust import expand,PARENTS
    from .train import SOURCE
    volume.reload();run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory=Path('/data/neural-robust')/run_id;directory.mkdir(parents=True)
    return expand(directory,Path('/data/swingup')/SOURCE,Path('/data/neural-radial')/PARENTS[trajectory],Path('/data/neural-robust')/collection,trajectory,units,sigma,volume.commit)


@app.local_entrypoint()
def robust_sweep():
    import json
    collections=list(robust_collect_job.map(range(3),order_outputs=False))
    def save(result):
        output=PROJECT/'acrobot_neural'/'artifacts'/result['id'];output.mkdir(parents=True,exist_ok=True)
        (output/'results.json').write_text(json.dumps(result,indent=2))
    for c in collections:save(c)
    tasks=[(c['trajectory'],c['id'],units,sigma) for c in collections for units in [512,2048] for sigma in [.01,.03]]
    results=[]
    for result in robust_train_job.starmap(tasks,order_outputs=False):results.append(result);save(result)
    (PROJECT/'acrobot_neural'/'artifacts'/'robust-summary.json').write_text(json.dumps(dict(collections=collections,runs=results),indent=2))


@app.function(image=image,cpu=2,memory=8192,timeout=3600,max_containers=12,volumes={'/data':volume})
def robust_audit_job(run_id: str,trajectory: int,baseline: bool=False):
    import re,json
    from pathlib import Path
    from .radial_robust import audit,PARENTS
    from .train import SOURCE
    if not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):raise ValueError('Invalid run')
    volume.reload();directory=Path('/data/neural-robust')/run_id
    if baseline:
        result=json.loads((directory/'results.json').read_text());result['variant']='original';(directory/'results.json').write_text(json.dumps(result,indent=2))
        policy=Path('/data/neural-radial')/PARENTS[trajectory]/'deployed.pt'
    else:policy=directory/'deployed.pt'
    return audit(directory,Path('/data/swingup')/SOURCE,policy,trajectory,volume.commit)


@app.local_entrypoint()
def robust_audit():
    import json
    summary=PROJECT/'acrobot_neural'/'artifacts'/'robust-summary.json';data=json.loads(summary.read_text())
    def score(r):return (r['exact_ground']['exact_ground_success'] and r['balance']['final_10s_rate']>=.9,sum(x['successes'] for x in r['development']))
    chosen=[max([r for r in data['runs'] if r['trajectory']==t],key=score) for t in range(3)]
    tasks=[(c['id'],c['trajectory'],True) for c in data['collections']]+[(r['id'],r['trajectory'],False) for r in chosen]
    audited=[]
    for result in robust_audit_job.starmap(tasks,order_outputs=False):
        audited.append(result);(PROJECT/'acrobot_neural'/'artifacts'/result['id']/'results.json').write_text(json.dumps(result,indent=2))
    data['audited']=audited;summary.write_text(json.dumps(data,indent=2))


@app.function(image=image,cpu=2,memory=8192,timeout=3600,max_containers=6,volumes={'/data':volume})
def top_recovery_job(trajectory: int):
    from pathlib import Path
    from datetime import datetime,timezone
    from uuid import uuid4
    from .top_recovery import run
    from .radial_robust import PARENTS
    volume.reload();run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    directory=Path('/data/neural-top')/run_id;directory.mkdir(parents=True)
    return run(directory,Path('/data/neural-radial')/PARENTS[trajectory],trajectory,volume.commit)


@app.local_entrypoint()
def top_recovery():
    import json
    results=[]
    for result in top_recovery_job.map(range(3),order_outputs=False):
        results.append(result);output=PROJECT/'acrobot_neural'/'artifacts'/result['id'];output.mkdir(parents=True,exist_ok=True)
        (output/'results.json').write_text(json.dumps(result,indent=2))
    (PROJECT/'acrobot_neural'/'artifacts'/'top-summary.json').write_text(json.dumps(results,indent=2))


browser_image=(modal.Image.debian_slim(python_version='3.14').apt_install('nodejs')
    .uv_sync(uv_project_dir=str(PROJECT),extras=['acrobot','feasibility'])
    .add_local_file(PROJECT/'modal_resources.py','/root/modal_resources.py'))
for package in ['acrobot_neural','acrobot_swingup','acrobot_feasibility','acrobot_comparison']:
    browser_image=browser_image.add_local_dir(PROJECT/package,f'/root/{package}',ignore=['__pycache__','artifacts'])


@app.function(image=browser_image,cpu=2,memory=4096,timeout=1200,volumes={'/data':volume})
def browser_verify_job():
    import json,subprocess
    from pathlib import Path
    import numpy as np
    from .radial import ExportedActor
    from .radial_robust import PARENTS
    from acrobot_comparison.env import advance_torque
    import torch
    torch.set_num_threads(1);volume.reload()
    actor=ExportedActor(Path('/data/neural-radial')/PARENTS[1]/'deployed.pt')
    rng=np.random.default_rng(391000)
    states=np.vstack([rng.uniform([-np.pi,-np.pi,-4,-8],[np.pi,np.pi,4,8],(1000,4)),np.array([np.pi,0,0,0])+rng.normal(size=(1000,4))*.05])
    torques=rng.uniform(-1,1,len(states));samples=Path('/tmp/browser-samples.json')
    samples.write_text(json.dumps(dict(states=states.tolist(),torques=torques.tolist())))
    root=Path('/root/acrobot_neural/browser')
    result=json.loads(subprocess.check_output(['node',str(root/'verify.js'),str(root/'policy.json'),str(samples)],text=True))
    error=np.abs(np.array(result.pop('torques'))-actor.act(states))
    physics_error=np.max(np.abs(np.array(result.pop('transitions'))-advance_torque(states,torques,.0025)))
    result.update(samples=len(states),max_torque_error=float(error.max()),p99_torque_error=float(np.quantile(error,.99)),max_physics_error=float(physics_error),runtime='Node JavaScript on Modal, same core used in browser')
    assert physics_error<1e-10,result
    assert error.max()<.01,result
    assert all(r['success'] for r in result['evaluations']),result
    assert next(r for r in result['evaluations'] if r['name']=='kick4')['preKick'],result
    output=Path('/data/browser-verification');output.mkdir(exist_ok=True)
    (output/'results.json').write_text(json.dumps(result,indent=2));volume.commit();return result


@app.local_entrypoint()
def browser_verify():
    import json
    result=browser_verify_job.remote()
    (PROJECT/'acrobot_neural'/'artifacts'/'browser-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
