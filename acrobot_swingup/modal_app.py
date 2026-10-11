"""Swing-up experiment orchestration; all simulation and fitting run on Modal."""
import modal
from modal_resources import PROJECT, project_image, volume

app = modal.App("robot-acrobot-swingup")
image = project_image(extras=["acrobot", "feasibility"])
for package in ("acrobot_swingup", "acrobot_comparison", "acrobot_feasibility"):
    image = image.add_local_dir(PROJECT / package, f"/root/{package}", ignore=["__pycache__", "artifacts"])


@app.function(image=image, cpu=2, memory=2048, timeout=1200, volumes={"/data": volume})
def search_controller(limit: float = 1.0):
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    from uuid import uuid4
    from acrobot_swingup.controller import candidate_configs, rollout

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    directory = Path("/data/swingup") / run_id
    directory.mkdir(parents=True)
    candidates, _ = rollout(candidate_configs(limit), count=8, seconds=60)
    candidates.sort(key=lambda r: (r["final_5s_rate"], r["mean_longest_hold"], r["mean_max_height"]), reverse=True)
    chosen = candidates[0]["config"]
    validation, _ = rollout([chosen], seed=42000, count=64, seconds=60, physics_dt=0.005)
    result = {"id": run_id, "kind": "controller_search", "limit": limit, "candidates": candidates,
              "chosen": chosen, "validation": validation[0]}
    (directory / "results.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print("Chosen controller:", chosen, "Validation:", validation[0], flush=True)
    return result


@app.local_entrypoint()
def search(limit: float = 1.0):
    import json
    result = search_controller.remote(limit)
    output = PROJECT / "acrobot_swingup" / "artifacts" / result["id"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(f"Saved controller search {result['id']}")


@app.function(image=image, cpu=2, memory=2048, timeout=1200, volumes={"/data": volume})
def record_controller(run_id: str):
    import json
    import re
    from pathlib import Path
    from acrobot_swingup.record import record

    if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", run_id):
        raise ValueError("Invalid run ID")
    volume.reload()
    directory = Path("/data/swingup") / run_id
    result = json.loads((directory / "results.json").read_text())
    if not result.get("recordings"):
        result["recordings"] = record(directory, result["chosen"])
        (directory / "results.json").write_text(json.dumps(result, indent=2))
        volume.commit()
    return result


@app.local_entrypoint()
def videos(run_id: str):
    import json
    result = record_controller.remote(run_id)
    output = PROJECT / "acrobot_swingup" / "artifacts" / run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(f"Saved {len(result['recordings'])} recordings for {run_id}")


@app.function(image=image, cpu=2, memory=4096, timeout=3600, volumes={"/data": volume})
def trajectory_search(seed: int, run_id: str, limit: float = 1.):
    import json
    from pathlib import Path
    from acrobot_swingup.trajectory import search
    directory = Path('/data/swingup') / run_id
    directory.mkdir(parents=True, exist_ok=True)
    result = search(seed, directory, limit)
    (directory / f'search-{seed}.json').write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def trajectories(limit: float = 1., seeds: int = 3):
    import json
    from datetime import datetime, timezone
    from uuid import uuid4
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
    print('Trajectory run:', run_id, flush=True)
    results = list(trajectory_search.starmap([(seed,run_id,limit) for seed in range(seeds)]))
    output=PROJECT/'acrobot_swingup'/'artifacts'/run_id
    output.mkdir(parents=True,exist_ok=True)
    (output/'searches.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results), flush=True)


@app.function(image=image, cpu=2, memory=4096, timeout=1800, volumes={"/data": volume})
def validate_trajectories(run_id: str):
    import re
    from pathlib import Path
    from acrobot_swingup.trajectory_report import publish
    if not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):
        raise ValueError('Invalid run ID')
    volume.reload()
    result=publish(Path('/data/swingup')/run_id)
    volume.commit()
    return result


@app.local_entrypoint()
def publish_trajectories(run_id: str):
    import json
    result=validate_trajectories.remote(run_id)
    output=PROJECT/'acrobot_swingup'/'artifacts'/run_id
    output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(result,indent=2))
    print('Published',run_id,'paths:',len(result['paths']))


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={"/data": volume})
def optimal_search(seed: int, run_id: str, limit: float = 1.):
    import json
    from pathlib import Path
    from acrobot_swingup.optimal import search
    directory=Path('/data/swingup')/run_id
    directory.mkdir(parents=True,exist_ok=True)
    result=search(seed,directory,limit)
    (directory/f'search-{seed}.json').write_text(json.dumps(result,indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def optimize(limit: float = 1., seeds: int = 3):
    import json
    from datetime import datetime,timezone
    from uuid import uuid4
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
    print('Optimal trajectory run:',run_id,flush=True)
    results=list(optimal_search.starmap([(seed,run_id,limit) for seed in range(seeds)]))
    output=PROJECT/'acrobot_swingup'/'artifacts'/run_id
    output.mkdir(parents=True,exist_ok=True)
    (output/'searches.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results),flush=True)


@app.function(image=image, cpu=2, memory=4096, timeout=900, volumes={"/data": volume})
def capture_region(run_id: str):
    import json
    import re
    from pathlib import Path
    from acrobot_swingup.capture import map_capture
    if not re.fullmatch(r'\d{8}T\d{6}-[a-f0-9]{8}',run_id):
        raise ValueError('Invalid run ID')
    result=map_capture()
    directory=Path('/data/swingup')/run_id
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'capture.json').write_text(json.dumps(result,indent=2))
    volume.commit()
    print('Capture level',result['empirical_level'],'independent successes',result['independent_successes'],
          'intercepts',result['coordinate_intercepts'],flush=True)
    return result


@app.local_entrypoint()
def capture(run_id: str):
    import json
    result=capture_region.remote(run_id)
    output=PROJECT/'acrobot_swingup'/'artifacts'/run_id
    output.mkdir(parents=True,exist_ok=True)
    (output/'capture.json').write_text(json.dumps(result,indent=2))
