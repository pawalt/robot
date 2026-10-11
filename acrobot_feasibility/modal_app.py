"""Feasibility study jobs; all simulation and recording execute on Modal."""
import modal
from modal_resources import PROJECT, project_image, volume

app = modal.App("robot-acrobot-feasibility")
image = (project_image(extras=["feasibility"])
         .add_local_dir(PROJECT / "acrobot_feasibility", "/root/acrobot_feasibility", ignore=["__pycache__", "artifacts"])
         .add_local_dir(PROJECT / "acrobot_comparison", "/root/acrobot_comparison", ignore=["__pycache__", "artifacts"]))


@app.function(image=image, cpu=2, memory=2048, timeout=900, max_containers=5)
def feasibility_interval(dt: float):
    from acrobot_feasibility.study import run_interval
    return run_interval(dt)


@app.function(image=image, cpu=2, memory=2048, timeout=900, volumes={"/data": volume})
def save_feasibility(results):
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from uuid import uuid4
    from acrobot_feasibility.record import record
    from acrobot_feasibility.study import simulate, COUNT, SEED, PHYSICS_DT, DURATION

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    directory = Path("/data/feasibility") / run_id
    directory.mkdir(parents=True)
    # Independently halve the physics integration step at the original control rate.
    checks = [{"controller": c, "limit": 1.0, "scale": s}
              for c in ("continuous", "discrete3") for s in (0.05, 1.0)]
    validation = simulate(0.05, checks, physics_dt=PHYSICS_DT / 2)[0]
    result = {"id": run_id, "duration": DURATION, "seed": SEED, "episodes_per_condition": COUNT,
              "physics_dt": PHYSICS_DT, "intervals": results, "integration_validation": validation,
              "recordings": record(directory)}
    (directory / "results.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def study():
    import json
    results = list(feasibility_interval.map([0.01, 0.02, 0.05, 0.1, 0.2]))
    result = save_feasibility.remote(results)
    output = PROJECT / "acrobot_feasibility" / "artifacts" / result["id"]
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(f"Feasibility study saved: {result['id']}")
