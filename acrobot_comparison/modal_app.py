"""Acrobot training, evaluation, recording, and experiment commands."""
import modal
from modal_resources import PROJECT, project_image, volume

app = modal.App("robot-acrobot-experiment")
comparison_image = project_image(extras=["acrobot"]).add_local_dir(
    PROJECT / "acrobot_comparison", remote_path="/root/acrobot_comparison",
    ignore=["__pycache__", "artifacts"],
)


@app.function(image=comparison_image, volumes={"/data": volume}, timeout=3600,
              cpu=2, memory=2048, max_containers=6)
def compare_train(experiment: str, method: str, seed: int, updates: int = 80):
    import json
    from pathlib import Path
    from acrobot_comparison.train import run_experiment
    from acrobot_comparison.record import record_checkpoints

    directory = Path("/data/acrobot") / experiment / f"{method}-{seed}"
    result = run_experiment(method, seed, directory, updates=updates,
                            report=lambda result: volume.commit())
    result["recordings"] = record_checkpoints(directory, result["config"])
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def compare(updates: int = 80):
    from datetime import datetime, timezone
    from uuid import uuid4
    import json

    experiment = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    tasks = [(experiment, method, seed, updates) for method in ("ppo", "grouped") for seed in range(3)]
    results = list(compare_train.starmap(tasks))
    directory = PROJECT / "acrobot_comparison" / "artifacts" / experiment
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Comparison complete: {experiment}. Results saved to {directory}")


@app.function(image=comparison_image, volumes={"/data": volume}, timeout=600, cpu=2, memory=2048)
def audit_comparison(experiment: str):
    import json
    import torch
    from pathlib import Path
    from acrobot_comparison.train import network, evaluate

    torch.set_num_threads(1)
    volume.reload()
    root = Path("/data/acrobot") / experiment
    results = []
    for path in sorted(root.glob("*/result.json")):
        result = json.loads(path.read_text())
        config = result["config"]
        policy = network(3)
        policy.load_state_dict(torch.load(path.parent / f"policy-{config['updates']}.pt", weights_only=True))
        result["heldout"] = {"seed": 20000, "episodes_per_condition": 128}
        for mode in ("balance", "swingup"):
            for stochastic in (False, True):
                label = mode + ("_sampled" if stochastic else "")
                result["heldout"][label] = evaluate(policy, seed=20000, count=128,
                    horizon=config["horizon"], mode=mode, stochastic=stochastic)[0]
        path.write_text(json.dumps(result, indent=2))
        results.append(result)
    # A passive baseline makes it clear when a learned policy merely coasts.
    passive = network(3)
    with torch.no_grad():
        for parameter in passive.parameters():
            parameter.zero_()
        passive[-1].bias[1] = 100
    baseline = {mode: evaluate(passive, seed=20000, count=128, mode=mode)[0]
                for mode in ("balance", "swingup")}
    (root / "summary.json").write_text(json.dumps({"passive_baseline": baseline}, indent=2))
    volume.commit()
    return {"id": experiment, "runs": results, "passive_baseline": baseline}


@app.local_entrypoint()
def audit(experiment: str):
    import json
    result = audit_comparison.remote(experiment)
    directory = PROJECT / "acrobot_comparison" / "artifacts" / experiment
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "audit.json").write_text(json.dumps(result, indent=2))
    print(f"Saved held-out evaluation to {directory / 'audit.json'}")


