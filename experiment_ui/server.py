"""Read-only dashboard. Video files are cached off-volume for safe range requests."""

import json
import re
import shutil
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse


def create_dashboard(data_root, refresh=lambda: None, cache_root="/tmp/robot-video-cache"):
    data_root, cache = Path(data_root), Path(cache_root)
    root = data_root / "runs"
    lock = Lock()
    app = FastAPI(title="Robot · Learning lab", docs_url=None, redoc_url=None)

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).with_name("directory.html"))

    @app.get("/experiments/grid-reaching")
    def grid_page():
        return FileResponse(Path(__file__).with_name("grid_reaching.html"))

    @app.get("/api/experiments")
    def experiment_directory():
        with lock:
            refresh()
            grid_runs = list(root.glob("*/run.json"))
            acrobot_runs = list((data_root / "acrobot").glob("*/*/result.json"))
            recorded = sum(bool(json.loads(p.read_text()).get("recordings")) for p in acrobot_runs)
            feasibility_runs = list((data_root / "feasibility").glob("*/results.json"))
            swingup_runs = list((data_root / "swingup").glob("*/results.json"))
            sweep_runs = list((data_root / "neural-sweeps").glob("*"))
            neural_runs = list((data_root / "neural").glob("*/results.json"))
            imitation_runs = list((data_root / "neural-imitation").glob("*"))
            return [
                {"id": "playground", "title": "Interactive Acrobot playground", "url": "/experiments/acrobot-play", "runs": 1, "status": "Play in your browser"},
                {"id": "top-recovery", "title": "Recovery from the top", "url": "/experiments/acrobot-top", "runs": len(list((data_root / "neural-top").glob("*/results.json"))), "status": "Top releases and kicks"},
                {"id": "disturbance", "title": "Larger starting disturbances", "url": "/experiments/acrobot-disturbance", "runs": len(list((data_root / "neural-robust").glob("*/results.json"))), "status": "Recovery comparison"},
                {"id": "radial", "title": "Neural ground-to-balance", "url": "/experiments/acrobot-radial", "runs": len(list((data_root / "neural-radial").glob("*/results.json"))), "status": "Neural imitation"},
                {"id": "rewards", "title": "PPO reward comparison", "url": "/experiments/acrobot-rewards", "runs": sum(bool(json.loads(p.read_text()).get("reward_sweep")) for p in neural_runs), "status": "Reward comparison"},
                {"id": "neural-imitation", "title": "Learning the full swing-up", "url": "/experiments/acrobot-imitation",
                 "runs": len(imitation_runs), "status": "Results available" if imitation_runs else "Ready"},
                {"id": "grid-reaching", "title": "Grid reaching", "url": "/experiments/grid-reaching",
                 "runs": len(grid_runs), "status": "Recorded" if grid_runs else "Ready to run"},
                {"id": "acrobot", "title": "Acrobot comparison", "url": "/experiments/acrobot",
                 "runs": len(acrobot_runs), "status": "Training / recording" if recorded < len(acrobot_runs) else "Recorded" if acrobot_runs else "Ready to run"},
                {"id": "neural-sweep", "title": "Neural architecture and forgetting comparison", "url": "/experiments/acrobot-neural-sweep",
                 "runs": len(sweep_runs), "status": "Results available" if sweep_runs else "Ready"},
                {"id": "neural", "title": "Acrobot neural controller", "url": "/experiments/acrobot-neural",
                 "runs": len(neural_runs), "status": "Results available" if neural_runs else "Training"},
                {"id": "swingup", "title": "Acrobot swing-up", "url": "/experiments/acrobot-swingup",
                 "runs": len(swingup_runs), "status": "Results available" if swingup_runs else "Ready to run"},
                {"id": "feasibility", "title": "Acrobot feasibility", "url": "/experiments/acrobot-feasibility",
                 "runs": len(feasibility_runs), "status": "Recorded" if feasibility_runs else "Ready to run"},
            ]

    @app.get("/experiments/acrobot-feasibility")
    def feasibility_page():
        return FileResponse(Path(__file__).with_name("feasibility.html"))

    @app.get("/api/feasibility")
    def feasibility_runs():
        with lock:
            refresh()
            return [json.loads(p.read_text()) for p in sorted((data_root / "feasibility").glob("*/results.json"), reverse=True)]

    @app.get("/feasibility-media/{run_id}/{filename}")
    def feasibility_media(run_id: str, filename: str):
        if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", run_id):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:continuous|discrete3|discrete9|passive)-\d+\.(?:mp4|jpg)|results\.json", filename):
            raise HTTPException(404)
        target = cache / "feasibility" / run_id / filename
        with lock:
            if not target.exists():
                refresh()
                source = data_root / "feasibility" / run_id / filename
                if not source.is_file():
                    raise HTTPException(404)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/experiments/acrobot-swingup")
    def swingup_page():
        return FileResponse(Path(__file__).with_name("swingup.html"))

    @app.get("/api/swingup")
    def swingup_runs():
        with lock:
            refresh()
            return [json.loads(p.read_text()) for p in sorted((data_root / "swingup").glob("*/results.json"), reverse=True)]

    @app.get("/swingup-media/{run_id}/{filename}")
    def swingup_media(run_id: str, filename: str):
        if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", run_id):
            raise HTTPException(404)
        if not re.fullmatch(r"swingup-\d+\.(?:mp4|jpg)|(?:optimal|attempt)-\d+-\d+\.npz|controller-\d+\.npz|results\.json", filename):
            raise HTTPException(404)
        target = cache / "swingup" / run_id / filename
        with lock:
            if not target.exists():
                refresh()
                source = data_root / "swingup" / run_id / filename
                if not source.is_file():
                    raise HTTPException(404)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/experiments/acrobot-play")
    def playground_page():
        return FileResponse(Path(__file__).with_name("playground") / "index.html")

    @app.get("/playground/{filename}")
    def playground_asset(filename: str):
        if filename == "app.js":
            target = Path(__file__).with_name("playground") / filename
        elif filename in {"core.js", "policy.json"}:
            target = Path(__file__).resolve().parent.parent / "acrobot_neural" / "browser" / filename
        else:
            raise HTTPException(404)
        return FileResponse(target, headers={"Cache-Control": "no-cache"})

    @app.get("/experiments/acrobot-top")
    def top_page():
        return FileResponse(Path(__file__).with_name("top_recovery.html"))

    @app.get("/api/neural-top")
    def top_runs():
        with lock:
            refresh()
            return [json.loads(p.read_text()) for p in sorted((data_root / "neural-top").glob("*/results.json"))]

    @app.get("/top-media/{run_id}/{filename}")
    def top_media(run_id: str, filename: str):
        if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", run_id):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:release_[pm](?:15|60)|free_fall|kick_4)-(?:exact|success|failure)\.(?:mp4|jpg)", filename):
            raise HTTPException(404)
        target = cache / "neural-top" / run_id / filename
        with lock:
            if not target.exists():
                refresh()
                source = data_root / "neural-top" / run_id / filename
                if not source.is_file():
                    raise HTTPException(404)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/experiments/acrobot-disturbance")
    def disturbance_page():
        return FileResponse(Path(__file__).with_name("disturbance.html"))

    @app.get("/api/neural-robust")
    def robust_runs():
        with lock:
            refresh()
            return [json.loads(p.read_text()) for p in sorted((data_root / "neural-robust").glob("*/results.json"))]

    @app.get("/robust-media/{run_id}/{filename}")
    def robust_media(run_id: str, filename: str):
        if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", run_id):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:large|very_large|velocity_4)-(?:first|success|failure)\.(?:mp4|jpg)|deployed\.pt", filename):
            raise HTTPException(404)
        target = cache / "neural-robust" / run_id / filename
        with lock:
            if not target.exists():
                refresh()
                source = data_root / "neural-robust" / run_id / filename
                if not source.is_file():
                    raise HTTPException(404)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/experiments/acrobot-radial")
    def radial_page():
        return FileResponse(Path(__file__).with_name("radial.html"))

    @app.get("/api/neural-radial")
    def radial_runs():
        with lock:
            refresh()
            return [json.loads(p.read_text()) for p in sorted((data_root / "neural-radial").glob("*/results.json"))]

    @app.get("/radial-media/{run_id}/{filename}")
    def radial_media(run_id: str, filename: str):
        if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", run_id):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:exact_ground|wider_ground)\.(?:mp4|jpg)|(?:policy|deployed)\.pt", filename):
            raise HTTPException(404)
        target = cache / "neural-radial" / run_id / filename
        with lock:
            if not target.exists():
                refresh()
                source = data_root / "neural-radial" / run_id / filename
                if not source.is_file():
                    raise HTTPException(404)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/experiments/acrobot-rewards")
    def rewards_page():
        return FileResponse(Path(__file__).with_name("rewards.html"))

    @app.get("/experiments/acrobot-neural")
    def neural_page():
        return FileResponse(Path(__file__).with_name("neural.html"))

    @app.get("/api/neural")
    def neural_runs():
        with lock:
            refresh()
            return [json.loads(p.read_text()) for p in sorted((data_root / "neural").glob("*/results.json"), reverse=True)]

    @app.get("/neural-media/{run_id}/{filename}")
    def neural_media(run_id: str, filename: str):
        if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", run_id):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:balance|imitation|ppo-\d+)(?:-(?:balance|bottom(?:-success)?|kick_0\.1|kick_0\.5|kick_1\.0|suffix_\d+(?:\.\d+)?s)\.(?:mp4|jpg)|\.pt)|results\.json", filename):
            raise HTTPException(404)
        target = cache / "neural" / run_id / filename
        with lock:
            if not target.exists():
                refresh()
                source = data_root / "neural" / run_id / filename
                if not source.is_file():
                    raise HTTPException(404)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/experiments/acrobot-neural-sweep")
    def neural_sweep_page():
        return FileResponse(Path(__file__).with_name("neural_sweep.html"))

    @app.get("/api/neural-sweeps")
    def neural_sweeps():
        with lock:
            refresh()
            return [{"id": directory.name, "jobs": [json.loads(p.read_text()) for p in sorted(directory.glob("*/results.json"))]}
                    for directory in sorted((data_root / "neural-sweeps").glob("*"), reverse=True) if directory.is_dir()]

    @app.get("/neural-sweep-media/{run_id}/{job}/{filename}")
    def neural_sweep_media(run_id: str, job: str, filename: str):
        if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", run_id):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:unprotected|rehearsal|distill|ewc|project|rollback|experts|wide256|wide512|residual256)-\d+", job):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:balance|suffix-2|bottom)\.(?:mp4|jpg)|(?:initial|bc-\d+|ppo-\d+|bc-retained|final-retained)\.pt", filename):
            raise HTTPException(404)
        target = cache / "neural-sweeps" / run_id / job / filename
        with lock:
            if not target.exists():
                refresh()
                source = data_root / "neural-sweeps" / run_id / job / filename
                if not source.is_file():
                    raise HTTPException(404)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/experiments/acrobot-imitation")
    def imitation_page():
        return FileResponse(Path(__file__).with_name("imitation.html"))

    @app.get("/api/neural-imitation")
    def imitation_runs():
        with lock:
            refresh()
            return [{"id": directory.name, "jobs": [json.loads(p.read_text()) for p in sorted(directory.glob("*/results.json"))]}
                    for directory in sorted((data_root / "neural-imitation").glob("*"), reverse=True) if directory.is_dir()]

    @app.get("/imitation-media/{run_id}/{job}/{filename}")
    def imitation_media(run_id: str, job: str, filename: str):
        if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", run_id):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:full|backward|corrective)-\d+", job):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:selected|final)-(?:balance|bottom|approach)\.(?:mp4|jpg)|(?:initial|round-\d+|final)\.pt", filename):
            raise HTTPException(404)
        target = cache / "neural-imitation" / run_id / job / filename
        with lock:
            if not target.exists():
                refresh()
                source = data_root / "neural-imitation" / run_id / job / filename
                if not source.is_file():
                    raise HTTPException(404)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/api/runs")
    def runs():
        with lock:
            refresh()
            manifests = sorted(root.glob("*/run.json"), reverse=True)
            return [json.loads(path.read_text()) for path in manifests]

    @app.get("/acrobot")
    @app.get("/experiments/acrobot")
    def acrobot_page():
        return FileResponse(Path(__file__).with_name("acrobot.html"))

    @app.get("/api/acrobot")
    def acrobot_runs():
        with lock:
            refresh()
            experiments = []
            for directory in sorted((data_root / "acrobot").glob("*"), reverse=True):
                results = [json.loads(path.read_text()) for path in sorted(directory.glob("*/result.json"))]
                if results:
                    summary = directory / "summary.json"
                    experiments.append({"id": directory.name, "runs": results,
                                        "audit": json.loads(summary.read_text()) if summary.is_file() else None,
                                        "complete": len(results) == 6 and all(r.get("recordings") for r in results)})
            return experiments

    @app.get("/acrobot-media/{experiment}/{run_id}/{filename}")
    def acrobot_media(experiment: str, run_id: str, filename: str):
        if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", experiment):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:ppo|grouped)-[0-2]", run_id):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:balance|swingup)-\d+\.(?:mp4|jpg)|policy-\d+\.pt", filename):
            raise HTTPException(404)
        target = cache / "acrobot" / experiment / run_id / filename
        with lock:
            if not target.exists():
                refresh()
                source = data_root / "acrobot" / experiment / run_id / filename
                if not source.is_file():
                    raise HTTPException(404)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/media/{run_id}/{filename}")
    def media(run_id: str, filename: str):
        if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", run_id):
            raise HTTPException(404)
        if not re.fullmatch(r"(?:learned|random)-\d+\.(?:mp4|jpg)|policy\.(?:npz|txt)|training\.csv|run\.json", filename):
            raise HTTPException(404)
        target = cache / run_id / filename
        with lock:
            if not target.exists():
                refresh()
                source = root / run_id / filename
                if not (root / run_id / "run.json").is_file() or not source.is_file():
                    raise HTTPException(404)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        # Starlette supports HTTP Range, so browser seek works without full downloads.
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})

    return app
