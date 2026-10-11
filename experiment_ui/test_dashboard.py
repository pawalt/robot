import json
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from fastapi.testclient import TestClient

from .server import create_dashboard
from grid_reaching.env import GridReachingEnv
from grid_reaching.record import record_episode


class DashboardTests(unittest.TestCase):
    def test_top_recovery_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20261011T120000-1234abcd"
            directory = root / "neural-top" / run_id
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(json.dumps({"id": run_id}))
            (directory / "release_p15-exact.mp4").write_bytes(bytes(range(64)))
            client = TestClient(create_dashboard(root, cache_root=root / "cache"))
            self.assertEqual(client.get("/experiments/acrobot-top").status_code, 200)
            self.assertEqual(client.get("/api/neural-top").json(), [{"id": run_id}])
            prefix = f"/top-media/{run_id}/"
            self.assertEqual(client.get(prefix + "release_p15-exact.mp4", headers={"Range": "bytes=0-15"}).status_code, 206)
            self.assertEqual(client.get(prefix + "results.json").status_code, 404)

    def test_disturbance_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20261011T120000-1234abcd"
            directory = root / "neural-robust" / run_id
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(json.dumps({"id": run_id}))
            (directory / "very_large-failure.mp4").write_bytes(bytes(range(64)))
            client = TestClient(create_dashboard(root, cache_root=root / "cache"))
            self.assertEqual(client.get("/experiments/acrobot-disturbance").status_code, 200)
            self.assertEqual(client.get("/api/neural-robust").json(), [{"id": run_id}])
            prefix = f"/robust-media/{run_id}/"
            self.assertEqual(client.get(prefix + "very_large-failure.mp4", headers={"Range": "bytes=0-15"}).status_code, 206)
            self.assertEqual(client.get(prefix + "demonstrations.npz").status_code, 404)

    def test_radial_policy_and_video_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20261011T120000-1234abcd"
            directory = root / "neural-radial" / run_id
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(json.dumps({"id": run_id}))
            (directory / "exact_ground.mp4").write_bytes(bytes(range(64)))
            (directory / "deployed.pt").write_bytes(b"test export")
            client = TestClient(create_dashboard(root, cache_root=root / "cache"))
            self.assertEqual(client.get("/experiments/acrobot-radial").status_code, 200)
            self.assertEqual(client.get("/api/neural-radial").json(), [{"id": run_id}])
            prefix = f"/radial-media/{run_id}/"
            self.assertEqual(client.get(prefix + "exact_ground.mp4", headers={"Range": "bytes=0-15"}).status_code, 206)
            self.assertEqual(client.get(prefix + "deployed.pt").content, b"test export")
            self.assertEqual(client.get(prefix + "results.json").status_code, 404)

    def test_imitation_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20261011T120000-1234abcd"
            directory = root / "neural-imitation" / run_id / "corrective-0"
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(json.dumps({"arm": "corrective"}))
            (directory / "selected-bottom.mp4").write_bytes(bytes(range(64)))
            client = TestClient(create_dashboard(root, cache_root=root / "cache"))
            self.assertEqual(client.get("/experiments/acrobot-imitation").status_code, 200)
            self.assertEqual(client.get("/api/neural-imitation").json()[0]["jobs"], [{"arm": "corrective"}])
            prefix = f"/imitation-media/{run_id}/corrective-0/"
            self.assertEqual(client.get(prefix + "selected-bottom.mp4", headers={"Range": "bytes=0-15"}).status_code, 206)
            self.assertEqual(client.get(prefix + "results.json").status_code, 404)

    def test_neural_sweep_results_and_media(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20261011T120000-1234abcd"
            directory = root / "neural-sweeps" / run_id / "distill-0"
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(json.dumps({"arm": "distill", "seed": 0}))
            (directory / "balance.mp4").write_bytes(bytes(range(64)))
            (directory / "final-retained.pt").write_bytes(b"checkpoint")
            client = TestClient(create_dashboard(root, cache_root=root / "cache"))
            self.assertEqual(client.get("/experiments/acrobot-neural-sweep").status_code, 200)
            self.assertEqual(client.get("/api/neural-sweeps").json()[0]["jobs"], [{"arm": "distill", "seed": 0}])
            url = f"/neural-sweep-media/{run_id}/distill-0/"
            self.assertEqual(client.get(url + "balance.mp4", headers={"Range": "bytes=0-15"}).status_code, 206)
            self.assertEqual(client.get(url + "final-retained.pt").content, b"checkpoint")
            self.assertEqual(client.get(url + "results.json").status_code, 404)
            self.assertEqual(client.get(url + "secret.env").status_code, 404)

    def test_recording_and_video_seeking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20261011T120000-1234abcd"
            run = root / "runs" / run_id
            run.mkdir(parents=True)
            env = GridReachingEnv(size=3)
            q = np.zeros((9, 4))
            q[:, 1] = 1  # One move right from state 7 reaches the goal.
            result = record_episode(env, q, 7, run / "learned-7.mp4")
            self.assertTrue(result["success"])
            self.assertEqual(result["steps"], 1)
            with imageio.get_reader(str(run / "learned-7.mp4")) as reader:
                self.assertEqual(reader.get_data(0).shape, (736, 640, 3))
                self.assertEqual(reader.count_frames(), 8)
            (run / "run.json").write_text(json.dumps({"id": run_id}))
            client = TestClient(create_dashboard(root, cache_root=root / "cache"))
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/api/runs").json(), [{"id": run_id}])
            response = client.get(f"/media/{run_id}/learned-7.mp4",
                                  headers={"Range": "bytes=0-31"})
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.headers["content-type"], "video/mp4")
            self.assertEqual(len(response.content), 32)
            self.assertEqual(client.get(f"/media/{run_id}/missing.mp4").status_code, 404)
            self.assertEqual(client.get("/media/bad/run.json").status_code, 404)
            self.assertEqual(client.get(f"/media/{run_id}/modal.env").status_code, 404)

    def test_empty_and_incomplete_runs_are_not_listed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "unfinished").mkdir()
            client = TestClient(create_dashboard(root))
            self.assertEqual(client.get("/api/runs").json(), [])

    def test_acrobot_results_and_media(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = "20261011T120000-1234abcd"
            directory = root / "acrobot" / experiment / "ppo-0"
            directory.mkdir(parents=True)
            (directory / "result.json").write_text(json.dumps({"config": {"method": "ppo"}, "history": []}))
            (directory / "balance-0.mp4").write_bytes(bytes(range(64)))
            client = TestClient(create_dashboard(root, cache_root=root / "cache"))
            self.assertEqual(client.get("/acrobot").status_code, 200)
            results = client.get("/api/acrobot").json()
            self.assertEqual(results[0]["id"], experiment)
            self.assertFalse(results[0]["complete"])
            url = f"/acrobot-media/{experiment}/ppo-0/"
            self.assertEqual(client.get(url + "balance-0.mp4", headers={"Range": "bytes=0-15"}).status_code, 206)
            self.assertEqual(client.get(url + "result.json").status_code, 404)
            self.assertEqual(client.get(url + "secret.env").status_code, 404)

    def test_feasibility_directory_and_media(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20261011T120000-1234abcd"
            directory = root / "feasibility" / run_id
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(json.dumps({"id": run_id}))
            (directory / "continuous-5.mp4").write_bytes(bytes(range(64)))
            client = TestClient(create_dashboard(root, cache_root=root / "cache"))
            self.assertEqual(client.get("/experiments/acrobot-feasibility").status_code, 200)
            self.assertEqual(client.get("/api/feasibility").json(), [{"id": run_id}])
            entry = next(e for e in client.get("/api/experiments").json() if e["id"] == "feasibility")
            self.assertEqual(entry["runs"], 1)
            url = f"/feasibility-media/{run_id}/"
            self.assertEqual(client.get(url + "continuous-5.mp4", headers={"Range": "bytes=0-15"}).status_code, 206)
            self.assertEqual(client.get(url + "secret.env").status_code, 404)

    def test_swingup_directory_and_media(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20261011T120000-1234abcd"
            directory = root / "swingup" / run_id
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(json.dumps({"id": run_id}))
            (directory / "swingup-0.mp4").write_bytes(bytes(range(64)))
            (directory / "controller-0.npz").write_bytes(b"checkpoint")
            client = TestClient(create_dashboard(root, cache_root=root / "cache"))
            self.assertEqual(client.get("/experiments/acrobot-swingup").status_code, 200)
            self.assertEqual(client.get("/api/swingup").json(), [{"id": run_id}])
            entry = next(e for e in client.get("/api/experiments").json() if e["id"] == "swingup")
            self.assertEqual(entry["runs"], 1)
            url = f"/swingup-media/{run_id}/"
            self.assertEqual(client.get(url + "swingup-0.mp4", headers={"Range": "bytes=0-15"}).status_code, 206)
            self.assertEqual(client.get(url + "controller-0.npz").content, b"checkpoint")
            self.assertEqual(client.get(url + "secret.env").status_code, 404)

    def test_neural_directory_and_media(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_id = "20261011T120000-1234abcd"
            directory = root / "neural" / run_id
            directory.mkdir(parents=True)
            (directory / "results.json").write_text(json.dumps({"id": run_id}))
            (directory / "balance-bottom.mp4").write_bytes(bytes(range(64)))
            (directory / "balance.pt").write_bytes(b"checkpoint")
            client = TestClient(create_dashboard(root, cache_root=root / "cache"))
            self.assertEqual(client.get("/experiments/acrobot-neural").status_code, 200)
            self.assertEqual(client.get("/api/neural").json(), [{"id": run_id}])
            entry = next(e for e in client.get("/api/experiments").json() if e["id"] == "neural")
            self.assertEqual(entry["runs"], 1)
            url = f"/neural-media/{run_id}/"
            self.assertEqual(client.get(url + "balance-bottom.mp4", headers={"Range": "bytes=0-15"}).status_code, 206)
            self.assertEqual(client.get(url + "balance.pt").content, b"checkpoint")
            self.assertEqual(client.get(url + "secret.env").status_code, 404)


if __name__ == "__main__":
    unittest.main()
