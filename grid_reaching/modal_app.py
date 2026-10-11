"""Grid experiment jobs. Run with modal run -m grid_reaching.modal_app::grid."""
import modal
from modal_resources import PROJECT, project_image, volume

app = modal.App("robot-grid-experiment")
grid_image = project_image().add_local_dir(
    PROJECT / "grid_reaching", remote_path="/root/grid_reaching",
    ignore=["__pycache__", "artifacts"],
)
DATA = "/data/runs"


@app.function(image=grid_image, volumes={"/data": volume}, timeout=900, cpu=2, memory=1024)
def train_and_record(episodes: int = 5000, seed: int = 0):
    from grid_reaching.record import create_run

    result = create_run(DATA, episodes=episodes, seed=seed)
    volume.commit()
    return result


@app.local_entrypoint()
def grid(episodes: int = 5000, seed: int = 0):
    result = train_and_record.remote(episodes, seed)
    print(f"Recorded run {result['id']}: {len(result['recordings'])} MP4s")
    print(f"Learned success rate: {result['learned']['success_rate']:.0%}")
