"""Serve the shared experiment directory and subpages."""
import modal
from modal_resources import PROJECT, project_image, volume

app = modal.App("robot-experiment-ui")
web_image = project_image().add_local_dir(
    PROJECT / "experiment_ui", remote_path="/root/experiment_ui",
    ignore=["__pycache__"],
).add_local_dir(PROJECT / "acrobot_neural" / "browser", remote_path="/root/acrobot_neural/browser")


@app.function(image=web_image, volumes={"/data": volume}, scaledown_window=60,
              max_containers=2, memory=512, env={"EXPERIMENT_UI_REVISION": "neural-sweep-1"})
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def dashboard():
    from experiment_ui.server import create_dashboard

    return create_dashboard("/data", refresh=volume.reload)
