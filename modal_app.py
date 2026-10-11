"""Compose the application. Deploy with: uv run modal deploy -m modal_app --env peyton-agents."""
import modal

from acrobot_comparison.modal_app import app as acrobot_app
from acrobot_feasibility.modal_app import app as feasibility_app
from acrobot_swingup.modal_app import app as swingup_app
from acrobot_neural.modal_app import app as neural_app
from experiment_ui.modal_app import app as ui_app
from grid_reaching.modal_app import app as grid_app

# Keep the existing deployment identity and URLs.
app = modal.App("robot-grid-learning")
app.include(ui_app)
app.include(grid_app)
app.include(acrobot_app)
app.include(feasibility_app)
app.include(swingup_app)
app.include(neural_app)
