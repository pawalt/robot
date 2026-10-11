# robot

Python project managed with [uv](https://docs.astral.sh/uv/).

This project is for learning reinforcement learning through small Gymnasium
environments and, eventually, basic robotics experiments.

Start with [grid reaching and tabular Q-learning](grid_reaching/README.md).

The [shared experiment directory](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/)
links to every experiment's metrics and videos:

- [Grid reaching](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/grid-reaching)
- [Acrobot: PPO vs grouped critic-free learning](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot)
- [Acrobot: controller feasibility](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-feasibility)
- [Acrobot: swing-up attempts](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-swingup)

- [Acrobot: neural-only control](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-neural)
- [Acrobot: architecture and forgetting comparison](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-neural-sweep)
- [Acrobot: full swing-up imitation](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/acrobot-imitation)

All experiments run on Modal in `peyton-agents`; see [AGENTS.md](AGENTS.md).

## Getting started

```sh
uv sync --locked
set -a
source ~/.codex/modal.env
set +a
uv run modal deploy --env peyton-agents -m modal_app
```

Add dependencies with `uv add <package>` and commit both `pyproject.toml` and
`uv.lock`.

## Project layout

- `modal_app.py`: assembles the UI and experiment apps; contains no experiment logic.
- `modal_resources.py`: shared image/dependency configuration and volume handle.
- `experiment_ui/`: shared directory, experiment pages, and web server.
- `grid_reaching/`: grid environment, Q-learning, recordings, and its Modal jobs/commands.
- `acrobot_comparison/`: Acrobot environment, training, evaluation, recordings, and its Modal jobs/commands.
- `acrobot_feasibility/`: LQR feasibility sweeps, torque-resolution comparisons, and its Modal jobs/commands.
- `acrobot_neural/`: teacher distillation, continuous PPO, and neural-only evaluation.
- `acrobot_swingup/`: energy-based swing-up controllers, held-out evaluation, and recordings.

Launch a grid run with `uv run modal run --env peyton-agents -m grid_reaching.modal_app::grid`.
See each experiment's README for its remote training and evaluation commands.

Reward-design comparison: [protocol](acrobot_neural/REWARDS.md), with a shared dashboard subpage at `/experiments/acrobot-rewards`.

Verified neural ground-to-balance policies: [radial imitation results](acrobot_neural/RADIAL.md), with recordings and policy downloads at `/experiments/acrobot-radial`.

Larger starting disturbances: [protocol and results](acrobot_neural/DISTURBANCE.md), dashboard `/experiments/acrobot-disturbance`.

Actual top releases and kicks: [protocol/results](acrobot_neural/TOP_RECOVERY.md), dashboard `/experiments/acrobot-top`.
