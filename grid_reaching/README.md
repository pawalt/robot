# Grid reaching with Q-learning

A first RL experiment: a point robot learns to reach the bottom-right cell of a
10 × 10 grid. The policy is a table of 400 numbers, not a neural network.

## Run it

From the project root:

```sh
uv sync --locked
uv run python -m grid_reaching.train
uv run python -m grid_reaching.demo
uv run python -m unittest grid_reaching.test_grid_reaching
```

The trained policy is already saved in `artifacts/policy.npz`. The demo loads it
without retraining and prints a route from the top-left corner. Use `--start 42`
to start from row 4, column 2. Coordinates are zero-based.

Training defaults to 5,000 episodes with seed 0. To experiment without
overwriting the original results:

```sh
uv run python -m grid_reaching.train --episodes 1000 --seed 7 --output /tmp/grid-run
uv run python -m grid_reaching.demo --policy /tmp/grid-run/policy.npz
```

## Environment

Read `env.py` first. It implements Gymnasium's `reset`, `step`, and text rendering
interfaces. You can instantiate `GridReachingEnv` directly or use
`gymnasium.make("grid_reaching:Robot/GridReaching-v0")`.

| Component | Meaning |
| --- | --- |
| Observation | Integer `row * size + column`, from 0 to 99 |
| Actions | 0: up, 1: right, 2: down, 3: left |
| Start | Uniformly sampled non-goal cell |
| Goal | Fixed bottom-right cell (state 99) |
| Reward | +10 on reaching the goal; −1 on every other step |
| Boundaries | An action into a wall leaves the robot in place and costs −1 |
| Termination | Reaching the goal |
| Truncation | 200 steps without reaching the goal |

The goal is fixed, so position alone describes the state. A moving goal would
also need to appear in the observation and in the Q-table's state index.

## Learning

Read `train.py` next. Each table entry `Q[state, action]` estimates the discounted
future reward for an action. The table starts at zero. Each transition updates
one entry:

```text
target = reward + gamma * max(Q[next_state])
Q[state, action] += alpha * (target - Q[state, action])
```

Here `alpha = 0.1` is the learning rate and `gamma = 0.95` discounts future rewards.
At the goal, the target is just the immediate reward: there is no future value.
A time limit ends the rollout but still uses the next state's value, because
the underlying reaching task could continue.

During training, epsilon-greedy exploration chooses a random action with
probability epsilon. Epsilon decreases from 1.0 to 0.05 over the first 80% of
episodes. Otherwise, the agent chooses an action with the largest Q-value,
breaking ties randomly. Evaluation turns exploration off and uses `argmax`.

## Saved results

`artifacts/` contains the Q-table (`policy.npz`), an arrow map (`policy.txt`),
per-episode training data (`training.csv`), and evaluation results (`results.json`).
Re-running training replaces these outputs.

The default run reached the goal from **all 99 non-goal starting cells**, using
a shortest path in every case. Mean route length was **9.09 steps**. A random
policy succeeded in **40 of 99 episodes**, averaging **150.56 steps** including
timeouts. Both policies use the same start cells and 200-step limit. Random
evaluation uses seed 10000; these are results for this small deterministic grid,
not evidence of generalization to other tasks.

## Things to try

1. Inspect the four Q-values for state 0 and compare them with state 98.
2. Reduce the episode count and inspect the arrow map for loops.
3. Change the step penalty or exploration schedule and compare training data.
4. Add an obstacle while keeping the goal reachable from every start.

Reference: [Gymnasium's custom environment guide](https://gymnasium.farama.org/introduction/create_custom_env/).

## Modal video dashboard

[Open the grid-reaching experiment](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/experiments/grid-reaching).

[All experiments](https://modal-labs-peyton-agents--robot-grid-learning-dashboard.modal.run/).

The dashboard shows paired learned/random MP4s from four starting cells, playback
speed controls, evaluation metrics, a training reward curve, and artifact downloads.
It is a read-only public web endpoint; training is launched through your authenticated
Modal CLI. The recordings contain only the simulated grid.

The root `modal_app.py` assembles the shared app. `grid_reaching/modal_app.py` owns
the grid experiment jobs. `train_and_record` trains on
CPU and writes each run to the `robot-grid-recordings` Modal Volume in the
`peyton-agents` environment. `dashboard` serves FastAPI via `@modal.asgi_app()`.
Runs have unique IDs, so new recordings preserve previous runs. Use **Refresh runs**
in the UI after a new job completes. The web app scales down when idle.

From the project root, load the configured credentials without printing them:

```sh
set -a
source ~/.codex/modal.env
set +a

# Train and save eight MP4 recordings on Modal.
uv run modal run --env peyton-agents -m grid_reaching.modal_app::grid

# Compare another training budget or seed.
uv run modal run --env peyton-agents -m grid_reaching.modal_app::grid --episodes 1000 --seed 7

# Publish dashboard/code changes. Existing volume recordings are preserved.
uv run modal deploy --env peyton-agents -m modal_app
```

`record.py` draws the robot, goal, trail, and step/reward labels into RGB frames
and encodes browser-compatible H.264 MP4s at six frames per second. `experiment_ui/server.py`
reloads the volume when listing runs, caches immutable media locally, and serves
HTTP range requests so seeking works. No credentials are baked into the image.
The image installs dependencies from the project's `uv.lock`.

Run the environment, video encoding, and web endpoint tests:

```sh
uv run python -m unittest grid_reaching.test_grid_reaching experiment_ui.test_dashboard
```

Modal references: [web apps](https://modal.com/docs/guide/webhooks) and
[persistent volumes](https://modal.com/docs/guide/volumes).
