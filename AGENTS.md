# Project instructions

These instructions apply to every project and experiment in this repository.

- Run all training jobs, experiments, benchmarks, evaluation sweeps, and rollout
  recording jobs on **Modal**, never on the local machine. Local work is for
  editing, static checks, and small correctness tests, not experimental runs.
- Always use the **`peyton-agents`** Modal environment. Source credentials from
  `~/.codex/modal.env` without printing or committing their values.
- Prioritize fast iteration: fan out independent Modal experiments and increase
  compute as useful. The user has explicitly welcomed more compute and GPUs when
  they improve turnaround; use measured bottlenecks to choose resources.
- Maintain **one shared experiment UI**: its homepage is a directory of all
  experiments, and each experiment has its own subpage. Add new experiments to
  this directory instead of creating disconnected dashboards.
- Keep the shared Modal entrypoint at the repository root (`modal_app.py`) and
  shared web UI in `experiment_ui/`. The root app may only organize/compose apps
  and orchestrate shared services; it must contain no experiment-specific logic.
  Each experiment's training, evaluation, recording, job definitions, and CLI
  commands belong in that experiment's own subfolder. Compose their Modal apps
  with `app.include(...)`. Shared infrastructure configuration is separate from
  experiment behavior.
- Store experiment outputs, checkpoints, metrics, and videos on Modal Volumes.
  Preserve previous runs and show failures as well as successes.
- Use **`jj`** for version control.
- Keep the user's stated learnings and open questions in the root
  `LEARNINGS.md`. Update it when the user shares new takeaways; preserve their
  meaning and do not turn unresolved questions into established conclusions.
- Use **`uv`**, a committed lockfile, and isolated environments for Python
  dependencies. Never install project dependencies at the system level.

- The user-facing interactive Acrobot playground runs physics and policy inference
  in the visitor’s browser, as requested. Automated training, rollouts, evaluation
  and benchmarks still run on Modal.
