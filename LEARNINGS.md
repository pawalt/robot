# My learnings

## 2026-10-11

- **Imitation learning is very effective.** In this project, it produced a neural controller that can swing up, balance, and recover from substantial disturbances.
- **Establish whether a task is achievable first.** One way to test this is to manually search for a successful trajectory before trying to learn a policy.
- **Combine deterministic methods to build teachers to imitate.** Different methods can handle different parts of the task, such as trajectory tracking for swing-up and LQR for balancing.
- **Open question: why didn't PPO work here, and what would make it work?** I still don't understand the failure or the changes needed to fix it. We have not established the cause; imitation succeeding does not explain why PPO failed.
