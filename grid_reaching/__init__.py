"""A first reinforcement-learning experiment: tabular grid reaching."""

from gymnasium.envs.registration import register

register(id="Robot/GridReaching-v0", entry_point="grid_reaching.env:GridReachingEnv")
