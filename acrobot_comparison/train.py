"""Identical clipped policy updates; only advantage estimation differs."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .env import DT, HORIZON, advance, measurements, observations, sample_starts


def network(outputs):
    model = nn.Sequential(nn.Linear(7, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, outputs))
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(model[-1].weight, gain=0.01 if outputs == 3 else 1)
    return model


def gae(rewards, values, lam=0.95):
    """Undiscounted finite-horizon returns, so the terminal bootstrap is zero."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        following = values[t + 1] if t + 1 < len(rewards) else 0
        delta = rewards[t] + following - values[t]
        carry = delta + lam * carry
        advantages[t] = carry
    return advantages, advantages + values


def grouped_advantages(rewards, group_size=8):
    """GRPO-inspired outcome advantages; no critic or reference-policy KL.

    Consecutive group members start from the exact same simulator state.
    Each member gets one group-standardized episode return, used at all steps.
    """
    returns = rewards.sum(0).reshape(-1, group_size)
    scale = returns.std(dim=1, unbiased=False, keepdim=True)
    relative = (returns - returns.mean(1, keepdim=True)) / scale.clamp_min(1e-6)
    advantages = relative.flatten().expand_as(rewards)
    return advantages, float((scale < 1e-6).float().mean())


def evaluate(policy, seed=10000, count=32, horizon=HORIZON, mode="balance", stochastic=False, capture=False):
    rng = np.random.default_rng(seed)
    state = sample_starts(rng, count, mode)
    returns, height_sum, upright_sum = (np.zeros(count) for _ in range(3))
    streak, longest = np.zeros(count), np.zeros(count)
    action_counts = np.zeros(3, dtype=np.int64)
    trace = [state[0].tolist()] if capture else []
    for step in range(horizon):
        with torch.no_grad():
            logits = policy(torch.from_numpy(observations(state, step, horizon)))
            if stochastic:
                probs = logits.softmax(-1).numpy()
                actions = (rng.random(count)[:, None] > probs.cumsum(-1)).sum(-1).clip(0, 2)
            else:
                actions = logits.argmax(-1).numpy()
        action_counts += np.bincount(actions, minlength=3)
        state = advance(state, actions)
        reward, height, upright = measurements(state)
        returns += reward
        height_sum += height
        upright_sum += upright
        streak = np.where(upright, streak + DT, 0)
        longest = np.maximum(longest, streak)
        if capture:
            trace.append(state[0].tolist())
    return {"mean_return": float(returns.mean()), "mean_height": float(height_sum.mean() / horizon),
            "upright_fraction": float(upright_sum.mean() / horizon),
            "mean_longest_hold": float(longest.mean()), "hold_5s_rate": float((longest >= 5 - 1e-6).mean()),
            "episodes": count, "action_fractions": (action_counts / action_counts.sum()).tolist()}, trace


def run_experiment(method, seed, output, updates=80, group_size=8, groups=8, horizon=HORIZON, report=None):
    if method not in ("ppo", "grouped"):
        raise ValueError(method)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = network(3)
    # Policy initialization and start-state RNG match across methods for each seed.
    critic = network(1) if method == "ppo" else None
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    value_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3, eps=1e-5) if critic else None
    batch_size = group_size * groups
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {"method": method, "seed": seed, "updates": updates, "group_size": group_size,
              "groups": groups, "horizon": horizon, "dt": DT, "gamma": 1.0, "gae_lambda": 0.95,
              "policy_hidden": [64, 64], "observations": 7, "policy_lr": 3e-4,
              "critic_lr": 1e-3, "epochs": 4, "minibatch": 1024, "clip": 0.2,
              "entropy_coefficient": 0.01, "near_upright_start_probability": 0.75,
              "training_steps": updates * batch_size * horizon}
    history = []
    training_seconds, evaluation_seconds = 0.0, 0.0

    def checkpoint(update, diagnostics):
        nonlocal evaluation_seconds
        before = time.perf_counter()
        metrics = {}
        for mode in ("balance", "swingup"):
            for stochastic in (False, True):
                label = mode + ("_sampled" if stochastic else "")
                metrics[label] = evaluate(policy, horizon=horizon, mode=mode, stochastic=stochastic)[0]
        evaluation_seconds += time.perf_counter() - before
        point = {"steps": update * batch_size * horizon, "training_seconds": training_seconds,
                 "evaluation_seconds": evaluation_seconds, **metrics, **diagnostics}
        history.append(point)
        torch.save(policy.state_dict(), output / f"policy-{update}.pt")
        result = {"config": config, "history": history, "status": "complete" if update == updates else "running"}
        (output / "result.json").write_text(json.dumps(result, indent=2))
        if report:
            report(result)
        print(f"{method} seed={seed} steps={point['steps']} balance_hold={metrics['balance']['mean_longest_hold']:.2f}s swingup_hold={metrics['swingup']['mean_longest_hold']:.2f}s", flush=True)

    checkpoint(0, {})
    for update in range(1, updates + 1):
        before = time.perf_counter()
        # Both methods use the same grouped starting states and total trajectory count.
        state = np.repeat(sample_starts(rng, groups), group_size, axis=0)
        obs_buffer, actions_buffer, logps_buffer, rewards_buffer, values_buffer = [], [], [], [], []
        for step in range(horizon):
            obs = torch.from_numpy(observations(state, step, horizon))
            with torch.no_grad():
                dist = Categorical(logits=policy(obs))
                action = dist.sample()
                logp = dist.log_prob(action)
                if critic:
                    values_buffer.append(critic(obs).squeeze(-1))
            state = advance(state, action.numpy())
            reward = measurements(state)[0]
            obs_buffer.append(obs)
            actions_buffer.append(action)
            logps_buffer.append(logp)
            rewards_buffer.append(torch.from_numpy(reward.astype(np.float32)))
        rewards = torch.stack(rewards_buffer)
        diagnostics = {"training_return": float(rewards.sum(0).mean())}
        if critic:
            values = torch.stack(values_buffer)
            advantages, targets = gae(rewards, values)
            diagnostics["value_explained_variance"] = float(1 - (targets - values).var() / targets.var().clamp_min(1e-8))
        else:
            advantages, diagnostics["zero_variance_group_fraction"] = grouped_advantages(rewards, group_size)
        observations_flat = torch.stack(obs_buffer).flatten(0, 1)
        actions_flat = torch.stack(actions_buffer).flatten()
        old_logps = torch.stack(logps_buffer).flatten()
        advantages = advantages.flatten()
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        if critic:
            targets = targets.flatten()
        kls, entropies, clips = [], [], []
        stop_policy = False
        for epoch in range(4):
            # Separate RNG: minibatch permutation must not alter future start states.
            order = torch.randperm(len(actions_flat))
            for indices in order.split(1024):
                dist = Categorical(logits=policy(observations_flat[indices]))
                logratio = dist.log_prob(actions_flat[indices]) - old_logps[indices]
                ratio = logratio.exp()
                advantage = advantages[indices]
                loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean() - 0.01 * dist.entropy().mean()
                kl = float(((ratio - 1) - logratio).mean().detach())
                if kl > 0.03:
                    stop_policy = True
                if not stop_policy:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()
                if critic:
                    value_loss = (critic(observations_flat[indices]).squeeze(-1) - targets[indices]).square().mean()
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    value_optimizer.step()
                kls.append(kl)
                entropies.append(float(dist.entropy().mean().detach()))
                clips.append(float(((ratio - 1).abs() > 0.2).float().mean().detach()))
        training_seconds += time.perf_counter() - before
        diagnostics.update(approx_kl=float(np.mean(kls)), entropy=float(np.mean(entropies)), clip_fraction=float(np.mean(clips)))
        if update % 10 == 0 or update == updates:
            checkpoint(update, diagnostics)
    return {"config": config, "history": history, "status": "complete"}
