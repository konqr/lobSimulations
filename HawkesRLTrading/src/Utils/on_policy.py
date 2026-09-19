"""Fresh-rollout recurrent PPO with a fixed hierarchical behavior likelihood."""
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def mixture_log_probs(logits, epsilon):
    e = torch.as_tensor(epsilon, dtype=logits.dtype, device=logits.device).unsqueeze(-1)
    return torch.logaddexp(F.log_softmax(logits, dim=-1) + torch.log1p(-e),
                          torch.log(e) - math.log(logits.shape[-1]))


def fixed_gae(rewards, values, dones, gamma, lam):
    advantage = torch.zeros_like(values)
    carry = torch.zeros((), device=values.device)
    for i in range(len(values) - 1, -1, -1):
        next_value = values[i + 1] if i + 1 < len(values) else 0.0
        continuation = 1.0 - dones[i]
        delta = rewards[i] + gamma * next_value * continuation - values[i]
        carry = delta + gamma * lam * continuation * carry
        advantage[i] = carry
    return advantage, advantage + values


def finalize_episode(agent, episode, terminated):
    """Fold a terminal event after another agent's wake into the last RL step.

    Match the existing cash/inventory reward convention. No new policy action
    occurred, so no action/count bonus or extra running penalty is earned.
    A wall-clock cutoff is not a true terminal and must not get zero bootstrap.
    """
    if not terminated:
        raise ValueError('On-policy episode ended without a true terminal')
    if not agent.trajectory_buffer or agent.trajectory_buffer[-1][0] != episode:
        raise ValueError('On-policy episode has no recorded RL transitions')
    ep, last = agent.trajectory_buffer[-1]
    if last[5]:
        return 0.0
    cash, inventory, mid = agent._on_policy_last_account
    q = agent.countInventory()
    factor = 1 - agent.transaction_cost * np.sign(q)
    tail = agent.cash - cash + (q * agent.mid - inventory * mid) * factor
    tail -= agent.terminal_invpenalty * q ** 2
    updated = list(last)
    updated[3] += float(tail)
    updated[5] = 1
    agent.trajectory_buffer[-1] = (ep, tuple(updated))
    return float(tail)


def replay(agent, states, epsilon, actions):
    d_logits, dv, _, _ = agent.Actor_Critic_d.forward_sequence(states.unsqueeze(0))
    u_logits, uv, _, _ = agent.Actor_Critic_u.forward_sequence(states.unsqueeze(0))
    dl = mixture_log_probs(d_logits.squeeze(0), epsilon)
    ul = mixture_log_probs(u_logits.squeeze(0), epsilon)
    joint = dl.gather(1, actions[:, :1]).squeeze(1)
    joint = joint + actions[:, 0] * ul.gather(1, actions[:, 1:2]).squeeze(1)
    entropy = -(dl.exp() * dl).sum(-1) + dl[:, 1].exp() * -(ul.exp() * ul).sum(-1)
    return joint, dv.flatten(), uv.flatten(), entropy


def train_fresh(agent, logger):
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    batches = []
    errors = []
    for episode in sorted({ep for ep, _ in agent.trajectory_buffer}):
        rows = [tr for ep, tr in agent.trajectory_buffer if ep == episode]
        if not rows[-1][5]:
            raise ValueError('Fresh PPO requires complete terminal episodes; no zero bootstrap at a cutoff')
        states = torch.cat([r[0] for r in rows]).to(agent.device)
        actions = torch.tensor([[r[1], r[2]] for r in rows], device=agent.device)
        def tensor(values):
            return torch.tensor(values, device=agent.device, dtype=torch.float32)
        epsilon = tensor([r[6]['epsilon'] for r in rows])
        old = tensor([r[6]['log_prob'] for r in rows])
        values = tensor([r[6]['value'] for r in rows])
        mask = torch.tensor([r[6]['actor'] for r in rows], device=agent.device, dtype=torch.bool)
        advantage, target = fixed_gae(tensor([r[3] for r in rows]), values,
                                     tensor([r[5] for r in rows]), agent.gamma, agent.gae_lambda)
        with torch.no_grad():
            reproduced, _, _, _ = replay(agent, states, epsilon, actions)
            error = (reproduced[mask] - old[mask]).abs().max().item() if mask.any() else 0.0
        if error > 2e-4:
            raise ValueError(f'Rollout/replay likelihood mismatch: {error}')
        errors.append(error)
        batches.append((states, actions, epsilon, old, mask, advantage.detach(), target.detach()))
    actor_adv = torch.cat([b[5][b[4]] for b in batches])
    mean = actor_adv.mean() if actor_adv.numel() else 0.0
    scale = actor_adv.std(unbiased=False).clamp_min(1e-8) if actor_adv.numel() else 1.0
    diagnostics = []
    stop = False
    for epoch in range(agent.on_policy_epochs):
        for index in np.random.permutation(len(batches)):
            states, actions, epsilon, old, mask, advantage, target = batches[index]
            joint, dv, uv, entropy = replay(agent, states, epsilon, actions)
            log_ratio = joint[mask] - old[mask]
            ratio = log_ratio.exp()
            kl = ((ratio - 1) - log_ratio).mean() if mask.any() else joint.sum() * 0
            if not torch.isfinite(kl):
                raise ValueError('Non-finite policy KL')
            if kl.item() > agent.on_policy_target_kl:
                stop = True
                break
            adv = ((advantage - mean) / scale)[mask]
            policy = -torch.minimum(ratio * adv, ratio.clamp(1-agent.clip_ratio, 1+agent.clip_ratio) * adv).mean() if mask.any() else joint.sum() * 0
            vd, vu = F.mse_loss(dv, target), F.mse_loss(uv, target)
            ent = entropy[mask].mean() if mask.any() else entropy.sum() * 0
            loss = policy + agent.value_loss_coef * (vd + vu) / 2 - agent.entropy_coef * ent
            if not torch.isfinite(loss):
                raise ValueError('Non-finite on-policy loss')
            agent.optimizer_d.zero_grad(); agent.optimizer_u.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(agent.Actor_Critic_d.parameters()) + list(agent.Actor_Critic_u.parameters()), agent.max_grad_norm)
            agent.optimizer_d.step(); agent.optimizer_u.step()
            agent.scheduler_d.step(); agent.scheduler_u.step()
            # One joint actor loss; u_policy_loss=0 is not a second objective.
            logger.log_losses(d_policy_loss=policy.item(), d_value_loss=vd.item(),
                              d_entropy_loss=ent.item(), u_policy_loss=0.0,
                              u_value_loss=vu.item(), u_entropy_loss=0.0)
            diagnostics.append({'epoch': epoch, 'batch_index': int(index), 'kl': kl.item(),
                                'clip_fraction': ((ratio - 1).abs() > agent.clip_ratio).float().mean().item() if mask.any() else 0.0,
                                'entropy': ent.item()})
        if stop:
            break
    info = {'rollout_episodes': sorted({ep for ep, _ in agent.trajectory_buffer}),
            'transitions': len(agent.trajectory_buffer), 'replay_logprob_max_error': max(errors),
            'optimizer_steps': len(diagnostics), 'early_stop_kl': stop,
            'seconds': time.time() - started, 'updates': diagnostics,
            'max_gpu_memory_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    path = Path(logger.log_dir) / f'on_policy_updates_{logger.label}.jsonl'
    with path.open('a') as file:
        file.write(json.dumps(info) + '\n')
    print('ON POLICY UPDATE', json.dumps(info), flush=True)
    agent.trajectory_buffer.clear()
    agent.episode_sides.clear()
    agent.Actor_Critic_d.reset_hidden_state(batch_size=1)
    agent.Actor_Critic_u.reset_hidden_state(batch_size=1)
    return [0] * 6
