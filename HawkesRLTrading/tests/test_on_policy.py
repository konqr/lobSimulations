import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HJBQVI.DGMTorch import ActorCriticSeparate
from HawkesRLTrading.src.Utils.on_policy import mixture_log_probs, fixed_gae, train_fresh, finalize_episode
from test_policy_action_attribution import Agent


class Logger:
    def __init__(self, directory):
        self.log_dir = directory; self.label = 'unit'; self.rows = []
    def log_losses(self, **kwargs):
        self.rows.append(kwargs)


def rollout():
    torch.manual_seed(13)
    agent = Agent(0, 0)
    agent.on_policy = True; agent.typeNN = 'LSTM'; agent.device = torch.device('cpu')
    agent.Actor_Critic_d = ActorCriticSeparate(10, 8, 1, 2, typeNN='LSTM', q_function=False).to('cpu')
    agent.Actor_Critic_u = ActorCriticSeparate(10, 8, 1, 12, typeNN='LSTM', q_function=False).to('cpu')
    agent.optimizer_d = torch.optim.Adam(agent.Actor_Critic_d.parameters(), lr=.001)
    agent.optimizer_u = torch.optim.Adam(agent.Actor_Critic_u.parameters(), lr=.001)
    agent.scheduler_d = torch.optim.lr_scheduler.LambdaLR(agent.optimizer_d, lambda _: 1)
    agent.scheduler_u = torch.optim.lr_scheduler.LambdaLR(agent.optimizer_u, lambda _: 1)
    agent.gamma = .99; agent.gae_lambda = .95; agent.clip_ratio = .2
    agent.value_loss_coef = .5; agent.entropy_coef = .01; agent.max_grad_norm = 1
    agent.on_policy_epochs = 2; agent.on_policy_target_kl = .02
    agent.trajectory_buffer = []
    data = {'Positions': {k: [] for k in ('Ask_L1', 'Ask_L2', 'Bid_L1', 'Bid_L2')},
            'LOB0': {'Ask_L1': (100.02, 10), 'Bid_L1': (100, 10)}}
    for step in range(16):
        agent.breach = step == 12
        result = agent.get_action(data, epsilon=.1)
        agent.store_transition(0, agent.last_state, result[1], -2 if agent.breach else .1,
                               agent.last_state, step == 15, rollout=result)
    return agent


class OnPolicyTests(unittest.TestCase):
    def test_epsilon_mixture_and_gae_terminal(self):
        logits = torch.tensor([[math.log(3), 0.]])
        torch.testing.assert_close(mixture_log_probs(logits, .2).exp(), torch.tensor([[.7, .3]]))
        torch.testing.assert_close(mixture_log_probs(logits, 1.).exp(), torch.tensor([[.5, .5]]))
        adv, ret = fixed_gae(torch.tensor([1., 2.]), torch.tensor([.5, .25]),
                             torch.tensor([0., 1.]), 1., 1.)
        torch.testing.assert_close(ret, torch.tensor([3., 2.]))

    def test_real_recurrent_rollout_matches_and_forced_reward_is_retained(self):
        agent = rollout()
        self.assertEqual(len(agent.trajectory_buffer), 16)
        self.assertFalse(agent.trajectory_buffer[12][1][6]['actor'])
        self.assertEqual(agent.trajectory_buffer[12][1][3], -2)
        old = [p.detach().clone() for p in agent.Actor_Critic_d.parameters()]
        with tempfile.TemporaryDirectory() as directory:
            logger = Logger(directory)
            train_fresh(agent, logger)
            self.assertTrue(logger.rows)
            self.assertTrue(all(math.isfinite(v) for row in logger.rows for v in row.values()))
        self.assertEqual(agent.trajectory_buffer, [])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old, agent.Actor_Critic_d.parameters())))

    def test_terminal_on_another_agent_wake_captures_tail_once(self):
        agent = rollout()
        ep, last = agent.trajectory_buffer[-1]
        row = list(last); row[5] = 0
        agent.trajectory_buffer[-1] = (ep, tuple(row))
        agent._on_policy_last_account = (1000., 2., 100.)
        agent.cash = 1001.; agent.Inventory = {'INTC': 2}; agent.mid = 101.
        agent.transaction_cost = 0.; agent.terminal_invpenalty = .1
        self.assertAlmostEqual(finalize_episode(agent, 0, True), 2.6)
        self.assertEqual(agent.trajectory_buffer[-1][1][5], 1)
        self.assertAlmostEqual(agent.trajectory_buffer[-1][1][3], last[3] + 2.6)
        self.assertEqual(finalize_episode(agent, 0, True), 0.)
        with self.assertRaises(ValueError):
            finalize_episode(agent, 0, False)

    def test_corrupted_rollout_probability_fails_before_update(self):
        agent = rollout()
        agent.trajectory_buffer[11][1][6]['log_prob'] += .2
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'likelihood mismatch'):
            train_fresh(agent, Logger(directory))


if __name__ == '__main__':
    unittest.main()
