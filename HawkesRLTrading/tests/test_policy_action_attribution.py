"""Exercise production action selection without starting a market simulation."""
import os
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from HawkesRLTrading.src.SimulationEntities.ICRLAgent import PPOAgent
from HawkesRLTrading.src.SimulationEntities.TradingAgent import TradingAgent


class Network:
    def __init__(self, size):
        self.logits = torch.arange(size, dtype=torch.float32).unsqueeze(0) / 10
        self.calls = 0

    def __call__(self, state):
        self.calls += 1
        return self.logits, torch.tensor([[2.5]])


class Agent(PPOAgent):
    def __init__(self, config, inventory=0):
        TradingAgent.__init__(self, Inventory={'INTC': inventory}, inventorylimit=25)
        self.action_space_config = config
        self.allowed_actions = self.actions[:12] if config == 0 else self.actions[:4]
        self.convert_dict = {} if config == 0 else {0: 2, 1: 3, 2: 8, 3: 9}
        self.Actor_Critic_d = Network(2)
        self.Actor_Critic_u = Network(12 if config == 0 else 4 if config == 1 else 3)
        self.typeNN = 'Dense'
        self.last_state = None
        self.last_action = None
        self.trajectory_buffer = [None] * 10
        self.episode_sides = {}
        self.exploration_bonus = False
        self.symmetric_mo_gating = True

    def readData(self, data):
        return torch.zeros((1, 10))


def select(config, decision, utility, exploration, inventory=0, spread=0.02, symmetric=True):
    agent = Agent(config, inventory)
    agent.symmetric_mo_gating = symmetric
    data = {'Positions': {level: [] for level in ('Ask_L1', 'Ask_L2', 'Bid_L1', 'Bid_L2')},
            'LOB0': {'Ask_L1': (100 + spread, 10), 'Bid_L1': (100, 10)}}
    with patch('random.random', return_value=0.0 if exploration else 1.0), \
         patch('random.randint', side_effect=[decision, utility]), \
         patch('torch.multinomial', side_effect=[torch.tensor([decision]), torch.tensor([utility])]):
        result = agent.get_action(data, epsilon=0.5)
    return agent, result


class PolicyActionTests(unittest.TestCase):
    def assert_sample_record(self, agent, result, utility):
        self.assertEqual(result[1], (1, utility))
        expected = torch.log_softmax(agent.Actor_Critic_u.logits, dim=1)[0, utility].item()
        self.assertAlmostEqual(result[3], expected, places=6)
        self.assertEqual(result[5], 2.5)
        agent.trajectory_buffer = []
        agent.store_transition(0, agent.last_state, result[1], -0.25, agent.last_state, False)
        self.assertEqual(agent.trajectory_buffer[0][1][1:4], (1, utility, -0.25))

    def test_no_decision_means_no_order_in_every_action_space(self):
        for config in (0, 1, 2):
            for explore in (False, True):
                with self.subTest(config=config, explore=explore):
                    agent, result = select(config, 0, 0, explore)
                    expected = 12 if config < 2 else ((12, 1), (12, 1))
                    self.assertEqual(result[0], expected)
                    self.assertEqual(agent.last_action, 12)
                    self.assertEqual(agent.Actor_Critic_d.calls, 1)
                    self.assertEqual(agent.Actor_Critic_u.calls, 1)

    def test_rejected_actions_keep_the_sampled_utility_in_buffer(self):
        # Empty cancel queues; impossible in-spread orders; inventory-gated MOs.
        cases = [(0, u, 0, 0.02) for u in (1, 3, 8, 10)]
        cases += [(1, u, 0, 0.02) for u in (1, 2)]
        cases += [(0, u, 0, 0.01) for u in (5, 6)]
        cases += [(0, 4, 0, 0.02), (0, 7, 23, 0.02)]
        for config, utility, inventory, spread in cases:
            for explore in (False, True):
                with self.subTest(config=config, utility=utility, explore=explore):
                    agent, result = select(config, 1, utility, explore, inventory, spread)
                    self.assertEqual(result[0], 12)
                    self.assertEqual(agent.last_action, 12)
                    self.assert_sample_record(agent, result, utility)

    def test_partial_composite_order_keeps_its_selected_utility(self):
        for utility, lo in ((1, (2, 1)), (2, (9, 1))):
            for explore in (False, True):
                with self.subTest(utility=utility, explore=explore):
                    agent, result = select(2, 1, utility, explore)
                    self.assertEqual(result[0], ((12, 1), lo))
                    self.assert_sample_record(agent, result, utility)

    def test_inventory_override_keeps_policy_intent(self):
        for inventory in (-24, 24):
            agent, result = select(0, 1, 11, False, inventory)
            self.assertEqual(result[0], 12)  # forced cancel has no resting order
            self.assert_sample_record(agent, result, 11)

    def test_gate_switch_does_not_change_actions_at_central_inventories(self):
        # Same state and same proposal: isolate gating from action-space changes.
        for inventory in (-16, -5, 0, 10, 22):
            for explore in (False, True):
                for utility in range(12):
                    with self.subTest(inventory=inventory, explore=explore, utility=utility):
                        _, legacy = select(0, 1, utility, explore, inventory, symmetric=False)
                        _, bounded = select(0, 1, utility, explore, inventory, symmetric=True)
                        self.assertEqual(legacy, bounded)

    def test_gate_switch_at_limits_changes_recovery_not_short_entry(self):
        _, legacy = select(0, 1, 7, False, -23, symmetric=False)
        _, bounded = select(0, 1, 7, False, -23, symmetric=True)
        self.assertEqual((legacy[0], bounded[0]), (12, 7))  # permits buying to recover
        _, legacy = select(0, 1, 4, False, 23, symmetric=False)
        _, bounded = select(0, 1, 4, False, 23, symmetric=True)
        self.assertEqual((legacy[0], bounded[0]), (12, 4))  # permits selling to recover

    def test_valid_limit_orders_keep_execution_and_sample(self):
        for config, utility, order in ((0, 11, 11), (1, 3, 9)):
            for explore in (False, True):
                agent, result = select(config, 1, utility, explore)
                self.assertEqual(result[0], order)
                self.assertEqual(agent.last_action, order)
                self.assert_sample_record(agent, result, utility)


if __name__ == '__main__':
    unittest.main()
