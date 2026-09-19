"""Frozen checkpoint tests; no market simulation is run locally."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HawkesRLTrading.src.Utils.frozen_eval import load_frozen_checkpoint, assert_frozen
from test_policy_action_attribution import Agent


class FrozenTests(unittest.TestCase):
    def test_empty_buffer_does_not_force_random_evaluation(self):
        data = {'Positions': {k: [] for k in ('Ask_L1', 'Ask_L2', 'Bid_L1', 'Bid_L2')},
                'LOB0': {'Ask_L1': (100.02, 10), 'Bid_L1': (100, 10)}}
        for config in (0, 1, 2):
            agent = Agent(config, 0)
            agent.trajectory_buffer = []
            with patch('random.random', side_effect=AssertionError('epsilon RNG used')), \
                 patch('random.randint', side_effect=AssertionError('uniform policy used')), \
                 patch('torch.multinomial', return_value=torch.tensor([0])):
                self.assertEqual(agent.get_action(data, epsilon=1, evaluation=True)[1][0], 0)

    def test_load_identity_and_detect_mutation(self):
        class Models:
            Actor_Critic_d = torch.nn.Linear(2, 2)
            Actor_Critic_u = torch.nn.Linear(2, 3)
        agent = Models()
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name in ('d', 'u'):
                p = Path(tmp) / (name + '.pt')
                torch.save(getattr(agent, 'Actor_Critic_' + name).state_dict(), p)
                paths[name] = str(p)
            meta = Path(tmp) / 'metadata.json'
            meta.write_text(json.dumps({'epoch': 16, 'models': paths}))
            info = load_frozen_checkpoint(agent, meta)
            self.assertEqual(info['epoch'], 16)
            self.assertFalse(agent.Actor_Critic_d.training)
            assert_frozen(agent, info)
            with torch.no_grad():
                agent.Actor_Critic_u.weight.add_(1)
            with self.assertRaises(RuntimeError):
                assert_frozen(agent, info)

    def test_training_and_checkpoint_writes_guarded_in_production(self):
        path = Path(__file__).resolve().parents[1] / 'AR_RL_Trainer.py'
        tree = ast.parse(path.read_text())
        guarded = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                   and ast.unparse(n.test).startswith(
                       'not EVAL_ONLY and episode % 4 == 0')]
        self.assertEqual(len(guarded), 1)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr in ('train', 'save_models')]
        self.assertTrue(calls)
        for call in calls:
            self.assertIn(call, list(ast.walk(guarded[0])))


if __name__ == '__main__':
    unittest.main()
