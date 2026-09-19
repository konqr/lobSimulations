import json
import os
import sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from HawkesRLTrading.src.Utils.action_flow import ActionFlowAudit


class ActionFlowTests(unittest.TestCase):
    def record(self, audit, result, **kwargs):
        options = dict(action_time=300, twap_present=True, twap_start=250,
                       twap_end=400, inventory_before=0, inventory_after=0)
        options.update(kwargs)
        audit.record(result, **options)

    def test_proposals_execution_forced_and_disabled_are_distinct(self):
        audit = ActionFlowAudit()
        self.record(audit, (12, (1, 3)))  # canceled proposal, not utility zero
        self.record(audit, (12, (0, 0)))
        self.record(audit, (4, (None, None)), inventory_before=25, inventory_after=24)
        self.record(audit, (12, (0, 0)), disabled=True)
        b = audit.snapshot()['phases']['during']
        self.assertEqual(b['n_steps'], 4)
        self.assertEqual(b['decisions'], {'0': 1, '1': 1})
        self.assertEqual(b['utilities_when_acting'], {'3': 1})
        self.assertEqual(b['proposal_to_execution'], {'d1:u3->12': 1, 'd0->12': 1,
                                                     'forced->4': 1, 'disabled->12': 1})
        self.assertEqual(b['inventory_step_delta'], -1)
        self.assertEqual(sum(b['executed'].values()), b['n_steps'])

    def test_phase_boundaries_absence_and_composite_actions(self):
        audit = ActionFlowAudit()
        for t in (249, 250, 400, 401):
            self.record(audit, (((12, 1), (9, 1)), (1, 2)), action_time=t)
        self.record(audit, (9, (1, 3)), twap_present=False)
        snap = audit.snapshot()
        self.assertEqual({k: v['n_steps'] for k, v in snap['phases'].items()},
                         {'pre': 1, 'during': 2, 'post': 1, 'absent': 1})
        self.assertIn('d1:u2->[[12,1],[9,1]]', snap['phases']['during']['proposal_to_execution'])
        self.assertEqual(json.loads(json.dumps(snap)), snap)
        self.record(audit, (12, (0, 0)))
        self.assertEqual(snap['phases']['during']['n_steps'], 2)


if __name__ == '__main__':
    unittest.main()
