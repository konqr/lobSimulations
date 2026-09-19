"""Bounded episode diagnostics for policy proposals versus executed commands."""
from collections import Counter
import copy
import json


class ActionFlowAudit:
    def __init__(self):
        self.phases = {}

    def record(self, result, *, action_time, twap_present, twap_start, twap_end,
               inventory_before, inventory_after, disabled=False):
        """Record one RL env.step, without affecting action selection.

        Inventory movement includes resting-order fills during this step, not
        just effects of the submitted command. It is not a causal fill count.
        The episode record supplies TWAP side and action-space configuration.
        """
        phase = ('absent' if not twap_present else 'pre' if action_time < twap_start
                 else 'post' if action_time > twap_end else 'during')
        bucket = self.phases.setdefault(phase, {
            'n_steps': 0, 'decisions': Counter(), 'utilities_when_acting': Counter(),
            'executed': Counter(), 'proposal_to_execution': Counter(),
            'n_forced': 0, 'n_disabled': 0, 'inventory_step_delta': 0.0,
        })
        executed, (decision, utility) = result[:2]
        execution_key = json.dumps(executed, separators=(',', ':'))
        if disabled:
            proposal = 'disabled'
            bucket['n_disabled'] += 1
        elif decision is None:
            proposal = 'forced'
            bucket['n_forced'] += 1
        else:
            bucket['decisions'][str(decision)] += 1
            proposal = f'd{decision}'
            if decision == 1:
                bucket['utilities_when_acting'][str(utility)] += 1
                proposal += f':u{utility}'
        bucket['n_steps'] += 1
        bucket['executed'][execution_key] += 1
        bucket['proposal_to_execution'][proposal + '->' + execution_key] += 1
        bucket['inventory_step_delta'] += float(inventory_after) - float(inventory_before)

    def snapshot(self):
        # Do not expose counters that will change beneath a stored episode.
        return {'schema_version': 1, 'phases': copy.deepcopy(self.phases)}
