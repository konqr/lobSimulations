from HJBQVI.DGMTorch import *
from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent
from typing import List, Tuple, Dict, Optional, Any
import torch
import numpy as np


def get_queue_priority(data, pos, label):
    ask_l1s = pos[label]
    n_as = []
    if len(ask_l1s):
        idxs = np.array([])
        for o in ask_l1s:
            idxs = np.append(idxs, np.where(np.array(data['lobL3'][label][1]) == o)).astype(int)
        n_as = np.cumsum(data['lobL3_sizes'][label][1])[idxs-1]
    return n_as


class ImpulseControlAgent(GymTradingAgent):
    def __init__(self, label:str,  epoch : int, model_dir: str, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random", Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5, wake_on_MO: bool=True, wake_on_Spread: bool=True , cashlimit=1000000):
        super().__init__(seed=seed, log_events = log_events, log_to_file = log_to_file, strategy=strategy, Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO, wake_on_Spread=wake_on_Spread , cashlimit=cashlimit)
        self.resetseed(seed)
        self.first_action = True
        self.second_action = True
        # model ids
        self.label = label
        self.model_dir = model_dir
        self.epoch = epoch
        ###
        self.model_u = None #PIANet(layer_widths[1], n_layers[1], 11, 10, typeNN = typeNN2)
        self.model_d = None #PIANet(layer_widths[2], n_layers[2], 11, 2, typeNN = typeNN2)
        self.layer_widths = [50,30,30]
        self.n_layers= [10,10,10]
        self.load_nets()

    def load_nets(self):
        model_manager = ModelManager(model_dir = self.model_dir, label = self.label)
        self.model_d =PIANet(self.layer_widths[1],self.n_layers[1],11, 2, typeNN='Dense')
        self.model_u = PIANet(self.layer_widths[2], self.n_layers[2], 11, 10, typeNN = 'Dense')
        _, self.model_d, self.model_u = model_manager.load_models(None, self.model_d, self.model_u, timestamp= self.label, epoch = self.epoch)
        return

    def get_action(self, data) -> Optional[Tuple[int, int]]:
        size = 1
        if self.first_action:
            action = 0 # lo_deep_ask
            self.first_action = False
            return action, size
        if self.second_action:
            action = 11 # lo_deep_bid
            self.second_action = False
            return action, size
        p_a, q_a = data['LOB0']['Ask_L1']
        p_b, q_b = data['LOB0']['Bid_L1']
        _, qD_a = data['LOB0']['Ask_L2']
        _, qD_b = data['LOB0']['Bid_L2']
        pos = data['Positions']
        n_as = get_queue_priority(data, pos, 'Ask_L1')
        n_as = np.append(n_as, get_queue_priority(data, pos, 'Ask_L2'))
        n_as = np.append(n_as, [q_a+qD_a])
        n_bs = get_queue_priority(data, pos, 'Bid_L1')
        n_bs = np.append(n_bs, get_queue_priority(data, pos, 'Bid_L2'))
        n_bs = np.append(n_bs, [q_b+qD_b])
        n_a, n_b = np.min(n_as), np.min(n_bs)
        state = torch.tensor([[self.cash, self.Inventory['INTC'], p_a, p_b, q_a, q_b, qD_a, qD_b, n_a, n_b, (p_a + p_b)*0.5]], dtype=torch.float32)
        t = torch.tensor([[self.current_time]], dtype=torch.float32)
        d, _ = self.model_d(t, state)
        if d:
            u, _ = self.model_u(t,state)
            if u > 0:
                u += 1
                if u > 9:
                    u += 1
            action = int(u.item())
        else:
            action = 12
        return action, size