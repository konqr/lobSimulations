from HJBQVI.DGMTorch_old import *
from HJBQVI.utils import TrainingLogger, ModelManager
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
    def __init__(self, label:str, epoch : int, model_dir: str,  rewardpenalty=1,seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random", Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5, wake_on_MO: bool=True, wake_on_Spread: bool=True , cashlimit=1000000,inventorylimit=5, start_trading_lag=0,
    truncation_enabled=False):
        super().__init__(seed=seed, log_events = log_events, log_to_file = log_to_file, strategy=strategy, Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO, wake_on_Spread=wake_on_Spread , cashlimit=cashlimit,inventorylimit=inventorylimit, start_trading_lag=start_trading_lag,
                         truncation_enabled=truncation_enabled)
        self.resetseed(seed)
        self.rewardpenalty=rewardpenalty
        self.transaction_cost=0
        # Trajectory storage
        self.trajectory_buffer = []
        self.first_action = True
        self.second_action = True
        # model ids
        self.label = label
        self.model_dir = model_dir
        self.epoch = epoch
        ###
        self.model_u = None #PIANet(layer_widths[1], n_layers[1], 11, 10, typeNN = typeNN2)
        self.model_d = None #PIANet(layer_widths[2], n_layers[2], 11, 2, typeNN = typeNN2)
        self.layer_widths = [50, 50, 50]
        self.n_layers= [5,5,5]
        self.include_time = False
        self.alt_state = False
        self.enhance_state = False
        self.device = torch.device('cpu')
        self.buffer = []
        self.init_cash = 2500

    def calculaterewards(self, termination) -> Any:
        penalty = self.rewardpenalty * (self.countInventory()**2)
        self.profit = self.cash - self.statelog[0][1]
        self.updatestatelog()
        deltaPNL = self.statelog[-1][2] - self.statelog[-2][2]
        deltaInv = self.statelog[-1][3]['INTC']*self.statelog[-1][-1]*(1- self.transaction_cost*np.sign(self.statelog[-1][3]['INTC'])) - self.statelog[-2][3]['INTC']*self.statelog[-2][-1]*(1-self.transaction_cost*np.sign(self.statelog[-1][3]['INTC']))
        # if self.istruncated or termination:
        #     deltaPNL += self.countInventory() * self.mid
        # reward shaping
        # if self.istruncated:
        #     penalty += 100
        # if self.last_action != 12:
        #     penalty -= self.rewardpenalty *10 # custom reward for incentivising actions rather than inaction for learning
        # if (not self.alt_state) and (self.last_state.cpu().numpy()[0][8] < self.last_state.cpu().numpy()[0][4] + self.last_state.cpu().numpy()[0][6]) and (self.last_state.cpu().numpy()[0][9] < self.last_state.cpu().numpy()[0][5] + self.last_state.cpu().numpy()[0][7]):
        #     penalty -= self.rewardpenalty *20 # custom reward for double sided quoting
        # if self.alt_state:
        #     if self.two_sided_reward:
        #         if (self.last_state.cpu().numpy()[0][3] <= 1) and (self.last_state.cpu().numpy()[0][4] <= 1):
        #             penalty -= self.rewardpenalty *20 # custom reward for double sided quoting
        #     if self.exploration_bonus:
        #         penalty -= self.visit_counter.get_exploration_bonus(self.last_state.cpu().numpy()[0][1:5], self.last_action)
        return deltaPNL + deltaInv - penalty

    def setupNNs(self, o):
        model_manager = ModelManager(model_dir = self.model_dir, label = self.label)
        self.model_d =PIANet(self.layer_widths[1],self.n_layers[1],23, 2, typeNN='LSTM',  hidden_activation='relu')
        self.model_u = PIANet(self.layer_widths[2], self.n_layers[2], 23, 12, typeNN = 'LSTM',  hidden_activation='relu')
        d = model_manager.load_models( d= self.model_d,u= self.model_u, timestamp= self.label, epoch = self.epoch)
        self.model_u = d['u']
        self.model_d = d['d']
        return

    def store_transition(self, ep, state, action, reward, next_state, done):
        """
        Store transition in trajectory buffer

        :param state: Current state
        :param action: Chosen action (tuple of d and u)
        :param reward: Received reward
        :param next_state: Next state
        :param done: Episode termination flag
        """
        # Unpack action components
        d, u = action
        if d is None: return
        # Store additional trajectory information
        transition = (
            state,       # Current state
            d,           # Decision action
            u,           # Utility action
            reward,      # Reward
            next_state,  # Next state
            int(done)         # Done flag
        )
        self.trajectory_buffer.append((ep, transition))

    def getState(self, data):
        """
        Read and preprocess trading data

        :param data: Input trading data
        :return: Processed state tensor
        """
        time = data['current_time']
        p_a, q_a = data['LOB0']['Ask_L1']
        p_b, q_b = data['LOB0']['Bid_L1']
        self.mid = 0.5*(p_a + p_b)
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
        lambdas = data['current_intensity']
        lambdas_norm = lambdas.flatten()/np.sum(lambdas.flatten())
        past_times = data['past_times']
        if self.Inventory['INTC'] ==0: self.init_cash = self.cash
        skew = (n_a - n_b)/(0.5*(q_a + q_b))
        avgFillPrice = 0
        if self.Inventory['INTC'] != 0 :
            avgFillPrice = (self.cash-self.init_cash)/self.Inventory['INTC']
        bool_mo_bid = np.argmax(lambdas_norm) == 4
        bool_mo_ask = np.argmax(lambdas_norm) == 7
        if self.alt_state:
            state = [[time, self.Inventory['INTC']] ]
            if self.ablation_params.get('spread', True):
                state[0] = state[0] + [p_a - p_b]
            if self.ablation_params.get('n_a', True):
                state[0] = state[0] + [n_a/q_a, n_b/q_b]
            if self.ablation_params.get('lambda', True):
                state[0] = state[0] + list(lambdas_norm)
            if self.ablation_params.get('hist', True):
                state[0] = state[0] + list(past_times.flatten())
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        else:
            state = torch.tensor([[ self.Inventory['INTC'], p_a, p_b, q_a, q_b, qD_a, qD_b, n_a, n_b, (p_a + p_b)*0.5] + list(lambdas.flatten()) + list(past_times.flatten())], dtype=torch.float32).to(self.device)
        if self.enhance_state:
            state = torch.cat([state, torch.tensor([[skew, p_a < avgFillPrice, p_b > avgFillPrice,  bool_mo_bid, bool_mo_ask]], dtype=torch.float32).to(self.device)], 1)
        if self.include_time:
            state = torch.cat([state,torch.tensor([[time]], dtype=torch.float32).to(self.device)], 1)
        return state

    def readData(self, data):
        return self.getState(data)

    def get_action(self, data, **kwargs):
        size = 1
        if self.first_action:
            action = 0 # lo_deep_ask
            self.first_action = False
            return action, (1, action)
        if self.second_action:
            action = 11 # lo_deep_bid
            self.second_action = False
            return action, (1, action)
        if self.breach:
            mo = 4 if self.countInventory() > 0 else 7
            return mo, (1, mo)
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
        n_a, n_b = np.min(n_as)/q_a, np.min(n_bs)/q_b
        state = np.array([[self.cash, self.Inventory['INTC'], p_a, p_b, q_a, q_b, qD_a, qD_b, n_a, n_b, (p_a + p_b)*0.5]], dtype=float)
        t = torch.tensor([[self.current_time]], dtype=torch.float32)/300
        # --- Standardize using known distributions ---
        p = 0.1
        p2 = p/2
        eps = 1e-8
        means = np.array([
            2500.0,   # X ~ N(0, 1000^2)
            0.0,   # Y ~ N(0, 2^2)
            100.0, # ask price ~ Pmid + spread/2
            100.0, # bid price ~ Pmid - spread/2
            (1-p)/p,   # mean of Geom(0.002)
            (1-p)/p,   # mean of Geom(0.002)
            (1-p2)/p2, # mean of Geom(0.0015)
            (1-p2)/p2, # mean of Geom(0.0015)
            0.,   # n_as ~ U(0, q_as) (data dependent, center later)
            0.,  # n_bs ~ U(0, q_bs)
            100.0  # Pmid
        ])

        stds = np.array([
            6.0,
            6.0,
            5.0,
            5.0,
            np.sqrt((1-p)/(p**2)),
            np.sqrt((1-p)/(p**2)),
            np.sqrt((1-p2)/(p2**2)),
            np.sqrt((1-p2)/(p2**2)),
            1.,
            1.,
            5.0
        ])

        features = (state - means) / (stds + eps)
        if 'hawkes' in self.label:
            lambdas = np.log(data['current_intensity'])
            lamb_means = np.array([[1.9164, 2.1349, 3.3938, 3.5451, 1.5297, 0.4841, 0.4841, 1.5341, 3.5451,
                                    3.3938, 2.1349, 1.9164]]).T
            lamb_std = np.array([[0.0000, 0.0000, 0.0000, 0.0000, 0.0395, 0.0000, 0.0000, 0.0470, 0.0000,
                                  0.0000, 0.0000, 0.0000]]).T

            self.buffer.append(lambdas)
            lambdas = (lambdas - lamb_means)/(lamb_std + 1e-6)
            lambdas[lamb_std==0] = 0
            features = np.hstack([features, lambdas.T])
        state = torch.tensor(features.astype('float'), dtype=torch.float32)
        d, _ = self.model_d(t, state)
        if d:
            u, _ = self.model_u(t,state)
            mapping = [2, 3, 8, 9]

            # if u > 0:
            #     u += 1
            #     if u > 9:
            #         u += 1
            action = int(u.item())
            action = mapping[action]
        else:
            action = 12
        u = action
        # if u in [4,7]:
        #     _u = u
        #     if _u == 4: u = 7
        #     elif _u == 7: u = 4
        d = 0 if action == 12 else 1

        # Validation checks
        if int(u) in [1, 3, 8, 10]:
            a = self.actions[int(u)]
            lvl = self.actionsToLevels[a]
            if len(data['Positions'][lvl]) == 0:
                self.last_action = 12
                return 12, (d, 0)

        if int(u) in [5, 6]:
            p_a, q_a = data['LOB0']['Ask_L1']
            p_b, q_b = data['LOB0']['Bid_L1']
            if p_a - p_b < 0.015:
                self.last_action = 12
                return 12, (d, 0)

        if (int(u) == 7) and (self.countInventory() >= self.inventorylimit - 2):
            self.last_action = 12
            return 12, (d, 0)

        if (int(u) == 4) and (self.countInventory() <= 2 - self.inventorylimit):
            self.last_action = 12
            return 12, (d, 0)
        return  u, (d, u)

class ImpulseControlAgentPoisson(ImpulseControlAgent):
    def __init__(self, label:str,  epoch : int, model_dir: str, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random", Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5, wake_on_MO: bool=True, wake_on_Spread: bool=True , cashlimit=1000000, **kwargs):
        super().__init__(label,  epoch , model_dir, seed=seed, log_events = log_events, log_to_file = log_to_file, strategy=strategy, Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO, wake_on_Spread=wake_on_Spread , cashlimit=cashlimit)
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
        # self.load_nets()

    def setupNNs(self,o):
        model_manager = ModelManager(model_dir = self.model_dir, label = self.label)
        self.model_d =PIANet(self.layer_widths[1],self.n_layers[1],11, 2, typeNN='Dense')
        self.model_u = PIANet(self.layer_widths[2], self.n_layers[2], 11, 10, typeNN = 'Dense')
        self.model_d, self.model_u = model_manager.load_models( d= self.model_d,u= self.model_u, timestamp= self.label, epoch = self.epoch)
        return

    def get_action(self, data = None, **kwargs) :
        size = 1
        if self.first_action:
            action = 0 # lo_deep_ask
            self.first_action = False
            return action, (1, action)
        if self.second_action:
            action = 11 # lo_deep_bid
            self.second_action = False
            return action, (1, action)
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
        return action, (d, action)