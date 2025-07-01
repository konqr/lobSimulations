from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent
from HawkesRLTrading.src.SimulationEntities.ICRLAgent import get_queue_priority
from typing import Dict, Optional, Any
import numpy as np

class ProbabilisticAgent(GymTradingAgent):
    def __init__(self, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random",
                 Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5,
                 wake_on_MO: bool=True, wake_on_Spread: bool=True, cashlimit=1000000,inventorylimit=100,
                 rewardpenalty = 0.1, transaction_cost=0, start_trading_lag=0):
        """
        Deals w probabilities of next MO
        """
        super().__init__(seed=seed, log_events=log_events, log_to_file=log_to_file, strategy=strategy,
                         Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO,
                         wake_on_Spread=wake_on_Spread, cashlimit=cashlimit, inventorylimit=inventorylimit, start_trading_lag=start_trading_lag,
                         truncation_enabled=False)

        self.resetseed(seed)

        self.rewardpenalty = rewardpenalty  # inventory penalty
        self.last_state, self.last_action = None, None
        self.inv_threshold = 10
        self.avgPrices = (0,0)
        self.init_cash = self.cash
        self.cols= ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                    "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
        self.trajectory_buffer = []
        self.transaction_cost = transaction_cost

    def readData(self, data):
        """
        Read and preprocess trading data

        :param data: Input trading data
        :return: Processed state tensor
        """
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
        past_times = data['past_times']
        state = [[self.cash, self.Inventory['INTC'], p_a, p_b, q_a, q_b, qD_a, qD_b, n_a, n_b, (p_a + p_b)*0.5] , lambdas.flatten(), past_times.flatten()]
        return state
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
        # d, u = action

        # Store additional trajectory information
        transition = (
            state,       # Current state
            action[0],           # Decision action
            action[1:],           # Utility action
            reward,      # Reward
            next_state,  # Next state
            int(done)         # Done flag
        )
        self.trajectory_buffer.append((ep, transition))
    def get_action(self, data, epsilon=0.1):
        """
        Epsilon-greedy policy to balance exploitation with exploration.

        Args:
            data: The input data
            epsilon: Probability of choosing a random action (default: 0.1)

        Returns:
            action, size, logits_d, logits_u
        """
        size = 1
        origData = data.copy()
        data = self.readData(data)

        # Store current state for learning
        self.last_state = data

        lambdas = data[1]
        lambdas_norm = lambdas/np.sum(lambdas)
        skew = (data[0][-3] - data[0][-2])/(0.5*(data[0][4] + data[0][5])) # (na - nb) / mean(qa, qb)
        inv = data[0][1]
        pa, pb = data[0][2], data[0][3]
        actions = []
        if np.argmax(lambdas_norm) == 4: # mo ask
            if inv < -self.inv_threshold:
                # if lambdas[7] + 1000 < lambdas[4]:
                    return ((12,size),(7,size)) # mo_bid
            elif (inv > self.inv_threshold) or ( (inv > 0)and(pa > (self.cash-self.init_cash)/inv) ): # goodfill
                if data[0][-3]/data[0][4] > 1:
                    actions.append((9,size)) # lo bid top
                if data[0][-2]/data[0][5]<1:
                    actions.append((2,size))
            elif ( (inv > 0)and(pa < (self.cash-self.init_cash)/inv) ): #badfill
                if len(origData['Positions']['Ask_L1']) != 0: actions.append((3, size))
                if data[0][-3]/data[0][4] > 1:
                    actions.append((9,size)) # lo bid top
                if data[0][-2]/data[0][5]<1:
                    actions.append((2,size))
        elif np.argmax(lambdas_norm) == 7: # mo bid
            if inv > self.inv_threshold:
                # if lambdas[4] + 1000 < lambdas[7]:
                    return ((12,size),(4,size)) # mo_ask
            elif (inv < -self.inv_threshold) or ( (inv < 0)and(pb < (self.cash-self.init_cash)/inv) ): # goodfill
                if data[0][-3]/data[0][4] < 1:
                    actions.append((9,size)) # lo bid top
                if data[0][-2]/data[0][5]>1:
                    actions.append((2,size))
            elif ( (inv < 0)and(pb > (self.cash-self.init_cash)/inv) ): #badfill
                if len(origData['Positions']['Bid_L1']) != 0: actions.append((8, size))
                if data[0][-3]/data[0][4] < 1:
                    actions.append((9,size)) # lo bid top
                if data[0][-2]/data[0][5] > 1:
                    actions.append((2,size))
        else:
            if skew > 0.5:
                if (data[0][4] - data[0][-2])/(0.5*(data[0][4] + data[0][5])) < skew:
                    if len(origData['Positions']['Ask_L2']) != 0: actions.append((1,size))
                    actions.append((2,size))
                if (data[0][-3] - data[0][5])/(0.5*(data[0][4] + data[0][5])) < skew:
                    if len(origData['Positions']['Bid_L2']) != 0: actions.append((11,size))
                    actions.append((9,size))
            if skew < -0.5:
                if (data[0][4] - data[0][-2])/(0.5*(data[0][4] + data[0][5])) > skew:
                    if len(origData['Positions']['Ask_L2']) != 0: actions.append((1,size))
                    actions.append((2,size))
                if (data[0][-3] - data[0][5])/(0.5*(data[0][4] + data[0][5])) > skew:
                    if len(origData['Positions']['Bid_L2']) != 0: actions.append((11,size))
                    actions.append((9,size))
        if len(actions) == 0:
            if len(np.concatenate(list(origData['Positions'].values()))) == 0:
                return ((2,size), (9,size))
            else:
                return ((12,size),(12,size))
        if len(actions) == 1:
            actions.append((12,size))
        return actions

    def calculaterewards(self, termination) -> Any:
        penalty = self.rewardpenalty * (self.countInventory()**2)
        self.profit = self.cash - self.statelog[0][1]
        self.updatestatelog()
        deltaPNL = self.statelog[-1][2] - self.statelog[-2][2]
        deltaInv = self.statelog[-1][3]['INTC']*self.statelog[-1][-1]*(1- self.transaction_cost*np.sign(self.statelog[-1][3]['INTC'])) - self.statelog[-2][3]['INTC']*self.statelog[-2][-1]*(1-self.transaction_cost*np.sign(self.statelog[-1][3]['INTC']))
        return deltaPNL + deltaInv - penalty
