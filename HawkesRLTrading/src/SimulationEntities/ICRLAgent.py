from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent
from typing import  Tuple, Dict, Optional, Any
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):

    #Takes in observations and outputs actions
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, action_space)

    #forward pass
    def forward(self, x):
        #input states
        x = self.input_layer(x)

        #relu activation
        x = F.relu(x)

        #actions
        actions = self.output_layer(x)

        #get softmax for a probability distribution
        action_probs = F.softmax(actions, dim=1)

        return action_probs

class StateValueNetwork(nn.Module):

    #Takes in state
    def __init__(self, observation_space):
        super(StateValueNetwork, self).__init__()

        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        #input layer
        x = self.input_layer(x)

        #activiation relu
        x = F.relu(x)

        #get state value
        state_value = self.output_layer(x)

        return state_value

class ICRLAgent(GymTradingAgent):
    def __init__(self, data0, env, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random", Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5, wake_on_MO: bool=True, wake_on_Spread: bool=True , cashlimit=1000000):
        '''
        Timing is Everything Mguni et al. implementation
        :param data0:
        :param env:
        :param seed:
        :param log_events:
        :param log_to_file:
        :param strategy:
        :param Inventory:
        :param cash:
        :param action_freq:
        :param wake_on_MO:
        :param wake_on_Spread:
        :param cashlimit:
        '''
        super().__init__(seed=seed, log_events = log_events, log_to_file = log_to_file, strategy=strategy, Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO, wake_on_Spread=wake_on_Spread , cashlimit=cashlimit)
        self.resetseed(seed)
        ###############
        self.eta =  100 # inventory penalty
        ###############
        self.Actor_d = PolicyNetwork(data0.shape, 2)
        self.Crtic_d = StateValueNetwork(data0.shape)
        self.Actor_u = PolicyNetwork(data0.shape, 10)
        self.Critic_u = StateValueNetwork(data0.shape)
        ###############

    def get_action(self, data):
        d = self.Actor_d(data)
        u = self.Actor_u(data)
        return d, u

    def train(self, data):
        #https://dilithjay.com/blog/actor-critic-methods
        return


