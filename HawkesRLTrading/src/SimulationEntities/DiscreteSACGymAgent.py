from HawkesRLTrading.src.SimulationEntities.Entity import Entity
from HawkesRLTrading.src.SimulationEntities.TradingAgent import TradingAgent
from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent
from HawkesRLTrading.src.Utils.networks import *
from typing import List, Tuple, Dict, Optional, Any
from abc import abstractmethod, ABC
import numpy as np
import torch
from torch.nn import functional as F

class DiscreteSACGymTradingAgent(GymTradingAgent):
    """
    Implements a SAC agent for a discrete space as described by this paper: https://arxiv.org/pdf/1910.07207
    """
    def __init__(self, seed=1, log_events = True, log_to_file = False, strategy = "SAC", Inventory = None, cash = 5000, action_freq = 0.5, wake_on_MO = True, wake_on_Spread=True, rewardpenalty: float=0.4, cashlimit=1000000, inventorylimit=100000, hyperparameter_config: dict=None, env=None):
        super().__init__(seed, log_events, log_to_file, strategy, Inventory, cash, action_freq, wake_on_MO, wake_on_Spread, cashlimit, inventorylimit)
        self.env=env
        if rewardpenalty is None:
            raise ValueError(f"Rewardpenalty value for agent {self.id} not specified")
        self.rewardpenalty=abs(rewardpenalty)
        self.resetseed(seed=seed)

        #set up hyperparameters
        if hyperparameter_config is None:
            raise ValueError(f"Must pass in hyperparameter configuration values for an SAC agent")
        else:
            hyperparams=["lr", "batch_size", "discount_rate", "learning_rate_decay", "tau", "alpha_0"] #replace with attribute names, note "tau is interpolation factor"
            for param in hyperparams:
                assert hyperparameter_config.get(param) is not None, f"Missing required hyperparameter: {param}"
                setattr(self, param, hyperparameter_config[param])
        
        self.n_actions=12 #how many actions can this agent take
        self.state_dim=3
        self.log_alpha=torch.tensor(np.log(self.alpha_0), requires_grad=True)
        self.entropy_target = 0.98 * (-np.log(1 / self.n_actions))
        #initialize networks
        self.policy_network=PolicyNetwork(state_shape=self.state_dim, n_actions=self.n_actions)
        self.critic_local1=QValueNetwork(state_shape=self.state_dim, n_actions=self.n_actions)
        self.critic_local2=QValueNetwork(state_shape=self.state_dim, n_actions=self.n_actions)
        self.critic_target_1=QValueNetwork(state_shape=self.state_dim, n_actions=self.n_actions)
        self.critic_target_2=QValueNetwork(state_shape=self.state_dim, n_actions=self.n_actions)
        #initialize optimisers
        self.critic_optimiser1=torch.optim.Adam(self.critic_local1.parameters(), lr=self.lr)
        self.critic_optimiser2=torch.optim.Adam(self.critic_local2.parameters(), lr=self.lr)
        self.policy_optimiser=torch.optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.alpha_optimiser=torch.optim.Adam([self.log_alpha], lr=self.lr)
        
        self.update_counter=0
        #do a hard copy
        self.soft_update_target_networks(tau=1)#function to update target networks

    #Training functions
    
    def soft_update_target_networks(self, tau=1):
        """
        Polyak update, when tau=1, it is equivalent to a hard copy/update
        """
        self.soft_update(self.critic_target_1, self.critic_local1, tau)
        self.soft_update(self.critic_target_2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)



    #Loss Functions:
    def critic_loss(self):
        pass

    def actor_loss(self):
        pass

    def temperature_loss(self, )
    #Utility functions


    def get_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            action=np.argmax(self.get_action_probabilities(state=state)) #deterministic
            return action
        else:
            action=np.random.choice(range(self.action_dim), p=self.get_action_probabilities(state=state)) #stochastic
            return action
    
    def get_action_probabilities(self, state):
        state_tensor=torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs=self.policy_network.forward(state_tensor)
        return action_probs.squeeze(0).detach().numpy()


    def calculaterewards(self) -> Any:
        penalty= self.rewardpenalty * self.countInventory()
        self.profit=self.cash - self.statelog[0][1]
        self.updatestatelog()
        deltaPNL=self.statelog[-1][2] - self.statelog[-2][2]
        return deltaPNL - penalty
    
    def resetseed(self, seed):
        return super().resetseed(seed)
    