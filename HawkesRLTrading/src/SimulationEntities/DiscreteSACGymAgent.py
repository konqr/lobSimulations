from HawkesRLTrading.src.SimulationEntities.Entity import Entity
from HawkesRLTrading.src.SimulationEntities.TradingAgent import TradingAgent
from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent
from HawkesRLTrading.src.Utils.networks import *
from typing import List, Tuple, Dict, Optional, Any
from abc import abstractmethod, ABC
import numpy as np
import torch
from torch.nn import functional as F
from HawkesRLTrading.src.Utils.ReplayBuffer import ReplayBuffer

class DiscreteSACGymTradingAgent(GymTradingAgent):
    """
    Implements a SAC agent for a discrete space as described by this paper: https://arxiv.org/pdf/1910.07207
    """
    def __init__(self, seed=1, log_events = True, log_to_file = False, strategy = "SAC", Inventory = None, cash = 5000, action_freq = 0.5, wake_on_MO = True, wake_on_Spread=True, rewardpenalty: float=0.4, cashlimit=1000000, inventorylimit=100000, hyperparameter_config: dict=None):
        super().__init__(seed, log_events, log_to_file, strategy, Inventory, cash, action_freq, wake_on_MO, wake_on_Spread, cashlimit, inventorylimit)
        if rewardpenalty is None:
            raise ValueError(f"Rewardpenalty value for agent {self.id} not specified")
        self.rewardpenalty=abs(rewardpenalty)
        self.resetseed(seed=seed)

        #set up hyperparameters
        if hyperparameter_config is None:
            raise ValueError(f"Must pass in hyperparameter configuration values for an SAC agent")
        else:
            hyperparams=["lr", "minibatch_size", "discount_rate", "tau", "alpha_0"] #replace with attribute names, note "tau is interpolation factor"
            for param in hyperparams:
                assert hyperparameter_config.get(param) is not None, f"Missing required hyperparameter: {param}"
                setattr(self, param, hyperparameter_config[param])
        
        self.n_actions=12 #how many actions can this agent take
        self.state_dim=3
        self.log_alpha=torch.tensor(np.log(self.alpha_0), requires_grad=True)
        self.alpha=self.log_alpha.exp()
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


        #Initialize Weights and Replay buffer
        self.replaybuffer=ReplayBuffer(obs_shape=self.state_dim, action_shape=self.n_actions)

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

    def learn(self,transition, log_interval=100, log_name="DiscreteSAC"):
        #Transition is a tuple of (state, action, rewards. next_state, done)
        self.replaybuffer.add_sample(transition=transition)
        if self.replaybuffer.count<self.minibatch_size:
            return
        else:
            self.critic_optimiser1.zero_grad()
            self.critic_optimiser2.zero_grad()
            self.policy_optimiser.zero_grad()
            self.alpha_optimiser.zero_grad()
            minibatch=self.replaybuffer.sample_minibatch(batchsize=self.minibatch_size)
            minibatch_separated=list(map(list, zip(*minibatch)))
            state_tensor = torch.tensor(np.array(minibatch_separated[0]))
            action_tensor = torch.tensor(np.array(minibatch_separated[1]))
            rewards_tensor = torch.tensor(np.array(minibatch_separated[2])).float()
            next_states_tensor = torch.tensor(np.array(minibatch_separated[3]))
            done_tensor = torch.tensor(np.array(minibatch_separated[4]))
            critic_loss1, critic_loss2=self.critic_loss(state_tensor, action_tensor, rewards_tensor, next_states_tensor, done_tensor)

            critic_loss1.backward()
            critic_loss2.backward()
            self.critic_optimiser1.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_probs = self.actor_loss(state_tensor)

            actor_loss.backward()
            self.policy_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probs)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks()



    #Loss Functions:
    def critic_loss(self, state, action, rewards, next_state, done):
        state_tensor=torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor=torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        reward_tensor=torch.tensor(rewards, dtype=torch.float32).unsqueeze(0)
        next_state_tensor=torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        done_tensor=torch.tensor(done, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            actionprobs=self.get_action_probabilities(state=state)
            logactionprobs=self.log_action_probabilities(state=state)
            next_q_1=self.critic_target_1.forward(next_state_tensor)
            next_q_2=self.critic_target_2.forward(next_state_tensor)
            soft_state_values=(actionprobs * (torch.min(next_q_1, next_q_2) - self.alpha * logactionprobs
            )).sum(dim=1)
            next_q_values = reward_tensor + torch.logical_not(done_tensor) * self.discount_rate *soft_state_values
        soft_q_1=self.critic_local1(state_tensor).gather(1, action_tensor.unsqueeze(-1)).squeeze(-1)
        soft_q_2=self.critic_local2(state_tensor).gather(1, action_tensor.unsqueeze(-1)).squeeze(-1)
        critic1_square_error = torch.nn.MSELoss(reduction="none")(soft_q_1, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q_2, next_q_values)
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic1_square_error, critic2_square_error)]
        self.replay_buffer.update_weights(weight_update)
        critic_loss = critic1_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss

    def actor_loss(self, state):
        actionprobs=self.get_action_probabilities(state=state)
        logactionprobs=self.log_action_probabilities(state=state)
        state_tensor=torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_local1=self.critic_local1(state_tensor)
        q_local2=self.critic_local2(state_tensor)
        loss=(actionprobs*(self.alpha*logactionprobs - torch.min(q_local1, q_local2))).sum(dim=1).mean()
        return loss

    def temperature_loss(self, log_action_probs):
        alpha_loss = -(self.log_alpha * (log_action_probs + self.target_entropy).detach()).mean()
        return alpha_loss

    #Utility functions


    def get_action(self, state, evaluation_episode=False):
        size=1
        if evaluation_episode:
            action=np.argmax(self.get_action_probabilities(state=state)) #deterministic
            return action, size
        else:
            action=np.random.choice(range(self.n_actions), p=self.get_action_probabilities(state=state)) #stochastic
            return action, size
    
    def get_action_probabilities(self, state):
        state_tensor=torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs=self.policy_network.forward(state_tensor)
        print(f"action_probs{action_probs}")
        return action_probs.squeeze(0).detach().numpy()

    def log_action_probabilities(self, state):
        actionprobs=self.get_action_probabilities(state=state)
        z= actionprobs==0
        z=z.float()*1e-8
        return torch.log(actionprobs+z)

    def calculaterewards(self) -> Any:
        penalty= self.rewardpenalty * self.countInventory()
        self.profit=self.cash - self.statelog[0][1]
        self.updatestatelog()
        deltaPNL=self.statelog[-1][2] - self.statelog[-2][2]
        return deltaPNL - penalty
    
    def resetseed(self, seed):
        return super().resetseed(seed)
    