import copy
from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent
from HJBQVI.DGMTorch import MLP, ActorCriticMLP, ActorCriticSGMLP, ActorCriticSeparate
from HJBQVI.utils import MinMaxScaler
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, defaultdict
import random
import gc

def get_queue_priority(data, pos, label):
    data = data.copy()
    ask_l1s = pos[label]
    n_as = []
    if len(ask_l1s):
        idxs = np.array([])
        for o in ask_l1s:
            idxs = np.append(idxs, np.where(np.array(data['lobL3'][label][1]) == o)).astype(int)
        n_as = np.cumsum(data['lobL3_sizes'][label][1])[idxs-1]
    return n_as

# class ICRLAgent(GymTradingAgent):
#     def __init__(self, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random", Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5, wake_on_MO: bool=True, wake_on_Spread: bool=True , cashlimit=1000000):
#         '''
#         Timing is Everything Mguni et al. implementation
#         :param data0:
#         :param seed:
#         :param log_events:
#         :param log_to_file:
#         :param strategy:
#         :param Inventory:
#         :param cash:
#         :param action_freq:
#         :param wake_on_MO:
#         :param wake_on_Spread:
#         :param cashlimit:
#         '''
#         super().__init__(seed=seed, log_events = log_events, log_to_file = log_to_file, strategy=strategy, Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO, wake_on_Spread=wake_on_Spread , cashlimit=cashlimit)
#         self.resetseed(seed)
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.device=device
#         ###############
#         self.rewardpenalty =  10 # inventory penalty
#         self.lr = 1e-2
#         ###############
#         self.experience_replay = []
#         self.mmscaler = MinMaxScaler()
#         self.mid = 0
#         torch.autograd.set_detect_anomaly(True)
#
#     def readData(self, data):
#         p_a, q_a = data['LOB0']['Ask_L1']
#         p_b, q_b = data['LOB0']['Bid_L1']
#         self.mid = 0.5*(p_a + p_b)
#         _, qD_a = data['LOB0']['Ask_L2']
#         _, qD_b = data['LOB0']['Bid_L2']
#         pos = data['Positions']
#         n_as = get_queue_priority(data, pos, 'Ask_L1')
#         n_as = np.append(n_as, get_queue_priority(data, pos, 'Ask_L2'))
#         n_as = np.append(n_as, [q_a+qD_a])
#         n_bs = get_queue_priority(data, pos, 'Bid_L1')
#         n_bs = np.append(n_bs, get_queue_priority(data, pos, 'Bid_L2'))
#         n_bs = np.append(n_bs, [q_b+qD_b])
#         n_a, n_b = np.min(n_as), np.min(n_bs)
#         lambdas = data['current_intensity']
#         state = torch.tensor([[self.cash, self.Inventory['INTC'], p_a, p_b, q_a, q_b, qD_a, qD_b, n_a, n_b, (p_a + p_b)*0.5] + list(lambdas.flatten())], dtype=torch.float32)
#         return state
#
#     def getState(self, state):
#         if type(state) == dict:
#             state = self.readData(state)
#         # if len(self.experience_replay):
#         #     self.mmscaler.fit(torch.tensor([i[0].numpy() for i in self.experience_replay]))
#         #     state = self.mmscaler.transform(state)
#         # else:
#         #     state = self.mmscaler.fit_transform(state)
#         return state
#
#     def setupNNs(self, data0):
#         ###############
#         data0 = self.getState(data0)
#         self.Actor_d = MLP(len(data0[0]),10,5, 2, hidden_activation='relu', final_activation='softmax')
#         self.Critic_d = MLP(len(data0[0]),10, 5,  1,hidden_activation='relu')
#         self.Actor_u = MLP(len(data0[0]),10, 5, 12, hidden_activation='relu', final_activation='softmax')
#         self.Critic_u = MLP(len(data0[0]),10, 5,1, hidden_activation='relu')
#
#     def get_action(self, data):
#         size = 1
#         origData = data.copy()
#         data = self.getState(data)
#         d, logits_d = self.Actor_d(data)
#         if d:
#             u, logits_u = self.Actor_u(data)
#             if int(u.item()) in [1,3,8,10]: # cancels
#                 a = self.actions[int(u.item())]
#                 lvl = self.actionsToLevels[a]
#                 if len(origData['Positions'][lvl]) == 0: #no position to cancel
#                     return 12, size, logits_d, []
#             if int(u.item()) in [5,6]:
#                 p_a, q_a = origData['LOB0']['Ask_L1']
#                 p_b, q_b = origData['LOB0']['Bid_L1']
#                 if p_a - p_b < 0.015: # reject if inspread not possible
#                     return 12, size, logits_d, []
#             return int(u.item()), size, logits_d, logits_u
#         else:
#             return 12, size, logits_d, []
#
#     def setupTraining(self, net):
#         def lr_lambda(epoch):
#             # Calculate decay rate
#             #decay_rate = np.log(1e-4) / (self.EPOCHS*phi_epochs - 1)
#             #return np.max([1e-5,np.exp(decay_rate * epoch )])
#             #return (1 - 1e-5)**epoch
#             return np.max([1e-4,0.1**(epoch//1000)])
#         optimizer = optim.Adam(net.parameters(), lr=self.lr)
#         scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#         return optimizer, scheduler
#
#     def appendER(self, sars):
#         self.experience_replay.append(sars)
#
#     def train(self, num= 0):
#         #https://dilithjay.com/blog/actor-critic-methods
#         if torch.cuda.device_count() > 1:
#             print("Let's use", torch.cuda.device_count(), "GPUs!")
#             self.Actor_d = nn.DataParallel(self.Actor_d)
#             self.Actor_u = nn.DataParallel(self.Actor_u)
#             self.Critic_d = nn.DataParallel(self.Critic_d)
#             self.Critic_u = nn.DataParallel(self.Critic_u)
#         self.Actor_d.to(self.device)
#         self.Actor_u.to(self.device)
#         self.Critic_u.to(self.device)
#         self.Critic_d.to(self.device)
#         optim_Ad, sched_Ad = self.setupTraining(self.Actor_d)
#         optim_Cd, sched_Cd = self.setupTraining(self.Critic_d)
#         optim_Au, sched_Au = self.setupTraining(self.Actor_u)
#         optim_Cu, sched_Cu = self.setupTraining(self.Critic_u)
#         if num==0: num = len(self.experience_replay)
#         for s, a, r, s_, done in self.experience_replay[-num:]:
#             s = self.getState(s.clone())
#             s_ = self.getState(s_.clone())
#             u, _, logits_d, logits_u = a
#             d = int(u != 12)
#             td_error_d = r + 0.99*self.Critic_d(s_)*(1-done) - self.Critic_d(s)
#             actor_loss_d = torch.log(logits_d[0][d])*td_error_d
#             if np.isnan(actor_loss_d.item()):
#                 print('dddd#')
#             critic_loss_d = torch.square(td_error_d)
#             for o, l, sc in [(optim_Ad, actor_loss_d, sched_Ad), (optim_Cd, critic_loss_d, sched_Cd)]:
#                 o.zero_grad()
#                 l.mean().backward(retain_graph=True)
#                 o.step()
#                 sc.step()
#             print(f'd Losses : Actor_d :{actor_loss_d.item():0.4f},Critic_d :{critic_loss_d.item():0.4f} ')
#             if d:
#                 td_error_u = r + self.Critic_u(s_)*(1-done) - self.Critic_u(s)
#                 actor_loss_u = torch.log(logits_u[0][u])*td_error_u
#                 critic_loss_u = torch.square(td_error_u)
#                 for o, l, sc in [(optim_Au, actor_loss_u, sched_Au), (optim_Cu, critic_loss_u, sched_Cu)]:
#                     o.zero_grad()
#                     l.mean().backward(retain_graph=True)
#                     o.step()
#                     sc.step()
#                 print(f'u Losses : Actor_u :{actor_loss_u.item():0.4f},Critic_u :{critic_loss_u.item():0.4f} ')
#
#     def calculaterewards(self, termination) -> Any:
#         penalty= self.rewardpenalty * (self.countInventory()**2)
#         self.profit=self.cash - self.statelog[0][1]
#         self.updatestatelog()
#         deltaPNL=self.statelog[-1][2] - self.statelog[-2][2]
#         if self.istruncated or termination:
#             deltaPNL += self.countInventory()*self.mid
#         return deltaPNL - penalty


'''
         For discrete actions
            logits = self.actor(states)
            probs = F.softmax(logits, dim=-1)
            
            # Compute advantage
            values = self.critic(states)
            
            # Use one-hot encoding for actions
            actions_onehot = F.one_hot(actions.long().squeeze(-1), num_classes=logits.size(-1)).float()
            
            # Compute log probabilities
            log_probs = torch.log(probs + 1e-10) * actions_onehot
            log_probs = log_probs.sum(dim=-1, keepdim=True)
            
            # Detach values to avoid backprop through critic
            advantages = (rewards + (1 - dones) * self.gamma * self.critic_target(next_states) - values).detach()
            
            # Policy gradient loss
            actor_loss = -(log_probs * advantages).mean()
        '''

class ICRLAgent(GymTradingAgent):
    def __init__(self, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random",
                 Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5,
                 wake_on_MO: bool=True, wake_on_Spread: bool=True, cashlimit=1000000,
                 buffer_capacity=10000, batch_size=64, tau=0.005):
        '''
        Timing is Everything Mguni et al. implementation with experience replay and target networks
        :param data0:
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
        :param buffer_capacity: Maximum size of replay buffer
        :param batch_size: Size of batch for training
        :param tau: Soft update parameter for target networks
        '''
        super().__init__(seed=seed, log_events=log_events, log_to_file=log_to_file, strategy=strategy,
                         Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO,
                         wake_on_Spread=wake_on_Spread, cashlimit=cashlimit)

        self.resetseed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        # Hyperparameters
        self.rewardpenalty = 0.1  # inventory penalty
        self.lr = 1e-3
        self.gamma = 0.99  # discount factor
        self.entropy_coef = 1000
        # Experience replay parameters
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.tau = tau  # for soft update of target networks

        # Initialize replay buffer as deque with fixed capacity
        self.replay_buffer = deque(maxlen=buffer_capacity)

        self.mmscaler = MinMaxScaler()
        self.mid = 0

        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)

    def readData(self, data):
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
        state = torch.tensor([[self.cash, self.Inventory['INTC'], p_a, p_b, q_a, q_b, qD_a, qD_b, n_a, n_b, (p_a + p_b)*0.5] + list(lambdas.flatten())], dtype=torch.float32)
        return state

    def getState(self, state):
        if type(state) == dict:
            state = self.readData(state)
        # Commented out scaling for now - can be uncommented if needed
        if len(self.replay_buffer) > 10:
            self.mmscaler.fit(torch.cat([i for i in self.replay_buffer]))
            state = self.mmscaler.transform(state)
        else:
            state = self.mmscaler.fit_transform(state)
        return state

    def setupNNs(self, data0):
        # Get state dimensions
        data0 = self.getState(data0)
        state_dim = len(data0[0])

        # Initialize main networks
        self.Actor_d = MLP(state_dim, 128, 3, 2, hidden_activation='relu', final_activation='softmax')
        self.Critic_d = MLP(state_dim, 128, 3, 1, hidden_activation='relu')
        self.Actor_u = MLP(state_dim, 128, 3, 12, hidden_activation='relu', final_activation='softmax')
        self.Critic_u = MLP(state_dim, 128, 3, 1, hidden_activation='relu')

        # Initialize target networks with same weights
        self.Actor_d_target = MLP(state_dim, 128, 3, 2, hidden_activation='relu', final_activation='softmax')
        self.Critic_d_target = MLP(state_dim, 128, 3, 1, hidden_activation='relu')
        self.Actor_u_target = MLP(state_dim, 128, 3, 12, hidden_activation='relu', final_activation='softmax')
        self.Critic_u_target = MLP(state_dim, 128, 3, 1, hidden_activation='relu')

        # Move all models to appropriate device
        self.Actor_d.to(self.device)
        self.Critic_d.to(self.device)
        self.Actor_u.to(self.device)
        self.Critic_u.to(self.device)
        self.Actor_d_target.to(self.device)
        self.Critic_d_target.to(self.device)
        self.Actor_u_target.to(self.device)
        self.Critic_u_target.to(self.device)

        # Copy parameters from main networks to target networks
        self._hard_update(self.Actor_d, self.Actor_d_target)
        self._hard_update(self.Critic_d, self.Critic_d_target)
        self._hard_update(self.Actor_u, self.Actor_u_target)
        self._hard_update(self.Critic_u, self.Critic_u_target)

        # Setup optimizers
        self.optimizer_actor_d, self.scheduler_actor_d = self.setupTraining(self.Actor_d)
        self.optimizer_critic_d, self.scheduler_critic_d = self.setupTraining(self.Critic_d)
        self.optimizer_actor_u, self.scheduler_actor_u = self.setupTraining(self.Actor_u)
        self.optimizer_critic_u, self.scheduler_critic_u = self.setupTraining(self.Critic_u)


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
        data = self.getState(data)

        # Store current state for learning
        self.last_state = data
        self.replay_buffer.append(self.readData(origData))
        # Generate random number to decide between exploration and exploitation
        if (random.random() < epsilon) or (len(self.replay_buffer) < 10):
            print('RANDOM ACTION')
            # EXPLORATION: Choose a random action
            # Get all valid actions in the current state
            valid_actions = []

            _, logits_d = self.Actor_d(data)


            # Check which actions are valid
            potential_actions = list(range(12))  # Actions 0-11

            # Filter out invalid actions based on state conditions
            for action in potential_actions:
                # Cancel orders check
                if action in [1, 3, 8, 10] and len(origData['Positions'][self.actionsToLevels[self.actions[action]]]) == 0:
                    continue

                # Inspread checks
                if action in [5, 6]:
                    p_a, q_a = origData['LOB0']['Ask_L1']
                    p_b, q_b = origData['LOB0']['Bid_L1']
                    if p_a - p_b < 0.015:  # reject if inspread not possible
                        continue

                # Ask market order check
                if action == 4 and self.countInventory() < 1:
                    continue

                valid_actions.append(action)

            # Add the "do nothing" action as always valid
            valid_actions.append(12)

            # Choose a random valid action
            random_action = random.choice(valid_actions)

            # Calculate logits just for logging/learning purposes
            if random_action != 12:
                _, logits_u = self.Actor_u(data)
                self.last_action_probs = (logits_d, logits_u)
            else:
                self.last_action_probs = (logits_d, torch.tensor([0,0]))

            self.last_action = random_action
            return random_action, size, logits_d, self.last_action_probs[1]

        else:
            # EXPLOITATION: Use the original policy logic
            d, logits_d = self.Actor_d(data)
            if d:
                u, logits_u = self.Actor_u(data)
                if int(u.item()) in [1, 3, 8, 10]:  # cancels
                    a = self.actions[int(u.item())]
                    lvl = self.actionsToLevels[a]
                    if len(origData['Positions'][lvl]) == 0:  # no position to cancel
                        self.last_action = 12
                        self.last_action_probs = (logits_d, torch.tensor([0,0]))
                        return 12, size, logits_d, torch.tensor([0,0])
                if int(u.item()) in [5, 6]:
                    p_a, q_a = origData['LOB0']['Ask_L1']
                    p_b, q_b = origData['LOB0']['Bid_L1']
                    if p_a - p_b < 0.015:  # reject if inspread not possible
                        self.last_action = 12
                        self.last_action_probs = (logits_d, torch.tensor([0,0]))
                        return 12, size, logits_d, torch.tensor([0,0])
                if (int(u.item()) == 4) and (self.countInventory() < 1): # ask mo
                    self.last_action = 12
                    self.last_action_probs = (logits_d, torch.tensor([0,0]))
                    return 12, size, logits_d, torch.tensor([0,0])

                self.last_action = int(u.item())
                self.last_action_probs = (logits_d, logits_u)
                return int(u.item()), size, logits_d, logits_u
            else:
                self.last_action = 12
                self.last_action_probs = (logits_d, torch.tensor([0,0]))
                return 12, size, logits_d, torch.tensor([0,0])

    def setupTraining(self, net):
        def lr_lambda(epoch):
            # Learning rate schedule
            return np.max([1e-4, 0.1**(epoch//200)])

        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def store_transition(self, s, a, r, s_, done):
        """Store transition in replay buffer"""
        self.replay_buffer.append((s, a, r, s_, done))

    def _hard_update(self, source, target):
        """Hard update: target <- source"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def _soft_update(self, source, target):
        """Soft update: target <- tau*source + (1-tau)*target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def sample_batch(self):
        """Sample a batch of experiences from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            # If not enough samples, return None
            return None

        # Sample random batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        batch = [torch.cat([i[0] for i in batch]).to(self.device),
                 torch.FloatTensor([i[1][0] for i in batch]).to(self.device).unsqueeze(1),
                 torch.FloatTensor([i[2] for i in batch]).to(self.device).unsqueeze(1),
                 torch.cat([i[3] for i in batch]).to(self.device),
                 torch.FloatTensor([i[4] for i in batch]).to(self.device).unsqueeze(1) ]
        return batch

    def train(self):
        """Train the agent using experience replay and target networks with batch processing"""
        # Move networks to appropriate device
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.Actor_d = nn.DataParallel(self.Actor_d)
            self.Actor_u = nn.DataParallel(self.Actor_u)
            self.Critic_d = nn.DataParallel(self.Critic_d)
            self.Critic_u = nn.DataParallel(self.Critic_u)
            self.Actor_d_target = nn.DataParallel(self.Actor_d_target)
            self.Actor_u_target = nn.DataParallel(self.Actor_u_target)
            self.Critic_d_target = nn.DataParallel(self.Critic_d_target)
            self.Critic_u_target = nn.DataParallel(self.Critic_u_target)

        # Move networks to device
        self.Actor_d.to(self.device)
        self.Actor_u.to(self.device)
        self.Critic_d.to(self.device)
        self.Critic_u.to(self.device)
        self.Actor_d_target.to(self.device)
        self.Actor_u_target.to(self.device)
        self.Critic_d_target.to(self.device)
        self.Critic_u_target.to(self.device)

        # Setup optimizers
        optim_Ad, sched_Ad = self.setupTraining(self.Actor_d)
        optim_Cd, sched_Cd = self.setupTraining(self.Critic_d)
        optim_Au, sched_Au = self.setupTraining(self.Actor_u)
        optim_Cu, sched_Cu = self.setupTraining(self.Critic_u)

        # Sample a batch from replay buffer
        batch = self.sample_batch()
        if batch is None:
            print("Not enough samples in replay buffer for training")
            return

        # Process batch all at once
        states, actions, rewards, next_states, dones = batch

        # Extract action components and convert to tensors
        u_values = torch.tensor([a[0] for a in actions], dtype=torch.long).to(self.device)
        logits_d_batch = torch.stack([a[2] for a in actions]).to(self.device)


        # Create decision masks (d=1 when u!=12, d=0 when u==12)
        d_values = (u_values != 12).long().to(self.device)
        d_mask = d_values.bool()  # Mask for utility network updates

        # Process states
        states = torch.cat([self.getState(s.clone()) for s in states]).to(self.device)
        next_states = torch.cat([self.getState(s_.clone()) for s_ in next_states]).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # ---- Train discriminator (d) networks ----
        # Use target network for computing target values
        with torch.no_grad():
            next_values_d = self.Critic_d_target(next_states)
            target_values_d = rewards + self.gamma * next_values_d * (1 - dones)

        # Current value estimates
        current_values_d = self.Critic_d(states)

        # Compute TD errors and losses
        critic_loss_d = F.mse_loss(current_values_d, target_values_d)
        # Update critic_d
        optim_Cd.zero_grad()
        critic_loss_d.backward(retain_graph=True)
        optim_Cd.step()

        # For actor loss, we need to extract the log probabilities of the chosen actions
        td_errors_d = (rewards + self.gamma*(1-dones)*self.Critic_d_target(next_states) - self.Critic_d(states)).detach()
        log_probs_d = torch.log(torch.stack([logits_d[0][d] for logits_d, d in zip(logits_d_batch.clone(), d_values)]) + 1e-10)
        actor_loss_d = -torch.mean(log_probs_d * td_errors_d.squeeze())

        # Update actor_d
        optim_Ad.zero_grad()
        actor_loss_d.backward(retain_graph=True)
        optim_Ad.step()

        # ---- Train utility (u) networks ----
        # Only update if we have at least one decision to act (d=1)
        if d_mask.any():
            logits_u_batch = torch.stack([a[3] for a in actions]).to(self.device)
            # Filter states, next_states, rewards, dones for only those where d=1
            states_u = states[d_mask]
            next_states_u = next_states[d_mask]
            rewards_u = rewards[d_mask]
            dones_u = dones[d_mask]
            u_values_u = u_values[d_mask]
            logits_u_batch_u = logits_u_batch[d_mask]

            # Use target network for computing target values
            with torch.no_grad():
                next_values_u = self.Critic_u_target(next_states_u)
                target_values_u = rewards_u + self.gamma * next_values_u * (1 - dones_u)

            # Current value estimates
            current_values_u = self.Critic_u(states_u)

            # Compute TD errors and losses
            td_errors_u = target_values_u - current_values_u
            critic_loss_u = torch.mean(torch.square(td_errors_u))

            # For actor loss, extract log probabilities of chosen utility actions
            log_probs_u = torch.log(torch.stack([logits_u[0][u] for logits_u, u in zip(logits_u_batch_u, u_values_u)]))
            actor_loss_u = -torch.mean(log_probs_u * td_errors_u.detach().squeeze())

            # Update critic_u
            optim_Cu.zero_grad()
            critic_loss_u.backward(retain_graph=True)
            optim_Cu.step()

            # Update actor_u
            optim_Au.zero_grad()
            actor_loss_u.backward(retain_graph=True)
            optim_Au.step()

            # Print average losses for utility networks
            print(f'u Losses: Actor_u: {actor_loss_u.item():0.4f}, Critic_u: {critic_loss_u.item():0.4f}')
        else:
            print("No utility actions in this batch")

        # Step schedulers
        sched_Ad.step()
        sched_Cd.step()
        sched_Au.step()
        sched_Cu.step()

        # Soft update target networks
        self._soft_update(self.Actor_d, self.Actor_d_target)
        self._soft_update(self.Critic_d, self.Critic_d_target)
        self._soft_update(self.Actor_u, self.Actor_u_target)
        self._soft_update(self.Critic_u, self.Critic_u_target)

        # Print average losses for discriminator networks
        print(f'd Losses: Actor_d: {actor_loss_d.item():0.4f}, Critic_d: {critic_loss_d.item():0.4f}')

    def calculate_entropy(self, probs):
        """Calculate the entropy of a probability distribution"""
        # Add small epsilon to avoid log(0)
        eps = 1e-3
        return -torch.sum((probs + eps)* torch.log(probs+1e-20))

    def learn(self, state, reward, next_state, done):
        """Online learning from a single transition with target networks"""
        if self.last_state is None or self.last_action is None or len(self.replay_buffer) < 10:
            return

        # Convert to tensors if not already
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(self.device)
        if isinstance(reward, (int, float)):
            reward = torch.FloatTensor([reward]).to(self.device)

        # Determine if decision net or action net was used
        if self.last_action == 12:  # No action was taken
            # Update decision network
            # Calculate TD error using target network for stability
            current_value = self.Critic_d(state)
            next_value = self.Critic_d_target(next_state)

            # Calculate target (bootstrapped if not done)
            target = reward + (1 - done) * self.gamma * next_value
            td_error = target - current_value

            # Update critic
            self.optimizer_critic_d.zero_grad()
            critic_loss_d = F.mse_loss(current_value, target.detach())
            critic_loss_d.backward()
            self.optimizer_critic_d.step()

            # Update actor using policy gradient
            _, logits_d = self.Actor_d(state) #self.last_action_probs[0][0]
            entropy_d = self.calculate_entropy(logits_d)
            self.optimizer_actor_d.zero_grad()
            actor_loss_d =  -torch.log(logits_d[0][0] + 1e-6) * td_error.detach() - self.entropy_coef * entropy_d# Policy gradient
            actor_loss_d.backward()
            self.optimizer_actor_d.step() # TODO: gradients are getting zero even with entropy - check this https://github.com/ikostrikov/pytorch-a3c/blob/48d95844755e2c3e2c7e48bbd1a7141f7212b63f/train.py

            # Soft update target networks
            self._soft_update(self.Critic_d, self.Critic_d_target)
            self._soft_update(self.Actor_d, self.Actor_d_target)

        else:  # An action was taken
            # Update both decision and action networks
            # Decision network update using target networks
            current_value_d = self.Critic_d(state)
            next_value_d = self.Critic_d_target(next_state)
            target_d = reward + (1 - done) * self.gamma * next_value_d
            td_error_d = target_d - current_value_d

            self.optimizer_critic_d.zero_grad()
            critic_loss_d = F.mse_loss(current_value_d, target_d.detach())
            critic_loss_d.backward()
            self.optimizer_critic_d.step()

            # Update decision actor
            _, logits_d = self.Actor_d(state)
            entropy_d = self.calculate_entropy(logits_d)
            self.optimizer_actor_d.zero_grad()
            actor_loss_d = -torch.log(logits_d[0][1] + 1e-6) * td_error_d.detach() - self.entropy_coef * entropy_d # Policy gradient
            actor_loss_d.backward()
            self.optimizer_actor_d.step()

            # Action network update using target networks
            current_value_u = self.Critic_u(state)
            next_value_u = self.Critic_u_target(next_state)
            target_u = reward + (1 - done) * self.gamma * next_value_u
            td_error_u = target_u - current_value_u

            self.optimizer_critic_u.zero_grad()
            critic_loss_u = F.mse_loss(current_value_u, target_u.detach())
            critic_loss_u.backward()
            self.optimizer_critic_u.step()

            # Update action actor
            _, logits_u = self.Actor_u(state)
            entropy_u = self.calculate_entropy(logits_u)
            if torch.numel(logits_u) > 0:  # Only if valid action logits exist
                self.optimizer_actor_u.zero_grad()
                actor_loss_u = -torch.log(logits_u[0][self.last_action] + 1e-6) * td_error_u.detach() - self.entropy_coef*entropy_u
                actor_loss_u.backward()
                self.optimizer_actor_u.step()
                print(f'u Losses: Actor_u: {actor_loss_u.item():0.4f}, Critic_u: {critic_loss_u.item():0.4f}')

            # Soft update target networks
            self._soft_update(self.Critic_d, self.Critic_d_target)
            self._soft_update(self.Actor_d, self.Actor_d_target)
            self._soft_update(self.Critic_u, self.Critic_u_target)
            self._soft_update(self.Actor_u, self.Actor_u_target)
            self.scheduler_actor_u.step()
            self.scheduler_critic_u.step()


        print(f'd Losses: Actor_d: {actor_loss_d.item():0.4f}, Critic_d: {critic_loss_d.item():0.4f}')
        # Step schedulers
        self.scheduler_actor_d.step()
        self.scheduler_critic_d.step()


    def calculaterewards(self, termination) -> Any:
        penalty = 0
        # penalty = self.rewardpenalty * (self.countInventory()**2)
        self.profit = self.cash - self.statelog[0][1]
        self.updatestatelog()
        deltaPNL = self.statelog[-1][2] - self.statelog[-2][2]
        if self.istruncated or termination:
            deltaPNL += self.countInventory() * self.mid
        if self.last_action != 12: penalty -= 10 # custom reward for incentivising actions rather than inaction for learning
        return deltaPNL - penalty

class ICRL2(ICRLAgent):

    def getState(self, state):
        if type(state) == dict:
            state = self.readData(state)
        # Commented out scaling for now - can be uncommented if needed
        if len(self.replay_buffer) > 10:
            self.mmscaler.fit(torch.cat([i[0] for i in self.replay_buffer]))
            state = self.mmscaler.transform(state)
        else:
            state = self.mmscaler.fit_transform(state)
        return state

    def get_action(self, data, epsilon=0.1):
        """
        Epsilon-greedy policy to balance exploitation with exploration.

        Args:
            data: The input data
            epsilon: Probability of choosing a random action (default: 0.1)

        Returns:
            action, size, action_probs_d, action_probs_u
        """
        size = 1
        origData = data.copy()
        data = self.getState(data)

        # Store current state for learning
        self.last_state = data
        # self.replay_buffer.append(self.readData(origData))

        # Generate random number to decide between exploration and exploitation
        if (random.random() < epsilon) or (len(self.replay_buffer) < 10):
            print('RANDOM ACTION')
            # EXPLORATION: Choose a random action
            # Get all valid actions in the current state
            valid_actions = []

            # Get decision probabilities from the shared network
            _, action_probs_d, _ = self.Actor_Critic_d(data)

            # Check which actions are valid
            potential_actions = list(range(12))  # Actions 0-11

            # Filter out invalid actions based on state conditions
            for action in potential_actions:
                # Cancel orders check
                if action in [1, 3, 8, 10] and len(origData['Positions'][self.actionsToLevels[self.actions[action]]]) == 0:
                    continue

                # Inspread checks
                if action in [5, 6]:
                    p_a, q_a = origData['LOB0']['Ask_L1']
                    p_b, q_b = origData['LOB0']['Bid_L1']
                    if p_a - p_b < 0.015:  # reject if inspread not possible
                        continue

                # Ask market order check
                if action == 4 and self.countInventory() < 1:
                    continue

                valid_actions.append(action)

            # Add the "do nothing" action as always valid
            valid_actions.append(12)

            # Choose a random valid action
            random_action = random.choice(valid_actions)

            # Calculate action probabilities just for logging/learning purposes
            if random_action != 12:
                _, action_probs_u, _ = self.Actor_Critic_u(data)
                self.last_action_probs = (action_probs_d, action_probs_u)
            else:
                self.last_action_probs = (action_probs_d, torch.tensor([0,0]))

            self.last_action = random_action
            return random_action, size, action_probs_d, self.last_action_probs[1]

        else:
            # EXPLOITATION: Use the policy networks
            # Get decision from decision network (d=0: do nothing, d=1: take action)
            d, action_probs_d, _ = self.Actor_Critic_d(data)

            # Check if decision is to take an action
            if d.item() == 1:  # Decision is to take an action
                # Get action from action network
                u, action_probs_u, _ = self.Actor_Critic_u(data)
                action_index = int(u.item())

                # Validate the selected action
                if action_index in [1, 3, 8, 10]:  # cancels
                    a = self.actions[action_index]
                    lvl = self.actionsToLevels[a]
                    if len(origData['Positions'][lvl]) == 0:  # no position to cancel
                        self.last_action = 12
                        self.last_action_probs = (action_probs_d, torch.tensor([0,0]))
                        return 12, size, action_probs_d, torch.tensor([0,0])

                if action_index in [5, 6]:  # inspread checks
                    p_a, q_a = origData['LOB0']['Ask_L1']
                    p_b, q_b = origData['LOB0']['Bid_L1']
                    if p_a - p_b < 0.015:  # reject if inspread not possible
                        self.last_action = 12
                        self.last_action_probs = (action_probs_d, torch.tensor([0,0]))
                        return 12, size, action_probs_d, torch.tensor([0,0])

                if (action_index == 4) and (self.countInventory() < 1):  # ask market order
                    self.last_action = 12
                    self.last_action_probs = (action_probs_d, torch.tensor([0,0]))
                    return 12, size, action_probs_d, torch.tensor([0,0])

                # If action is valid, return it
                self.last_action = action_index
                self.last_action_probs = (action_probs_d, action_probs_u)
                return action_index, size, action_probs_d, action_probs_u

            else:  # Decision is to do nothing
                self.last_action = 12
                self.last_action_probs = (action_probs_d, torch.tensor([0,0]))
                return 12, size, action_probs_d, torch.tensor([0,0])

    def setupNNs(self, data0):
        self.alpha = 0.5
        # Get state dimensions
        data0 = self.getState(data0)
        state_dim = len(data0[0])

        # Initialize main networks with shared architecture
        self.Actor_Critic_d = ActorCriticMLP(state_dim, 128, 3, 2, actor_activation='softmax', hidden_activation='leaky_relu')
        self.Actor_Critic_u = ActorCriticMLP(state_dim, 128, 3, 12, actor_activation='softmax', hidden_activation='leaky_relu')

        # Initialize target networks with shared architecture
        self.Actor_Critic_d_target = ActorCriticMLP(state_dim, 128, 3, 2, actor_activation='softmax', hidden_activation='leaky_relu')
        self.Actor_Critic_u_target = ActorCriticMLP(state_dim, 128, 3, 12, actor_activation='softmax', hidden_activation='leaky_relu')

        # Move all models to appropriate device
        self.Actor_Critic_d.to(self.device)
        self.Actor_Critic_u.to(self.device)
        self.Actor_Critic_d_target.to(self.device)
        self.Actor_Critic_u_target.to(self.device)

        # Copy parameters from main networks to target networks
        self._hard_update(self.Actor_Critic_d, self.Actor_Critic_d_target)
        self._hard_update(self.Actor_Critic_u, self.Actor_Critic_u_target)

        # Setup optimizers
        self.optimizer_d, self.scheduler_d = self.setupTraining(self.Actor_Critic_d)
        self.optimizer_u, self.scheduler_u = self.setupTraining(self.Actor_Critic_u)

    def learn(self, state, reward, next_state, done):
        """Online learning from a single transition with target networks using shared actor-critic architecture"""
        if self.last_state is None or self.last_action is None or len(self.replay_buffer) < 10:
            return

        # Convert to tensors if not already
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(self.device)
        if isinstance(reward, (int, float)):
            reward = torch.FloatTensor([reward]).to(self.device)

        # Determine if decision net or action net was used
        if self.last_action == 12:  # No action was taken
            # Update decision network (Actor_Critic_d)

            # Forward pass through current and target networks
            _, action_probs_d, value_d = self.Actor_Critic_d(state)
            with torch.no_grad():
                _, _, next_value_d = self.Actor_Critic_d_target(next_state)

            # Calculate target (bootstrapped if not done)
            target_d = reward + (1 - done) * self.gamma * next_value_d
            td_error_d = target_d - value_d

            # Calculate losses
            critic_loss_d = F.mse_loss(value_d, target_d.detach())
            entropy_d = self.calculate_entropy(action_probs_d)
            actor_loss_d = -torch.log(action_probs_d[0][0] + 1e-6) * td_error_d.detach() - self.entropy_coef * entropy_d

            # Combined loss for the shared network
            total_loss_d = actor_loss_d + critic_loss_d

            # Update network
            self.optimizer_d.zero_grad()
            total_loss_d.backward()
            self.optimizer_d.step()

            # Soft update target network
            self._soft_update(self.Actor_Critic_d, self.Actor_Critic_d_target)

            # Print losses
            print(f'd Losses: Actor_d: {actor_loss_d.item():0.4f}, Critic_d: {critic_loss_d.item():0.4f}')

        else:  # An action was taken
            # Update both decision and action networks

            # Decision network update
            _, action_probs_d, value_d = self.Actor_Critic_d(state)
            with torch.no_grad():
                _, _, next_value_d = self.Actor_Critic_d_target(next_state)

            target_d = reward + (1 - done) * self.gamma * next_value_d
            td_error_d = target_d - value_d

            critic_loss_d = F.mse_loss(value_d, target_d.detach())
            entropy_d = self.calculate_entropy(action_probs_d)
            actor_loss_d = -torch.log(action_probs_d[0][1] + 1e-6) * td_error_d.detach() - self.entropy_coef * entropy_d

            # Combined loss for decision network
            total_loss_d = actor_loss_d + critic_loss_d

            # Update decision network
            self.optimizer_d.zero_grad()
            total_loss_d.backward()
            self.optimizer_d.step()

            # Action network update
            _, action_probs_u, value_u = self.Actor_Critic_u(state)
            with torch.no_grad():
                _, _, next_value_u = self.Actor_Critic_u_target(next_state)

            target_u = reward + (1 - done) * self.gamma * next_value_u
            td_error_u = target_u - value_u

            critic_loss_u = F.mse_loss(value_u, target_u.detach())
            entropy_u = self.calculate_entropy(action_probs_u)

            if torch.numel(action_probs_u) > 0:  # Only if valid action logits exist
                actor_loss_u = -torch.log(action_probs_u[0][self.last_action] + 1e-6) * td_error_u.detach() - self.entropy_coef * entropy_u

                # Combined loss for action network
                total_loss_u = actor_loss_u + critic_loss_u

                # Update action network
                self.optimizer_u.zero_grad()
                total_loss_u.backward()
                self.optimizer_u.step()

                print(f'u Losses: Actor_u: {actor_loss_u.item():0.4f}, Critic_u: {critic_loss_u.item():0.4f}')

            # Soft update target networks
            self._soft_update(self.Actor_Critic_d, self.Actor_Critic_d_target)
            self._soft_update(self.Actor_Critic_u, self.Actor_Critic_u_target)

        # Step schedulers
        self.scheduler_d.step()
        if self.last_action != 12:  # Only step u scheduler if action was taken
            self.scheduler_u.step()

    def learnSAC(self):
        """
        Soft Actor-Critic (SAC) learning from a single transition with target networks
        using shared actor-critic architecture.
        """
        if self.last_state is None or self.last_action is None or len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from replay buffer if available
        batch = self.sample_batch()
        # Process the batch
        states, actions, rewards, next_states, dones = batch
        actions_u = actions.clone()
        actions_u[actions_u == 12] = -1
        actions_d = actions.clone()
        actions_d[actions_d != 12] = 1
        actions_d[actions_d == 12] = 0
        networks = [(self.Actor_Critic_d, self.Actor_Critic_d_target, self.optimizer_d, actions_d),
                        (self.Actor_Critic_u, self.Actor_Critic_u_target, self.optimizer_u, actions_u)]

        # Process each network (decision and/or action)
        states = self.getState(states)
        next_states = self.getState(next_states)
        # networks_to_process = networks if 'networks' in locals() else [(network, target_network, optimizer)]
        networks_to_process = []

        for current_network, current_target, current_optimizer, actions in networks_to_process:
            # =================== SAC LEARNING ALGORITHM ===================
            valid_idxs = torch.where(actions != -1)[0]
            states = states[valid_idxs, :]
            actions = actions[valid_idxs, :]
            next_states = next_states[valid_idxs,:]
            dones = dones[valid_idxs,:]
            rewards = rewards[valid_idxs, :]
            # 1. Get current Q-values and action probabilities
            _, action_probs, q_values = current_network(states)
            log_probs = torch.log(action_probs + 1e-10)

            # 2. Calculate entropy term
            entropy = -torch.sum((action_probs + 1e-3) * log_probs, dim=1, keepdim=True)

            # 3. Get target Q-values
            with torch.no_grad():
                # For target value, we use the target network
                next_actions, next_action_probs, next_q_values = current_target(next_states)
                next_log_probs = torch.log(next_action_probs + 1e-10)

                # Calculate the expected Q-value across all possible next actions
                next_batch_indices = torch.arange(next_q_values.size(0)).to(self.device)
                next_q = next_q_values[next_batch_indices,next_actions.long().squeeze(1)].unsqueeze(1)
                # next_q = torch.sum(next_action_probs * next_q_values, dim=1, keepdim=True)

                # Include entropy term in the target for maximum entropy RL
                next_value = next_q + self.alpha * (-torch.sum((next_action_probs + 1e-3) * next_log_probs, dim=1, keepdim=True))

                # Compute the target Q value: r + γ(1-d)(Q + αH)
                target_q = rewards + (1 - dones) * self.gamma * next_value

            # 4. Critic loss: MSE between current Q and target Q
            batch_indices = torch.arange(q_values.size(0)).to(self.device)
            current_q = q_values[batch_indices,actions.long().squeeze(1)].unsqueeze(1)

            critic_loss = F.mse_loss(current_q, target_q.detach())

            # 5. Actor loss: Policy gradient with entropy regularization
            # In SAC, the actor aims to maximize Q-value and entropy
            # L = E[α*log(π(a|s)) - Q(s,a)]

            # Calculate the expected Q-value under the current policy
            policy_q = torch.sum((action_probs + 1e-3)* q_values, dim=1, keepdim=True)

            # Actor loss with entropy regularization
            actor_loss = torch.mean(self.alpha * log_probs - q_values.detach())

            # 6. Optional: temperature parameter auto-tuning
            # If using adaptive temperature (alpha), update it here
            # Omitted for simplicity, but could be added

            # 7. Combined loss and optimization step
            total_loss = actor_loss + critic_loss

            current_optimizer.zero_grad()
            total_loss.backward()
            current_optimizer.step()

            # 8. Soft update of the target networks
            self._soft_update(current_network, current_target)

            # Print losses for monitoring
            if current_network == self.Actor_Critic_d:
                print(f'd Losses: total: {total_loss.item(): 0.4f}, Actor_d: {actor_loss.item():0.4f}, Critic_d: {critic_loss.item():0.4f}')
            else:
                print(f'u Losses: total: {total_loss.item(): 0.4f}, Actor_u: {actor_loss.item():0.4f}, Critic_u: {critic_loss.item():0.4f}')

        # Step the schedulers
        self.scheduler_d.step()
        if self.last_action != 12:  # Only step u scheduler if action was taken
            self.scheduler_u.step()

class ICRLSG(ICRL2):

    '''problem: this is made for cts action spaces : https://arxiv.org/pdf/2209.10081 '''

    def setupNNs(self, data0):
        self.alpha = 0.5
        # Get state dimensions
        data0 = self.getState(data0)
        state_dim = len(data0[0])

        # Initialize main networks with shared architecture
        self.Actor_Critic_d = ActorCriticSGMLP(state_dim, 128, 3, 2, hidden_activation='leaky_relu')
        self.Actor_Critic_u = ActorCriticSGMLP(state_dim, 128, 3, 12,  hidden_activation='leaky_relu')

        # Initialize target networks with shared architecture
        self.Actor_Critic_d_target = ActorCriticSGMLP(state_dim, 128, 3, 2, hidden_activation='leaky_relu')
        self.Actor_Critic_u_target = ActorCriticSGMLP(state_dim, 128, 3, 12, hidden_activation='leaky_relu')

        # Move all models to appropriate device
        self.Actor_Critic_d.to(self.device)
        self.Actor_Critic_u.to(self.device)
        self.Actor_Critic_d_target.to(self.device)
        self.Actor_Critic_u_target.to(self.device)

        # Copy parameters from main networks to target networks
        self._hard_update(self.Actor_Critic_d, self.Actor_Critic_d_target)
        self._hard_update(self.Actor_Critic_u, self.Actor_Critic_u_target)

        # Setup optimizers
        self.optimizer_d, self.scheduler_d = self.setupTraining(self.Actor_Critic_d)
        self.optimizer_u, self.scheduler_u = self.setupTraining(self.Actor_Critic_u)

    def get_action(self, data, epsilon=0.1):
        """
        Epsilon-greedy policy to balance exploitation with exploration.

        Args:
            data: The input data
            epsilon: Probability of choosing a random action (default: 0.1)

        Returns:
            action, size, action_probs_d, action_probs_u
        """
        size = 1
        origData = data.copy()
        data = self.getState(data)

        # Store current state for learning
        self.last_state = data

        # Generate random number to decide between exploration and exploitation
        if (random.random() < epsilon) or (len(self.replay_buffer) < 10):
            print('RANDOM ACTION')
            # EXPLORATION: Choose a random action
            # Get all valid actions in the current state
            valid_actions = []

            # Get decision distribution from the shared network
            _, (d_mean, d_log_std, d_log_prob), _ = self.Actor_Critic_d(data)

            # Check which actions are valid
            potential_actions = list(range(12))  # Actions 0-11

            # Filter out invalid actions based on state conditions
            for action in potential_actions:
                # Cancel orders check
                if action in [1, 3, 8, 10] and len(origData['Positions'][self.actionsToLevels[self.actions[action]]]) == 0:
                    continue

                # Inspread checks
                if action in [5, 6]:
                    p_a, q_a = origData['LOB0']['Ask_L1']
                    p_b, q_b = origData['LOB0']['Bid_L1']
                    if p_a - p_b < 0.015:  # reject if inspread not possible
                        continue

                # Ask market order check
                if action == 4 and self.countInventory() < 1:
                    continue

                valid_actions.append(action)

            # Add the "do nothing" action as always valid
            valid_actions.append(12)

            # Choose a random valid action
            random_action = random.choice(valid_actions)

            # Calculate action probabilities just for logging/learning purposes
            if random_action != 12:
                # Get action distribution from the action network
                _, (u_mean, u_log_std, u_log_prob), _ = self.Actor_Critic_u(data)
                self.last_action_probs = (d_mean, u_mean)
            else:
                self.last_action_probs = (d_mean, torch.zeros_like(d_mean))

            self.last_action = random_action
            return random_action, size, d_mean, self.last_action_probs[1]

        else:
            # EXPLOITATION: Use the policy networks
            # Get decision from decision network (soft decision based on tanh)
            d, (d_mean, d_log_std, d_log_prob), _ = self.Actor_Critic_d(data)

            # Check if decision is to take an action (convert from tanh-squashed action to binary decision)
            if torch.abs(d.item()) > 0.5:  # Decision is to take an action
                # Get action from action network
                u, (u_mean, u_log_std, u_log_prob), _ = self.Actor_Critic_u(data)

                # Scale the squashed action from [-1, 1] to [0, 11]
                action_index = int((u.item() + 1) * 5.5)

                # Validate the selected action
                if action_index in [1, 3, 8, 10]:  # cancels
                    a = self.actions[action_index]
                    lvl = self.actionsToLevels[a]
                    if len(origData['Positions'][lvl]) == 0:  # no position to cancel
                        self.last_action = 12
                        self.last_action_probs = (d_mean, torch.zeros_like(d_mean))
                        return 12, size, d_mean, torch.zeros_like(d_mean)

                if action_index in [5, 6]:  # inspread checks
                    p_a, q_a = origData['LOB0']['Ask_L1']
                    p_b, q_b = origData['LOB0']['Bid_L1']
                    if p_a - p_b < 0.015:  # reject if inspread not possible
                        self.last_action = 12
                        self.last_action_probs = (d_mean, torch.zeros_like(d_mean))
                        return 12, size, d_mean, torch.zeros_like(d_mean)

                if (action_index == 4) and (self.countInventory() < 1):  # ask market order
                    self.last_action = 12
                    self.last_action_probs = (d_mean, torch.zeros_like(d_mean))
                    return 12, size, d_mean, torch.zeros_like(d_mean)

                # If action is valid, return it
                self.last_action = action_index
                self.last_action_probs = (d_mean, u_mean)
                return action_index, size, d_mean, u_mean

            else:  # Decision is to do nothing
                self.last_action = 12
                self.last_action_probs = (d_mean, torch.zeros_like(d_mean))
                return 12, size, d_mean, torch.zeros_like(d_mean)

    def learnSAC(self):
        """
        Soft Actor-Critic (SAC) learning from a single transition with target networks
        using shared actor-critic architecture and squashed Gaussian policy.
        """
        if self.last_state is None or self.last_action is None or len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from replay buffer if available
        batch = self.sample_batch()
        # Process the batch
        states, actions, rewards, next_states, dones = batch
        actions_u = actions.clone()
        actions_u[actions_u == 12] = -1
        actions_d = actions.clone()
        actions_d[actions_d != 12] = 1
        actions_d[actions_d == 12] = 0
        networks = [(self.Actor_Critic_d, self.Actor_Critic_d_target, self.optimizer_d, actions_d),
                    (self.Actor_Critic_u, self.Actor_Critic_u_target, self.optimizer_u, actions_u)]

        # Process each network (decision and/or action)
        states = self.getState(states)
        next_states = self.getState(next_states)
        networks_to_process = networks

        for current_network, current_target, current_optimizer, actions in networks_to_process:
            # =================== SAC LEARNING ALGORITHM ===================
            valid_idxs = torch.where(actions != -1)[0]
            states = states[valid_idxs, :]
            actions = actions[valid_idxs, :]
            next_states = next_states[valid_idxs,:]
            dones = dones[valid_idxs,:]
            rewards = rewards[valid_idxs, :]

            # 1. Get current Q-values and policy distribution
            sampled_actions, (actor_mean, actor_log_std, log_probs), q_values = current_network(states)

            # 2. Calculate entropy term (using pre-computed log probabilities)
            entropy = -log_probs

            # 3. Get target Q-values
            with torch.no_grad():
                # For target value, we use the target network
                next_sampled_actions, (next_actor_mean, next_actor_log_std, next_log_probs), next_q_values = current_target(next_states)

                # Calculate the expected Q-value for the sampled next actions
                next_batch_indices = torch.arange(next_q_values.size(0)).to(self.device)
                next_q = next_q_values[next_batch_indices].unsqueeze(1)

                # Include entropy term in the target for maximum entropy RL
                next_value = next_q + self.alpha * (-next_log_probs)

                # Compute the target Q value: r + γ(1-d)(Q + αH)
                target_q = rewards + (1 - dones) * self.gamma * next_value

            # 4. Critic loss: MSE between current Q and target Q
            batch_indices = torch.arange(q_values.size(0)).to(self.device)
            current_q = q_values[batch_indices].unsqueeze(1)

            critic_loss = F.mse_loss(current_q, target_q.detach())

            # 5. Actor loss: Policy gradient with entropy regularization
            # Compute log probabilities for the current actions
            action_log_probs = current_network.get_log_prob(states, actions)

            # Actor loss with entropy regularization
            actor_loss = torch.mean(self.alpha * (-action_log_probs) - current_q.detach())

            # 6. Combined loss and optimization step
            total_loss = actor_loss + critic_loss

            current_optimizer.zero_grad()
            total_loss.backward()
            current_optimizer.step()

            # 7. Soft update of the target networks
            self._soft_update(current_network, current_target)

            # 8. Print losses for monitoring
            if current_network == self.Actor_Critic_d:
                print(f'd Losses: total: {total_loss.item(): 0.4f}, Actor_d: {actor_loss.item():0.4f}, Critic_d: {critic_loss.item():0.4f}')
            else:
                print(f'u Losses: total: {total_loss.item(): 0.4f}, Actor_u: {actor_loss.item():0.4f}, Critic_u: {critic_loss.item():0.4f}')

        # Step the schedulers
        self.scheduler_d.step()
        if self.last_action != 12:  # Only step u scheduler if action was taken
            self.scheduler_u.step()

class StateActionVisitCounter:
    def __init__(self, lambda_exploration=1.0, first_visit_bonus=0.2):
        """
        Efficient state-action visit counter with fast exploration bonus computation.

        Args:
            lambda_exploration: Exploration bonus coefficient
            first_visit_bonus: Bonus returned for a never-before-seen (s,a) pair
        """
        self.lambda_exploration = lambda_exploration
        self.first_visit_bonus = first_visit_bonus

        # Use defaultdict for O(1) access and automatic initialization
        self.visit_counts = defaultdict(int)

        # Cache for sqrt calculations to avoid repeated computation
        self.sqrt_cache = {}

        # Pre-compute common sqrt values for better performance
        self._precompute_common_sqrt_values()

    def _precompute_common_sqrt_values(self, max_precompute=1000):
        """Pre-compute sqrt values for first 1000 visit counts"""
        for i in range(1, max_precompute + 1):
            self.sqrt_cache[i] = np.sqrt(i)

    def _state_to_key(self, state):
        """
        Convert state tuple to hashable key for dictionary lookup.
        Handles floating point precision issues by rounding.

        Args:
            state: (inventory_intc, price_diff, n_a_over_q_a, n_b_over_q_b,
                    twap_side, intensity_bucket, twap_phase)
        """
        # Round to avoid floating point precision issues. The queue ratios
        # (n_a/q_a, n_b/q_b) are heavy-tailed near-continuous, so bucket them
        # into 3 bins (edges at 1.0, 2.0): <1 = ahead of the visible queue
        # (likely to fill), 1-2 = front-ish, >2 = buried. Rounding them instead
        # shatters the count table (~71% of pairs globally unique), pinning the
        # exploration bonus at the first-visit floor.
        return (
            round(state[0], 0),  # inventory_intc
            round(state[1], 2),  # p_a - p_b
            int(np.digitize(state[2], [1.0, 2.0])),  # n_a/q_a bucket (0/1/2)
            int(np.digitize(state[3], [1.0, 2.0])),  # n_b/q_b bucket (0/1/2)
            round(state[4], 0),  # twap_side: TWAPPresent (-1/0/+1)
            int(state[5]),       # Hawkes net-flow-imbalance bucket (-1/0/+1)
            int(state[6])        # twap_phase: -1 before / 0 during / +1 after
        )

    def update_visit_count(self, state, action):
        """
        Increment visit count for state-action pair.

        Args:
            state: (inventory_intc, price_diff, n_a_over_q_a, n_b_over_q_b)
            action: int from 0 to 12
        """
        key = (self._state_to_key(state), action)
        self.visit_counts[key] += 1

        # Update sqrt cache for this new count if not too large
        new_count = self.visit_counts[key]
        if new_count <= 10000 and new_count not in self.sqrt_cache:
            self.sqrt_cache[new_count] = np.sqrt(new_count)

    def get_visit_count(self, state, action):
        """Get visit count for state-action pair (O(1) lookup)"""
        key = (self._state_to_key(state), action)
        return self.visit_counts[key]

    def get_exploration_bonus(self, state, action):
        """
        Get exploration bonus lambda/sqrt(N(s,a)) with fast lookup.

        Args:
            state: (inventory_intc, price_diff, n_a_over_q_a, n_b_over_q_b)
            action: int from 0 to 12

        Returns:
            float: Exploration bonus value
        """
        key = (self._state_to_key(state), action)
        count = self.visit_counts[key]

        if count == 0:
            # First visit - return configurable bonus
            return self.first_visit_bonus

        # Use cached sqrt if available, otherwise compute
        if count in self.sqrt_cache:
            sqrt_count = self.sqrt_cache[count]
        else:
            sqrt_count = np.sqrt(count)
            # Cache if reasonable size
            if count <= 10000:
                self.sqrt_cache[count] = sqrt_count

        return self.lambda_exploration / sqrt_count

    def get_stats(self):
        """Get statistics about the visit counter"""
        total_states = len(set(key[0] for key in self.visit_counts.keys()))
        total_sa_pairs = len(self.visit_counts)
        total_visits = sum(self.visit_counts.values())

        return {
            'unique_states': total_states,
            'unique_state_action_pairs': total_sa_pairs,
            'total_visits': total_visits,
            'cache_size': len(self.sqrt_cache)
        }

class PPOAgent(GymTradingAgent):
    def __init__(self, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random",
                 Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5,
                 wake_on_MO: bool=True, wake_on_Spread: bool=True, cashlimit=1000000, inventorylimit=100,
                 buffer_capacity=10000, batch_size=64, epochs=1000, layer_widths = 128, n_layers = 3, clip_ratio=0.2,
                 value_loss_coef=0.5, entropy_coef=10, max_grad_norm=0.5, gae_lambda=0.95, gamma=0.99, rewardpenalty = 0.1, hidden_activation='leaky_relu',
                 transaction_cost = 0.01, start_trading_lag=0, truncation_enabled=True, action_space_config = 0, include_time = False, alt_state=False, enhance_state=False,
                 policy_loss_coef = 1, optim_type = 'ADAM',lr=1e-3, exploration_bonus = 0, first_visit_bonus = 0.2, two_sided_reward = True, ablation_params= {}, typeNN = "dense", chunk_length=64, terminal_invpenalty=0, cem_full_episode=False, phase_a_refresh_every=25, action_bonus=0.5, running_invpenalty=0.0, symmetric_mo_gating=False, cem_elite_floor=0.0, cem_n_elites=None):
        """
        PPO Agent with Generalized Advantage Estimation (GAE)
        Maintains two networks: one for decision (d) and one for utility (u)

        :param seed: Random seed
        :param log_events: Whether to log events
        :param log_to_file: Whether to log to file
        :param strategy: Trading strategy
        :param Inventory: Initial inventory
        :param cash: Initial cash
        :param action_freq: Action frequency
        :param wake_on_MO: Wake on market order
        :param wake_on_Spread: Wake on spread
        :param cashlimit: Cash limit
        :param buffer_capacity: Maximum size of trajectory buffer
        :param batch_size: Size of batch for training
        :param epochs: Number of epochs for PPO update
        :param clip_ratio: PPO clipping parameter
        :param value_loss_coef: Coefficient for value loss
        :param entropy_coef: Coefficient for entropy bonus
        :param max_grad_norm: Maximum gradient norm for clipping
        :param gae_lambda: GAE lambda parameter
        """
        super().__init__(seed=seed, log_events=log_events, log_to_file=log_to_file, strategy=strategy,
                         Inventory=Inventory, cash=cash, action_freq=action_freq, wake_on_MO=wake_on_MO,
                         wake_on_Spread=wake_on_Spread, cashlimit=cashlimit, inventorylimit=inventorylimit, start_trading_lag=start_trading_lag,
                         truncation_enabled=truncation_enabled)

        self.resetseed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        #allowed actions:

        self.action_space_config = action_space_config
        if action_space_config == 0:
            self.allowed_actions= ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                               "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
            self.convert_dict = {}
        elif action_space_config == 1:
            self.allowed_actions= ["lo_top_Ask","co_top_Ask","co_top_Bid", "lo_top_Bid" ]
            self.convert_dict = {0:2, 1:3, 2:8, 3:9}
        elif action_space_config == 2:
            self.allowed_actions = ['lo-lo', 'bid-co-lo', 'ask-co-lo']
        self.include_time = include_time
        self.alt_state = alt_state
        self.enhance_state = enhance_state
        # PPO Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.policy_loss_coef = policy_loss_coef
        self.init_cash = self.cash
        # Training parameters
        self.layer_widths = layer_widths
        self.n_layers = n_layers
        self.hidden_activation = hidden_activation
        self.lr = lr
        self.gamma = gamma  # discount factor
        self.rewardpenalty = rewardpenalty  # inventory penalty
        self.last_state, self.last_action = None, None
        self.transaction_cost = transaction_cost
        self.optim = optim_type
        # Trajectory storage
        self.trajectory_buffer = []
        self.buffer_capacity = buffer_capacity
        self.exploration_bonus = bool(exploration_bonus)
        self.visit_counter = StateActionVisitCounter(lambda_exploration=exploration_bonus, first_visit_bonus=first_visit_bonus)
        self._twap_seen = False          # before/during/after-TWAP state machine (exploration key)
        self.last_intensity_bucket = 0   # cached Hawkes net-flow-imbalance bucket (exploration key)
        self.two_sided_reward = two_sided_reward
        # Flat per-step bonus for taking any non-no-op action. Measured against
        # the thing it competes with: on the 2026-09-07 runs the per-step PnL
        # change had median EXACTLY 0 (85-88% of steps move no money) and mean
        # |dW| of 0.006-0.018 dollars, so the default 0.5 is 30-90x the typical
        # PnL signal and, unlike PnL, it is deterministic and always positive.
        # Summed over ~2,480 steps/episode it is worth hundreds against a
        # terminal PnL of +-0.2. Kept at 0.5 by default so existing runs are
        # unchanged; lower it to make PnL the dominant term.
        self.action_bonus = action_bonus
        # Per-step inventory penalty, lambda*inv**2. Off by default, which is
        # the status quo -- the running penalty is commented out in
        # calculaterewards and the only inventory control is the TERMINAL
        # penalty, one event ~2,480 steps away through 64-step LSTM chunks.
        # With nothing opposing it, inventory drifts to a per-run direction that
        # does not depend on the TWAP side at all.
        self.running_invpenalty = running_invpenalty
        # Market-order gating. The legacy gates (symmetric_mo_gating=False, the
        # default, which every run to date is on) are:
        #   mo_Ask (SELL) blocked when inv < 1  OR  inv >= inventorylimit - 2
        #   mo_Bid (BUY)  blocked when inv <= 2 - inventorylimit
        # At inventorylimit=25 that permits SELLING only on inv in [1, 22] but
        # BUYING on inv in [-22, +inf). The `inv >= limit - 2` clause blocks
        # selling while very long, which is backwards for a position limit, and
        # its mirror -- block buying while very long -- is simply absent, so the
        # aggressive action set is long-biased. This is DORMANT in every run to
        # date because action_space_config=1 never emits u=4 or u=7, but it
        # would corrupt any action_space_config=0 run.
        # True makes the two gates mirror images. The separate "no shorting via
        # market order" rule (inv < 1) is a modelling choice, not a bug, and is
        # left in place in both modes.
        self.symmetric_mo_gating = symmetric_mo_gating
        self.cem_elite_floor = cem_elite_floor  # see get_max_contiguous_rewards
        self.cem_n_elites = cem_n_elites        # total elites, held constant across arms
        # State scaler
        self.mmscaler = MinMaxScaler()
        self.ablation_params = ablation_params
        # NN Type
        self.typeNN = typeNN
        self.chunk_length = chunk_length
        #Optional kappa, inventory penalty at termination
        self.terminal_invpenalty = terminal_invpenalty
        self.cem_full_episode = cem_full_episode
        self.episode_sides = {}  # ep -> TWAPPresent at storage time, used for side-balanced CEM
        # How often (in Phase B PPO epochs) to refresh Phase A (recompute old log probs,
        # GAE, and saved hidden states with current network weights). The saved hidden
        # states would otherwise grow stale as PPO updates move the LSTM weights.
        self.phase_a_refresh_every = phase_a_refresh_every
        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)

    def readData(self, data):
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
        self.last_intensity_bucket = self._intensity_bucket(lambdas_norm)
        past_times = data['past_times']
        if self.Inventory['INTC'] ==0: self.init_cash = self.cash
        skew = (n_a - n_b)/(0.5*(q_a + q_b))
        # print(f"skew: {skew}")
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

    def getState(self, state):
        """
        Scale and transform state

        :param state: Input state
        :return: Scaled state
        """
        if type(state) == dict:
            state = self.readData(state)

        # # Scaling the state
        # if len(self.trajectory_buffer) > 10:
        #     states = torch.cat([torch.cat([tr[1][0] for tr in self.trajectory_buffer])])
        #     self.mmscaler.fit(states)
        #     state = self.mmscaler.transform(state)
        # else:
        #     state = self.mmscaler.fit_transform(state)

        return state

    def setupTraining(self, net):
        def lr_lambda(epoch):
            # Learning rate schedule
            x =1
            # if epoch > 500:
            #     x= 1e-1
            # x = x*np.max([1e-1,(.5)**(epoch//5000)])
            return x
        if self.optim == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(net.parameters(), lr = self.lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def setupNNs(self, data0 ,type='separate'):
        """
        Setup neural networks for PPO with two networks (d and u)

        :param data0: Initial data for determining state dimensions
        """
        # Get state dimensions
        data0 = self.getState(data0)
        state_dim = len(data0[0])

        # Initialize main networks with shared architecture
        if type == 'separate':
            self.Actor_Critic_d = ActorCriticSeparate(state_dim, self.layer_widths, self.n_layers, 2, actor_activation=None, hidden_activation=self.hidden_activation, q_function = False, typeNN=self.typeNN)
            self.Actor_Critic_u = ActorCriticSeparate(state_dim, self.layer_widths, self.n_layers, len(self.allowed_actions), actor_activation=None, hidden_activation=self.hidden_activation,q_function = False, typeNN=self.typeNN)
        else:
            self.Actor_Critic_d = ActorCriticMLP(state_dim, self.layer_widths, self.n_layers, 2, actor_activation='tanh', hidden_activation=self.hidden_activation, q_function = False, typeNN = self.typeNN)
            self.Actor_Critic_u = ActorCriticMLP(state_dim, self.layer_widths, self.n_layers, len(self.allowed_actions), actor_activation='tanh', hidden_activation=self.hidden_activation,q_function = False, typeNN = self.typeNN)

        # Move all models to appropriate device
        self.Actor_Critic_d.to(self.device)
        self.Actor_Critic_u.to(self.device)

        # Setup optimizers
        self.optimizer_d, self.scheduler_d = self.setupTraining(self.Actor_Critic_d)
        self.optimizer_u, self.scheduler_u = self.setupTraining(self.Actor_Critic_u)

    def get_weights(self):
        return {'d':self.Actor_Critic_d.state_dict(), 'u':self.Actor_Critic_u.state_dict()}

    def update_weights(self, new_weights):
        self.Actor_Critic_d.load_state_dict(new_weights['d'])
        self.Actor_Critic_u.load_state_dict(new_weights['u'])
        return 0

    def _twap_phase(self):
        # -1 = before TWAP, 0 = during, +1 = after.
        # before and after both have TWAPPresent == 0, so _twap_seen disambiguates them.
        if getattr(self, 'TWAPPresent', 0) != 0:
            self._twap_seen = True
            return 0
        return 1 if self._twap_seen else -1

    def _intensity_bucket(self, lambdas_norm, deadzone=0.05):
        # net background-flow imbalance: ask group (idx 0:6) vs bid group (idx 6:12).
        # lambdas_norm sums to 1, so imb is in [-1, 1].
        imb = float(np.sum(lambdas_norm[:6]) - np.sum(lambdas_norm[6:12]))
        if imb > deadzone:
            return 1
        if imb < -deadzone:
            return -1
        return 0

    def _mo_ask_blocked(self, exploration=False):
        """Should a SELL market order be suppressed at the current inventory?

        Legacy (symmetric_mo_gating=False) reproduces the historical gates
        exactly, including the exploration branch's missing position-limit
        clause. See __init__ for why the legacy form is long-biased."""
        inv = self.countInventory()
        if inv < 1:                      # no shorting via market order
            return True
        if self.symmetric_mo_gating:     # selling reduces a long: the long limit must not block it
            return inv <= 2 - self.inventorylimit
        if exploration:
            return False                 # legacy exploration branch had no limit clause
        return inv >= self.inventorylimit - 2

    def _mo_bid_blocked(self, exploration=False):
        """Should a BUY market order be suppressed at the current inventory?"""
        inv = self.countInventory()
        if self.symmetric_mo_gating:     # mirror of the sell gate: stop at the long limit
            return inv >= self.inventorylimit - 2
        if exploration:
            return False                 # legacy exploration branch had no bid clause at all
        return inv <= 2 - self.inventorylimit

    def _exploration_key_state(self):
        # [inventory, spread, n_a/q_a, n_b/q_b] + [twap_side, intensity bucket, twap phase]
        base = self.last_state.cpu().numpy()[0][1:5]
        return [base[0], base[1], base[2], base[3],
                getattr(self, 'TWAPPresent', 0), self.last_intensity_bucket, self._twap_phase()]


    def calculaterewards(self, termination) -> Any:
        penalty = 0
        # penalty = self.rewardpenalty * (self.countInventory()**2)
        self.profit = self.cash - self.statelog[0][1]
        self.updatestatelog()
        deltaPNL = self.statelog[-1][2] - self.statelog[-2][2]
        deltaInv = self.statelog[-1][3]['INTC']*self.statelog[-1][-1]*(1- self.transaction_cost*np.sign(self.statelog[-1][3]['INTC'])) - self.statelog[-2][3]['INTC']*self.statelog[-2][-1]*(1-self.transaction_cost*np.sign(self.statelog[-1][3]['INTC']))
        # if self.istruncated or termination:
        #     deltaPNL += self.countInventory() * self.mid
        # reward shaping
        if self.istruncated:
            penalty += 100
        if self.last_action != 12:
            penalty -= self.action_bonus # flat bonus for acting vs no-op (see __init__ for the scale it competes with)
        if self.running_invpenalty:
            penalty += self.running_invpenalty * (self.countInventory()**2)
        if self.terminal_invpenalty and termination: 
            penalty += self.terminal_invpenalty * (self.countInventory()**2) # terminal inventory penalty (kappa)

        # last_state can be None when an episode opens with an inventory breach: get_action
        # returns the forced market order early without ever calling readData/setting last_state.
        # All the state-dependent shaping terms below would crash on None, so skip them.
        if self.last_state is not None:
            if (not self.alt_state) and (self.last_state.cpu().numpy()[0][8] < self.last_state.cpu().numpy()[0][4] + self.last_state.cpu().numpy()[0][6]) and (self.last_state.cpu().numpy()[0][9] < self.last_state.cpu().numpy()[0][5] + self.last_state.cpu().numpy()[0][7]):
                penalty -= self.rewardpenalty *20 # custom reward for double sided quoting
            if self.alt_state:
                if self.two_sided_reward:
                    if (self.last_state.cpu().numpy()[0][3] <= 1) and (self.last_state.cpu().numpy()[0][4] <= 1):
                        penalty -= self.rewardpenalty *20 # custom reward for double sided quoting
                if self.exploration_bonus:
                    penalty -= self.visit_counter.get_exploration_bonus(self._exploration_key_state(), self.last_action)
        return deltaPNL + deltaInv - penalty

    def get_action(self, data, epsilon=0.1, evaluation=False):
        """
        Get action using current policy with epsilon-greedy exploration

        :param data: Input trading data
        :param epsilon: Exploration probability
        :return: Executed action, sampled (d, u), policy log probabilities, values.

        Gating changes execution, not the sampled utility used by PPO. A blocked
        proposal must retain its utility index: index 0 is a real quote action,
        not a no-op sentinel. last_action retains the existing execution/shaping
        convention. For d=0 the utility index is a placeholder and is masked out
        of the utility actor loss.
        """
        # Reset per-episode TWAP-phase tracker at episode start
        if self.last_state is None:
            self._twap_seen = False
        # Reset LSTM hidden state at episode start (when last_state is None)
        if self.typeNN == "LSTM" and self.last_state is None:
            self.Actor_Critic_d.reset_hidden_state(batch_size=1)
            self.Actor_Critic_u.reset_hidden_state(batch_size=1)

        on_policy = getattr(self, 'on_policy', False)
        self._behavior_epsilon = (0.0 if evaluation else
            (1.0 if getattr(self, '_on_policy_steps', 0) < 10 else epsilon)) if on_policy else 0.0
        if on_policy:
            from HawkesRLTrading.src.Utils.on_policy import mixture_log_probs
            self._on_policy_steps = getattr(self, '_on_policy_steps', 0) + 1
            if self.breach:
                # Forced liquidation is actor-masked, but retains reward and
                # recurrent/critic context in the rollout.
                state = self.readData(data)
                self.last_state = state
                with torch.no_grad():
                    _, value_d = self.Actor_Critic_d(state)
                    _, value_u = self.Actor_Critic_u(state)
                mo = 4 if self.countInventory() > 0 else 7
                return mo, (None, None), 0, 0, value_d.item(), value_u.item()

        if self.action_space_config <2:
            if self.breach:
                mo = 4 if self.countInventory() > 0 else 7
                return mo, (None, None), 0, 0, 0, 0
            origData = data.copy()
            state = self.readData(data)
            self.last_state = state
            # Exploration
            if not on_policy and not evaluation and ((random.random() < epsilon) or (len(self.trajectory_buffer) < 10)):
                # Random decision
                d = random.randint(0, 1)
                u = random.randint(0, len(self.allowed_actions) - 1)
                _u = copy.deepcopy(u)
                u = self.convert_dict.get(u, u)
                # Compute dummy logits for logging — no_grad: exploration action is random,
                # we don't need gradients here. Without this the LSTM forward leaks autograd
                # graph (esp. with set_detect_anomaly=True).
                with torch.no_grad():
                    d_logits, d_value = self.Actor_Critic_d(state)
                    u_logits, u_value = self.Actor_Critic_u(state)
                    d_log_prob = torch.log_softmax(d_logits, dim=1)[0, d]
                    u_log_prob = torch.log_softmax(u_logits, dim=1)[0, _u]

                # Both branches implement the same hierarchical decision: d=0
                # submits no order. Still forward both LSTMs above on every step.
                if d == 0:
                    self.last_action = 12
                    return 12, (d, 0), d_log_prob.item(), 0, d_value.item(), 0

                # Validation checks (similar to original implementation)
                if int(u) in [1, 3, 8, 10]:  # cancels
                    a = self.actions[int(u)]
                    lvl = self.actionsToLevels[a]
                    if len(origData['Positions'][lvl]) == 0:  # no position to cancel
                        self.last_action = 12
                        return 12, (d, _u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

                if int(u) in [5, 6]:
                    p_a, q_a = origData['LOB0']['Ask_L1']
                    p_b, q_b = origData['LOB0']['Bid_L1']
                    if p_a - p_b < 0.015:  # reject if inspread not possible
                        self.last_action = 12
                        return 12, (d, _u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

                if (int(u) == 4) and self._mo_ask_blocked(exploration=True):  # ask mo -> since it hits the bid
                    self.last_action = 12
                    return 12, (d, _u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

                if (int(u) == 7) and self._mo_bid_blocked(exploration=True):  # bid mo
                    self.last_action = 12
                    return 12, (d, _u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

                self.last_action = u
                return u, (d, _u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

            # Exploitation
            with torch.no_grad():
                # Decision network
                d_logits, d_value = self.Actor_Critic_d(state)
                d_probs = (mixture_log_probs(d_logits, self._behavior_epsilon).exp() if on_policy else torch.softmax(d_logits, dim=1)).squeeze()
                d = torch.multinomial(d_probs, 1).item()
                d_log_prob = torch.log(d_probs[d])

                # Always forward u-network here (even when d will turn out to be 0) so its
                # LSTM hidden state advances on every step. Training's Phase A re-runs the
                # u-LSTM on every stored state regardless of d, so rollout must do the same
                # to keep `u_log_probs_old` valid as a PPO importance-sampling baseline.
                u_logits, u_value = self.Actor_Critic_u(state)
                u_probs = (mixture_log_probs(u_logits, self._behavior_epsilon).exp() if on_policy else torch.softmax(u_logits, dim=1)).squeeze()

                # If no decision to act (d=0)
                if d == 0:
                    self.last_action = 12
                    return 12, (d, 0), d_log_prob.item(), 0, d_value.item(), 0

                u = torch.multinomial(u_probs, 1).item()
                _u = copy.deepcopy(u)
                u = self.convert_dict.get(u, u)

                # Validation checks
                if np.abs(self.countInventory()) >= self.inventorylimit -1: # cancel top orders if at inventory limit
                    if self.countInventory() < 0:
                        u = 3
                    elif self.countInventory() > 0:
                        u = 8
                u_log_prob = torch.log(u_probs[_u])

                if int(u) in [1, 3, 8, 10]:  # cancels
                    a = self.actions[int(u)]
                    lvl = self.actionsToLevels[a]
                    if len(origData['Positions'][lvl]) == 0:  # no position to cancel
                        self.last_action = 12
                        return 12, (d, _u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

                if int(u) in [5, 6]:
                    p_a, q_a = origData['LOB0']['Ask_L1']
                    p_b, q_b = origData['LOB0']['Bid_L1']
                    if p_a - p_b < 0.015:  # reject if inspread not possible
                        self.last_action = 12
                        return 12, (d, _u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

                if (int(u) == 4) and self._mo_ask_blocked():  # ask mo -> hits the bid
                    self.last_action = 12
                    return 12, (d, _u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

                if (int(u)==7) and self._mo_bid_blocked(): # bid mo
                    self.last_action = 12
                    return 12, (d, _u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

                self.last_action = u
                return u, (d, _u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

        else:
            if self.breach:
                mo = 4 if self.countInventory() > 0 else 7
                return ((mo,1),(12,1)), (None, None), 0, 0, 0, 0

            origData = data.copy()
            state = self.readData(data)
            self.last_state = state

            # Mapping from u to composite action
            u_to_action = {
                0: ((2, 1), (9, 1)),  # lo top bid and ask
                1: ((3, 1), (2, 1)),  # co bid and lo top bid
                2: ((8, 1), (9, 1))   # co ask and lo top ask
            }

            # Exploration
            if not on_policy and not evaluation and ((random.random() < epsilon) or (len(self.trajectory_buffer) < 10)):
                d = random.randint(0, 1)
                u = random.randint(0, 2)  # because u in {0, 1, 2}

                # Dummy logits for logging — no_grad: action is random, no gradient needed.
                with torch.no_grad():
                    d_logits, d_value = self.Actor_Critic_d(state)
                    u_logits, u_value = self.Actor_Critic_u(state)
                    d_log_prob = torch.log_softmax(d_logits, dim=1)[0, d]
                    u_log_prob = torch.log_softmax(u_logits, dim=1)[0, u]

                if d == 0:
                    self.last_action = 12
                    return ((12, 1), (12, 1)), (d, u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

                action = u_to_action[u]
                self.last_action = action
                if int(u) in [1, 2]:  # cancels
                    a = self.actions[u_to_action[u][0][0]]
                    lo = u_to_action[u][1]
                    lvl = self.actionsToLevels[a]
                    if len(origData['Positions'][lvl]) == 0:  # no position to cancel
                        self.last_action = 12
                        return ((12,1),lo), (d, u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()
                return action, (d, u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

            # Exploitation
            with torch.no_grad():
                d_logits, d_value = self.Actor_Critic_d(state)
                d_probs = (mixture_log_probs(d_logits, self._behavior_epsilon).exp() if on_policy else torch.softmax(d_logits, dim=1)).squeeze()
                d = torch.multinomial(d_probs, 1).item()
                d_log_prob = torch.log(d_probs[d])

                # Always forward u-network so its LSTM hidden state stays in sync with
                # training Phase A (which sees every state regardless of d).
                u_logits, u_value = self.Actor_Critic_u(state)
                u_probs = (mixture_log_probs(u_logits, self._behavior_epsilon).exp() if on_policy else torch.softmax(u_logits, dim=1)).squeeze()

                if d == 0:
                    self.last_action = 12
                    return ((12, 1), (12, 1)), (d, 0), d_log_prob.item(), 0, d_value.item(), 0

                u = torch.multinomial(u_probs, 1).item()
                u_log_prob = torch.log(u_probs[u])

                action = u_to_action[u]

                self.last_action = action
            if int(u) in [1, 2]:  # cancels
                a = self.actions[u_to_action[u][0][0]]
                lo = u_to_action[u][1]
                lvl = self.actionsToLevels[a]
                if len(origData['Positions'][lvl]) == 0:  # no position to cancel
                    self.last_action = 12
                    return ((12,1),lo), (d, u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()
            return action, (d, u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

    def store_transition(self, ep, state, action, reward, next_state, done, rollout=None):
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
        on_policy = getattr(self, 'on_policy', False)
        actor_valid = d is not None
        if d is None and not on_policy: return
        if on_policy:
            if rollout is None:
                raise ValueError('On-policy transitions require rollout likelihood/value')
            d, u = (d, u) if actor_valid else (0, 0)
            state = state.detach().clone()
            next_state = next_state.detach().clone()
        # Store additional trajectory information
        transition = (
            state,       # Current state
            d,           # Decision action
            u,           # Utility action
            reward,      # Reward
            next_state,  # Next state
            int(done)         # Done flag
        )
        if on_policy:
            self._on_policy_last_account = (self.cash, self.countInventory(), self.mid)
            transition += ({'actor': actor_valid, 'epsilon': self._behavior_epsilon,
                            'log_prob': float(rollout[2] + (rollout[3] if d == 1 else 0)),
                            'value': float(rollout[4])},)
        self.trajectory_buffer.append((ep, transition))
        side = getattr(self, 'TWAPPresent', 0)
        # Keep any nonzero side once seen — TWAPPresent flips 0→±1→0 across an episode
        if side != 0 or ep not in self.episode_sides:
            self.episode_sides[ep] = side
        if self.exploration_bonus:
            self.visit_counter.update_visit_count(self._exploration_key_state(), self.last_action)

    def compute_gae(self, rewards, values_d, values_u, dones):
        """
        Compute Generalized Advantage Estimation for both networks

        :param rewards: Rewards trajectory
        :param values_d: Decision network value estimates
        :param values_u: Utility network value estimates
        :param dones: Episode termination flags
        :return: Advantages and returns for both networks
        """
        advantages_d = []
        returns_d = [0]
        advantages_u = []
        returns_u = [0]
        gae_d = 0
        gae_u = 0

        # Append terminal zero to values for computing TD error
        values_d = values_d + [0]
        values_u = values_u + [0]

        for i in reversed(range(len(rewards))):
            # Decision network GAE
            delta_d = (rewards[i] +
                       self.gamma * values_d[i+1] * (1 - dones[i]) -
                       values_d[i])
            gae_d = delta_d + self.gamma * self.gae_lambda * (1 - dones[i]) * gae_d
            advantages_d.insert(0, gae_d)
            # returns_d.insert(0, gae_d + values_d[i])
            returns_d.insert(0, rewards[i] + self.gamma*returns_d[0])
            # Utility network GAE
            delta_u = (rewards[i] +
                       self.gamma * values_u[i+1] * (1 - dones[i]) -
                       values_u[i])
            gae_u = delta_u + self.gamma * self.gae_lambda * (1 - dones[i]) * gae_u
            advantages_u.insert(0, gae_u)
            # returns_u.insert(0, gae_u + values_u[i])
            returns_u.insert(0, rewards[i] + self.gamma*returns_u[0])
        # Convert to tensors
        advantages_d = torch.tensor(advantages_d, dtype=torch.float32).to(self.device)
        returns_d = torch.tensor(returns_d[:-1], dtype=torch.float32).to(self.device)
        advantages_u = torch.tensor(advantages_u, dtype=torch.float32).to(self.device)
        returns_u = torch.tensor(returns_u[:-1], dtype=torch.float32).to(self.device)

        # Normalize advantages
        # advantages_d = (advantages_d - advantages_d.mean()) / (advantages_d.std() + 1e-8)
        # advantages_u = (advantages_u - advantages_u.mean()) / (advantages_u.std() + 1e-8)
        # returns_d = (returns_d - returns_d.mean()) / (returns_d.std() + 1e-8)
        # returns_u = (returns_u - returns_u.mean()) / (returns_u.std() + 1e-8)
        return advantages_d, returns_d, advantages_u, returns_u

    def train(self, train_logger, use_CEM = False):
        """
        PPO training method using entire episode trajectory.
        Dense branch: random timestep sampling (stateless networks).
        LSTM branch: chunk-based PPO with preserved temporal context.
        """
        # Ensure we have a full trajectory
        if len(self.trajectory_buffer) < 2:
            return

        if getattr(self, 'on_policy', False):
            if use_CEM or self.typeNN != 'LSTM':
                raise ValueError('Fresh PPO requires recurrent mode and CEM disabled')
            from HawkesRLTrading.src.Utils.on_policy import train_fresh
            return train_fresh(self, train_logger)
        if self.typeNN == "LSTM":
            self._train_lstm(train_logger, use_CEM)
        else:
            self._train_dense(train_logger, use_CEM)

        # Clear trajectory buffer after training
        while len(self.trajectory_buffer) > self.buffer_capacity:
            eID = self.trajectory_buffer[0][0]
            end_idx = np.max([i for i in range(len(self.trajectory_buffer)) if self.trajectory_buffer[i][0] == eID]) + 1
            self.trajectory_buffer = self.trajectory_buffer[end_idx:]
            self.episode_sides.pop(eID, None)
            gc.collect()
        return [0]*6

    def _train_dense(self, train_logger, use_CEM=False):
        """Dense (stateless) PPO training — original random timestep sampling."""
        tmp_buffer = []
        eIDs = np.unique([tr[0] for tr in self.trajectory_buffer])
        for eID in eIDs:
            states = torch.cat([tr[1][0] for tr in self.trajectory_buffer if tr[0] == eID]).to(self.device)
            d_actions = torch.tensor([tr[1][1] for tr in self.trajectory_buffer if tr[0] == eID]).to(self.device)
            u_actions = torch.tensor([tr[1][2] for tr in self.trajectory_buffer if tr[0] == eID]).to(self.device)
            rewards = [tr[1][3] for tr in self.trajectory_buffer if tr[0] == eID]
            dones = [tr[1][5] for tr in self.trajectory_buffer if tr[0] == eID]

            with torch.no_grad():
                d_logits_old, values_d_old = self.Actor_Critic_d(states)
                d_log_probs_old = F.log_softmax(d_logits_old, dim=1).gather(1, d_actions.unsqueeze(1)).squeeze()
                u_logits_old, values_u_old = self.Actor_Critic_u(states)
                u_log_probs_old = F.log_softmax(u_logits_old, dim=1).gather(1, u_actions.unsqueeze(1)).squeeze()

            advantages_d, returns_d, advantages_u, returns_u = self.compute_gae(
                rewards,
                values_d_old.cpu().numpy().flatten().tolist(),
                values_u_old.cpu().numpy().flatten().tolist(),
                dones
            )
            tmp_buffer.append([states, d_actions, u_actions, d_logits_old, values_d_old, d_log_probs_old, u_logits_old, values_u_old, u_log_probs_old, advantages_d, returns_d, advantages_u, returns_u])
        _states, _d_actions, _u_actions, _d_logits_old, _values_d_old, _d_log_probs_old, _u_logits_old, _values_u_old, _u_log_probs_old, _advantages_d, _returns_d, _advantages_u, _returns_u = [torch.cat([element[i] for element in tmp_buffer]) for i in range(len(tmp_buffer[0]))]
        del tmp_buffer
        if use_CEM:
            cem_states_d, cem_d_actions = self.get_CEM_data(type='d')
            cem_states_u, cem_u_actions = self.get_CEM_data(type='u')
        # PPO training for multiple epochs
        for _ in range(self.epochs):
            idxs =np.random.choice(np.arange(len(_states)), self.batch_size)
            states, d_actions, u_actions, d_logits_old, values_d_old, d_log_probs_old, u_logits_old, values_u_old, u_log_probs_old, advantages_d, returns_d, advantages_u, returns_u = _states[idxs,:], _d_actions[idxs], _u_actions[idxs], _d_logits_old[idxs,:], _values_d_old[idxs,:], _d_log_probs_old[idxs], _u_logits_old[idxs,:], _values_u_old[idxs,:], _u_log_probs_old[idxs], _advantages_d[idxs], _returns_d[idxs], _advantages_u[idxs], _returns_u[idxs]

            # Decision Network Training
            # Current policy output
            d_logits, d_values_pred = self.Actor_Critic_d(states)
            if use_CEM:
                idxs =np.random.choice(np.arange(len(cem_states_d)), self.batch_size)
                # Separate variable so d_logits (PPO batch) stays available for entropy loss below.
                # Previously this overwrote d_logits, so entropy regularisation landed on the
                # elite batch and directly opposed the CE supervised signal there.
                d_logits_cem, _ = self.Actor_Critic_d(torch.cat(cem_states_d)[idxs,:])
                d_policy_loss = F.cross_entropy(d_logits_cem, torch.tensor(cem_d_actions)[idxs].to(self.device))

            else:
                d_log_probs = F.log_softmax(d_logits, dim=1).gather(1, d_actions.unsqueeze(1)).squeeze()

                # Compute ratios
                d_ratios = torch.exp(d_log_probs - d_log_probs_old)

                # PPO Clipped Objective for Decision Network
                d_surr1 = d_ratios * advantages_d
                d_surr2 = torch.clamp(d_ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_d
                d_policy_loss = -torch.min(d_surr1, d_surr2).mean()

            # Value loss for Decision Network
            d_value_loss = F.mse_loss(d_values_pred.squeeze(), returns_d)

            # Entropy for Decision Network
            d_entropy_loss = -(torch.softmax(d_logits, dim=1) * F.log_softmax(d_logits, dim=1)).sum(dim=1).mean()

            # Total Decision Network Loss
            d_loss = (self.policy_loss_coef*d_policy_loss +
                      self.value_loss_coef * d_value_loss -
                      self.entropy_coef * d_entropy_loss)

            # Optimize Decision Network
            self.optimizer_d.zero_grad()
            d_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.Actor_Critic_d.parameters(), self.max_grad_norm)
            self.optimizer_d.step()
            self.scheduler_d.step()
            # Utility Network Training
            # Only train utility network for trajectories with d=1
            d_mask = (d_actions == 1)
            if d_mask.any():
                # Filter states and other tensors
                states_u = states[d_mask]
                u_actions_u = u_actions[d_mask]
                advantages_u_filtered = advantages_u[d_mask]
                returns_u_filtered = returns_u[d_mask]
                u_log_probs_old_filtered = u_log_probs_old[d_mask]

                # Current policy output
                u_logits, u_values_pred = self.Actor_Critic_u(states_u)
                if use_CEM:
                    idxs =np.random.choice(np.arange(len(cem_states_u)), self.batch_size)
                    # Separate variable so u_logits (PPO batch) stays available for entropy loss below.
                    u_logits_cem, _ = self.Actor_Critic_u(torch.cat(cem_states_u)[idxs,:])
                    u_policy_loss = F.cross_entropy(u_logits_cem, torch.tensor(cem_u_actions)[idxs].to(self.device))
                else:
                    u_log_probs = F.log_softmax(u_logits, dim=1).gather(1, u_actions_u.unsqueeze(1)).squeeze()

                    # Compute ratios
                    u_ratios = torch.exp(u_log_probs - u_log_probs_old_filtered)

                    # PPO Clipped Objective for Utility Network
                    u_surr1 = u_ratios * advantages_u_filtered
                    u_surr2 = torch.clamp(u_ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_u_filtered
                    u_policy_loss = -torch.min(u_surr1, u_surr2).mean()

                u_value_loss = F.mse_loss(u_values_pred.squeeze(), returns_u_filtered)

                # Entropy for Utility Network
                u_entropy_loss = -(torch.softmax(u_logits, dim=1) * F.log_softmax(u_logits, dim=1)).sum(dim=1).mean()

                # Total Utility Network Loss
                u_loss = (u_policy_loss +
                          self.value_loss_coef * u_value_loss -
                          self.entropy_coef * u_entropy_loss)

                # Optimize Utility Network
                self.optimizer_u.zero_grad()
                u_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor_Critic_u.parameters(), self.max_grad_norm)
                self.optimizer_u.step()
                self.scheduler_u.step()
                # Print losses for monitoring
                print(f'Utility Network - Policy Loss: {u_policy_loss.item():.4f}, '
                      f'Value Loss: {u_value_loss.item():.4f}, '
                      f'Entropy Loss: {u_entropy_loss.item():.4f}')

            # Print losses for monitoring
            print(f'Decision Network - Policy Loss: {d_policy_loss.item():.4f}, '
                  f'Value Loss: {d_value_loss.item():.4f}, '
                  f'Entropy Loss: {d_entropy_loss.item():.4f}')

            train_logger.log_losses(d_policy_loss  = d_policy_loss.item(), d_value_loss = d_value_loss.item(), d_entropy_loss = d_entropy_loss.item(), u_policy_loss = u_policy_loss.item(), u_value_loss = u_value_loss.item(), u_entropy_loss = u_entropy_loss.item())
            del d_policy_loss, d_value_loss, d_entropy_loss
            del u_policy_loss, u_value_loss, u_entropy_loss
            del u_logits, u_values_pred, d_logits, d_values_pred

    def _build_phase_a_chunks(self, use_CEM):
        """
        Phase A of chunk-based PPO-LSTM: replay the trajectory buffer through the
        current network to (re)compute old log probs, values, GAE, and saved
        per-chunk hidden states. Returns a list of chunk dicts ready for Phase B.

        Called by `_train_lstm` once at the start and every `phase_a_refresh_every`
        PPO epochs so the "old policy" baseline tracks the slowly-drifting current
        network instead of being frozen for all 1000 epochs.
        """
        chunk_length = self.chunk_length
        eIDs = np.unique([tr[0] for tr in self.trajectory_buffer])
        all_chunks = []

        if use_CEM:
            elite_segments = self.get_max_contiguous_rewards(K=10)

        for eID in eIDs:
            ep_data = [(tr[1]) for tr in self.trajectory_buffer if tr[0] == eID]
            states = torch.cat([t[0] for t in ep_data]).to(self.device)
            d_actions = torch.tensor([t[1] for t in ep_data]).to(self.device)
            u_actions = torch.tensor([t[2] for t in ep_data]).to(self.device)
            rewards = [t[3] for t in ep_data]
            dones = [t[5] for t in ep_data]
            T = len(states)

            # Initialize hidden states to zeros (episode boundary)
            n_layers_d = self.Actor_Critic_d.actor.n_layers
            hidden_dim_d = self.Actor_Critic_d.actor.hidden_dim
            n_layers_u = self.Actor_Critic_u.actor.n_layers
            hidden_dim_u = self.Actor_Critic_u.actor.hidden_dim
            # Critic may have different dims if architecture differs, but typically same
            hidden_dim_d_c = self.Actor_Critic_d.critic.hidden_dim
            hidden_dim_u_c = self.Actor_Critic_u.critic.hidden_dim

            d_actor_h = torch.zeros(n_layers_d, 1, hidden_dim_d, device=self.device)
            d_actor_c = torch.zeros(n_layers_d, 1, hidden_dim_d, device=self.device)
            d_critic_h = torch.zeros(n_layers_d, 1, hidden_dim_d_c, device=self.device)
            d_critic_c = torch.zeros(n_layers_d, 1, hidden_dim_d_c, device=self.device)
            u_actor_h = torch.zeros(n_layers_u, 1, hidden_dim_u, device=self.device)
            u_actor_c = torch.zeros(n_layers_u, 1, hidden_dim_u, device=self.device)
            u_critic_h = torch.zeros(n_layers_u, 1, hidden_dim_u_c, device=self.device)
            u_critic_c = torch.zeros(n_layers_u, 1, hidden_dim_u_c, device=self.device)

            # Collect per-chunk hidden states and logits/values
            chunk_boundaries = []  # (chunk_start, chunk_end, saved_hidden_dict)
            all_d_logits = []
            all_d_values = []
            all_u_logits = []
            all_u_values = []

            with torch.no_grad():
                for chunk_start in range(0, T, chunk_length):
                    chunk_end = min(chunk_start + chunk_length, T)
                    chunk_states = states[chunk_start:chunk_end].unsqueeze(0)  # (1, actual_len, features)

                    # Save hidden states BEFORE processing this chunk
                    saved_hidden = {
                        'd_actor_h': d_actor_h.clone(), 'd_actor_c': d_actor_c.clone(),
                        'd_critic_h': d_critic_h.clone(), 'd_critic_c': d_critic_c.clone(),
                        'u_actor_h': u_actor_h.clone(), 'u_actor_c': u_actor_c.clone(),
                        'u_critic_h': u_critic_h.clone(), 'u_critic_c': u_critic_c.clone(),
                    }

                    # Forward through d-network
                    d_logits_chunk, d_values_chunk, d_actor_final, d_critic_final = \
                        self.Actor_Critic_d.forward_sequence(
                            chunk_states, (d_actor_h, d_actor_c), (d_critic_h, d_critic_c))
                    d_actor_h, d_actor_c = d_actor_final
                    d_critic_h, d_critic_c = d_critic_final

                    # Forward through u-network
                    u_logits_chunk, u_values_chunk, u_actor_final, u_critic_final = \
                        self.Actor_Critic_u.forward_sequence(
                            chunk_states, (u_actor_h, u_actor_c), (u_critic_h, u_critic_c))
                    u_actor_h, u_actor_c = u_actor_final
                    u_critic_h, u_critic_c = u_critic_final

                    # Remove batch dim: (1, seq_len, dim) -> (seq_len, dim)
                    all_d_logits.append(d_logits_chunk.squeeze(0))
                    all_d_values.append(d_values_chunk.squeeze(0))
                    all_u_logits.append(u_logits_chunk.squeeze(0))
                    all_u_values.append(u_values_chunk.squeeze(0))
                    chunk_boundaries.append((chunk_start, chunk_end, saved_hidden))

            # Concatenate for full episode
            d_logits_all = torch.cat(all_d_logits, dim=0)  # (T, action_dim)
            d_values_all = torch.cat(all_d_values, dim=0)  # (T, 1)
            u_logits_all = torch.cat(all_u_logits, dim=0)
            u_values_all = torch.cat(all_u_values, dim=0)

            # Compute old log probs
            d_log_probs_old = F.log_softmax(d_logits_all, dim=-1).gather(1, d_actions.unsqueeze(1)).squeeze(1)
            u_log_probs_old = F.log_softmax(u_logits_all, dim=-1).gather(1, u_actions.unsqueeze(1)).squeeze(1)

            # Compute GAE
            advantages_d, returns_d, advantages_u, returns_u = self.compute_gae(
                rewards,
                d_values_all.cpu().numpy().flatten().tolist(),
                u_values_all.cpu().numpy().flatten().tolist(),
                dones
            )

            # Build CEM mask for this episode
            if use_CEM and eID in elite_segments and elite_segments[eID] is not None:
                elite_start, elite_end, _ = elite_segments[eID]
                cem_mask_ep = torch.zeros(T, device=self.device)
                cem_mask_ep[elite_start:elite_end+1] = 1.0
            else:
                cem_mask_ep = torch.zeros(T, device=self.device)

            # Package each chunk with padded data and validity mask
            for chunk_start, chunk_end, saved_hidden in chunk_boundaries:
                actual_len = chunk_end - chunk_start
                pad_len = chunk_length - actual_len

                cs, ce = chunk_start, chunk_end
                chunk_s = states[cs:ce]
                chunk_da = d_actions[cs:ce]
                chunk_ua = u_actions[cs:ce]
                chunk_dlp = d_log_probs_old[cs:ce]
                chunk_ulp = u_log_probs_old[cs:ce]
                chunk_adv_d = advantages_d[cs:ce]
                chunk_ret_d = returns_d[cs:ce]
                chunk_adv_u = advantages_u[cs:ce]
                chunk_ret_u = returns_u[cs:ce]
                chunk_cem = cem_mask_ep[cs:ce]

                mask = torch.ones(chunk_length, device=self.device)

                if pad_len > 0:
                    chunk_s = F.pad(chunk_s, (0, 0, 0, pad_len))
                    chunk_da = F.pad(chunk_da, (0, pad_len))
                    chunk_ua = F.pad(chunk_ua, (0, pad_len))
                    chunk_dlp = F.pad(chunk_dlp, (0, pad_len))
                    chunk_ulp = F.pad(chunk_ulp, (0, pad_len))
                    chunk_adv_d = F.pad(chunk_adv_d, (0, pad_len))
                    chunk_ret_d = F.pad(chunk_ret_d, (0, pad_len))
                    chunk_adv_u = F.pad(chunk_adv_u, (0, pad_len))
                    chunk_ret_u = F.pad(chunk_ret_u, (0, pad_len))
                    chunk_cem = F.pad(chunk_cem, (0, pad_len))
                    mask[actual_len:] = 0

                all_chunks.append({
                    'states': chunk_s,           # (chunk_length, features)
                    'd_actions': chunk_da,       # (chunk_length,)
                    'u_actions': chunk_ua,
                    'd_log_probs_old': chunk_dlp,
                    'u_log_probs_old': chunk_ulp,
                    'advantages_d': chunk_adv_d,
                    'returns_d': chunk_ret_d,
                    'advantages_u': chunk_adv_u,
                    'returns_u': chunk_ret_u,
                    'mask': mask,
                    'cem_mask': chunk_cem,
                    **saved_hidden,  # d_actor_h/c, d_critic_h/c, u_actor_h/c, u_critic_h/c
                })

        return all_chunks

    def _train_lstm(self, train_logger, use_CEM=False):
        """
        Chunk-based PPO-LSTM training. Splits episodes into sequential chunks,
        samples chunks randomly, and restores hidden states at chunk boundaries
        for temporal context. When use_CEM=True, replaces PPO policy loss with
        cross-entropy on elite (highest-reward) subsequences for self-imitation.

        Phase A (recompute old log probs / GAE / saved hidden states from the
        current network) is refreshed every self.phase_a_refresh_every Phase B
        epochs so the old-policy baseline tracks the moving weights instead of
        being frozen for all `self.epochs` updates.
        """
        chunk_length = self.chunk_length
        refresh_every = max(1, self.phase_a_refresh_every)

        # ---- Phase A: build chunks from current network ----
        all_chunks = self._build_phase_a_chunks(use_CEM)
        if not all_chunks:
            return
        num_chunks = len(all_chunks)
        num_chunks_per_batch = max(1, self.batch_size // chunk_length)

        # ---- Phase B: PPO minibatch updates using chunks ----
        for epoch in range(self.epochs):
            # Periodically rebuild Phase A so saved hidden states + old log probs
            # don't drift too far from the current (PPO-updated) network.
            if epoch > 0 and epoch % refresh_every == 0:
                all_chunks = self._build_phase_a_chunks(use_CEM)
                if not all_chunks:
                    return
                num_chunks = len(all_chunks)

            chunk_idxs = np.random.choice(num_chunks, min(num_chunks_per_batch, num_chunks), replace=False)
            B = len(chunk_idxs)

            # Stack sampled chunks into batched tensors
            batch_states = torch.stack([all_chunks[i]['states'] for i in chunk_idxs])         # (B, chunk_length, features)
            batch_d_actions = torch.stack([all_chunks[i]['d_actions'] for i in chunk_idxs])   # (B, chunk_length)
            batch_u_actions = torch.stack([all_chunks[i]['u_actions'] for i in chunk_idxs])
            batch_d_lp_old = torch.stack([all_chunks[i]['d_log_probs_old'] for i in chunk_idxs])
            batch_u_lp_old = torch.stack([all_chunks[i]['u_log_probs_old'] for i in chunk_idxs])
            batch_adv_d = torch.stack([all_chunks[i]['advantages_d'] for i in chunk_idxs])
            batch_ret_d = torch.stack([all_chunks[i]['returns_d'] for i in chunk_idxs])
            batch_adv_u = torch.stack([all_chunks[i]['advantages_u'] for i in chunk_idxs])
            batch_ret_u = torch.stack([all_chunks[i]['returns_u'] for i in chunk_idxs])
            batch_mask = torch.stack([all_chunks[i]['mask'] for i in chunk_idxs])
            batch_cem_mask = torch.stack([all_chunks[i]['cem_mask'] for i in chunk_idxs])

            # Stack hidden states along batch dim: (n_layers, 1, hidden_dim) -> (n_layers, B, hidden_dim)
            batch_d_actor_h = torch.cat([all_chunks[i]['d_actor_h'] for i in chunk_idxs], dim=1)
            batch_d_actor_c = torch.cat([all_chunks[i]['d_actor_c'] for i in chunk_idxs], dim=1)
            batch_d_critic_h = torch.cat([all_chunks[i]['d_critic_h'] for i in chunk_idxs], dim=1)
            batch_d_critic_c = torch.cat([all_chunks[i]['d_critic_c'] for i in chunk_idxs], dim=1)
            batch_u_actor_h = torch.cat([all_chunks[i]['u_actor_h'] for i in chunk_idxs], dim=1)
            batch_u_actor_c = torch.cat([all_chunks[i]['u_actor_c'] for i in chunk_idxs], dim=1)
            batch_u_critic_h = torch.cat([all_chunks[i]['u_critic_h'] for i in chunk_idxs], dim=1)
            batch_u_critic_c = torch.cat([all_chunks[i]['u_critic_c'] for i in chunk_idxs], dim=1)

            # === Decision Network ===
            d_logits_seq, d_values_seq, _, _ = self.Actor_Critic_d.forward_sequence(
                batch_states, (batch_d_actor_h, batch_d_actor_c), (batch_d_critic_h, batch_d_critic_c))

            # Flatten: (B, chunk_length, dim) -> (B*chunk_length, dim)
            d_logits_flat = d_logits_seq.reshape(B * chunk_length, -1)
            d_values_flat = d_values_seq.reshape(B * chunk_length, -1)
            d_actions_flat = batch_d_actions.reshape(B * chunk_length)
            d_lp_old_flat = batch_d_lp_old.reshape(B * chunk_length)
            adv_d_flat = batch_adv_d.reshape(B * chunk_length)
            ret_d_flat = batch_ret_d.reshape(B * chunk_length)
            mask_flat = batch_mask.reshape(B * chunk_length)
            cem_flat = batch_cem_mask.reshape(B * chunk_length)

            valid = mask_flat.bool()
            d_logits_v = d_logits_flat[valid]
            d_values_v = d_values_flat[valid]
            d_actions_v = d_actions_flat[valid]
            d_lp_old_v = d_lp_old_flat[valid]
            adv_d_v = adv_d_flat[valid]
            ret_d_v = ret_d_flat[valid]

            cem_d_valid = valid & cem_flat.bool()
            if use_CEM and cem_d_valid.any():
                d_logits_cem = d_logits_flat[cem_d_valid]
                d_actions_cem = d_actions_flat[cem_d_valid]
                d_policy_loss = F.cross_entropy(d_logits_cem, d_actions_cem)
            else:
                d_log_probs = F.log_softmax(d_logits_v, dim=1).gather(1, d_actions_v.unsqueeze(1)).squeeze(1)
                d_ratios = torch.exp(d_log_probs - d_lp_old_v)
                d_surr1 = d_ratios * adv_d_v
                d_surr2 = torch.clamp(d_ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_d_v
                d_policy_loss = -torch.min(d_surr1, d_surr2).mean()
            d_value_loss = F.mse_loss(d_values_v.squeeze(), ret_d_v)
            d_entropy_loss = -(torch.softmax(d_logits_v, dim=1) * F.log_softmax(d_logits_v, dim=1)).sum(dim=1).mean()

            d_loss = (self.policy_loss_coef * d_policy_loss +
                      self.value_loss_coef * d_value_loss -
                      self.entropy_coef * d_entropy_loss)

            self.optimizer_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Actor_Critic_d.parameters(), self.max_grad_norm)
            self.optimizer_d.step()
            self.scheduler_d.step()

            # === Utility Network: process full chunks for temporal context, loss only on d==1 ===
            u_logits_seq, u_values_seq, _, _ = self.Actor_Critic_u.forward_sequence(
                batch_states, (batch_u_actor_h, batch_u_actor_c), (batch_u_critic_h, batch_u_critic_c))

            u_logits_flat = u_logits_seq.reshape(B * chunk_length, -1)
            u_values_flat = u_values_seq.reshape(B * chunk_length, -1)
            u_actions_flat = batch_u_actions.reshape(B * chunk_length)
            u_lp_old_flat = batch_u_lp_old.reshape(B * chunk_length)
            adv_u_flat = batch_adv_u.reshape(B * chunk_length)
            ret_u_flat = batch_ret_u.reshape(B * chunk_length)

            # Mask: valid timesteps AND d_actions == 1
            u_mask = valid & (d_actions_flat == 1)

            if u_mask.any():
                u_logits_v = u_logits_flat[u_mask]
                u_values_v = u_values_flat[u_mask]
                u_actions_v = u_actions_flat[u_mask]
                u_lp_old_v = u_lp_old_flat[u_mask]
                adv_u_v = adv_u_flat[u_mask]
                ret_u_v = ret_u_flat[u_mask]

                cem_u_valid = u_mask & cem_flat.bool()
                if use_CEM and cem_u_valid.any():
                    u_logits_cem = u_logits_flat[cem_u_valid]
                    u_actions_cem = u_actions_flat[cem_u_valid]
                    u_policy_loss = F.cross_entropy(u_logits_cem, u_actions_cem)
                else:
                    u_log_probs = F.log_softmax(u_logits_v, dim=1).gather(1, u_actions_v.unsqueeze(1)).squeeze(1)
                    u_ratios = torch.exp(u_log_probs - u_lp_old_v)
                    u_surr1 = u_ratios * adv_u_v
                    u_surr2 = torch.clamp(u_ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_u_v
                    u_policy_loss = -torch.min(u_surr1, u_surr2).mean()
                u_value_loss = F.mse_loss(u_values_v.squeeze(), ret_u_v)
                u_entropy_loss = -(torch.softmax(u_logits_v, dim=1) * F.log_softmax(u_logits_v, dim=1)).sum(dim=1).mean()

                u_loss = (u_policy_loss +
                          self.value_loss_coef * u_value_loss -
                          self.entropy_coef * u_entropy_loss)

                self.optimizer_u.zero_grad()
                u_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor_Critic_u.parameters(), self.max_grad_norm)
                self.optimizer_u.step()
                self.scheduler_u.step()

                print(f'Utility Network - Policy Loss: {u_policy_loss.item():.4f}, '
                      f'Value Loss: {u_value_loss.item():.4f}, '
                      f'Entropy Loss: {u_entropy_loss.item():.4f}')
            else:
                u_policy_loss = torch.tensor(0.0)
                u_value_loss = torch.tensor(0.0)
                u_entropy_loss = torch.tensor(0.0)

            print(f'Decision Network - Policy Loss: {d_policy_loss.item():.4f}, '
                  f'Value Loss: {d_value_loss.item():.4f}, '
                  f'Entropy Loss: {d_entropy_loss.item():.4f}')

            train_logger.log_losses(
                d_policy_loss=d_policy_loss.item(), d_value_loss=d_value_loss.item(),
                d_entropy_loss=d_entropy_loss.item(), u_policy_loss=u_policy_loss.item(),
                u_value_loss=u_value_loss.item(), u_entropy_loss=u_entropy_loss.item())

    def get_max_contiguous_rewards(self, K):
        """
        Find best segments for CEM imitation learning.

        Default: best contiguous subarray of rewards per episode (minimum length K).
        If self.cem_full_episode=True: rank episodes by total reward, pick top 5 full episodes.

        :param K: Minimum length of subarray
        :return: Dictionary mapping episode -> (start_idx, end_idx, max_sum) or None
        """
        if not self.trajectory_buffer:
            return {}

        # Group trajectories by episode
        episodes = {}
        for ep, transition in self.trajectory_buffer:
            if ep not in episodes:
                episodes[ep] = []
            episodes[ep].append(transition)

        if self.cem_full_episode:
            # Top-5 full-episode selection
            episode_totals = {}
            for ep, transitions in episodes.items():
                if len(transitions) < K:
                    continue
                episode_totals[ep] = sum(t[3] for t in transitions)

            if not episode_totals:
                return {ep: None for ep in episodes}

            buy_eps  = [e for e in episode_totals if self.episode_sides.get(e, 0) > 0]
            sell_eps = [e for e in episode_totals if self.episode_sides.get(e, 0) < 0]
            # THREE regimes, not two. On a TWAP-absent episode TWAPPresent is
            # pinned to 0 all the way through, so episode_sides[ep] == 0 and the
            # episode falls into neither pool. In an alternating run both buy
            # and sell pools are always non-empty, so the global fallback below
            # never fires and absent episodes could never be elites -- meaning
            # self-imitation never reinforced a single standalone
            # market-making episode, in exactly the runs that exist to test
            # criterion 2 (profitable when the TWAP is absent).
            none_eps = [e for e in episode_totals if self.episode_sides.get(e, 0) == 0]
            pools = [p for p in (buy_eps, sell_eps, none_eps) if p]
            if len(pools) > 1:
                # Balance elites across regimes so none is starved, and so a
                # regime with systematically higher totals cannot monopolise the
                # set -- absent episodes have no meta-order to trade against, so
                # their totals are not comparable with present ones.
                #
                # The TOTAL is held fixed rather than the per-pool count. Taking
                # top-3 of each pool would give 9 elites in a three-regime run,
                # 6 in a two-regime one and 5 in a single-regime one, so the
                # elite count -- and with it the elite FRACTION of the ~40
                # episodes the buffer holds, i.e. the strength of the
                # intervention -- would vary with the arm being tested and
                # confound it. A looser elite fraction pulls the imitation
                # target toward the buffer mean, so CEM would do systematically
                # less in exactly the multi-regime arms.
                # Water-fill: smallest pool first, so a pool that cannot fill
                # its equal share passes the shortfall on to the larger pools
                # instead of silently shrinking the total. An even split alone
                # would give 5 elites, not 6, whenever one regime is thin --
                # which is exactly the early-buffer situation.
                top_eps = set()
                # None = legacy multi-regime total (top-3 of each of two pools)
                remaining = 6 if self.cem_n_elites is None else self.cem_n_elites
                ordered = sorted(pools, key=len)
                for i, pool in enumerate(ordered):
                    share = -(-remaining // (len(ordered) - i))   # ceil division
                    k = min(share, len(pool))
                    top_eps.update(sorted(pool, key=episode_totals.get, reverse=True)[:k])
                    remaining -= k
            else:
                # Single regime (or untagged): global top-N.
                sorted_eps = sorted(episode_totals, key=episode_totals.get, reverse=True)
                # None = legacy single-regime total, which was 5, not 6
                n = 5 if self.cem_n_elites is None else self.cem_n_elites
                top_eps = set(sorted_eps[:n])

            if self.cem_elite_floor:
                # `sorted(...)[:3]` returns three episodes regardless of quality.
                # Once the reward is PnL-dominated, episode totals are near
                # zero-mean noise (+-0.2 over ~2,480 steps, 85-88% of which move
                # no money), so the top-3 of ~40 buffered episodes is roughly the
                # +1.7 sigma tail -- mostly luck. Cross-entropy toward the
                # luckiest trajectories turns CEM from a bias into a variance
                # injector. Require elites to clear the buffer mean by this many
                # SDs, and skip them otherwise. Standard CEM practice.
                vals = list(episode_totals.values())
                if len(vals) > 1:
                    mu = sum(vals) / len(vals)
                    sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                    thresh = mu + self.cem_elite_floor * sd
                    top_eps = {e for e in top_eps if episode_totals[e] > thresh}

            results = {}
            for ep, transitions in episodes.items():
                if ep in top_eps:
                    results[ep] = (0, len(transitions) - 1, episode_totals[ep])
                else:
                    results[ep] = None
            return results

        # Default: best contiguous subarray per episode
        results = {}
        for ep, transitions in episodes.items():
            if len(transitions) < K:
                results[ep] = None
                continue

            rewards = [t[3] for t in transitions]

            max_sum = float('-inf')
            best_start = 0
            best_end = K - 1

            for start in range(len(rewards) - K + 1):
                current_sum = sum(rewards[start:start + K])

                if current_sum > max_sum:
                    max_sum = current_sum
                    best_start = start
                    best_end = start + K - 1

                for end in range(start + K, len(rewards)):
                    current_sum += rewards[end]
                    if current_sum > max_sum:
                        max_sum = current_sum
                        best_start = start
                        best_end = end

            results[ep] = (best_start, best_end, max_sum)

        return results

    def get_subarray_data(self, episode, start_idx, end_idx):
        """
        Helper method to extract state, d, u data for a given episode and index range

        :param episode: Episode number
        :param start_idx: Start index (inclusive)
        :param end_idx: End index (inclusive)
        :return: List of (state, d, u) tuples for the specified range
        """
        episode_data = []
        for ep, transition in self.trajectory_buffer:
            if ep == episode:
                episode_data.append(transition)

        if start_idx < 0 or end_idx >= len(episode_data) or start_idx > end_idx:
            return []

        result = []
        for i in range(start_idx, end_idx + 1):
            state, d, u, reward, next_state, done = episode_data[i]
            result.append((state, d, u))

        return result

    def get_CEM_data(self, type='d'):
        states = []
        actions = []
        results = self.get_max_contiguous_rewards(K=10)

        for episode, result in results.items():
            if result is not None:
                start_idx, end_idx, max_sum = result
                print(f"Episode {episode}: indices {start_idx}-{end_idx}, sum={max_sum}")

                # Get the corresponding state, d, u data
                subarray_data = self.get_subarray_data(episode, start_idx, end_idx)
                if type=='d':
                    for s, d, u in subarray_data:
                        states.append(s)
                        actions.append(d)
                else:
                    for s, d, u in subarray_data:
                        states.append(s)
                        actions.append(u)
            else:
                print(f"Episode {episode}: too short (< K steps)")

        return states, actions
    
class AdversarialPPOAgent(PPOAgent):
    def __init__(self, *args, TWAPPresent, **kwargs):
        super().__init__(*args, **kwargs)
        self.TWAPPresent = TWAPPresent
    
    def readData(self, data):
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
        self.last_intensity_bucket = self._intensity_bucket(lambdas_norm)
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
        state = torch.cat([state, torch.tensor([[self.TWAPPresent]], dtype=torch.float32).to(self.device)], 1)
        if self.enhance_state:
            state = torch.cat([state, torch.tensor([[skew, p_a < avgFillPrice, p_b > avgFillPrice,  bool_mo_bid, bool_mo_ask]], dtype=torch.float32).to(self.device)], 1)
        if self.include_time:
            state = torch.cat([state,torch.tensor([[time]], dtype=torch.float32).to(self.device)], 1)
        return state
