from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent
from HJBQVI.DGMTorch import MLP, ActorCriticMLP, ActorCriticSGMLP
from HJBQVI.utils import MinMaxScaler
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

def get_queue_priority(data, pos, label):
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        penalty = self.rewardpenalty * (self.countInventory()**2)
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
        networks_to_process = networks if 'networks' in locals() else [(network, target_network, optimizer)]

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

class PPOAgent(GymTradingAgent):
    def __init__(self, seed=1, log_events: bool = True, log_to_file: bool = False, strategy: str= "Random",
                 Inventory: Optional[Dict[str, Any]]=None, cash: int=5000, action_freq: float =0.5,
                 wake_on_MO: bool=True, wake_on_Spread: bool=True, cashlimit=1000000,
                 buffer_capacity=10000, batch_size=64, epochs=1000, layer_widths = 128, n_layers = 3, clip_ratio=0.2,
                 value_loss_coef=0.5, entropy_coef=10, max_grad_norm=0.5, gae_lambda=0.95, rewardpenalty = 0.1, hidden_activation='leaky_relu'):
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
                         wake_on_Spread=wake_on_Spread, cashlimit=cashlimit)

        self.resetseed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # PPO Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda

        # Training parameters
        self.layer_widths = layer_widths
        self.n_layers = n_layers
        self.hidden_activation = hidden_activation
        self.lr = 1e-3
        self.gamma = 0.99  # discount factor
        self.rewardpenalty = rewardpenalty  # inventory penalty
        self.last_state, self.last_action = None, None
        # Trajectory storage
        self.trajectory_buffer = []
        self.buffer_capacity = buffer_capacity

        # State scaler
        self.mmscaler = MinMaxScaler()

        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)

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
        state = torch.tensor([[self.cash, self.Inventory['INTC'], p_a, p_b, q_a, q_b, qD_a, qD_b, n_a, n_b, (p_a + p_b)*0.5] + list(lambdas.flatten()) + list(past_times.flatten())], dtype=torch.float32).to(self.device)
        return state

    def getState(self, state):
        """
        Scale and transform state

        :param state: Input state
        :return: Scaled state
        """
        if type(state) == dict:
            state = self.readData(state)

        # Scaling the state
        if len(self.trajectory_buffer) > 10:
            states = torch.cat([torch.cat([tr[1][0] for tr in self.trajectory_buffer])])
            self.mmscaler.fit(states)
            state = self.mmscaler.transform(state)
        else:
            state = self.mmscaler.fit_transform(state)

        return state

    def setupTraining(self, net):
        def lr_lambda(epoch):
            # Learning rate schedule
            return np.max([1e-6, 0.1**(epoch//10000)])

        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def setupNNs(self, data0):
        """
        Setup neural networks for PPO with two networks (d and u)

        :param data0: Initial data for determining state dimensions
        """
        # Get state dimensions
        data0 = self.getState(data0)
        state_dim = len(data0[0])

        # Initialize main networks with shared architecture
        self.Actor_Critic_d = ActorCriticMLP(state_dim, self.layer_widths, self.n_layers, 2, actor_activation='tanh', hidden_activation=self.hidden_activation, q_function = False)
        self.Actor_Critic_u = ActorCriticMLP(state_dim, self.layer_widths, self.n_layers, 12, actor_activation='tanh', hidden_activation=self.hidden_activation,q_function = False)

        # Move all models to appropriate device
        self.Actor_Critic_d.to(self.device)
        self.Actor_Critic_u.to(self.device)

        # Setup optimizers
        self.optimizer_d, self.scheduler_d = self.setupTraining(self.Actor_Critic_d)
        self.optimizer_u, self.scheduler_u = self.setupTraining(self.Actor_Critic_u)

    def calculaterewards(self, termination) -> Any:
        penalty = self.rewardpenalty * (self.countInventory()**2)
        self.profit = self.cash - self.statelog[0][1]
        self.updatestatelog()
        deltaPNL = self.statelog[-1][2] - self.statelog[-2][2]
        deltaInv = self.statelog[-1][3]*self.statelog[-1][-1] - self.statelog[-2][3]*self.statelog[-2][-1]
        # if self.istruncated or termination:
        #     deltaPNL += self.countInventory() * self.mid
        # reward shaping
        if self.last_action != 12:
            penalty -= 10 # custom reward for incentivising actions rather than inaction for learning
        if (self.last_state.cpu().numpy()[0][8] < self.last_state.cpu().numpy()[0][4] + self.last_state.cpu().numpy()[0][6]) and (self.last_state.cpu().numpy()[0][9] < self.last_state.cpu().numpy()[0][5] + self.last_state.cpu().numpy()[0][7]):
            penalty -= 20 # custom reward for double sided quoting
        return deltaPNL + deltaInv - penalty

    def get_action(self, data, epsilon=0.1):
        """
        Get action using current policy with epsilon-greedy exploration

        :param data: Input trading data
        :param epsilon: Exploration probability
        :return: Chosen action and its log probabilities
        """
        origData = data.copy()
        state = self.readData(data)
        self.last_state = state
        # Exploration
        if (random.random() < epsilon) or (len(self.trajectory_buffer) < 10):
            # Random decision
            d = random.randint(0, 1)
            u = random.randint(0, 11)

            # Compute dummy logits for logging
            d_logits, d_value = self.Actor_Critic_d(state)
            u_logits, u_value = self.Actor_Critic_u(state)

            # Get log probabilities
            d_log_prob = torch.log_softmax(d_logits, dim=1)[0, d]
            u_log_prob = torch.log_softmax(u_logits, dim=1)[0, u]

            # Validation checks (similar to original implementation)
            if int(u) in [1, 3, 8, 10]:  # cancels
                a = self.actions[int(u)]
                lvl = self.actionsToLevels[a]
                if len(origData['Positions'][lvl]) == 0:  # no position to cancel
                    self.last_action = 12
                    return 12, (d, 0), d_log_prob.item(), 0, d_value.item(), 0

            if int(u) in [5, 6]:
                p_a, q_a = origData['LOB0']['Ask_L1']
                p_b, q_b = origData['LOB0']['Bid_L1']
                if p_a - p_b < 0.015:  # reject if inspread not possible
                    self.last_action = 12
                    return 12, (d, 0), d_log_prob.item(), 0, d_value.item(), 0

            if (int(u) == 4) and (self.countInventory() < 1):  # ask mo
                self.last_action = 12
                return 12, (d, 0), d_log_prob.item(), 0, d_value.item(), 0

            self.last_action = u
            return u, (d, u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

        # Exploitation
        with torch.no_grad():
            # Decision network
            d_logits, d_value = self.Actor_Critic_d(state)
            d_probs = torch.softmax(d_logits, dim=1).squeeze()
            d = torch.multinomial(d_probs, 1).item()
            d_log_prob = torch.log(d_probs[d])

            # If no decision to act (d=0)
            if d == 0:
                self.last_action = 12
                return 12, (d, 0), d_log_prob.item(), 0, d_value.item(), 0

            # Utility network
            u_logits, u_value = self.Actor_Critic_u(state)
            u_probs = torch.softmax(u_logits, dim=1).squeeze()
            u = torch.multinomial(u_probs, 1).item()
            u_log_prob = torch.log(u_probs[u])

            # Validation checks (similar to original implementation)
            if int(u) in [1, 3, 8, 10]:  # cancels
                a = self.actions[int(u)]
                lvl = self.actionsToLevels[a]
                if len(origData['Positions'][lvl]) == 0:  # no position to cancel
                    self.last_action = 12
                    return 12, (d, 0), d_log_prob.item(), 0, d_value.item(), 0

            if int(u) in [5, 6]:
                p_a, q_a = origData['LOB0']['Ask_L1']
                p_b, q_b = origData['LOB0']['Bid_L1']
                if p_a - p_b < 0.015:  # reject if inspread not possible
                    self.last_action = 12
                    return 12, (d, 0), d_log_prob.item(), 0, d_value.item(), 0

            if (int(u) == 4) and (self.countInventory() < 1):  # ask mo
                self.last_action = 12
                return 12, (d, 0), d_log_prob.item(), 0, d_value.item(), 0
            self.last_action = u
            return u, (d, u), d_log_prob.item(), u_log_prob.item(), d_value.item(), u_value.item()

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

    def train(self):
        """
        PPO training method using entire episode trajectory
        """
        # Ensure we have a full trajectory
        if len(self.trajectory_buffer) < 2:
            return

        # Prepare training data
        states = torch.cat([tr[1][0] for tr in self.trajectory_buffer]).to(self.device)
        d_actions = torch.tensor([tr[1][1] for tr in self.trajectory_buffer]).to(self.device)
        u_actions = torch.tensor([tr[1][2] for tr in self.trajectory_buffer]).to(self.device)
        rewards = [tr[1][3] for tr in self.trajectory_buffer]
        dones = [tr[1][5] for tr in self.trajectory_buffer]

        # Compute values and log probabilities
        with torch.no_grad():
            # Decision network
            d_logits_old, values_d_old = self.Actor_Critic_d(states)
            d_log_probs_old = F.log_softmax(d_logits_old, dim=1).gather(1, d_actions.unsqueeze(1)).squeeze()
            # values_d_old = torch.stack([self.Critic_d(s)[1] for s in states]).squeeze()

            # Utility network
            u_logits_old, values_u_old = self.Actor_Critic_u(states)
            u_log_probs_old = F.log_softmax(u_logits_old, dim=1).gather(1, u_actions.unsqueeze(1)).squeeze()
            # values_u_old = torch.stack([self.Critic_u(s)[1] for s in states]).squeeze()

        # Compute Generalized Advantage Estimation
        advantages_d, returns_d, advantages_u, returns_u = self.compute_gae(
            rewards,
            values_d_old.cpu().numpy().flatten().tolist(),
            values_u_old.cpu().numpy().flatten().tolist(),
            dones
        )

        # PPO training for multiple epochs
        for _ in range(self.epochs):
            # Decision Network Training
            # Current policy output
            d_logits, d_values_pred = self.Actor_Critic_d(states)
            d_log_probs = F.log_softmax(d_logits, dim=1).gather(1, d_actions.unsqueeze(1)).squeeze()

            # Compute ratios
            d_ratios = torch.exp(d_log_probs - d_log_probs_old)

            # PPO Clipped Objective for Decision Network
            d_surr1 = d_ratios * advantages_d
            d_surr2 = torch.clamp(d_ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_d
            d_policy_loss = -torch.min(d_surr1, d_surr2).mean()

            # Value loss for Decision Network
            # d_values_pred, _ = self.Critic_d(states)
            d_value_loss = F.mse_loss(d_values_pred.squeeze(), returns_d)

            # Entropy for Decision Network
            d_entropy_loss = -(torch.softmax(d_logits, dim=1) * F.log_softmax(d_logits, dim=1)).sum(dim=1).mean()

            # Total Decision Network Loss
            d_loss = (d_policy_loss +
                      self.value_loss_coef * d_value_loss -
                      self.entropy_coef * d_entropy_loss)

            # Optimize Decision Network
            self.optimizer_d.zero_grad()
            d_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.Actor_Critic_d.parameters(), self.max_grad_norm)
            self.optimizer_d.step()

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

                # Print losses for monitoring
                print(f'Utility Network - Policy Loss: {u_policy_loss.item():.4f}, '
                      f'Value Loss: {u_value_loss.item():.4f}, '
                      f'Entropy Loss: {u_entropy_loss.item():.4f}')

            # Print losses for monitoring
            print(f'Decision Network - Policy Loss: {d_policy_loss.item():.4f}, '
                  f'Value Loss: {d_value_loss.item():.4f}, '
                  f'Entropy Loss: {d_entropy_loss.item():.4f}')

            return d_policy_loss.item(), d_value_loss.item(), d_entropy_loss.item(), u_policy_loss.item(), u_value_loss.item(), u_entropy_loss.item()

        # Clear trajectory buffer after training
        while len(self.trajectory_buffer) > self.buffer_capacity:
            eID = self.trajectory_buffer[0][0]
            end_idx = np.max([i for i in range(len(self.trajectory_buffer)) if self.trajectory_buffer[i][0] == eID]) + 1
            self.trajectory_buffer = self.trajectory_buffer[end_idx:]