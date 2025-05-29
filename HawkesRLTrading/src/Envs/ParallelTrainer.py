from multiprocessing import Process, Queue, Event
import numpy as np
import copy
import logging
from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent
from HawkesRLTrading.src.SimulationEntities.ICRLAgent import PPOAgent
from HJBQVI.utils import TrainingLogger, ModelManager
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import tradingEnv, preprocessdata
import pickle
import torch
import matplotlib.pyplot as plt
import os
from collections import defaultdict

logging.basicConfig()
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

class ParallelPPOTrainer:
    def __init__(self,
                 n_actors: int = 4,
                 rollout_steps: int = 400,  # T stop_time
                 ppo_epochs: int = 4,
                 batch_size: int = 256,
                 agent_config: dict = None,
                 env_kwargs: dict = None,
                 log_dir: str = None,
                 model_dir: str = None,
                 label: str = None,
                 transaction_cost: float = 0.0001):

        self.n_actors = n_actors
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.agent_config = agent_config or {}
        self.env_kwargs = env_kwargs or {}
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.label = label
        self.tc = transaction_cost

        # Initialize shared components
        self.train_logger = TrainingLogger(
            layer_widths=agent_config.get('layer_widths', [256, 256]),
            n_layers=agent_config.get('n_layers', 2),
            log_dir=log_dir,
            label=label
        )
        self.model_manager = ModelManager(model_dir=model_dir, label=label)

        # Create master agent for training
        self.master_agent = self._create_agent(agent_config)

        # Statistics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = []

        # Plotting data tracking
        self.cash_history = []
        self.inventory_history = []
        self.time_history = []
        self.actions_history = []
        self.episode_boundaries = [0]
        self.pnl_history = []
        self.avgEpisodicRewards = []
        self.stdEpisodicRewards = []
        self.finalcash = []
        self.profit_counter = 0

        # Episode-wise tracking for individual episodes
        self.episode_data = defaultdict(list)  # Store data per episode for detailed plotting

    def _create_agent(self, config):
        """Create a PPO agent instance"""
        return PPOAgent(
            seed=config.get('seed', 1),
            log_events=config.get('log_events', True),
            log_to_file=config.get('log_to_file', True),
            strategy=config.get('strategy', 'default'),
            Inventory=config.get('Inventory', 0),
            cash=config.get('cash', 100000),
            action_freq=config.get('action_freq', 1),
            wake_on_MO=config.get('wake_on_MO', False),
            wake_on_Spread=config.get('wake_on_Spread', False),
            cashlimit=config.get('cashlimit', 1000000),
            inventorylimit=config.get('inventorylimit', 1000),
            batch_size=self.batch_size,
            layer_widths=config.get('layer_widths', [256, 256]),
            n_layers=config.get('n_layers', 2),
            buffer_capacity=config.get('buffer_capacity', 100000),
            rewardpenalty=config.get('rewardpenalty', 0.5),
            epochs=config.get('epochs',10),
            start_trading_lag = config.get('start_trading_lag', 0),
            transaction_cost=self.tc
        )

    def _actor_worker(self, actor_id: int, pause_event:Event, shared_weights_queue: Queue,
                      experience_queue: Queue, control_queue: Queue):
        """Worker function for parallel actors"""

        # Create local agent and environment
        local_agent_config = copy.deepcopy(self.agent_config)
        local_agent_config['seed'] = self.agent_config.get('seed', 1) + actor_id
        local_agent = self._create_agent(local_agent_config)

        # Setup environment kwargs
        local_env_kwargs = copy.deepcopy(self.env_kwargs)
        j = local_env_kwargs['GymTradingAgent'][0]
        j['agent_instance'] = local_agent
        local_env_kwargs['GymTradingAgent'] = [j]

        episode_count = 0

        while pause_event.wait():
            # Check for control signals
            try:
                if not control_queue.empty():
                    signal = control_queue.get_nowait()
                    if signal == "STOP":
                        break
            except:
                pass

            # Update weights from master if available
            try:
                if not shared_weights_queue.empty():
                    new_weights = shared_weights_queue.get_nowait()
                    local_agent.update_weights(new_weights)
            except:
                pass

            # Run rollout
            experience_batch = self._run_rollout(local_agent, local_env_kwargs, actor_id, episode_count)

            # Send experience to training process
            experience_queue.put((actor_id, experience_batch))
            episode_count += 1

    def _run_rollout(self, agent, env_kwargs, actor_id, episode_id):
        """Run a single rollout of T timesteps"""

        # Create environment
        env = tradingEnv(stop_time=self.rollout_steps, wall_time_limit=23400, **env_kwargs)

        # Initialize episode
        Simstate, observations, termination, truncation = env.step(action=None)
        AgentsIDs = [k for k, v in Simstate["Infos"].items() if v == True]
        gym_agent: GymTradingAgent = env.getAgent(ID=AgentsIDs[0])

        # Setup neural networks if first time
        if not hasattr(gym_agent, 'networks_setup'):
            gym_agent.setupNNs(observations)
            gym_agent.networks_setup = True

        # Collect experience for this rollout
        experiences = []
        step_count = 0
        episode_reward = 0

        # Episode-specific tracking for plotting
        episode_cash = []
        episode_inventory = []
        episode_time = []
        episode_actions = []

        while (Simstate["Done"] == False and
               termination != True):

            AgentsIDs = [k for k, v in Simstate["Infos"].items() if v == True]

            if len(AgentsIDs) > 1:
                raise Exception("Multiple gym agents not yet implemented")

            gym_agent: GymTradingAgent = env.getAgent(ID=AgentsIDs[0])

            # Get action from agent
            epsilon = 0.1 if step_count > 100 else 0.5
            agent_action = gym_agent.get_action(data=observations, epsilon=epsilon)
            action = (gym_agent.id, (agent_action[0], 1))

            # Store previous observations
            observations_prev = observations.copy()

            # Take step in environment
            Simstate, observations, termination, truncation = env.step(action=action)

            # Calculate reward
            reward = gym_agent.calculaterewards(termination)
            episode_reward += reward

            # Store experience
            experience = {
                'state': gym_agent.readData(observations_prev),
                'action': agent_action[1],
                'reward': reward,
                'next_state': gym_agent.readData(observations),
                'done': (termination or truncation),
                'log_prob': agent_action[2] if len(agent_action) > 2 else None,
                'value': agent_action[3] if len(agent_action) > 3 else None
            }
            experiences.append(experience)

            # Collect data for plotting
            episode_cash.append(observations['Cash'])
            episode_inventory.append(observations['Inventory'])
            episode_time.append(Simstate['TimeCode'])
            episode_actions.append(action[1][0])

            step_count += 1

        # Calculate final PnL for this episode
        final_pnl = episode_cash[-1] + episode_inventory[-1] * gym_agent.mid * (1 - self.tc * np.sign(episode_inventory[-1])) if episode_cash else 0

        return {
            'experiences': experiences,
            'episode_reward': episode_reward,
            'episode_length': step_count,
            'actor_id': actor_id,
            'episode_id': episode_id,
            'plotting_data': {
                'cash': episode_cash,
                'inventory': episode_inventory,
                'time': episode_time,
                'actions': episode_actions,
                'final_pnl': final_pnl,
                'mid_price': gym_agent.mid if hasattr(gym_agent, 'mid') else 100
            }
        }

    def _update_plotting_data(self, experiences_batch):
        """Update plotting data from collected experiences"""

        for actor_id, batch in experiences_batch:
            plotting_data = batch['plotting_data']

            # Check for episode boundary (reset in time)
            if (len(self.time_history) > 0 and
                    len(plotting_data['time']) > 0 and
                    plotting_data['time'][0] < self.time_history[-1]):
                self.episode_boundaries.append(len(self.cash_history))

            # Append data to global history
            self.cash_history.extend(plotting_data['cash'])
            self.inventory_history.extend(plotting_data['inventory'])
            self.time_history.extend(plotting_data['time'])
            self.actions_history.extend(plotting_data['actions'])

            # Calculate PnL for each step
            mid_price = plotting_data['mid_price']
            for i, (cash, inventory) in enumerate(zip(plotting_data['cash'], plotting_data['inventory'])):
                pnl = cash + inventory * mid_price * (1 - self.tc * np.sign(inventory) if inventory != 0 else 0)
                self.pnl_history.append(pnl)

            # Store episode final cash
            if plotting_data['cash']:
                final_cash = plotting_data['cash'][-1] + plotting_data['inventory'][-1] * mid_price
                self.finalcash.append(final_cash)

            self.profit_counter += len(plotting_data['cash'])

    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio using episode boundaries"""
        all_log_returns = []
        pft = np.array(self.pnl_history)

        # Calculate log returns for each episode separately
        for i in range(len(self.episode_boundaries)):
            start_idx = self.episode_boundaries[i]
            end_idx = self.episode_boundaries[i + 1] if i + 1 < len(self.episode_boundaries) else len(pft)

            if end_idx - start_idx > 1:  # Need at least 2 points for log returns
                episode_pnl = pft[start_idx:end_idx]
                episode_log_returns = np.diff(np.log(np.maximum(episode_pnl, 1e-8)))  # Avoid log(0)
                all_log_returns.extend(episode_log_returns)

        # Calculate Sharpe on concatenated log returns from all episodes
        if len(all_log_returns) > 0:
            all_log_returns = np.array(all_log_returns)
            sr = np.mean(all_log_returns) / np.std(all_log_returns) if np.std(all_log_returns) > 0 else 0
        else:
            sr = 0

        return sr

    def _plot_profit_overlay(self, save_path=None):
        """Plot profit with episode overlay similar to original"""
        if len(self.pnl_history) == 0:
            return

        plt.figure(figsize=(12, 8))

        pft = np.array(self.pnl_history)
        t_array = np.array(self.time_history)
        sr = self._calculate_sharpe_ratio()

        # Plot each episode as a separate line
        for i in range(len(self.episode_boundaries)):
            start_idx = self.episode_boundaries[i]
            end_idx = self.episode_boundaries[i + 1] if i + 1 < len(self.episode_boundaries) else len(pft)

            if end_idx > start_idx:  # Valid episode
                episode_t = t_array[start_idx:end_idx]
                episode_pnl = pft[start_idx:end_idx]
                episode_profit = episode_pnl - self.agent_config.get('cash', 2500)  # Profit relative to starting capital

                # Plot this episode
                if i == 0:
                    plt.plot(episode_t, episode_profit, alpha=0.7, label=f'Sharpe:{sr:0.4f}')
                else:
                    plt.plot(episode_t, episode_profit, alpha=0.7)

        plt.legend()
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.xlabel('Time in seconds')
        plt.ylabel('Profit in Dollars')
        plt.title('Final Profit - All Episodes Overlaid')

        if save_path:
            plt.savefig(save_path)
            np.save(save_path.replace('.png', ''), np.array([self.time_history, self.pnl_history]))
        plt.close()

    def _plot_policy_summary(self, save_path=None):
        """Plot policy summary (cash, inventory, actions)"""
        if len(self.cash_history) == 0:
            return

        plt.figure(figsize=(12, 8))

        plt.subplot(221)
        plt.plot(np.arange(len(self.cash_history)), self.cash_history)
        plt.title('Cash')

        plt.subplot(222)
        plt.plot(np.arange(len(self.inventory_history)), self.inventory_history)
        plt.title('Inventory')

        plt.subplot(223)
        plt.scatter(np.arange(len(self.actions_history)), self.actions_history, alpha=0.6)
        if hasattr(self.master_agent, 'actions'):
            plt.yticks(np.arange(len(self.master_agent.actions)), self.master_agent.actions)
        plt.title('Actions')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _plot_episodic_rewards(self, save_path=None):
        """Plot average episodic rewards with standard deviation"""
        if len(self.avgEpisodicRewards) == 0:
            return

        # Calculate moving average of final profits
        pft = np.array(self.finalcash) - self.agent_config.get('cash', 2500) if self.finalcash else np.array([])
        ma = np.convolve(pft, np.ones(min(5, len(pft)))/min(5, len(pft)), mode='valid') if len(pft) >= 5 else pft

        plt.figure(figsize=(12, 8))

        plt.subplot(211)
        if len(self.avgEpisodicRewards) > 0:
            plt.plot(np.arange(len(self.avgEpisodicRewards)), self.avgEpisodicRewards)
            if len(self.stdEpisodicRewards) > 0:
                plt.fill_between(np.arange(len(self.avgEpisodicRewards)),
                                 np.array(self.avgEpisodicRewards) - np.array(self.stdEpisodicRewards),
                                 np.array(self.avgEpisodicRewards) + np.array(self.stdEpisodicRewards),
                                 alpha=0.3)
        plt.title('Avg Episodic Rewards')

        plt.subplot(212)
        if len(ma) > 0:
            plt.plot(np.arange(len(ma)), ma)
            plt.ticklabel_format(useOffset=False, style='plain')
        plt.title('Final Profit MA')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _update_episodic_rewards(self):
        """Update episodic rewards statistics"""
        if hasattr(self.master_agent, 'trajectory_buffer') and len(self.master_agent.trajectory_buffer) > 0:
            episodic_rewards = []
            r = 0
            tmp = self.master_agent.trajectory_buffer[0][0]

            for ij in self.master_agent.trajectory_buffer:
                if ij[0] == tmp:
                    r += ij[1][3]  # Assuming reward is at index 3 in trajectory
                else:
                    episodic_rewards.append(r)
                    r = ij[1][3]
                    tmp = ij[0]

            if episodic_rewards:
                self.avgEpisodicRewards.append(np.mean(episodic_rewards))
                self.stdEpisodicRewards.append(np.std(episodic_rewards))

    def _update_master_weights(self, experiences_batch):
        """Update master agent with collected experiences and perform PPO update"""

        # Update plotting data first
        self._update_plotting_data(experiences_batch)

        # Aggregate all experiences
        all_experiences = []
        total_reward = 0
        total_episodes = 0
        eID = 0
        if len(self.master_agent.trajectory_buffer) > 0:
            eID = self.master_agent.trajectory_buffer[-1][0] + 1

        for actor_id, batch in experiences_batch:
            all_experiences.append((eID, batch['experiences']))
            total_reward += batch['episode_reward']
            total_episodes += 1
            eID += 1
            # Log episode statistics
            self.episode_rewards.append(batch['episode_reward'])
            self.episode_lengths.append(batch['episode_length'])

        # Store experiences in master agent's buffer
        for epID, exp in all_experiences:
            for one_exp in exp:
                self.master_agent.store_transition(
                    epID,
                    one_exp['state'],
                    one_exp['action'],
                    one_exp['reward'],
                    one_exp['next_state'],
                    one_exp['done']
                )
                if not hasattr(self.master_agent, 'networks_setup'):
                    self.master_agent.setupNNs(one_exp['state'])
                    self.master_agent.networks_setup = True

        # Perform PPO training
        training_stats = []
        for epoch in range(self.ppo_epochs):
            stats = self.master_agent.train(self.train_logger)
            training_stats.append(stats)

        # Update episodic rewards
        self._update_episodic_rewards()

        # Log training statistics
        avg_reward = total_reward / max(total_episodes, 1)
        print(f"Average episode reward: {avg_reward:.4f}")
        print(f"Total experiences collected: {len(all_experiences)}")

        return self.master_agent.get_weights(), training_stats

    def train(self, num_iterations: int = 500, checkpoint_params=None):
        """Main training loop"""

        # Load checkpoint if provided
        if checkpoint_params is not None:
            loaded_models = self.model_manager.load_models(
                timestamp=checkpoint_params[0],
                epoch=checkpoint_params[1],
                d=self.master_agent.Actor_Critic_d,
                u=self.master_agent.Actor_Critic_u
            )
            self.master_agent.Actor_Critic_d = loaded_models['d']
            self.master_agent.Actor_Critic_u = loaded_models['u']

        # Initialize multiprocessing components
        shared_weights_queues = [Queue() for _ in range(self.n_actors)]
        experience_queue = Queue()
        control_queues = [Queue() for _ in range(self.n_actors)]
        pause_events = [Event() for _ in range(self.n_actors)]

        # Start actor processes
        actors = []
        for i in range(self.n_actors):
            actor = Process(
                target=self._actor_worker,
                args=(i, pause_events[i], shared_weights_queues[i], experience_queue, control_queues[i])
            )
            pause_events[i].set()
            actor.start()
            actors.append(actor)

        try:
            for iteration in range(num_iterations):
                print(f"\n=== Training Iteration {iteration + 1}/{num_iterations} ===")

                # Collect experiences from all actors
                experiences_batch = []
                actors_completed = []

                # Wait for all actors to complete their rollouts
                while len(actors_completed) < self.n_actors:
                    try:
                        actor_id, batch = experience_queue.get(timeout=300)  # 5 minute timeout
                        experiences_batch.append((actor_id, batch))
                        actors_completed += [actor_id]
                        actors_completed = np.unique(actors_completed).tolist()
                        print(f"Received experience from actor {actor_id}")
                        pause_events[actor_id].clear()
                    except:
                        print("Timeout waiting for actor experience")
                        break

                # Update master agent and get new weights
                if experiences_batch:
                    new_weights, training_stats = self._update_master_weights(experiences_batch)
                    self.training_stats.extend(training_stats)

                    # Distribute new weights to all actors
                    for i in range(self.n_actors):
                        try:
                            shared_weights_queues[i].put(new_weights)
                            pause_events[i].set()
                        except Exception as e:
                            print(f"Failed to send weights to actor {i}: {e}")

                # Log progress
                if len(self.episode_rewards) > 0:
                    recent_rewards = self.episode_rewards[-self.n_actors:]
                    avg_recent_reward = np.mean(recent_rewards)
                    print(f"Recent average reward: {avg_recent_reward:.4f}")

                # Generate plots periodically
                if (iteration + 1) % 10 == 0 or iteration == 0:  # Plot every 10 iterations
                    if self.log_dir and self.label:
                        # Create profit overlay plot
                        profit_path = os.path.join(self.log_dir, f"{self.label}_profit.png")
                        self._plot_profit_overlay(save_path=profit_path)

                        # Create policy summary plot
                        policy_path = os.path.join(self.log_dir, f"{self.label}_policy.png")
                        self._plot_policy_summary(save_path=policy_path)

                        # Create episodic rewards plot
                        rewards_path = os.path.join(self.log_dir, f"{self.label}_avgepisodicreward.png")
                        self._plot_episodic_rewards(save_path=rewards_path)

                        print(f"Plots saved at iteration {iteration + 1}")

                # Save model periodically
                if (iteration + 1) % 10 == 0:
                    self.model_manager.save_models(
                        epoch=iteration + 1,
                        d=self.master_agent.Actor_Critic_d,
                        u=self.master_agent.Actor_Critic_u
                    )
                    print(f"Model saved at iteration {iteration + 1}")

                torch.cuda.empty_cache()
                self.train_logger.save_logs()
                self.train_logger.plot_losses(show=False, save=True)

        finally:
            # Clean shutdown
            print("Shutting down actors...")
            for i in range(self.n_actors):
                control_queues[i].put("STOP")

            # Wait for actors to finish
            for actor in actors:
                actor.join(timeout=10)
                if actor.is_alive():
                    actor.terminate()
                    actor.join()

            # Generate final plots
            if self.log_dir and self.label:
                profit_path = os.path.join(self.log_dir, f"{self.label}_final_profit.png")
                self._plot_profit_overlay(save_path=profit_path)

                policy_path = os.path.join(self.log_dir, f"{self.label}_final_policy.png")
                self._plot_policy_summary(save_path=policy_path)

                rewards_path = os.path.join(self.log_dir, f"{self.label}_final_avgepisodicreward.png")
                self._plot_episodic_rewards(save_path=rewards_path)

            print("Training completed!")

    def get_training_statistics(self):
        """Return training statistics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_stats': self.training_stats,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'final_cash': self.finalcash,
            'total_profit': np.sum(np.array(self.finalcash) - self.agent_config.get('cash', 2500)) if self.finalcash else 0
        }

# Usage example
def main():
    # Configuration
    with open("D:\\PhD\\calibrated params\\INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_Symm_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
        kernelparams = pickle.load(f)
    kernelparams = preprocessdata(kernelparams)
    # with open("D:\\PhD\\calibrated params\\INTC.OQ_Params_2019-01-02_2019-03-29_dictTOD_constt", 'rb') as f:
    #     tod = pickle.load(f)
    cols= ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
           "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    # kernelparams = [[np.zeros((12,12))]*4, np.array([[kernelparams[c]] for c in cols])]
    faketod = {}
    for k in cols:
        faketod[k] = {}
        for k1 in np.arange(13):
            faketod[k][k1] = 1.0
    tod=np.zeros(shape=(len(cols), 13))
    for i in range(len(cols)):
        tod[i]=[faketod[cols[i]][k] for k in range(13)]
    Pis={'Bid_L2': [0.,
                    [(1, 1.)]],
         'Bid_inspread': [0.,
                          [(1, 1.)]],
         'Bid_L1': [0.,
                    [(1, 1.)]],
         'Bid_MO': [0.,
                    [(1, 1.)]]}
    Pis["Ask_MO"] = Pis["Bid_MO"]
    Pis["Ask_L1"] = Pis["Bid_L1"]
    Pis["Ask_inspread"] = Pis["Bid_inspread"]
    Pis["Ask_L2"] = Pis["Bid_L2"]
    Pi_Q0= {'Ask_L1': [0.,
                       [(10, 1.)]],
            'Ask_L2': [0.,
                       [(10, 1.)]],
            'Bid_L1': [0.,
                       [(10, 1.)]],
            'Bid_L2': [0.,
                       [(10, 1.)]]}
    agent_config = {
        'seed': 1,
        'log_events': True,
        'log_to_file': True,
        'strategy': 'ICRL',
        'Inventory': {"INTC": 0},
        'cash': 2500,
        'action_freq': .2,
        'wake_on_MO': True,
        'wake_on_Spread': True,
        'cashlimit': 1000000,
        'inventorylimit': 25,
        'layer_widths': 128,
        'n_layers': 3,
        'buffer_capacity': 100000,
        'rewardpenalty': 0.5,
        'epochs':1000,
        'start_trading_lag' : 10,
        'agent_instance': None
    }
    env_kwargs={
        "TradingAgent": [],

        "GymTradingAgent": [agent_config], # Will be filled by trainer
        "Exchange": {"symbol": "INTC",
                     "ticksize":0.01,
                     "LOBlevels": 2,
                     "numOrdersPerLevel": 10,
                     "PriceMid0": 100,
                     "spread0": 0.03},
        "Arrival_model": {"name": "Hawkes",
                          "parameters": {"kernelparams": kernelparams,
                                         "tod": tod,
                                         "Pis": Pis,
                                         "beta": 0.941,
                                         "avgSpread": 0.0101,
                                         "Pi_Q0": Pi_Q0}}

        }

    # Create trainer
    trainer = ParallelPPOTrainer(
        n_actors=8,           # Number of parallel actors
        rollout_steps=20,    # T stop_time for each rollout
        ppo_epochs=1,         # Number of PPO epochs per training step
        batch_size=512,
        agent_config=agent_config,
        env_kwargs=env_kwargs,
        log_dir='logs',
        model_dir='models',
        label='parallel_ppo'
    )

    # Start training
    trainer.train(num_iterations=125)  # 125 * 4 actors = 500 total episodes equivalent

    # Get statistics
    stats = trainer.get_training_statistics()
    print(f"Final average reward: {stats['avg_reward']:.4f} ± {stats['std_reward']:.4f}")

if __name__ == "__main__":
    main()