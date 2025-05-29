import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import time
from typing import List, Dict, Any, Tuple
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import gymnasium as gym
import numpy as np
import copy
from typing import Any, Optional
import logging
import matplotlib.pyplot as plt
from HawkesRLTrading.src.SimulationEntities.GymTradingAgent import GymTradingAgent, RandomGymTradingAgent
from HawkesRLTrading.src.SimulationEntities.MetaOrderTradingAgents import TWAPGymTradingAgent
from HawkesRLTrading.src.SimulationEntities.ImpulseControlAgent import ImpulseControlAgent
from HawkesRLTrading.src.SimulationEntities.ProbabilisticAgent import ProbabilisticAgent
from HawkesRLTrading.src.SimulationEntities.ICRLAgent import ICRLAgent, ICRL2, ICRLSG, PPOAgent
from HawkesRLTrading.src.Stochastic_Processes.Arrival_Models import ArrivalModel, HawkesArrival
from HawkesRLTrading.src.SimulationEntities.Exchange import Exchange
from HawkesRLTrading.src.Kernel import Kernel
from HJBQVI.utils import TrainingLogger, ModelManager, get_gpu_specs
from HawkesRLTradingEnv import tradingEnv, preprocessdata
import pickle
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
                 label: str = None):

        self.n_actors = n_actors
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.agent_config = agent_config or {}
        self.env_kwargs = env_kwargs or {}
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.label = label

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
            start_trading_lag = config.get('start_trading_lag', 0)
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
            experience_batch = self._run_rollout(local_agent, local_env_kwargs, actor_id)

            # Send experience to training process
            experience_queue.put((actor_id, experience_batch))
            episode_count += 1

    def _run_rollout(self, agent, env_kwargs, actor_id):
        """Run a single rollout of T timesteps"""

        # Create environment
        env = tradingEnv( stop_time=self.rollout_steps, wall_time_limit=23400, **env_kwargs)

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
                'log_prob': agent_action[2] if len(agent_action) > 2 else None,  # Store log prob if available
                'value': agent_action[3] if len(agent_action) > 3 else None     # Store value if available
            }
            experiences.append(experience)

            step_count += 1

        return {
            'experiences': experiences,
            'episode_reward': episode_reward,
            'episode_length': step_count,
            'actor_id': actor_id
        }

    def _update_master_weights(self, experiences_batch):
        """Update master agent with collected experiences and perform PPO update"""

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
                    epID,  # Episode number not used in this context
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
        pause_events =  [Event() for _ in range(self.n_actors)]
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
                            print(e)
                            print(f"Failed to send weights to actor {i}")

                # Log progress
                if len(self.episode_rewards) > 0:
                    recent_rewards = self.episode_rewards[-self.n_actors:]
                    avg_recent_reward = np.mean(recent_rewards)
                    print(f"Recent average reward: {avg_recent_reward:.4f}")

                # Save model periodically
                if (iteration + 1) % 50 == 0:
                    self.model_manager.save_models(
                        epoch=iteration + 1,
                        d=self.master_agent.Actor_Critic_d,
                        u=self.master_agent.Actor_Critic_u
                    )
                    print(f"Model saved at iteration {iteration + 1}")

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

            print("Training completed!")

    def get_training_statistics(self):
        """Return training statistics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_stats': self.training_stats,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0
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