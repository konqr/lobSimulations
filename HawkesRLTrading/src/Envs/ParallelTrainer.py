import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe
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
            rewardpenalty=config.get('rewardpenalty', 0.5)
        )

    def _actor_worker(self, actor_id: int, shared_weights_queue: Queue,
                      experience_queue: Queue, control_queue: Queue):
        """Worker function for parallel actors"""

        # Create local agent and environment
        local_agent_config = copy.deepcopy(self.agent_config)
        local_agent_config['seed'] = self.agent_config.get('seed', 1) + actor_id
        local_agent = self._create_agent(local_agent_config)

        # Setup environment kwargs
        local_env_kwargs = copy.deepcopy(self.env_kwargs)
        local_env_kwargs['agent_instance'] = local_agent

        episode_count = 0

        while True:
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

        while (Simstate["Done"] == False and
               termination != True and
               step_count < self.rollout_steps):

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

        for actor_id, batch in experiences_batch:
            all_experiences.extend(batch['experiences'])
            total_reward += batch['episode_reward']
            total_episodes += 1

            # Log episode statistics
            self.episode_rewards.append(batch['episode_reward'])
            self.episode_lengths.append(batch['episode_length'])

        # Store experiences in master agent's buffer
        for exp in all_experiences:
            self.master_agent.store_transition(
                episode=0,  # Episode number not used in this context
                state=exp['state'],
                action=exp['action'],
                reward=exp['reward'],
                next_state=exp['next_state'],
                done=exp['done']
            )

        # Perform PPO training
        training_stats = []
        if len(all_experiences) >= self.batch_size:
            for epoch in range(self.ppo_epochs):
                stats = self.master_agent.train_step()
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

        # Start actor processes
        actors = []
        for i in range(self.n_actors):
            actor = Process(
                target=self._actor_worker,
                args=(i, shared_weights_queues[i], experience_queue, control_queues[i])
            )
            actor.start()
            actors.append(actor)

        try:
            for iteration in range(num_iterations):
                print(f"\n=== Training Iteration {iteration + 1}/{num_iterations} ===")

                # Collect experiences from all actors
                experiences_batch = []
                actors_completed = 0

                # Wait for all actors to complete their rollouts
                while actors_completed < self.n_actors:
                    try:
                        actor_id, batch = experience_queue.get(timeout=300)  # 5 minute timeout
                        experiences_batch.append((actor_id, batch))
                        actors_completed += 1
                        print(f"Received experience from actor {actor_id}")
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
                        except:
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
    agent_config = {
        'seed': 1,
        'log_events': True,
        'log_to_file': True,
        'strategy': 'your_strategy',
        'Inventory': 0,
        'cash': 100000,
        'action_freq': 1,
        'wake_on_MO': False,
        'wake_on_Spread': False,
        'cashlimit': 1000000,
        'inventorylimit': 1000,
        'layer_widths': [256, 256],
        'n_layers': 2,
        'buffer_capacity': 100000,
        'rewardpenalty': 0.5
    }

    # Environment kwargs (adapt to your environment setup)
    env_kwargs = {
        'GymTradingAgent': [{'agent_instance': None}]  # Will be filled by trainer
    }

    # Create trainer
    trainer = ParallelPPOTrainer(
        n_actors=4,           # Number of parallel actors
        rollout_steps=400,    # T stop_time for each rollout
        ppo_epochs=4,         # Number of PPO epochs per training step
        batch_size=256,
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