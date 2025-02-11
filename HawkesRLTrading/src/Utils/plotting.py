import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Action labels
ACTIONS = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask", "co_top_Ask", "mo_Ask", "lo_inspread_Ask",
           "lo_inspread_Bid", "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid", "lo_deep_Bid", 'None']

def plot_inventory_cash_positions(timestamps, inventories, cash_holdings):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(timestamps, inventories, label='Inventory', color='blue')
    ax2.plot(timestamps, cash_holdings, label='Cash Holdings', color='green')

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Inventory", color='blue')
    ax2.set_ylabel("Cash Holdings", color='green')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title("Inventory and Cash Holdings Over Time")
    plt.show()

def plot_lob_evolution(timestamps, lob_data):
    fig, ax = plt.subplots(figsize=(12, 6))

    ask_L1 = [lob["LOB0"]["Ask_L1"][0] for lob in lob_data]
    ask_L2 = [lob["LOB0"]["Ask_L2"][0] for lob in lob_data]
    bid_L1 = [lob["LOB0"]["Bid_L1"][0] for lob in lob_data]
    bid_L2 = [lob["LOB0"]["Bid_L2"][0] for lob in lob_data]

    ax.plot(timestamps, ask_L1, label='Ask L1', color='red')
    ax.plot(timestamps, ask_L2, label='Ask L2', linestyle='dashed', color='red')
    ax.plot(timestamps, bid_L1, label='Bid L1', color='green')
    ax.plot(timestamps, bid_L2, label='Bid L2', linestyle='dashed', color='green')

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    plt.title("LOB Evolution Over Time")
    plt.show()

def plot_cumulative_rewards(timestamps, rewards):
    cum_rewards = np.cumsum(rewards)
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, cum_rewards, label="Cumulative Rewards", color="purple")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Rewards")
    plt.title("Cumulative Rewards Over Time")
    plt.legend()
    plt.show()

def plot_actions(timestamps, actions):
    plt.figure(figsize=(12, 6))
    print(actions)
    plt.scatter(timestamps, [ACTIONS[a[0]] for a in actions], c=[a[0] for a in actions])
    plt.xlabel("Time")
    plt.ylabel("Actions")
    plt.title("Actions Over Time")
    plt.legend()
    plt.show()

def run_and_plot_env(tradingEnv, **kwargs):
    env = tradingEnv(stop_time=3000, seed=1, **kwargs)
    observations_log = []
    actions_log = []
    rewards_log = []
    timestamps = []

    print("Initial Observations", env.getobservations())
    Simstate, observations, termination = env.step(action=None)
    i = 0

    while not Simstate["Done"] and not termination:
        AgentsIDs = [k for k, v in Simstate["Infos"].items() if v]
        if len(AgentsIDs) > 1:
            raise Exception("Multiple gym agents are not yet implemented")
        agent = env.getAgent(ID=AgentsIDs[0])
        assert isinstance(agent, GymTradingAgent), "Agent must be a GymTradingAgent"
        action = (agent.id, agent.get_action(data=observations))

        # print(f"Limit Order Book: {observations['LOB0']}")
        # print(f"Action: {action}")

        Simstate, observations, termination = env.step(action=action)
        observations_log.append(observations)
        actions_log.append(action[1])
        # print
        rewards_log.append(agent.calculaterewards())
        timestamps.append(i)
        i += 1

    plot_inventory_cash_positions(timestamps,
                                  [obs['Inventory'] for obs in observations_log],
                                  [obs['Cash'] for obs in observations_log])
    plot_lob_evolution(timestamps, observations_log)
    plot_cumulative_rewards(timestamps, np.array(rewards_log))
    plot_actions(timestamps, actions_log)

# Example call:
# run_and_plot_env(tradingEnv, **kwargs)
