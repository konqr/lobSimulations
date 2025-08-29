import sys
import os
sys.path.append(os.path.abspath('/home/ajafree/lobSimulations'))
# sys.path.append(os.path.abspath('/Users/alirazajafree/Documents/GitHub/lobSimulations/'))
from HawkesRLTrading.src.Envs.HawkesRLTradingEnv import *
# import matplotlib.pyplot as plt
            
# from scipy.optimize import curve_fit
# import numpy as np

# with open("/Users/alirazajafree/researchprojects/otherdata/INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_Symm_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
with open("/home/ajafree/researchprojects/otherdata/Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_2019-01-02_2019-12-31_CLSLogLin_10", 'rb') as f: # INTC.OQ_ParamsInferredWCutoff_2019-01-02_2019-03-31_poisson
    kernelparams = pickle.load(f)
kernelparams = preprocessdata(kernelparams)
cols= ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
        "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
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
kwargs={
            "TradingAgent": [],
            "GymTradingAgent":
                                [{"cash":10000000,
                                "cashlimit": 1000000000,
                                "strategy": "POV",
                                "on_trade":False,
                                "total_order_size":2300,
                                "order_target":"INTC",
                                "participation_rate":0.05,
                                "total_time":2600,
                                "window_size":60, #window size, measured in seconds
                                "side":"buy", #buy or sell
                                "action_freq":1,
                                "Inventory": {"INTC":0},
                                'start_trading_lag': 300,
                                "wake_on_MO": False,
                                "wake_on_Spread": False}],
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
agents = kwargs['GymTradingAgent']
i=0
t = []
inventoryhistories: Dict[int, Dict[int, list]] = {}  # episode -> agent_id -> list of (time, inventory)
cashs:Dict[int, List] = {}
inventories:Dict[int, List] = {}
actionss:Dict[int, List] = {}

starting_midprice:int = 0
price_paths = []
execution_history:List[Tuple] = []
price_paths_non_agent = []
times = []
market_volumes = []

cash_differences = 0

new_mv = True

env=tradingEnv(stop_time=2600, wall_time_limit=23400, seed=1, **kwargs)
prev_inventory = 0

Simstate, observations, termination, truncation =env.step(action=None) 
AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
observationsDict:Dict[int, Dict] = {agentid: {"Inventory": agent.Inventory, "Positions": []} for agent, agentid in zip(agents, AgentsIDs)}
while Simstate["Done"]==False and termination!=True:
    logger.debug(f"ENV TERMINATION: {termination}")
    AgentsIDs=[k for k,v in Simstate["Infos"].items() if v==True]
    print(f"Agents with IDs {AgentsIDs} have an action available")
    agents:List[GymTradingAgent] = [env.getAgent(ID=agentid) for agentid in AgentsIDs]
    action:list[Tuple] = []
    # price_paths_non_agent.append(float(observations.get('LOB0').get('Ask_L1')[0] + observations.get('LOB0').get('Bid_L1')[0])/2)
    times.append(Simstate["TimeCode"])
    
    for agent in agents:
        
        assert isinstance(agent, GymTradingAgent), "Agent with action should be a GymTradingAgent"
        agentAction:Tuple[int, int] = agent.get_action(data=env.getobservations(agentID=agent.id))
        action = (agent.id, agentAction)
        observations_prev = copy.deepcopy(observationsDict.get(agent.id, {}))
        print(f"Limit Order Book: {observationsDict.get(agent.id, {}).get('LOB0', '')}")
        print(f"Inventory: {observationsDict.get(agent.id, {}).get('Inventory', '')}")

        Simstate, observations, termination, truncation=env.step(action=action) #do not try and use this data before this line in the loop

        # midprice = float(observations.get('LOB0').get('Ask_L1')[0] + observations.get('LOB0').get('Bid_L1')[0])

        # price_paths.append(midprice)

        # if(i==0):
        #     starting_midprice = float(observations.get('LOB0').get('Ask_L1')[0] + observations.get('LOB0').get('Bid_L1')[0])/2

        observationsDict.update({agent.id:observations})
        logger.debug(f"\n Agent: {agent.id}\n Simstate: {Simstate}\nObservations: {observations}\nTermination: {termination}\nTruncation: {truncation}")

        cashs.update({agent.id:cashs.get(agent.id, [])+[observations['Cash']]})

        inventories.update({agent.id:inventories.get(agent.id, []) + [observations['Inventory']]})

        diff = abs(inventories[agent.id][-1] - prev_inventory)

        if((observations["current_time"]-100)%60 == 0):
            new_mv = True

        if new_mv:
             if observations["market_volume"] > 1:
                if(observations["market_volume"]*60 + observations["Inventory"] > 120):
                    market_volumes.append(observations["market_volume"]*60)
                else:
                    market_volumes.append(120-observations["Inventory"])
                new_mv = False

        # if(diff != 0):
        #     #inventory has changed, order has gone through
        #     if kwargs['GymTradingAgent'][0]["side"] == 'sell':
        #         execution_history.append((observationsDict.get(agent.id, {}).get('LOB0', '').get('Bid_L1'), diff))  
        #         cash_differences += (observationsDict.get(agent.id, {}).get('LOB0', '').get('Bid_L1')[0])*diff
        #     else:
        #         execution_history.append((observationsDict.get(agent.id, {}).get('LOB0', '').get('Ask_L1'), diff)) 
        #         cash_differences += (observationsDict.get(agent.id, {}).get('LOB0', '').get('Ask_L1')[0])*diff

        prev_inventory = observations['Inventory']
        actionss.update({agent.id: actionss.get(agent.id, []) + [action[1][0]]})
        print(f"ACTION DONE{i}")
        t += [Simstate['TimeCode']]
        i+=1
        # if episode not in inventoryhistories:
        #     inventoryhistories[episode] = {}
        # if agent.id not in inventoryhistories[episode]:
        #     inventoryhistories[episode][agent.id] = []
        # inventoryhistories[episode][agent.id].append((Simstate['TimeCode'], observations['Inventory']))

# Plot the final inventory trajectory after the simulation completes
# plt.figure(figsize=(10, 6))
# plt.plot(times, inventories[1], marker='o', linestyle='-', color='blue', label='Agent Inventory')
# plt.xlabel("Time Step")
# plt.ylabel("Inventory")
# plt.title("Agent Inventory Over Time")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# agent_ids = set()
# for ep in inventoryhistories:
#     agent_ids.update(inventoryhistories[ep].keys())

# final_cash_diff = abs(kwargs["GymTradingAgent"][0]["cash"] - cashs[1][-1])
# print(f"Calculated diff: {cash_differences}. Actual difference: {final_cash_diff}")
# price_paths = [p - price_paths[0] for p in price_paths]


# agent_percentage_change_price = [(p - price_paths[0])/price_paths[0] for p in price_paths]

# for episode in range(len(inventoryhistories)):
#     plt.plot(inventories[1], agent_percentage_change_price, alpha=0.5)
#     plt.xlabel("Cumulative executed volume")
#     plt.ylabel("Percentage change price")
#     plt.title("Checking SQL for impact of TWAP agent")
#     plt.legend(fontsize='small')
#     plt.show()

# percentage_change_price = [(p - price_paths_non_agent[0])/price_paths_non_agent[0] for p in price_paths_non_agent]

# plt.figure()
# plt.plot(times, percentage_change_price, alpha=0.5)
# plt.xlabel("Time step")
# plt.ylabel("Midprice")
# plt.title("Price Path Tracking")
# plt.show()

# for agent_id in agent_ids:
#     plt.figure()
#     for ep in inventoryhistories:
#         if agent_id in inventoryhistories[ep]:
#             history = inventoryhistories[ep][agent_id]
#             times, invs = zip(*history)
#             plt.plot(times, invs, alpha=0.5, label=f"Ep {ep+1}")
#     plt.xlabel("Time")
#     plt.ylabel("Inventory")
#     plt.title(f"Inventory Trajectories Across All Episodes for agent {agent_id}")
#     plt.legend(fontsize='small', ncol=2)
#     plt.show()

            
       
# for agent in agents:
#     goal = kwargs["GymTradingAgent"][agent.id-1]["total_order_size"]
#     final_inventory = inventories[agent.id]
        

# Add this after your existing plotting code
# def fit_sqrt_market_impact():
#     """
#     Fit square root function to TWAP market impact: Price_Impact = a * sqrt(Volume) + b
#     """
#     # Get the data
#     volumes = inventories[1]  # Cumulative executed volume
#     # Since we don't have price data, we'll simulate market impact based on volume
#     # This is a placeholder - in practice you'd use actual price changes
#     price_impacts = [0.001 * v**0.5 for v in volumes]  # Simulated square root impact
    
#     # Remove any zero volumes to avoid sqrt(0) issues
#     non_zero_mask = np.array(volumes) > 0
#     volumes_clean = np.array(volumes)[non_zero_mask]
#     impacts_clean = np.array(price_impacts)[non_zero_mask]
    
#     # Define square root function: Impact = a * sqrt(Volume) + b
#     def sqrt_function(volume, a, b):
#         return a * np.sqrt(volume) + b
    
#     try:
#         # Fit the curve
#         params, covariance = curve_fit(sqrt_function, volumes_clean, impacts_clean)
#         a, b = params
        
#         # Generate fitted curve
#         volume_range = np.linspace(min(volumes_clean), max(volumes_clean), 100)
#         fitted_curve = sqrt_function(volume_range, a, b)
        
#         # Calculate R-squared
#         fitted_values = sqrt_function(volumes_clean, a, b)
#         ss_res = np.sum((impacts_clean - fitted_values) ** 2)
#         ss_tot = np.sum((impacts_clean - np.mean(impacts_clean)) ** 2)
#         r_squared = 1 - (ss_res / ss_tot)
        
#         # Create the plot
#         plt.figure(figsize=(12, 8))
        
#         # Plot original data
#         plt.scatter(volumes, price_impacts, alpha=0.6, color='blue', s=20, label='Actual Impact')
        
#         # Plot fitted curve
#         plt.plot(volume_range, fitted_curve, color='red', linewidth=2, 
#                 label=f'√ Fit: {a:.6f}*√Volume + {b:.6f}')
        
#         # Add statistics to plot
#         plt.text(0.02, 0.98, f'R² = {r_squared:.4f}\na = {a:.6f}\nb = {b:.6f}', 
#                 transform=plt.gca().transAxes, verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
#         plt.xlabel("Cumulative Executed Volume")
#         plt.ylabel("Percentage Price Change (%)")
#         plt.title("TWAP Market Impact - Square Root Law Fit")
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig("TWAP_sqrt_market_impact_fit.png", dpi=300, bbox_inches='tight')
#         plt.show()
        
#         # Print results
#         print(f"Square Root Law Fit Results:")
#         print(f"Price Impact = {a:.6f} * √(Volume) + {b:.6f}")
#         print(f"R-squared: {r_squared:.4f}")
#         print(f"Standard errors: a={np.sqrt(covariance[0,0]):.6f}, b={np.sqrt(covariance[1,1]):.6f}")
        
#         return params, r_squared
        
#     except Exception as e:
#         print(f"Square root fitting failed: {e}")
#         return None, None

# # Call the function after your existing code
# sqrt_params, r_squared = fit_sqrt_market_impact()

# # Alternative: Replace your existing plot with the fitted version
# plt.figure(figsize=(12, 8))

# # Calculate price impacts for plotting (simulated since we don't have actual price data)
# price_impacts_for_plot = [0.001 * v**0.5 for v in inventories[1]]

# # Plot original data points
# plt.scatter(inventories[1], price_impacts_for_plot, alpha=0.6, color='blue', s=20, label='Actual Impact')

# # If fit was successful, add the fitted curve
# if sqrt_params is not None:
#     volumes_for_fit = np.array(inventories[1])
#     non_zero_mask = volumes_for_fit > 0
#     volumes_clean = volumes_for_fit[non_zero_mask]
    
#     volume_range = np.linspace(min(volumes_clean), max(volumes_clean), 100)
#     fitted_curve = sqrt_params[0] * np.sqrt(volume_range) + sqrt_params[1]
    
#     plt.plot(volume_range, fitted_curve, color='red', linewidth=2, 
#             label=f'√ Law: {sqrt_params[0]:.6f}*√V + {sqrt_params[1]:.6f}')

# plt.xlabel("Cumulative Executed Volume")
# plt.ylabel("Percentage Change Price (%)")
# plt.title("TWAP Market Impact with Square Root Law Fit")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("TWAP_market_impact_sqrt_fitted.png", dpi=300, bbox_inches='tight')
# plt.show()

# Plot inventory and participation rate on the same graph
def plot_inventory_with_participation_rate():
    """
    Plot inventory trajectory with participation rate step function overlaid on the same graph
    """
    participation_rate = kwargs["GymTradingAgent"][0]["participation_rate"]  # 0.05
    window_size = kwargs["GymTradingAgent"][0]["window_size"]  # 60 seconds
    total_time = kwargs["GymTradingAgent"][0]["total_time"]  # 220 seconds
    start_trading_lag = kwargs["GymTradingAgent"][0]["start_trading_lag"]  # 100 seconds
    
    # Create time windows for trading (starting from lag time)
    trading_start_time = start_trading_lag
    trading_end_time = min(trading_start_time + total_time, max(times) if times else trading_start_time + total_time)
    
    # Generate window boundaries
    window_times = list(range(trading_start_time, int(trading_end_time) + window_size, window_size))
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot inventory trajectory on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Agent Inventory', color=color)
    line1 = ax1.plot(times, inventories[1], marker='o', linestyle='-', color=color, 
                     label='Agent Inventory', linewidth=2, markersize=3)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Add trading start/end lines
    ax1.axvline(x=trading_start_time, color='green', linestyle='--', alpha=0.7, 
                label=f'Trading Start ({trading_start_time}s)')
    ax1.axvline(x=trading_end_time, color='red', linestyle='--', alpha=0.7, 
                label=f'Trading End ({int(trading_end_time)}s)')
    
    # Create secondary y-axis for participation rate
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Participation Volume', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Calculate participation volume for each window and plot step function
    participation_volumes = []
    window_midpoints = []
    
    for i, market_vol in enumerate(market_volumes):
        if i < len(window_times) - 1:
            window_start = window_times[i]
            window_end = window_times[i + 1]
            window_midpoint = (window_start + window_end) / 2
            participation_vol = participation_rate * market_vol
            
            participation_volumes.append(participation_vol)
            window_midpoints.append(window_midpoint)
    
    # Plot step function using horizontal lines on secondary axis
    step_lines = []
    for i, (time_point, part_vol) in enumerate(zip(window_midpoints, participation_volumes)):
        if i < len(window_times) - 1:
            window_start = window_times[i]
            window_end = window_times[i + 1]
            
            # Draw horizontal line for this window
            line = ax2.hlines(y=part_vol, xmin=window_start, xmax=window_end, 
                            colors='red', linewidth=3, alpha=0.8)
            if i == 0:
                step_lines.append(line)
            
            # Draw vertical connectors between steps
            if i > 0:
                prev_vol = participation_volumes[i-1]
                ax2.vlines(x=window_start, ymin=min(prev_vol, part_vol), ymax=max(prev_vol, part_vol),
                          colors='red', linewidth=2, alpha=0.6)
    
    # Add window boundaries as vertical lines
    for window_time in window_times:
        if trading_start_time <= window_time <= trading_end_time:
            ax1.axvline(x=window_time, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    # Set title
    plt.title(f"Agent Inventory vs Participation Rate ({participation_rate}) × Market Volume")
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2 = step_lines
    labels2 = ['Participation Rate x Market Volume']
    
    # Combine legends
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
    ax1.legend(all_lines, all_labels, loc='upper left')
    
    # Print summary statistics
    # if participation_volumes:
    #     print(f"\nCombined Plot Summary:")
    #     print(f"Participation Rate: {participation_rate}")
    #     print(f"Window Size: {window_size} seconds")
    #     print(f"Number of Windows: {len(participation_volumes)}")
    #     print(f"Average Participation Volume: {np.mean(participation_volumes):.2f}")
    #     print(f"Max Participation Volume: {max(participation_volumes):.2f}")
    #     print(f"Min Participation Volume: {min(participation_volumes):.2f}")
    #     print(f"Final Agent Inventory: {inventories[1][-1] if inventories[1] else 0}")
    #     print(f"Market Volumes: {market_volumes}")
    #     print(f"Participation Volumes: {participation_volumes}")
    
    plt.tight_layout()
    plt.savefig("/home/ajafree/POV/inventory_trajectory/inventory_with_participation_rate.png", dpi=300, bbox_inches='tight')
    # plt.show()

# Alternative: Plot both on same axis with normalized scales
def plot_inventory_and_participation_normalized():
    """
    Plot both inventory and participation rate on same axis with normalized scales
    """
    participation_rate = kwargs["GymTradingAgent"][0]["participation_rate"]  # 0.05
    window_size = kwargs["GymTradingAgent"][0]["window_size"]  # 60 seconds
    total_time = kwargs["GymTradingAgent"][0]["total_time"]  # 220 seconds
    start_trading_lag = kwargs["GymTradingAgent"][0]["start_trading_lag"]  # 100 seconds
    
    # Create time windows for trading
    trading_start_time = start_trading_lag
    trading_end_time = min(trading_start_time + total_time, max(times) if times else trading_start_time + total_time)
    window_times = list(range(trading_start_time, int(trading_end_time) + window_size, window_size))
    
    # Calculate participation volumes
    participation_volumes = []
    for i, market_vol in enumerate(market_volumes):
        if i < len(window_times) - 1:
            participation_vol = participation_rate * market_vol
            participation_volumes.append(participation_vol)
    
    plt.figure(figsize=(14, 8))
    
    # Plot inventory
    plt.plot(times, inventories[1], marker='o', linestyle='-', color='blue', 
             label='Agent Inventory', linewidth=2, markersize=3)
    
    # Plot participation rate step function as horizontal lines
    for i, part_vol in enumerate(participation_volumes):
        if i < len(window_times) - 1:
            window_start = window_times[i]
            window_end = window_times[i + 1]
            
            # Draw horizontal line for this window
            plt.hlines(y=part_vol, xmin=window_start, xmax=window_end, 
                      colors='red', linewidth=3, alpha=0.8,
                      label='Participation Rate × Market Volume' if i == 0 else "")
            
            # Draw vertical connectors between steps
            if i > 0:
                prev_vol = participation_volumes[i-1]
                plt.vlines(x=window_start, ymin=min(prev_vol, part_vol), ymax=max(prev_vol, part_vol),
                          colors='red', linewidth=2, alpha=0.6)
    
    # Add window boundaries and trading period markers
    for window_time in window_times:
        if trading_start_time <= window_time <= trading_end_time:
            plt.axvline(x=window_time, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    plt.axvline(x=trading_start_time, color='green', linestyle='--', alpha=0.7, 
                label=f'Trading Start ({trading_start_time}s)')
    plt.axvline(x=trading_end_time, color='red', linestyle='--', alpha=0.7, 
                label=f'Trading End ({int(trading_end_time)}s)')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Volume')
    plt.title(f"Agent Inventory and Participation Rate ({participation_rate}) × Market Volume")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("inventory_and_participation_normalized.png", dpi=300, bbox_inches='tight')
    plt.show()

# Call both plotting functions
plot_inventory_with_participation_rate()
# plot_inventory_and_participation_normalized()
