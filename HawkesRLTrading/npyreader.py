import numpy as np
import glob
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib.pyplot as plt

# data = np.load('/Users/alirazajafree/researchprojects/Training Results/slippages/Outputfile/single_runs/multiple_runs/this_cant_be_right/with_randomised_starttimestest_UNTRAINED_RL,TWAP_randomisedstart_profit.npy')
# twap_executions_by_episode = np.load('/Users/alirazajafree/researchprojects/Training Results/slippages/Outputfile/single_runs/multiple_runs/this_cant_be_right/with_randomised_starttimestest_UNTRAINED_RL,TWAP_randomisedstart_twap_executions.npz')


# buy_sell_runs = np.load('/Users/alirazajafree/researchprojects/Training Results/slippages/Outputfile/New_data/train_RLAgent_vs_TWAP_standardised_updatedslippagegraphs_profit.npy')


# file_pattern = "/Users/alirazajafree/30JuneCopy/Market Impact/Price path/TWAP/*.npy"
# file_list = glob.glob(file_pattern)
# data_list = [np.load(f) for f in file_list]

# times = np.load('/Users/alirazajafree/30JuneCopy/Market Impact/Price path/times.npy')


# times = data[0]

# portfolios = data[1]

# boundaryIndices = [0]

# for i in range(1, len(times)):
#     if(times[i] < times[i-1]):
#         boundaryIndices.append(i)
# print(boundaryIndices)
# logReturns = []

# for i in range(len(boundaryIndices)-1):
#     episodicValues = [portfolios[boundaryIndices[i]], portfolios[boundaryIndices[i+1]]]
#     logReturns.extend(np.diff(np.log(episodicValues)))

# logReturns = np.array(logReturns)
# std = np.std(logReturns)
# mean = np.mean(logReturns)

# sharpe = mean/std
# print(f"bad sharpe {sharpe}")


def getSharpe(data):
    arr = data
    # print(np.shape(data))
    episode_boundaries = np.where(np.diff(arr[0]) <0)[0]
    start_idxs = episode_boundaries[:-1] + 1
    end_idxs = episode_boundaries[1:]
    log_ret2 = []
    for s, e in zip(start_idxs, end_idxs):
        log_ret2.append(np.log(arr[1][e]/arr[1][s]))
    sharpe = np.mean(log_ret2)/np.std(log_ret2)
    ann_sharpe = sharpe*np.sqrt(6.5*3600/(arr[0][e] - arr[0][s])*252)

    print(f"Sharpe: {sharpe}")
    print(f"Ann_sharpe: {ann_sharpe}")


def getSharpeComplete(data_array, times):
    """
    Calculate Sharpe ratio including ALL episodes (first and last episodes included).
    
    Args:
        data_array: numpy array with shape (2, n) where data_array[0] is time, data_array[1] is portfolio values
    
    Returns:
        dict: Contains Sharpe ratio and detailed statistics
    """
    # print("=== COMPLETE SHARPE CALCULATION (ALL EPISODES) ===")
    # print(f"Data array shape: {data_array.shape}")
    
    arr = data_array
    times = arr[0]
    values = arr[1]
    
    # Find episode boundaries by detecting time decreases
    episode_boundaries = np.where(np.diff(times) < 0)[0]
    
    # Properly include first and last episodes
    if len(episode_boundaries) > 0:
        # start_idxs: [0, boundary1+1, boundary2+1, ...]
        # end_idxs: [boundary1, boundary2, ..., len-1]
        start_idxs = np.concatenate([[0], episode_boundaries + 1])
        end_idxs = np.concatenate([episode_boundaries, [len(times) - 1]])
    else:
        # Single episode case
        start_idxs = np.array([0])
        end_idxs = np.array([len(times) - 1])
    
    # print(f"Episode boundaries found: {len(episode_boundaries)}")
    # print(f"Total episodes to process: {len(start_idxs)}")
    
    log_returns = []
    valid_episodes = 0
    
    for i, (s, e) in enumerate(zip(start_idxs, end_idxs)):
        start_val = values[s]
        end_val = values[e]
        
        if start_val > 0 and end_val > 0 and start_val != end_val:
            log_ret = np.log(end_val / start_val)
            log_returns.append(log_ret)
            valid_episodes += 1
            # print(f"Episode {i}: [{s}:{e}] start={start_val:.6f} end={end_val:.6f} return={log_ret:.8f}")
        # else:
            # print(f"Episode {i}: [{s}:{e}] SKIPPED (invalid values: start={start_val}, end={end_val})")
    
    log_returns = np.array(log_returns)
    
    if len(log_returns) > 1:
        mean_return = np.mean(log_returns)
        std_return = np.std(log_returns)
        
        if std_return > 0:
            sharpe = mean_return / std_return
            ann_sharpe = sharpe * np.sqrt(6.5 * 3600/(arr[0][e] - arr[0][s]) * 252)
        else:
            sharpe = 0
            ann_sharpe = 0
    else:
        mean_return = 0
        std_return = 0
        sharpe = 0
        ann_sharpe = 0
    
    # print(f"\n=== COMPLETE RESULTS ===")
    # print(f"Valid episodes processed: {valid_episodes} / {len(start_idxs)}")
    # print(f"Mean return: {mean_return:.8f}")
    # print(f"Std return: {std_return:.8f}")
    print(f"Sharpe: {sharpe:.6f}")
    print(f"Ann_sharpe: {ann_sharpe:.6f}")
    
    return {
        'episodes_processed': valid_episodes,
        'total_episodes': len(start_idxs),
        'log_returns': log_returns,
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe': sharpe,
        'ann_sharpe': ann_sharpe
    }


def getSlippagesFromFinalCashInventory(finalcash, total_executed, starting_midprices, starting_cash=1000000):
    slippages = []
    for (cash, executed, starting_midprice) in zip(finalcash, total_executed, starting_midprices):
        if(cash > starting_cash): 
            side = "sell"
        else:
            side = "buy"
        print(cash)
        print(executed)
        print(side)
        # executed = 6
        print(starting_midprice)
        benchmark_cost = executed*starting_midprice
        executed_cost = starting_cash-cash if side == "buy" else cash-starting_cash
        assert(executed_cost>0, "error")
        diff = executed_cost-benchmark_cost
        slippage = diff/benchmark_cost
        slippage *= 10000
        slippages.append(float(slippage))
    return slippages




def getSharpeWithTWAPSplit_buysell(data):
    """
    Calculate separate Sharpe ratios for periods without TWAP, with TWAP buy, and with TWAP sell.
    First half of each episode: WITHOUT TWAP
    Second half of each episode: WITH TWAP (split by buy/sell episodes)
    
    Buy episodes: 3, 4, 5, 7, 9, 18, 20, 25, 26, 34
    Sell episodes: all others
    """
    arr = data
    episode_boundaries = np.where(np.diff(arr[0]) < 0)[0]
    start_idxs = episode_boundaries[:-1] + 1
    end_idxs = episode_boundaries[1:]
    
    # Add the first episode start and last episode end
    if len(episode_boundaries) > 0:
        start_idxs = np.concatenate([[0], start_idxs])
        end_idxs = np.concatenate([end_idxs, [len(arr[0]) - 1]])
    else:
        # Single episode case
        start_idxs = np.array([0])
        end_idxs = np.array([len(arr[0]) - 1])
    
    # Define buy and sell episodes (matching splitBuySellRunsByEpisodes)
    buy_episodes = {3, 4, 5, 7, 9, 18, 20, 25, 26, 34}
    
    without_twap_log_returns = []     # First half - RL agent trading alone
    with_twap_buy_log_returns = []    # Second half - RL agent + TWAP buying
    with_twap_sell_log_returns = []   # Second half - RL agent + TWAP selling
    
    print(f"Found {len(start_idxs)} episodes")
    print(f"Buy episodes: {sorted(buy_episodes)}")
    print(f"Total episodes expected: 35")
    
    for episode_idx, (s, e) in enumerate(zip(start_idxs, end_idxs)):
        episode_length = e - s + 1
        mid_point = s + episode_length // 2
        
        # First half: WITHOUT TWAP (RL agent trading alone)
        without_twap_start_value = arr[1][s]
        without_twap_end_value = arr[1][mid_point]
        
        # Second half: WITH TWAP (TWAP agent joins)
        with_twap_start_value = arr[1][mid_point]
        with_twap_end_value = arr[1][e]
        
        # Calculate log returns for first half (always add to without_twap)
        if without_twap_start_value > 0:
            without_twap_log_return = np.log(without_twap_end_value / without_twap_start_value)
            without_twap_log_returns.append(without_twap_log_return)
        
        # Calculate log returns for second half (split by episode type)
        if with_twap_start_value > 0:
            with_twap_log_return = np.log(with_twap_end_value / with_twap_start_value)
            
            if episode_idx in buy_episodes:
                with_twap_buy_log_returns.append(with_twap_log_return)
                episode_type = "BUY"
            else:
                with_twap_sell_log_returns.append(with_twap_log_return)
                episode_type = "SELL"
            
            print(f"Episode {episode_idx} ({episode_type}): WITHOUT TWAP [{s}:{mid_point}] = {without_twap_log_return:.6f}, "
                  f"WITH TWAP {episode_type} [{mid_point}:{e}] = {with_twap_log_return:.6f}")
    
    # Convert to numpy arrays
    without_twap_log_returns = np.array(without_twap_log_returns)
    with_twap_buy_log_returns = np.array(with_twap_buy_log_returns)
    with_twap_sell_log_returns = np.array(with_twap_sell_log_returns)
    
    # Calculate Sharpe ratios for WITHOUT TWAP
    if len(without_twap_log_returns) > 1:
        without_twap_mean = np.mean(without_twap_log_returns)
        without_twap_std = np.std(without_twap_log_returns)
        without_twap_sharpe = without_twap_mean / without_twap_std if without_twap_std > 0 else 0
        without_twap_ann_sharpe = without_twap_sharpe * np.sqrt(6.5 * 12 * 252)
    else:
        without_twap_sharpe = without_twap_ann_sharpe = 0
    
    # Calculate Sharpe ratios for WITH TWAP BUY
    if len(with_twap_buy_log_returns) > 1:
        with_twap_buy_mean = np.mean(with_twap_buy_log_returns)
        with_twap_buy_std = np.std(with_twap_buy_log_returns)
        with_twap_buy_sharpe = with_twap_buy_mean / with_twap_buy_std if with_twap_buy_std > 0 else 0
        with_twap_buy_ann_sharpe = with_twap_buy_sharpe * np.sqrt(6.5 * 12 * 252)
    else:
        with_twap_buy_sharpe = with_twap_buy_ann_sharpe = 0
    
    # Calculate Sharpe ratios for WITH TWAP SELL
    if len(with_twap_sell_log_returns) > 1:
        with_twap_sell_mean = np.mean(with_twap_sell_log_returns)
        with_twap_sell_std = np.std(with_twap_sell_log_returns)
        with_twap_sell_sharpe = with_twap_sell_mean / with_twap_sell_std if with_twap_sell_std > 0 else 0
        with_twap_sell_ann_sharpe = with_twap_sell_sharpe * np.sqrt(6.5 * 12 * 252)
    else:
        with_twap_sell_sharpe = with_twap_sell_ann_sharpe = 0
    
    print(f"\n=== RESULTS ===")
    print(f"WITHOUT TWAP (First Half - RL Solo):")
    print(f"  Episodes: {len(without_twap_log_returns)}")
    # print(f"  Mean Return: {np.mean(without_twap_log_returns):.6f}")
    # print(f"  Std Return: {np.std(without_twap_log_returns):.6f}")
    # print(f"  Sharpe: {without_twap_sharpe:.4f}")
    print(f"  Ann_sharpe: {without_twap_ann_sharpe:.4f}")
    
    print(f"\nWITH TWAP BUY (Second Half - RL vs TWAP Buying):")
    print(f"  Episodes: {len(with_twap_buy_log_returns)} (from {len(buy_episodes)} buy episodes)")
    # print(f"  Mean Return: {np.mean(with_twap_buy_log_returns) if len(with_twap_buy_log_returns) > 0 else 0:.6f}")
    # print(f"  Std Return: {np.std(with_twap_buy_log_returns) if len(with_twap_buy_log_returns) > 0 else 0:.6f}")
    # print(f"  Sharpe: {with_twap_buy_sharpe:.4f}")
    print(f"  Ann_sharpe: {with_twap_buy_ann_sharpe:.4f}")
    
    print(f"\nWITH TWAP SELL (Second Half - RL vs TWAP Selling):")
    print(f"  Episodes: {len(with_twap_sell_log_returns)} (from {31 - len(buy_episodes)} sell episodes)")
    # print(f"  Mean Return: {np.mean(with_twap_sell_log_returns) if len(with_twap_sell_log_returns) > 0 else 0:.6f}")
    # print(f"  Std Return: {np.std(with_twap_sell_log_returns) if len(with_twap_sell_log_returns) > 0 else 0:.6f}")
    # print(f"  Sharpe: {with_twap_sell_sharpe:.4f}")
    print(f"  Ann_sharpe: {with_twap_sell_ann_sharpe:.4f}")
    
    
    
    return {
        'without_twap_returns': without_twap_log_returns,
        'with_twap_buy_returns': with_twap_buy_log_returns,
        'with_twap_sell_returns': with_twap_sell_log_returns,
        'without_twap_sharpe': without_twap_sharpe,
        'with_twap_buy_sharpe': with_twap_buy_sharpe,
        'with_twap_sell_sharpe': with_twap_sell_sharpe,
        'without_twap_ann_sharpe': without_twap_ann_sharpe,
        'with_twap_buy_ann_sharpe': with_twap_buy_ann_sharpe,
        'with_twap_sell_ann_sharpe': with_twap_sell_ann_sharpe
    }

def splitBuySellRunsByEpisodes(data_array):
    """
    Split the buy_sell_runs data into two arrays based on episode indices.
    
    Args:
        data_array: numpy array with shape (2, n) where data_array[0] is time, data_array[1] is portfolio values
    
    Returns:
        dict: Contains two arrays - one for specified episodes and one for remaining episodes
    """
    print("=== SPLITTING BUY_SELL_RUNS BY EPISODES ===")
    print(f"Input data shape: {data_array.shape}")
    
    # Define the episodes to extract
    target_episodes = {3, 4, 5, 7, 9, 18, 20, 25, 26, 34}
    
    # Find episode boundaries using same logic as other functions
    episode_boundaries = np.where(np.diff(data_array[0]) < 0)[0]
    
    if len(episode_boundaries) > 0:
        start_idxs = np.concatenate([[0], episode_boundaries + 1])
        end_idxs = np.concatenate([episode_boundaries, [len(data_array[0]) - 1]])
    else:
        # Single episode case
        start_idxs = np.array([0])
        end_idxs = np.array([len(data_array[0]) - 1])
    
    print(f"Found {len(start_idxs)} total episodes")
    print(f"Target episodes to extract: {sorted(target_episodes)}")
    
    # Collect data for target episodes and remaining episodes
    target_times = []
    target_portfolios = []
    remaining_times = []
    remaining_portfolios = []
    
    target_episode_count = 0
    remaining_episode_count = 0
    
    for episode_idx, (s, e) in enumerate(zip(start_idxs, end_idxs)):
        episode_times = data_array[0][s:e+1]
        episode_portfolios = data_array[1][s:e+1]
        
        if episode_idx in target_episodes:
            target_times.extend(episode_times)
            target_portfolios.extend(episode_portfolios)
            target_episode_count += 1
            print(f"Episode {episode_idx}: Added to TARGET array [{s}:{e+1}] - {len(episode_times)} points")
        else:
            remaining_times.extend(episode_times)
            remaining_portfolios.extend(episode_portfolios)
            remaining_episode_count += 1
            print(f"Episode {episode_idx}: Added to REMAINING array [{s}:{e+1}] - {len(episode_times)} points")
    
    # Convert to numpy arrays
    target_times = np.array(target_times)
    target_portfolios = np.array(target_portfolios)
    remaining_times = np.array(remaining_times)
    remaining_portfolios = np.array(remaining_portfolios)
    
    # Build final arrays with shape (2, n)
    target_array = np.vstack([target_times, target_portfolios]) if len(target_times) > 0 else np.array([[], []])
    remaining_array = np.vstack([remaining_times, remaining_portfolios]) if len(remaining_times) > 0 else np.array([[], []])
    
    print(f"\n=== SPLIT RESULTS ===")
    print(f"TARGET episodes ({sorted([i for i in range(len(start_idxs)) if i in target_episodes])}):")
    print(f"  Episodes count: {target_episode_count}")
    print(f"  Data points: {target_array.shape[1] if target_array.size > 0 else 0}")
    print(f"  Array shape: {target_array.shape}")
    
    print(f"\nREMAINING episodes:")
    print(f"  Episodes count: {remaining_episode_count}")
    print(f"  Data points: {remaining_array.shape[1] if remaining_array.size > 0 else 0}")
    print(f"  Array shape: {remaining_array.shape}")
    
    return {
        'target_episodes_data': target_array,
        'remaining_episodes_data': remaining_array,
        'target_episodes': sorted([i for i in range(len(start_idxs)) if i in target_episodes]),
        'remaining_episodes': sorted([i for i in range(len(start_idxs)) if i not in target_episodes]),
        'target_episode_count': target_episode_count,
        'remaining_episode_count': remaining_episode_count
    }

def getSharpeWithTWAPSplit(data_array):
    """
    Splits each episode into first half (without TWAP) and second half (with TWAP),
    combining both buy and sell episodes together. Returns Sharpe ratios for comparison.
    
    Args:
        data_array: numpy array with shape (2, n) where data_array[0] is time, data_array[1] is portfolio values
    
    Returns:
        dict: Contains log returns and Sharpe ratios for both periods
    """
    # print("=== RL PERFORMANCE: WITH vs WITHOUT TWAP ===")
    # print(f"Data array shape: {data_array.shape}")
    
    # Use same episode detection logic as getSharpe()
    # episode_boundaries contains indices i where time[i+1] < time[i] (episode end at i)
    episode_boundaries = np.where(np.diff(data_array[0]) < 0)[0]

    # Correct start and end indices so episodes don't overlap:
    # starts = [0] + (episode_boundaries + 1)
    # ends   = episode_boundaries + [len-1]
    if len(episode_boundaries) > 0:
        start_idxs = np.concatenate([[0], episode_boundaries + 1])
        end_idxs = np.concatenate([episode_boundaries, [len(data_array[0]) - 1]])
    else:
        # Single episode case
        start_idxs = np.array([0])
        end_idxs = np.array([len(data_array[0]) - 1])
    
    print(f"Found {len(start_idxs)} episodes. Episode boundaries: {list(zip(start_idxs[:5], end_idxs[:5]))}...")
    
    # Store log returns for each period
    without_twap_log_returns = []  # First half of episodes (RL solo)
    with_twap_log_returns = []     # Second half of episodes (RL vs TWAP)
    
    # Process each episode using same logic as getSharpe()
    for episode_idx, (s, e) in enumerate(zip(start_idxs, end_idxs)):
        episode_length = e - s + 1
        # midpoint defines first half as s..mid-1 and second half as mid..e
        mid_point = s + episode_length // 2

        # Skip if episode is too short
        if episode_length < 2:
            print(f"Episode {episode_idx}: Too short ({episode_length} timesteps), skipping")
            continue

        # Define indices for halves (non-overlapping)
        first_end_idx = mid_point - 1
        second_start_idx = mid_point

        # If first half has no length (happens when episode_length==1), skip
        if first_end_idx < s or second_start_idx > e:
            print(f"Episode {episode_idx}: invalid half indices, skipping")
            continue

        # First half: WITHOUT TWAP (RL agent trading alone)
        without_twap_start_value = data_array[1][s]
        without_twap_end_value = data_array[1][first_end_idx]

        # Second half: WITH TWAP (RL vs TWAP)
        with_twap_start_value = data_array[1][second_start_idx]
        with_twap_end_value = data_array[1][e]

        # Calculate log returns for first half (always add to without_twap)
        without_twap_log_return = np.nan
        with_twap_log_return = np.nan

        if without_twap_start_value > 0 and without_twap_end_value > 0:
            without_twap_log_return = np.log(without_twap_end_value / without_twap_start_value)
            without_twap_log_returns.append(without_twap_log_return)

        # Calculate log returns for second half
        if with_twap_start_value > 0 and with_twap_end_value > 0:
            with_twap_log_return = np.log(with_twap_end_value / with_twap_start_value)
            with_twap_log_returns.append(with_twap_log_return)

        # print(f"Episode {episode_idx}: length={episode_length}, "
        #       f"without_twap [{s}:{first_end_idx}] = {without_twap_log_return if not np.isnan(without_twap_log_return) else 'nan'}, "
        #       f"with_twap [{second_start_idx}:{e}] = {with_twap_log_return if not np.isnan(with_twap_log_return) else 'nan'}")
    
    # Convert to numpy arrays
    without_twap_log_returns = np.array(without_twap_log_returns)
    with_twap_log_returns = np.array(with_twap_log_returns)
    
    # Calculate Sharpe ratios for WITHOUT TWAP period
    if len(without_twap_log_returns) > 0:
        without_twap_mean = np.mean(without_twap_log_returns)
        without_twap_std = np.std(without_twap_log_returns)
        without_twap_sharpe = without_twap_mean / without_twap_std if without_twap_std > 0 else 0
        without_twap_ann_sharpe = without_twap_sharpe * np.sqrt(6.5 * 12 * 252)  # Annualize
    else:
        without_twap_sharpe = without_twap_ann_sharpe = 0
    
    # Calculate Sharpe ratios for WITH TWAP period
    if len(with_twap_log_returns) > 0:
        with_twap_mean = np.mean(with_twap_log_returns)
        with_twap_std = np.std(with_twap_log_returns)
        with_twap_sharpe = with_twap_mean / with_twap_std if with_twap_std > 0 else 0
        with_twap_ann_sharpe = with_twap_sharpe * np.sqrt(6.5 * 12 * 252)  # Annualize
    else:
        with_twap_sharpe = with_twap_ann_sharpe = 0
    
    # # Print results
    # print(f"\n=== RESULTS ===")
    # print(f"WITHOUT TWAP (First Half - RL Solo):")
    # print(f"  Returns: {len(without_twap_log_returns)}")
    # print(f"  Mean Return: {np.mean(without_twap_log_returns) if len(without_twap_log_returns) > 0 else 0:.6f}")
    # print(f"  Std Return: {np.std(without_twap_log_returns) if len(without_twap_log_returns) > 0 else 0:.6f}")
    # print(f"  Sharpe: {without_twap_sharpe:.4f}")
    # print(f"  Ann_sharpe: {without_twap_ann_sharpe:.4f}")
    
    # print(f"\nWITH TWAP (Second Half - RL vs TWAP):")
    # print(f"  Returns: {len(with_twap_log_returns)}")
    # print(f"  Mean Return: {np.mean(with_twap_log_returns) if len(with_twap_log_returns) > 0 else 0:.6f}")
    # print(f"  Std Return: {np.std(with_twap_log_returns) if len(with_twap_log_returns) > 0 else 0:.6f}")
    # print(f"  Sharpe: {with_twap_sharpe:.4f}")
    # print(f"  Ann_sharpe: {with_twap_ann_sharpe:.4f}")
    
    # Performance comparison
    improvement = with_twap_sharpe - without_twap_sharpe if without_twap_sharpe != 0 else 0
    ann_improvement = with_twap_ann_sharpe - without_twap_ann_sharpe if without_twap_ann_sharpe != 0 else 0
    
    # print(f"\n=== PERFORMANCE COMPARISON ===")
    # print(f"Sharpe Improvement: {improvement:.4f}")
    # print(f"Ann. Sharpe Improvement: {ann_improvement:.4f}")
    
    return {
        'without_twap_returns': without_twap_log_returns,
        'with_twap_returns': with_twap_log_returns,
        'without_twap_sharpe': without_twap_sharpe,
        'with_twap_sharpe': with_twap_sharpe,
        'without_twap_ann_sharpe': without_twap_ann_sharpe,
        'with_twap_ann_sharpe': with_twap_ann_sharpe,
        'sharpe_improvement': improvement,
        'ann_sharpe_improvement': ann_improvement
    }



def getSharpeWithTWAPExecutionData(data_array, twap_executions_by_episode, starting_trade_lag=100):
    """
    Calculate Sharpe ratios using actual TWAP execution timing and side information.
    
    Args:
        data_array: numpy array with shape (2, n) where data_array[0] is time, data_array[1] is portfolio values
        twap_executions_by_episode: dict with keys like "episode_0", "episode_1" etc.
                                   Each value is np.array with dtype=[('price', 'f8'), ('quantity', 'f8'), ('side', 'U4')]
        starting_trade_lag: time offset before TWAP starts (default 100)
    
    Returns:
        dict: Contains log returns and Sharpe ratios for without TWAP, with TWAP buy, and with TWAP sell
    """
    print("=== RL PERFORMANCE: TWAP EXECUTION-BASED ANALYSIS ===")
    print(f"Data array shape: {data_array.shape}")
    
    # Use same episode detection logic as getSharpe()
    episode_boundaries = np.where(np.diff(data_array[0]) < 0)[0]
    start_idxs = episode_boundaries[:-1] + 1
    end_idxs = episode_boundaries[1:]
    
    # Add the first episode start and last episode end
    if len(episode_boundaries) > 0:
        start_idxs = np.concatenate([[0], start_idxs])
        end_idxs = np.concatenate([end_idxs, [len(data_array[0]) - 1]])
    else:
        # Single episode case
        start_idxs = np.array([0])
        end_idxs = np.array([len(data_array[0]) - 1])
    
    print(f"Found {len(start_idxs)} episodes")
    print(f"TWAP execution data available for {len(twap_executions_by_episode)} episodes")
    
    # Store log returns for each period
    without_twap_log_returns = []      # Before TWAP starts - RL agent trading alone
    with_twap_buy_log_returns = []     # During TWAP buying - RL agent vs TWAP buying
    with_twap_sell_log_returns = []    # During TWAP selling - RL agent vs TWAP selling
    
    # Process each episode
    for episode_idx, (s, e) in enumerate(zip(start_idxs, end_idxs)):
        episode_key = f"episode_{episode_idx}"
        
        # Check if we have TWAP execution data for this episode
        if episode_key not in twap_executions_by_episode:
            print(f"Episode {episode_idx}: No TWAP execution data, skipping")
            continue
        
        twap_executions = twap_executions_by_episode[episode_key]
        
        # Skip if no executions
        if len(twap_executions) == 0:
            print(f"Episode {episode_idx}: Empty TWAP executions, skipping")
            continue
        
        # Determine TWAP side from first execution
        twap_side = twap_executions[0]['side']
        num_twap_executions = len(twap_executions)
        
        # Calculate TWAP start time index
        # TWAP starts at: starting_trade_lag + 300 - num_twap_executions
        twap_start_offset = starting_trade_lag + 300 - num_twap_executions
        twap_start_idx = s + twap_start_offset
        
        # Ensure TWAP start is within episode bounds
        if twap_start_idx >= e:
            print(f"Episode {episode_idx}: TWAP start ({twap_start_idx}) beyond episode end ({e}), skipping")
            continue
        
        if twap_start_idx <= s:
            print(f"Episode {episode_idx}: TWAP start ({twap_start_idx}) before episode start ({s}), skipping")
            continue
        
        # WITHOUT TWAP period: from episode start to TWAP start
        without_twap_start_value = data_array[1][s]
        without_twap_end_value = data_array[1][twap_start_idx]
        
        # WITH TWAP period: from TWAP start to episode end
        with_twap_start_value = data_array[1][twap_start_idx]
        with_twap_end_value = data_array[1][e]
        
        # Calculate log returns for WITHOUT TWAP period
        if without_twap_start_value > 0:
            without_twap_log_return = np.log(without_twap_end_value / without_twap_start_value)
            without_twap_log_returns.append(without_twap_log_return)
        else:
            without_twap_log_return = np.nan
        
        # Calculate log returns for WITH TWAP period
        if with_twap_start_value > 0:
            with_twap_log_return = np.log(with_twap_end_value / with_twap_start_value)
            
            # Add to appropriate list based on TWAP side
            if twap_side.lower() == 'buy':
                with_twap_buy_log_returns.append(with_twap_log_return)
            elif twap_side.lower() == 'sell':
                with_twap_sell_log_returns.append(with_twap_log_return)
            else:
                print(f"Episode {episode_idx}: Unknown TWAP side '{twap_side}', skipping")
                continue
        else:
            with_twap_log_return = np.nan
        
        print(f"Episode {episode_idx}: {twap_side.upper()} TWAP, {num_twap_executions} executions, "
              f"TWAP starts at idx {twap_start_idx} (offset {twap_start_offset})")
        print(f"  WITHOUT TWAP [{s}:{twap_start_idx}] = {without_twap_log_return:.6f}")
        print(f"  WITH TWAP {twap_side.upper()} [{twap_start_idx}:{e}] = {with_twap_log_return:.6f}")
    
    # Convert to numpy arrays
    without_twap_log_returns = np.array([x for x in without_twap_log_returns if not np.isnan(x)])
    with_twap_buy_log_returns = np.array([x for x in with_twap_buy_log_returns if not np.isnan(x)])
    with_twap_sell_log_returns = np.array([x for x in with_twap_sell_log_returns if not np.isnan(x)])
    
    # Calculate Sharpe ratios for WITHOUT TWAP
    if len(without_twap_log_returns) > 1:
        without_twap_mean = np.mean(without_twap_log_returns)
        without_twap_std = np.std(without_twap_log_returns)
        without_twap_sharpe = without_twap_mean / without_twap_std if without_twap_std > 0 else 0
        without_twap_ann_sharpe = without_twap_sharpe * np.sqrt(6.5 * 12 * 252)
    else:
        without_twap_sharpe = without_twap_ann_sharpe = 0
    
    # Calculate Sharpe ratios for WITH TWAP BUY
    if len(with_twap_buy_log_returns) > 1:
        with_twap_buy_mean = np.mean(with_twap_buy_log_returns)
        with_twap_buy_std = np.std(with_twap_buy_log_returns)
        with_twap_buy_sharpe = with_twap_buy_mean / with_twap_buy_std if with_twap_buy_std > 0 else 0
        with_twap_buy_ann_sharpe = with_twap_buy_sharpe * np.sqrt(6.5 * 12 * 252)
    else:
        with_twap_buy_sharpe = with_twap_buy_ann_sharpe = 0
    
    # Calculate Sharpe ratios for WITH TWAP SELL
    if len(with_twap_sell_log_returns) > 1:
        with_twap_sell_mean = np.mean(with_twap_sell_log_returns)
        with_twap_sell_std = np.std(with_twap_sell_log_returns)
        with_twap_sell_sharpe = with_twap_sell_mean / with_twap_sell_std if with_twap_sell_std > 0 else 0
        with_twap_sell_ann_sharpe = with_twap_sell_sharpe * np.sqrt(6.5 * 12 * 252)
    else:
        with_twap_sell_sharpe = with_twap_sell_ann_sharpe = 0
    
    # Print results
    print(f"\n=== RESULTS (EXECUTION-BASED TIMING) ===")
    print(f"WITHOUT TWAP (RL Solo - Before TWAP Starts):")
    print(f"  Returns: {len(without_twap_log_returns)}")
    print(f"  Mean Return: {np.mean(without_twap_log_returns) if len(without_twap_log_returns) > 0 else 0:.6f}")
    print(f"  Std Return: {np.std(without_twap_log_returns) if len(without_twap_log_returns) > 0 else 0:.6f}")
    print(f"  Sharpe: {without_twap_sharpe:.4f}")
    print(f"  Ann_sharpe: {without_twap_ann_sharpe:.4f}")
    
    print(f"\nWITH TWAP BUY (RL vs TWAP Buying):")
    print(f"  Returns: {len(with_twap_buy_log_returns)}")
    print(f"  Mean Return: {np.mean(with_twap_buy_log_returns) if len(with_twap_buy_log_returns) > 0 else 0:.6f}")
    print(f"  Std Return: {np.std(with_twap_buy_log_returns) if len(with_twap_buy_log_returns) > 0 else 0:.6f}")
    print(f"  Sharpe: {with_twap_buy_sharpe:.4f}")
    print(f"  Ann_sharpe: {with_twap_buy_ann_sharpe:.4f}")
    
    print(f"\nWITH TWAP SELL (RL vs TWAP Selling):")
    print(f"  Returns: {len(with_twap_sell_log_returns)}")
    print(f"  Mean Return: {np.mean(with_twap_sell_log_returns) if len(with_twap_sell_log_returns) > 0 else 0:.6f}")
    print(f"  Std Return: {np.std(with_twap_sell_log_returns) if len(with_twap_sell_log_returns) > 0 else 0:.6f}")
    print(f"  Sharpe: {with_twap_sell_sharpe:.4f}")
    print(f"  Ann_sharpe: {with_twap_sell_ann_sharpe:.4f}")
    
    # Performance comparisons
    buy_improvement = with_twap_buy_sharpe - without_twap_sharpe if without_twap_sharpe != 0 else 0
    sell_improvement = with_twap_sell_sharpe - without_twap_sharpe if without_twap_sharpe != 0 else 0
    
    print(f"\n=== PERFORMANCE COMPARISON ===")
    print(f"Buy TWAP vs No TWAP Improvement: {buy_improvement:.4f}")
    print(f"Sell TWAP vs No TWAP Improvement: {sell_improvement:.4f}")
    
    return {
        'without_twap_returns': without_twap_log_returns,
        'with_twap_buy_returns': with_twap_buy_log_returns,
        'with_twap_sell_returns': with_twap_sell_log_returns,
        'without_twap_sharpe': without_twap_sharpe,
        'with_twap_buy_sharpe': with_twap_buy_sharpe,
        'with_twap_sell_sharpe': with_twap_sell_sharpe,
        'without_twap_ann_sharpe': without_twap_ann_sharpe,
        'with_twap_buy_ann_sharpe': with_twap_buy_ann_sharpe,
        'with_twap_sell_ann_sharpe': with_twap_sell_ann_sharpe,
        'buy_improvement': buy_improvement,
        'sell_improvement': sell_improvement
    }



def aggregatePricePaths(data_list):
    plt.figure(figsize=(12, 8))
    
    for i, data in enumerate(data_list):
        plt.plot(data, color = 'lightblue', alpha=0.7, label=f"Run {i+1}")
    # Find the maximum length among all datasets
    max_len = max([d.shape[0] for d in data_list])

    # Pad shorter arrays with np.nan for proper averaging
    padded = np.full((len(data_list), max_len), np.nan)
    for i, d in enumerate(data_list):
        padded[i, :d.shape[0]] = d

    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(padded, axis=0)

    # Remove NaN values for curve fitting
    valid_mask = ~np.isnan(mean_path)
    x_valid = np.arange(len(mean_path))[valid_mask]
    y_valid = mean_path[valid_mask]

    #plot mean
    plt.plot(mean_path, color='orange', linewidth=2, label='Average')
 
    plt.xlabel("Time step")
    plt.ylabel("Percentage change in midprice")
    plt.title("Aggregated Price Paths from Multiple Runs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("", dpi=300, bbox_inches='tight')
    plt.show()



def aggregatePricePaths_decay(data_list, times):
    plt.figure(figsize=(12, 8))
    
    time_threshold_idx = np.where(times >= 1300)[0]
    if len(time_threshold_idx) == 0:
        print("No data after 1300 seconds")
        return
    
    start_idx = time_threshold_idx[0]
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data after 1300 seconds
        if len(data) > start_idx:
            filtered_data = data[start_idx:]
            filtered_times = times[start_idx:start_idx + len(filtered_data)]
            filtered_data_list.append(filtered_data)
            plt.plot(filtered_data, color='lightblue', alpha=0.7, label=f"Run {i+1}")
    
    # Find the maximum length among filtered datasets
    if filtered_data_list:
        max_len = max([d.shape[0] for d in filtered_data_list])
        
        # Pad shorter arrays with np.nan for proper averaging
        padded = np.full((len(filtered_data_list), max_len), np.nan)
        for i, d in enumerate(filtered_data_list):
            padded[i, :d.shape[0]] = d
        
        # Compute mean across runs, ignoring nan
        mean_path = np.nanmean(padded, axis=0)
        
        # Plot mean with actual times
        mean_times = times[start_idx:start_idx + len(mean_path)]
        plt.plot(mean_path, color='orange', linewidth=2, label='Average')
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Percentage change in midprice")
    plt.title("Aggregated Price Paths from Multiple Runs (After 1300 seconds)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price Decay/TWAP/TWAP_price_decay_unfitted_with_time_1.png", dpi=300, bbox_inches='tight')
    plt.show()

def aggregatePricePaths_decay_fitted_propogatormodel(data_list, times):
    time_threshold_idx = np.where(times >= 1300)[0]
    if len(time_threshold_idx) == 0:
        print("No data after 1300 seconds")
        return
    
    start_idx = time_threshold_idx[0]
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data after 1300 seconds
        if len(data) > start_idx:
            filtered_data = data[start_idx:]
            filtered_data_list.append(filtered_data)
    if not filtered_data_list:
        return
    
    max_len = max([d.shape[0] for d in filtered_data_list])
        
    # Pad shorter arrays with np.nan for proper averaging
    padded = np.full((len(filtered_data_list), max_len), np.nan)
    for i, d in enumerate(filtered_data_list):
        padded[i, :d.shape[0]] = d
    
    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(padded, axis=0)
    mean_times = times[start_idx:start_idx + len(mean_path)]

    # Remove NaN values for fitting
    valid_mask = ~np.isnan(mean_path)
    t_fit = mean_times[valid_mask]
    y_fit = mean_path[valid_mask]

    # Calculate z values - since we normalized by subtracting 100, use total time range
    z = [(t-100)/1200 for t in mean_times]  # Total time range is 1200s (100 to 1300)
    z_fit = [(t-100)/1200 for t in t_fit]
    
    # Filter to only include z between 1 and 2 inclusive
    z_mask = np.array([(z_val >= 1.0) and (z_val <= 2.0) for z_val in z])
    z_fit_mask = np.array([(z_val >= 1.0) and (z_val <= 2.0) for z_val in z_fit])
    
    # Apply the mask to filter data
    z_filtered = np.array(z)[z_mask]
    mean_path_filtered = mean_path[z_mask]
    z_fit_filtered = np.array(z_fit)[z_fit_mask]
    y_fit_filtered = y_fit[z_fit_mask]
    
    print(f"Original data points: {len(z_fit)}")
    print(f"Filtered data points (1 ≤ z ≤ 2): {len(z_fit_filtered)}")
    print(f"z range: {min(z_fit_filtered):.3f} to {max(z_fit_filtered):.3f}")

    plt.figure(figsize=(12, 8))
    
    # Plot only the average on z-axis (filtered)
    plt.plot(z_filtered, mean_path_filtered, color='orange', linewidth=3, label='Average', alpha=0.8)

    # Define the propagator model varying both β and a
    def propagator_model(z_array, a, beta):
        result = []
        for z_val in z_array:
            if z_val <= 0:
                # Handle edge case where z <= 0
                term = 0
            elif z_val >= 1:
                # When z >= 1, both terms are positive
                term = z_val**(1-beta) - (z_val-1)**(1-beta)
            else:
                # When 0 < z < 1, (z-1) is negative
                # Use absolute value and handle sign carefully
                term1 = z_val**(1-beta)
                # For (z-1)^(1-beta) where z-1 < 0 and 1-beta is fractional
                # Use: |z-1|^(1-beta) * (-1)^(1-beta) = (1-z)^(1-beta) * (-1)
                term2 = (1-z_val)**(1-beta) * (-1)
                term = term1 - term2
            
            result.append(a * term)
        return np.array(result)
    
    try:
        # Check if we have enough filtered data points
        if len(z_fit_filtered) < 3:
            print("Not enough data points in z range [1, 2] for fitting")
            return
            
        # Fit both parameters 'a' and 'β' using filtered data
        initial_guess = [0.000175*np.sqrt(1200)-0.000272, 0.22]  # Initial guess for a and β
        
        # Bounds: a > 0, 0.1 < β < 1.9
        bounds = ([0, 0.1], [np.inf, 1.9])
        
        # Fit the propagator model using filtered data
        popt, pcov = curve_fit(propagator_model, z_fit_filtered, y_fit_filtered, 
                              p0=initial_guess, bounds=bounds, maxfev=5000)
        
        a_fit, beta_fit = popt
        
        # Generate fitted curve for plotting (only for z between 1 and 2)
        z_smooth = np.linspace(min(z_fit_filtered), max(z_fit_filtered), 100)
        y_propagator_fit = propagator_model(z_smooth, a_fit, beta_fit)
        
        plt.plot(z_smooth, y_propagator_fit, color='red', linewidth=2, linestyle='--',
                label=f'Propagator Model: {a_fit:.4f} * (z^{1-beta_fit:.3f} - (z-1)^{1-beta_fit:.3f})')
        
        # Calculate R-squared using filtered data
        y_fitted_points = propagator_model(z_fit_filtered, a_fit, beta_fit)
        ss_res = np.sum((y_fit_filtered - y_fitted_points) ** 2)
        ss_tot = np.sum((y_fit_filtered - np.mean(y_fit_filtered)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"Propagator Model Fit - R²: {r_squared:.4f}")
        print(f"Parameters: a={a_fit:.4f}, β={beta_fit:.3f}")
        print(f"Equation: y = {a_fit:.4f} * (z^{1-beta_fit:.3f} - (z-1)^{1-beta_fit:.3f})")
        print(f"Where z = (t-100)/1200, constrained to [1, 2]")
        
        # Add R² to plot
        plt.text(0.02, 0.98, f'R² = {r_squared:.4f}\nβ = {beta_fit:.3f}\na = {a_fit:.4f}\n1 ≤ z ≤ 2', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Calculate parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov))
        print(f"Parameter uncertainties: a ± {param_errors[0]:.4f}, β ± {param_errors[1]:.3f}")
        
    except Exception as e:
        print(f"Propagator model fit failed: {e}")
        import traceback
        traceback.print_exc()

    plt.xlabel("z = (t-100)/1200")
    plt.ylabel("Percentage change in midprice")
    plt.title("TWAP Price Impact Decay - Propagator Model Fit (1 ≤ z ≤ 2)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1.0, 2.0)  # Set x-axis limits to show only the filtered range
    plt.tight_layout()
    # plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price Decay/TWAP/TWAP_price_decay_propagator_z_1_to_2.png", 
    #             dpi=300, bbox_inches='tight')
    plt.show()
    

    


def aggregatePricePaths_decay_fitted(data_list, times):
    time_threshold_idx = np.where(times >= 1300)[0]
    if len(time_threshold_idx) == 0:
        print("No data after 1300 seconds")
        return
    
    start_idx = time_threshold_idx[0]
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data after 1300 seconds
        if len(data) > start_idx:
            filtered_data = data[start_idx:]
            filtered_data_list.append(filtered_data)
    if not filtered_data_list:
        return
    
    max_len = max([d.shape[0] for d in filtered_data_list])
        
    # Pad shorter arrays with np.nan for proper averaging
    padded = np.full((len(filtered_data_list), max_len), np.nan)
    for i, d in enumerate(filtered_data_list):
        padded[i, :d.shape[0]] = d
    
    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(padded, axis=0)
    mean_times = times[start_idx:start_idx + len(mean_path)]

    # Remove NaN values for fitting
    valid_mask = ~np.isnan(mean_path)
    t_fit = mean_times[valid_mask]
    y_fit = mean_path[valid_mask]

    # Use original time values (no normalization)
    # t_normalized = t_fit - t_fit[0]  # REMOVED
    
    # --- Plot 1: Power Law Fit ---
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    for i, data in enumerate(filtered_data_list):
        filtered_times = times[start_idx:start_idx + len(data)]
        plt.plot(filtered_times, data, color='lightblue', alpha=0.3)
    
    # Plot mean
    plt.plot(mean_times, mean_path, color='orange', linewidth=3, label='Average', alpha=0.8)
    
    # Power law fit: y = a * t^(-b) using original time values
    def power_law(t, a, b):
        return a * np.power(t, -b)  # Changed to t^(-b)
    
    try:
        power_params, power_cov = curve_fit(power_law, t_fit, y_fit, 
                                          p0=[y_fit[0] * t_fit[0]**0.5, 0.5], 
                                          bounds=([0, 0], [np.inf, 2]))
        
        power_fitted = power_law(t_fit, *power_params)
        plt.plot(t_fit, power_fitted, color='red', linewidth=2, linestyle='--',
                label=f'Power Law: {power_params[0]:.4f} * t^(-{power_params[1]:.3f})')
        
        # Calculate R-squared
        ss_res = np.sum((y_fit - power_fitted) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared_power = 1 - (ss_res / ss_tot)
        
        print(f"Power Law - R²: {r_squared_power:.4f}")
        print(f"Power Law parameters: a={power_params[0]:.4f}, b={power_params[1]:.3f}")
        
        # Add R² to plot
        plt.text(0.02, 0.98, f'R² = {r_squared_power:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    except Exception as e:
        print(f"Power law fit failed: {e}")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Percentage change in midprice")
    plt.title("TWAP Price Impact Decay - Power Law Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price Decay/TWAP/TWAP_price_decay_power_law.png", 
    #             dpi=300, bbox_inches='tight')
    plt.show()

    # --- Plot 2: Exponential Decay Fit ---
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    for i, data in enumerate(filtered_data_list):
        filtered_times = times[start_idx:start_idx + len(data)]
        plt.plot(filtered_times, data, color='lightblue', alpha=0.3)
    
    # Plot mean
    plt.plot(mean_times, mean_path, color='orange', linewidth=3, label='Average', alpha=0.8)
    
    # Exponential decay fit: y = a * exp(-b*t) + c using original time values
    def exponential_decay(t, a, b, c):
        return a * np.exp(-b * t) + c
    
    try:
        # Initial guess for original time values
        initial_a = y_fit[0] - y_fit[-1]
        initial_b = 0.0001  # Much smaller decay rate for large time values
        initial_c = y_fit[-1]
        
        exp_params, exp_cov = curve_fit(exponential_decay, t_fit, y_fit,
                                       p0=[initial_a, initial_b, initial_c],
                                       bounds=([0, 0, -np.inf], [np.inf, 0.01, np.inf]))
        
        exp_fitted = exponential_decay(t_fit, *exp_params)
        plt.plot(t_fit, exp_fitted, color='green', linewidth=2, linestyle='-.',
                label=f'Exponential: {exp_params[0]:.4f}*exp(-{exp_params[1]:.6f}*t) + {exp_params[2]:.4f}')
        
        # Calculate R-squared
        ss_res = np.sum((y_fit - exp_fitted) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared_exp = 1 - (ss_res / ss_tot)
        
        print(f"Exponential Decay - R²: {r_squared_exp:.4f}")
        print(f"Exponential parameters: a={exp_params[0]:.4f}, b={exp_params[1]:.6f}, c={exp_params[2]:.4f}")
        
        # Add R² to plot
        plt.text(0.02, 0.98, f'R² = {r_squared_exp:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    except Exception as e:
        print(f"Exponential decay fit failed: {e}")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Percentage change in midprice")
    plt.title("TWAP Price Impact Decay - Exponential Decay Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price Decay/TWAP/TWAP_price_decay_exponential.png", 
    #             dpi=300, bbox_inches='tight')
    plt.show()

def price_quantity_graph_predecay(data_list, times):
    # Since TWAP is linear with time step window size, the executed quantity over time can be assumed to be linear
    time_threshold_idx = np.where(times <= 1300)[0]
    if len(time_threshold_idx) == 0:
        print("No data before 1300 seconds")
        return
    
    end_idx = time_threshold_idx[-1]
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data after 1300 seconds
        filtered_data = data[:end_idx]
        filtered_data_list.append(filtered_data)
    if not filtered_data_list:
        return
    
    
    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(filtered_data_list, axis=0)
    mean_times = times[:end_idx][:len(mean_path)]

    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    for i, data in enumerate(filtered_data_list):
        filtered_times = times[0:len(data)]
        plt.plot(filtered_times, data, color='lightblue', alpha=0.3)
    
    # Plot mean
    plt.plot(mean_times, mean_path, color='orange', linewidth=3, label='Average', alpha=0.8)

    # plt.show()

    # Remove NaN values for fitting
    valid_mask = ~np.isnan(mean_path)
    t_fit = mean_times[valid_mask]
    y_fit = mean_path[valid_mask]
    
    # Fit a square root function: y = a * sqrt(t) + b
    def sqrt_func(t, a, b):
        return a * np.sqrt(t) + b

    try:
        # Use only t > 0 to avoid sqrt(0) issues
        mask = t_fit > 0
        popt, pcov = curve_fit(sqrt_func, t_fit[mask], y_fit[mask], p0=[1, 0])
        y_sqrt_fit = sqrt_func(t_fit, *popt)
        plt.plot(t_fit, y_sqrt_fit, color='red', linewidth=2, linestyle='--',
                 label=f'Sqrt Fit: {popt[0]:.4f} * sqrt(t) + {popt[1]:.4f}')
        # Calculate R-squared
        ss_res = np.sum((y_fit[mask] - sqrt_func(t_fit[mask], *popt)) ** 2)
        ss_tot = np.sum((y_fit[mask] - np.mean(y_fit[mask])) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"Square Root Fit - R²: {r_squared:.4f}")
        print(f"Square Root parameters: a={popt[0]:.4f}, b={popt[1]:.4f}")
        plt.legend()
        plt.xlabel("Quantity")
        plt.ylabel("Price impact")
        plt.title("Price vs Quantity (Square Root Fit)")

        plt.show()
    except Exception as e:
        print(f"Square root fit failed: {e}")


def price_quantity_graph_predecay_mortised(data_list, times):
    """
    Same as price_quantity_graph_predecay but sampling data every 50 seconds with tolerance for float imprecision
    """
    # Since TWAP is linear with time step window size, the executed quantity over time can be assumed to be linear
    time_threshold_idx = np.where(times <= 1300)[0]
    if len(time_threshold_idx) == 0:
        print("No data before 1300 seconds")
        return
    
    end_idx = time_threshold_idx[-1]
    
    # Find indices for every 50 seconds with tolerance
    target_times = np.arange(0, 1300, 50)  # [0, 50, 100, 150, ..., 1250]
    tolerance = 0.5  # Allow ±0.5 seconds tolerance
    
    sampled_indices = []
    for target_time in target_times:
        # Find the closest time within tolerance
        time_diffs = np.abs(times[:end_idx] - target_time)
        closest_idx = np.argmin(time_diffs)
        
        if time_diffs[closest_idx] <= tolerance:
            sampled_indices.append(closest_idx)
        else:
            print(f"Warning: No time found within {tolerance}s of target time {target_time}")
    
    print(f"Found {len(sampled_indices)} sample points every ~50 seconds")
    print(f"Sample times: {times[sampled_indices]}")
    
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data to only include sampled indices (before 1300 seconds)
        if len(data) > max(sampled_indices):
            filtered_data = data[sampled_indices]
            filtered_data_list.append(filtered_data)
        else:
            # If data is shorter, take what we can
            valid_indices = [idx for idx in sampled_indices if idx < len(data)]
            if valid_indices:
                filtered_data = data[valid_indices]
                filtered_data_list.append(filtered_data)
    
    if not filtered_data_list:
        print("No filtered data available")
        return
    
    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(filtered_data_list, axis=0)
    mean_times = times[sampled_indices[:len(mean_path)]]

    plt.figure(figsize=(12, 8))
    
    # Plot individual runs with markers since we have fewer points
    for i, data in enumerate(filtered_data_list):
        sample_times = times[sampled_indices[:len(data)]]
        plt.plot(sample_times, data, color='lightblue', alpha=0.5, marker='o', markersize=3)
    
    # Plot mean with larger markers
    plt.plot(mean_times, mean_path, color='orange', linewidth=3, marker='o', markersize=6,
             label='Average (50s intervals)', alpha=0.8)

    # Remove NaN values for fitting
    valid_mask = ~np.isnan(mean_path)
    t_fit = mean_times[valid_mask]
    y_fit = mean_path[valid_mask]
    
    # Fit a square root function: y = a * sqrt(t) + b
    def sqrt_func(t, a, b):
        return a * np.sqrt(t + 1e-6) + b  # Add small epsilon to avoid sqrt(0)

    try:
        # Use only t > 0 to avoid sqrt(0) issues
        mask = t_fit > 0
        if np.sum(mask) > 2:  # Need at least 3 points for fitting
            popt, pcov = curve_fit(sqrt_func, t_fit[mask], y_fit[mask], p0=[0.001, 0])
            
            # Generate smooth curve for plotting
            t_smooth = np.linspace(min(t_fit[mask]), max(t_fit[mask]), 100)
            y_sqrt_fit_smooth = sqrt_func(t_smooth, *popt)
            
            plt.plot(t_smooth, y_sqrt_fit_smooth, color='red', linewidth=2, linestyle='--',
                     label=f'√ Fit: {popt[0]:.6f} * √t + {popt[1]:.6f}')
            
            # Calculate R-squared
            y_sqrt_fit_points = sqrt_func(t_fit[mask], *popt)
            ss_res = np.sum((y_fit[mask] - y_sqrt_fit_points) ** 2)
            ss_tot = np.sum((y_fit[mask] - np.mean(y_fit[mask])) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"Square Root Fit (50s intervals) - R²: {r_squared:.4f}")
            print(f"Square Root parameters: a={popt[0]:.6f}, b={popt[1]:.6f}")
            
            # Add R² to plot
            plt.text(0.02, 0.98, f'R² = {r_squared:.4f}\n50s sampling', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            print("Not enough valid data points for fitting")
            
    except Exception as e:
        print(f"Square root fit failed: {e}")

    plt.xlabel("Quantity")
    plt.ylabel("Price Impact (%)")
    plt.title("TWAP Price Impact Build-up - 50 Second Intervals (√t Fit)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price path/TWAP/TWAP_predecay_50s_intervals.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_times, mean_path, sampled_indices



def price_quantity_graph_predecay_mortised_starting_at_0(data_list, times):
    """
    Same as price_quantity_graph_predecay but sampling data every 50 seconds with tolerance for float imprecision
    """
    # Since TWAP is linear with time step window size, the executed quantity over time can be assumed to be linear
    time_threshold_idx = np.where(times <= 1350)[0]
    if len(time_threshold_idx) == 0:
        print("No data before 1300 seconds")
        return
    
    end_idx = time_threshold_idx[-1]
    
    # Find indices for every 50 seconds with tolerance
    target_times = np.arange(100, 1350, 50)  # [100, 150, ..., 1250]
    tolerance = 0.5  # Allow ±0.5 seconds tolerance
    target_times[0] = 101 #the simulation has a 1 second delay at the start
    
    sampled_indices = []
    for target_time in target_times:
        # Find the closest time within tolerance
        time_diffs = np.abs(times[:end_idx] - target_time)
        closest_idx = np.argmin(time_diffs)
        
        if time_diffs[closest_idx] <= tolerance:
            sampled_indices.append(closest_idx)
        else:
            print(f"Warning: No time found within {tolerance}s of target time {target_time}")
    
    print(f"Found {len(sampled_indices)} sample points every ~50 seconds")
    print(f"Sample times: {times[sampled_indices]}")
    
    filtered_data_list = []
    
    for i, data in enumerate(data_list):
        # Filter data to only include sampled indices (before 1300 seconds)
        if len(data) > max(sampled_indices):
            filtered_data = data[sampled_indices]
            filtered_data_list.append(filtered_data)
        else:
            # If data is shorter, take what we can
            valid_indices = [idx for idx in sampled_indices if idx < len(data)]
            if valid_indices:
                filtered_data = data[valid_indices]
                filtered_data_list.append(filtered_data)
    
    if not filtered_data_list:
        print("No filtered data available")
        return
    
    # Compute mean across runs, ignoring nan
    mean_path = np.nanmean(filtered_data_list, axis=0)
    mean_times = times[sampled_indices[:len(mean_path)]]

    plt.figure(figsize=(12, 8))
        
    # Plot individual runs with markers since we have fewer points
    for i, data in enumerate(filtered_data_list):
        sample_times = times[sampled_indices[:len(data)]]
        # Shift time axis to start from 0 (quantity = 0)
        quantity_axis = sample_times - sample_times[0]
        plt.plot(quantity_axis, data, color='lightblue', alpha=0.5, marker='o', markersize=3)
    
    # Plot mean with larger markers
    # Shift time axis to start from 0 (quantity = 0)
    quantity_axis_mean = mean_times - mean_times[0]
    plt.plot(quantity_axis_mean, mean_path, color='orange', linewidth=3, marker='o', markersize=6,
             label='Average (50s intervals)', alpha=0.8)
    
    # Remove NaN values for fitting
    valid_mask = ~np.isnan(mean_path)
    q_fit = quantity_axis_mean[valid_mask]  # Use shifted quantity axis instead of t_fit
    y_fit = mean_path[valid_mask]
    
    # Fit a square root function: y = a * sqrt(q) + b
    def sqrt_func(q, a, b):
        return a * np.sqrt(q + 1e-6) + b  # Add small epsilon to avoid sqrt(0)
    
    try:
        # Use only q > 0 to avoid sqrt(0) issues
        mask = q_fit >= 0 
        if np.sum(mask) > 2:  # Need at least 3 points for fitting
            popt, pcov = curve_fit(sqrt_func, q_fit[mask], y_fit[mask], p0=[0.001, 0])
            
            # Generate smooth curve for plotting
            q_smooth = np.linspace(min(q_fit[mask]), max(q_fit[mask]), 100)
            y_sqrt_fit_smooth = sqrt_func(q_smooth, *popt)
            
            plt.plot(q_smooth, y_sqrt_fit_smooth, color='red', linewidth=2, linestyle='--',
                     label=f'√ Fit: {popt[0]:.6f} * √q + {popt[1]:.6f}')
            
            # Calculate R-squared
            y_sqrt_fit_points = sqrt_func(q_fit[mask], *popt)
            ss_res = np.sum((y_fit[mask] - y_sqrt_fit_points) ** 2)
            ss_tot = np.sum((y_fit[mask] - np.mean(y_fit[mask])) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"Square Root Fit (50s intervals) - R²: {r_squared:.4f}")
            print(f"Square Root parameters: a={popt[0]:.6f}, b={popt[1]:.6f}")
            
            # Add R² to plot
            plt.text(0.02, 0.98, f'R² = {r_squared:.4f}\n50s sampling', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            print("Not enough valid data points for fitting")
            
    except Exception as e:
        print(f"Square root fit failed: {e}")
    
    plt.xlabel("Quantity (normalized to start at 0)")
    plt.ylabel("Price Impact (%)")
    plt.title("TWAP Price Impact Build-up - 50 Second Intervals (√q Fit)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/alirazajafree/30JuneCopy/Market Impact/Price path/TWAP/TWAP_predecay_50s_intervals.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return mean_times, mean_path, sampled_indices

# side = "sell"

# twap_obsv_episodes = np.load(f"/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/retest/logs/observations/retest_uRL_versus_{side}twap_observations.npy", allow_pickle=True)

# print(f"Number of episodes: {len(twap_obsv_episodes)}")

# # Since all episodes contain accumulated data, we only need episode 19 (contains all data)
# # Then we'll split it by detecting time resets
# all_observations = twap_obsv_episodes[-1]  # Take the last episode (has all data)

# print(f"Total observations in final episode: {len(all_observations)}")

# # Detect episode boundaries by time decreases
# episode_boundaries = [0]
# for i in range(1, len(all_observations)):
#     current_time = all_observations[i].get('current_time', 999999)
#     prev_time = all_observations[i-1].get('current_time', 0)
    
#     # Episode boundary: time decreases or resets
#     if current_time < prev_time:
#         episode_boundaries.append(i)
#         print(f"Episode boundary at index {i}: time went from {prev_time} to {current_time}")

# episode_boundaries.append(len(all_observations))

# print(f"\nFound {len(episode_boundaries)-1} episodes")
# print(f"Episode boundaries: {episode_boundaries[:5]}... (first 5)")

# # Now extract starting midprices and other data
# starting_midprices = []
# decoded_episodes = {}

# for ep_idx in range(len(episode_boundaries) - 1):
#     start_idx = episode_boundaries[ep_idx]
#     end_idx = episode_boundaries[ep_idx + 1]
    
#     episode_observations = all_observations[start_idx:end_idx]
    
#     episode_data = {
#         'episode': ep_idx,
#         'num_observations': len(episode_observations),
#         'cash': [],
#         'inventory': [],
#         'current_time': [],
#         'lob_data': []
#     }
    
#     for obs in episode_observations:
#         episode_data['cash'].append(obs.get('Cash', None))
#         episode_data['inventory'].append(obs.get('Inventory', None))
#         episode_data['current_time'].append(obs.get('current_time', None))
        
#         lob = obs.get('LOB0', {})
#         lob_snapshot = {
#             'ask_l1': lob.get('Ask_L1'),
#             'bid_l1': lob.get('Bid_L1'),
#             'ask_l2': lob.get('Ask_L2'),
#             'bid_l2': lob.get('Bid_L2')
#         }
#         episode_data['lob_data'].append(lob_snapshot)
    
#     # Get starting midprice from first observation of this episode
#     if len(episode_observations) > 0:
#         first_lob = episode_observations[0].get('LOB0', {})
#         ask = first_lob.get('Ask_L1', [None])[0]
#         bid = first_lob.get('Bid_L1', [None])[0]
#         if ask and bid:
#             starting_midprices.append((ask + bid) / 2)
#         else:
#             starting_midprices.append(100.0)  # fallback
    
#     decoded_episodes[ep_idx] = episode_data
#     print(f"Episode {ep_idx}: {len(episode_observations)} observations, "
#           f"time range: {episode_data['current_time'][0]:.2f} - {episode_data['current_time'][-1]:.2f}")

# # # Now calculate slippages
# # logdir = f"/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/retest/twap_alone/regPOVoldPIs/"
# twap_finalcash = np.load(f"/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/retest/logs/twap_slippages/retest_uRL_versus_{side}final_cash.npy")
# twap_totalexecuted = np.load(f"/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/retest/logs/twap_slippages/retest_uRL_versus_{side}total_executed.npy")
# starting_midprices = np.load(f"{logdir}/retest_twap_regularpov_oldpis_{side}_alone_start_midprices.npy")

# temp = [0 for i in range(len(twap_totalexecuted))]
# for i in range(len(twap_totalexecuted)-1, 0, -1):
#     temp[i] = twap_totalexecuted[i]-twap_totalexecuted[i-1]
# temp[0] = twap_totalexecuted[0]
# twap_totalexecuted = temp

# print(twap_finalcash)
# print(f"\nStarting midprices count: {len(starting_midprices)}")
# print(f"Final cash count: {len(twap_finalcash)}")
# print(f"Total executed count: {len(twap_totalexecuted)}")

# slippages = getSlippagesFromFinalCashInventory(twap_finalcash, twap_totalexecuted, starting_midprices, side)
# print(f"\nSlippages: {slippages}")
# print(f"Mean slippage: {np.mean(slippages):.4f} bps")
# print(f"Number of episodes: {len(slippages)}")

# print(f"{side} slippages")
# twap_obsv_episodes = np.load(f"/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/retest/fRL/logs/observations/retest_fRL_versus_{side}twap_observations.npy", allow_pickle=True)
# twap_finalcash = np.load(f"/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/retest/fRL/logs/twap_slippages/retest_fRL_versus_{side}final_cash.npy")
# twap_totalexecuted = np.load(f"/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/retest/fRL/logs/twap_slippages/retest_fRL_versus_{side}total_executed.npy")
# starting_midprices = [] #np.load(f"/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/retest/fRL/logs/logsretest_twap_{side}_alone_start_midprices.npy")
# decoded_episodes = {}
# for episode_idx, episode_obsv in enumerate(twap_obsv_episodes):
#     # print(episode_idx)
#     print(len(episode_obsv))
#     episode_data = {
#             'episode': episode_idx,
#             'timestamps': [],
#             'cash': [],
#             'inventory': [],
#             'lob_data': [],
#             'market_volume': [],
#             'current_time': [],
#             'num_observations': len(episode_obsv)
#     }

#     # if episode_idx == 0:
#     #     episode_start_idx = 0
#     # else:
#     #     # Sum up lengths of all previous episodes
#     #     episode_start_idx = sum(len(twap_obsv_episodes[i]) for i in range(episode_idx))

#     for obs_idx, obs in enumerate(episode_obsv):
#         episode_data['cash'].append(obs.get('Cash', None)) 
#         episode_data['inventory'].append(obs.get('Inventory', None))
#         episode_data['current_time'].append(obs.get('current_time', None))
#         episode_data['market_volume'].append(obs.get('market_volume', None))
#         lob = obs.get('LOB0', {})
#         lob_snapshot = {
#             'ask_l1': lob.get('Ask_L1'),
#             'bid_l1': lob.get('Bid_L1'),
#             'ask_l2': lob.get('Ask_L2'),
#             'bid_l2': lob.get('Bid_L2')
#         }
#         episode_data['lob_data'].append(lob_snapshot)

#         if obs_idx == 0:
#             starting_midprices.append((lob_snapshot['ask_l1'][0]+lob_snapshot['bid_l1'][0])/2)

# # print("starting midprices" + str(len(starting_midprices)))

# slippages = getSlippagesFromFinalCashInventory(twap_finalcash, twap_totalexecuted, starting_midprices, side)
# print(slippages)
# print(np.mean(slippages))
# print(len(slippages))



# rl_alone_data = np.load('/Users/alirazajafree/researchprojects/LSTM_fRL/with_expo/alone/scaled/testing_logs/test_LSTMAgent_alone_profit.npy')
# getSharpe(rl_alone_data)



#SLIPPAGE DISTRIBUTION CALCULATION
# slip = getSlippagesFromFinalCashInventory(np.load("/Users/alirazajafree/researchprojects/LSTM_fRL/without_expo/testing/test_LSTM_TWAPfinal_cash.npy"), np.load("/Users/alirazajafree/researchprojects/LSTM_fRL/without_expo/testing/test_LSTM_TWAPtotal_executed.npy"), np.load("/Users/alirazajafree/researchprojects/LSTM_fRL/without_expo/testing/test_LSTM_TWAP_start_midprices.npy"))
# print(slip)

# import matplotlib.pyplot as plt
# from matplotlib.scale import SymmetricalLogTransform

# final_cash = np.load("/Users/alirazajafree/researchprojects/LSTM_fRL/without_expo/testing/test_LSTM_TWAPfinal_cash.npy")
# starting_cash = 1000000
# buy_mask = final_cash <= starting_cash
# sell_mask = final_cash > starting_cash
# slip_buy = [s for s, m in zip(slip, buy_mask) if m]
# slip_sell = [s for s, m in zip(slip, sell_mask) if m]

# fig, ax = plt.subplots()
# # symlog-spaced bins: linear near zero, log-spaced in tails
# linthresh = max(np.median(np.abs(slip)) * 0.1, 1e-6)
# smin, smax = np.min(slip), np.max(slip)
# transform = SymmetricalLogTransform(base=10, linthresh=linthresh, linscale=1)
# inv_transform = transform.inverted()
# bins = inv_transform.transform(np.linspace(transform.transform(smin), transform.transform(smax), 31))
# ax.hist(slip_buy, bins=bins, edgecolor='black', alpha=0.6, label=f'Buy ({len(slip_buy)} eps)', color='tab:red')
# ax.hist(slip_sell, bins=bins, edgecolor='black', alpha=0.6, label=f'Sell ({len(slip_sell)} eps)', color='tab:green')
# mean_val = np.mean(slip)
# median_val = np.median(slip)
# ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.4f}')
# ax.axvline(median_val, color='blue', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.4f}')
# ax.set_xscale('symlog', linthresh=linthresh)
# ax.set_xlabel('Slippage')
# ax.set_ylabel('Frequency')
# ax.set_title('Slippage Distribution')
# ax.legend()
# plt.tight_layout()
# plt.show()

# rl_alone = np.load("/Users/alirazajafree/researchprojects/RL_alone/RL_alonetest_episodes_RL_alone_profit.npy")
# print("uRL Alone")
# getSharpeComplete(rl_alone)

# rl_before_buy = np.load("/Users/alirazajafree/researchprojects/untrained_RL/profits/untrained_rl_testingtest_untrained_RL_versus_buy_profit_wout_twap.npy")
# print("uRL Before Buy")
# getSharpeComplete(rl_before_buy)
# # getSharpe(rl_before_buy)
# rl_after_buy = np.load("/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/retest/logs/profit/retest_uRL_versus_buy_profit_w_twap.npy")
# print("uRL With buy")
# getSharpeComplete(rl_after_buy)
# # getSharpe(rl_after_buy)
# rl_after_buy = np.load("/Users/alirazajafree/researchprojects/untrained_RL/profits/untrained_rl_testingtest_untrained_RL_versus_buy_profit_w_twap.npy")
# print("uRL With buy (bigpov)")
# getSharpeComplete(rl_after_buy)

# rl_before_sell = np.load("/Users/alirazajafree/researchprojects/untrained_RL/profits/untrained_rl_testingtest_untrained_RL_versus_sell_profit_wout_twap.npy")
# print("uRL Before Sell")
# getSharpeComplete(rl_before_sell)
# # getSharpe(rl_before_sell)

# rl_after_sell = np.load("/Users/alirazajafree/researchprojects/uRL_testing_scaledTWAP/retest/logs/profit/retest_uRL_versus_sell_profit_w_twap.npy")
# print("uRL With sell")
# getSharpeComplete(rl_after_sell)
# getSharpe(rl_after_sell)
# rl_after_sell = np.load("/Users/alirazajafree/researchprojects/untrained_RL/profits/untrained_rl_testingtest_untrained_RL_versus_sell_profit_w_twap.npy")
# print("uRL With sell (bigpov)")
# getSharpeComplete(rl_after_sell)


# print("total sell sharpe fRL")
# getSharpeComplete(np.load("/Users/alirazajafree/researchprojects/fRL_september/RL_info/outputstest_fRL_versus_sell_profit.npy"))

# print("total sell sharpe uRL")
# getSharpeComplete(np.load("/Users/alirazajafree/researchprojects/untrained_RL/profits/untrained_rl_testingtest_untrained_RL_versus_sell_profit.npy"))



# getSharpe(np.load("/Users/alirazajafree/researchprojects/LSTM_fRL/with_expo/alone/testing/test_LSTMRLAgent_alone_profit.npy"))


# Find and calculate Sharpe ratio for all files with 'profit' in the directory
import os

target_directory = "/Users/alirazajafree/researchprojects/LSTM_fRL/without_expo/testing/self_imitation/onesided/buy/termpenalty/"
file_pattern = os.path.join(target_directory, "*profit*npy")

# Get all files matching the pattern
profit_files = glob.glob(file_pattern)

print(f"\n{'='*80}")
print(f"Found {len(profit_files)} files with 'profit' in {target_directory}")
print(f"{'='*80}\n")

# Process each file
for i, file_path in enumerate(sorted(profit_files), 1):
    filename = os.path.basename(file_path)
    print(f"\n[{i}/{len(profit_files)}] Processing: {filename}")
    print("-" * 80)
    
    try:
        # Load the numpy file
        data = np.load(file_path)
        
        # Calculate Sharpe ratio using the existing function
        results = getSharpe(data)
        
        
    except Exception as e:
        print(f"ERROR loading or processing file: {e}")
    
    print("-" * 80)

print(f"\n{'='*80}")
print("Completed processing all files")
print(f"{'='*80}\n")
