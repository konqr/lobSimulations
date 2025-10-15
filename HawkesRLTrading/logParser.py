import re
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
# import seaborn as sns
import pickle
import numpy as np
from typing import Dict, Any, Tuple, List
import glob
from collections import Counter
def parse_log_file(filename):
    """
    Parse the limit order book log file and extract relevant information
    """
    data = []

    with open(filename, 'r') as file:
        content = file.read()

    # Find all Limit Order Book entries
    lob_matches = list(re.finditer(r"Limit Order Book: ({.*?})", content))

    # Find all Statelog entries
    statelog_matches = list(re.finditer(r"Statelog: \((.*?)\)(?=\n|$)", content, re.MULTILINE | re.DOTALL))

    print(f"Found {len(lob_matches)} Limit Order Book entries")
    print(f"Found {len(statelog_matches)} Statelog entries")

    # Match LOB entries with their corresponding Statelog entries
    for i, lob_match in enumerate(lob_matches):
        try:
            # Extract limit order book snapshot
            lob_str = lob_match.group(1)
            lob_dict = ast.literal_eval(lob_str)

            # Find the corresponding statelog entry (should be the next one after LOB)
            statelog_match = None
            lob_end_pos = lob_match.end()

            for statelog in statelog_matches:
                if statelog.start() > lob_end_pos:
                    statelog_match = statelog
                    break

            if not statelog_match:
                continue

            # Parse the statelog tuple
            statelog_str = statelog_match.group(1)

            # Extract timestamp (first element)
            timestamp_match = re.search(r'^([\d.]+),', statelog_str)
            timestamp = float(timestamp_match.group(1)) if timestamp_match else None

            # Extract inventory (looks like {'INTC': -6})
            inventory_match = re.search(r"({'[^']*': -?\d+})", statelog_str)
            inventory = ast.literal_eval(inventory_match.group(1)) if inventory_match else {}

            # Extract positions from the detailed order book info
            # Look for Ask_L1 and Bid_L1 sections
            ask_l1_match = re.search(r"'Ask_L1': \[(.*?)\]", statelog_str)
            bid_l1_match = re.search(r"'Bid_L1': \[(.*?)\]", statelog_str)

            # Count positions by counting LimitOrder occurrences
            ask_l1_positions = 0
            bid_l1_positions = 0

            if ask_l1_match:
                ask_l1_content = ask_l1_match.group(1)
                ask_l1_positions = ask_l1_content.count('LimitOrder(')

            if bid_l1_match:
                bid_l1_content = bid_l1_match.group(1)
                bid_l1_positions = bid_l1_content.count('LimitOrder(')

            # Calculate mid-price
            if 'Ask_L1' in lob_dict and 'Bid_L1' in lob_dict:
                ask_price = lob_dict['Ask_L1'][0]
                bid_price = lob_dict['Bid_L1'][0]
                mid_price = (ask_price + bid_price) / 2
            else:
                mid_price = None

            # Store the extracted data
            data.append({
                'timestamp': timestamp,
                'mid_price': mid_price,
                'inventory': inventory.get('INTC', 0) if 'INTC' in inventory else 0,
                'ask_l1_positions': ask_l1_positions,
                'bid_l1_positions': bid_l1_positions,
                'total_l1_positions': ask_l1_positions + bid_l1_positions,
                'lob_snapshot': lob_dict
            })

        except Exception as e:
            print(f"Error parsing entry {i}: {e}")
            continue

    return pd.DataFrame(data)

def detect_episodes(df):
    """
    Detect episode boundaries based on non-increasing timestamps
    and split data into episodes
    """
    episodes = []
    current_episode = []
    episode_num = 0

    for i, row in df.iterrows():
        if i == 0:
            current_episode.append(row)
            continue

        # Check if timestamp decreased (new episode starts)
        if row['timestamp'] < df.iloc[i-1]['timestamp']:
            # Save current episode if it has data
            if current_episode:
                episode_df = pd.DataFrame(current_episode)
                episode_df['episode'] = episode_num
                episodes.append(episode_df)
                episode_num += 1
                current_episode = []

        current_episode.append(row)

    # Don't forget the last episode
    if current_episode:
        episode_df = pd.DataFrame(current_episode)
        episode_df['episode'] = episode_num
        episodes.append(episode_df)

    return episodes

def plot_analysis(df):
    """
    Create plots for mid-price, inventory PDF, and positions PDF
    """
    # Detect episodes
    episodes = detect_episodes(df)
    print(f"Detected {len(episodes)} episodes")

    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Limit Order Book Analysis', fontsize=16)

    # Plot 1: Mid-price over time with episodes overlaid
    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))

    for episode_idx, episode_df in enumerate(episodes):
        if len(episode_df) > 1:  # Only plot episodes with more than 1 point
            # Reset timestamp to start from 0 for each episode
            episode_timestamps = episode_df['timestamp'].values - episode_df['timestamp'].iloc[0]
            axes[0, 0].plot(episode_timestamps, episode_df['mid_price'],
                            color=colors[episode_idx], linewidth=1.5, alpha=0.7,
                            label=f'Episode {episode_idx + 1}')

    axes[0, 0].set_title('Mid-Price Over Time (Episodes Overlaid)')
    axes[0, 0].set_xlabel('Time (relative to episode start)')
    axes[0, 0].set_ylabel('Mid-Price')
    axes[0, 0].grid(True, alpha=0.3)

    # Only show legend if there are not too many episodes
    if len(episodes) <= 10:
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: Inventory PDF
    inventory_values = df['inventory'].dropna()
    if len(inventory_values) > 0:
        axes[0, 1].hist(inventory_values, bins=30, density=True, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Inventory Distribution (PDF)')
        axes[0, 1].set_xlabel('Inventory (INTC)')
        axes[0, 1].set_ylabel('Probability Density')
        axes[0, 1].grid(True, alpha=0.3)

        # Add vertical line at mean
        mean_inv = inventory_values.mean()
        axes[0, 1].axvline(mean_inv, color='red', linestyle='--',
                           label=f'Mean: {mean_inv:.2f}')
        axes[0, 1].legend()

    # Plot 3: Ask_L1 Positions PDF
    ask_positions = df['ask_l1_positions'].dropna()
    if len(ask_positions) > 0:
        axes[1, 0].hist(ask_positions, bins=20, density=True, alpha=0.7,
                        edgecolor='black', color='orange')
        axes[1, 0].set_title('Ask_L1 Positions Distribution (PDF)')
        axes[1, 0].set_xlabel('Number of Ask_L1 Positions')
        axes[1, 0].set_ylabel('Probability Density')
        axes[1, 0].grid(True, alpha=0.3)

        # Add vertical line at mean
        mean_ask = ask_positions.mean()
        axes[1, 0].axvline(mean_ask, color='red', linestyle='--',
                           label=f'Mean: {mean_ask:.2f}')
        axes[1, 0].legend()

    # Plot 4: Bid_L1 Positions PDF
    bid_positions = df['bid_l1_positions'].dropna()
    if len(bid_positions) > 0:
        axes[1, 1].hist(bid_positions, bins=20, density=True, alpha=0.7,
                        edgecolor='black', color='green')
        axes[1, 1].set_title('Bid_L1 Positions Distribution (PDF)')
        axes[1, 1].set_xlabel('Number of Bid_L1 Positions')
        axes[1, 1].set_ylabel('Probability Density')
        axes[1, 1].grid(True, alpha=0.3)

        # Add vertical line at mean
        mean_bid = bid_positions.mean()
        axes[1, 1].axvline(mean_bid, color='red', linestyle='--',
                           label=f'Mean: {mean_bid:.2f}')
        axes[1, 1].legend()

    plt.tight_layout()
    return fig

def print_summary_stats(df):
    """
    Print summary statistics
    """
    episodes = detect_episodes(df)

    print("=== SUMMARY STATISTICS ===")
    print(f"Total records parsed: {len(df)}")
    print(f"Number of episodes detected: {len(episodes)}")
    print(f"Time range: {df['timestamp'].min():.2f} - {df['timestamp'].max():.2f}")
    print()

    # Episode statistics
    episode_lengths = [len(ep) for ep in episodes]
    print("Episode Statistics:")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f} records")
    print(f"  Min episode length: {np.min(episode_lengths)} records")
    print(f"  Max episode length: {np.max(episode_lengths)} records")
    print()

    print("Mid-Price Statistics:")
    print(f"  Mean: {df['mid_price'].mean():.4f}")
    print(f"  Std:  {df['mid_price'].std():.4f}")
    print(f"  Min:  {df['mid_price'].min():.4f}")
    print(f"  Max:  {df['mid_price'].max():.4f}")
    print()

    print("Inventory Statistics:")
    print(f"  Mean: {df['inventory'].mean():.2f}")
    print(f"  Std:  {df['inventory'].std():.2f}")
    print(f"  Min:  {df['inventory'].min()}")
    print(f"  Max:  {df['inventory'].max()}")
    print()

    print("Ask_L1 Positions Statistics:")
    print(f"  Mean: {df['ask_l1_positions'].mean():.2f}")
    print(f"  Std:  {df['ask_l1_positions'].std():.2f}")
    print(f"  Min:  {df['ask_l1_positions'].min()}")
    print(f"  Max:  {df['ask_l1_positions'].max()}")
    print()

    print("Bid_L1 Positions Statistics:")
    print(f"  Mean: {df['bid_l1_positions'].mean():.2f}")
    print(f"  Std:  {df['bid_l1_positions'].std():.2f}")
    print(f"  Min:  {df['bid_l1_positions'].min()}")
    print(f"  Max:  {df['bid_l1_positions'].max()}")

def main():
    """
    Main function to run the analysis
    """
    # Replace with your log file path
    log_filename = "D:\\PhD\\results - icrl\\test_fullstate_pl.o5742934"  # Change this to your file path

    print("Parsing log file...")
    df = parse_log_file(log_filename)

    if df.empty:
        print("No data could be parsed from the log file.")
        return

    print(f"Successfully parsed {len(df)} records.")

    # Print summary statistics
    print_summary_stats(df)

    # Create plots
    print("\nGenerating plots...")
    fig = plot_analysis(df)

    plt.savefig(log_filename.split('.o')[0]+'.png')
    plt.show()
    # Save the processed data
    # output_filename = "processed_lob_data.csv"
    # df.to_csv(output_filename, index=False)
    # print(f"\nProcessed data saved to: {output_filename}")

    return df

def calcSharpe(file):
    arr = np.load(file)
    episode_boundaries = np.where(np.diff(arr[0]) <0)[0]
    start_idxs = episode_boundaries[:-1] + 1
    end_idxs = episode_boundaries[1:]
    log_ret2 = []
    for s, e in zip(start_idxs, end_idxs):
        log_ret2.append(np.log(arr[1][e]/arr[1][s]))
    sharpe=np.mean(log_ret2)/np.std(log_ret2)
    annualized_sharpe = np.sqrt(6.5*252*3600/(arr[0][e] - arr[0][s]))*sharpe
    return sharpe, annualized_sharpe

def calcSharpe_fRL(file):
    arr = np.load(file)  # arr[0] = times, arr[1] = values
    times = arr[0]
    values = arr[1]

    def window_sharpe(t_start, t_end):
        # Restrict to the current window
        mask = (times >= t_start) & (times < t_end)
        t_win = times[mask]
        v_win = values[mask]

        if len(t_win) < 2:
            return np.nan, np.nan

        # Recalculate episode boundaries inside this window
        episode_boundaries = np.where(np.diff(t_win) < 0)[0]
        print('Num Episodes = ', len(episode_boundaries))
        start_idxs = np.insert(episode_boundaries + 1, 0, 0)
        end_idxs = np.append(episode_boundaries, len(t_win) - 1)

        log_ret2 = []
        for s, e in zip(start_idxs, end_idxs):
            if v_win[s] > 0 and v_win[e] > 0:
                log_ret2.append(np.log(v_win[e] / v_win[s]))

        if len(log_ret2) == 0:
            return np.nan, np.nan

        sharpe = np.mean(log_ret2) / np.std(log_ret2)
        annualized_sharpe = np.sqrt(6.5 * 252 * 3600 / (t_end - t_start)) * sharpe
        return sharpe, annualized_sharpe

    # Compute Sharpe for the two windows
    sharpe_100_250, ann_sharpe_100_250 = window_sharpe(100, 250)
    sharpe_250_400, ann_sharpe_250_400 = window_sharpe(250, 400)

    return (sharpe_100_250, ann_sharpe_100_250,
            sharpe_250_400, ann_sharpe_250_400)




def check_bid_ask_symmetry(params: Dict[str, Any], tolerance: float = 1e-10) -> Dict[str, Any]:
    """
    Check if a parameters dictionary is symmetric between Bid and Ask values.

    Args:
        params: Dictionary containing bid/ask parameters
        tolerance: Numerical tolerance for floating point comparisons

    Returns:
        Dictionary with symmetry analysis results
    """
    results = {
        'is_symmetric': True,
        'symmetric_pairs': [],
        'asymmetric_pairs': [],
        'bid_only_keys': [],
        'ask_only_keys': [],
        'details': []
    }

    # Find all bid and ask keys
    bid_keys = [k for k in params.keys() if 'Bid' in k]
    ask_keys = [k for k in params.keys() if 'Ask' in k]

    # Check symmetry: compare transitions where bid/ask sides are swapped
    # For example: lo_deep_Ask->co_top_Bid should equal lo_deep_Bid->co_top_Ask
    processed_pairs = set()

    for key in params.keys():
        if '->' in key and ('Bid' in key or 'Ask' in key):
            # Parse the transition: from_state->to_state
            from_state, to_state = key.split('->')

            # Create the symmetric counterpart by swapping Bid/Ask in both states
            symmetric_from = swap_bid_ask(from_state)
            symmetric_to = swap_bid_ask(to_state)
            symmetric_key = f"{symmetric_from}->{symmetric_to}"

            # Create a pair identifier to avoid duplicate checking
            pair_id = tuple(sorted([key, symmetric_key]))
            if pair_id in processed_pairs:
                continue
            processed_pairs.add(pair_id)

            if symmetric_key not in params:
                results['details'].append(f"Missing symmetric counterpart: {key} (expected: {symmetric_key})")
                results['is_symmetric'] = False
                continue

            # Compare values
            val1 = params[key]
            val2 = params[symmetric_key]

            is_pair_symmetric = compare_values(val1, val2, tolerance)

            if is_pair_symmetric:
                results['symmetric_pairs'].append((key, symmetric_key))
                results['details'].append(f"✓ Symmetric: {key} ≈ {symmetric_key}")
            else:
                results['asymmetric_pairs'].append((key, symmetric_key))
                results['is_symmetric'] = False
                results['details'].append(f"✗ Asymmetric: {key} ≠ {symmetric_key}")
                results['details'].append(f"  Value 1: {val1}")
                results['details'].append(f"  Value 2: {val2}")

    # Also check non-transition parameters (simple Bid/Ask pairs)
    for bid_key in bid_keys:
        if '->' not in bid_key:  # Skip transition keys, already handled above
            ask_key = bid_key.replace('Bid', 'Ask')

            if ask_key not in params:
                results['bid_only_keys'].append(bid_key)
                results['is_symmetric'] = False
                results['details'].append(f"No corresponding Ask key for: {bid_key}")
                continue

            # Compare values
            bid_val = params[bid_key]
            ask_val = params[ask_key]

            is_pair_symmetric = compare_values(bid_val, ask_val, tolerance)

            if is_pair_symmetric:
                results['symmetric_pairs'].append((bid_key, ask_key))
                results['details'].append(f"✓ Symmetric: {bid_key} ≈ {ask_key}")
            else:
                results['asymmetric_pairs'].append((bid_key, ask_key))
                results['is_symmetric'] = False
                results['details'].append(f"✗ Asymmetric: {bid_key} ≠ {ask_key}")
                results['details'].append(f"  Bid value: {bid_val}")
                results['details'].append(f"  Ask value: {ask_val}")

    # Check for ask keys without corresponding bid keys (for non-transition keys)
    for ask_key in ask_keys:
        if '->' not in ask_key:  # Skip transition keys, already handled above
            bid_key = ask_key.replace('Ask', 'Bid')
            if bid_key not in params:
                results['ask_only_keys'].append(ask_key)
                results['is_symmetric'] = False
                results['details'].append(f"No corresponding Bid key for: {ask_key}")

    return results

def swap_bid_ask(state: str) -> str:
    """
    Swap Bid and Ask in a state name.

    Args:
        state: State name that may contain 'Bid' or 'Ask'

    Returns:
        State name with Bid/Ask swapped
    """
    if 'Bid' in state:
        return state.replace('Bid', 'Ask')
    elif 'Ask' in state:
        return state.replace('Ask', 'Bid')
    else:
        return state

def compare_values(val1: Any, val2: Any, tolerance: float = 1e-10) -> bool:
    """
    Compare two values for equality, handling different data types.

    Args:
        val1, val2: Values to compare
        tolerance: Numerical tolerance for floating point comparisons

    Returns:
        True if values are considered equal
    """
    # Handle numpy scalars
    if hasattr(val1, 'dtype') and np.isscalar(val1):
        val1 = val1.item()
    if hasattr(val2, 'dtype') and np.isscalar(val2):
        val2 = val2.item()

    # Handle tuples (like your transition matrix entries)
    if isinstance(val1, tuple) and isinstance(val2, tuple):
        if len(val1) != len(val2):
            return False
        return all(compare_values(v1, v2, tolerance) for v1, v2 in zip(val1, val2))

    # Handle numpy arrays
    if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
        if val1.shape != val2.shape:
            return False
        return np.allclose(val1, val2, rtol=tolerance, atol=tolerance)

    # Handle numeric values
    if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
        return abs(val1 - val2) <= tolerance

    # Handle other types with direct comparison
    return val1 == val2

def print_symmetry_report(results: Dict[str, Any]) -> None:
    """Print a formatted report of the symmetry analysis."""
    print("=" * 60)
    print("BID-ASK SYMMETRY ANALYSIS")
    print("=" * 60)

    if results['is_symmetric']:
        print("✓ RESULT: Parameters are SYMMETRIC")
    else:
        print("✗ RESULT: Parameters are NOT SYMMETRIC")

    print(f"\nSummary:")
    print(f"  Symmetric pairs: {len(results['symmetric_pairs'])}")
    print(f"  Asymmetric pairs: {len(results['asymmetric_pairs'])}")
    print(f"  Bid-only keys: {len(results['bid_only_keys'])}")
    print(f"  Ask-only keys: {len(results['ask_only_keys'])}")

    if results['details']:
        print(f"\nDetails:")
        for detail in results['details']:
            print(f"  {detail}")

def calcSlippage_trained(start_midprices_file, twap_executions_file):
    start_times = np.load("/Users/alirazajafree/researchprojects/FinalTWAPTesting/start_times.npy")
    startimes = [time - 100 for time in start_times]
    # start_midprices = np.load(start_midprices_file)
    twap_executions = np.load(twap_executions_file)
    slippage_total = 0
    differences = 0
    epis = 0
    for episode in range(61):
        time = startimes[episode]
        episode_executions = twap_executions[f"episode_{episode}"]
        episode_total_execution_value = 0
        episode_side = episode_executions[0]["side"]
        total_executed = 0
        for execution in episode_executions:
            # print(execution["price"])
            total_executed += execution["quantity"]
            episode_total_execution_value += execution["price"]*execution["quantity"]
        start_midprice = episode_executions[0]["price"]
        arrival = start_midprice * total_executed
        if episode_side == "buy":
            if (episode_total_execution_value - arrival) < 0:
                thing+=1
                print(f"Execution value: {episode_total_execution_value}. Arrival cost: {arrival}")
            differences += (episode_total_execution_value - arrival)
            
            slippage = (episode_total_execution_value - arrival) / arrival
            epis+=1
            slippage_total += slippage
        # else:
        #     differences += (arrival - episode_total_execution_value)
        #     slippage = (arrival - episode_total_execution_value) / arrival
            
    print(epis)
    print(f"Differences: {differences}")
    print((slippage_total/epis)*100*100)

def calcSlippage(start_midprices_file, twap_executions_file):
    start_midprices = np.load(start_midprices_file)
    twap_executions = np.load(twap_executions_file)
    slippage_total = 0
    differences = 0
    for episode in range(61):
        episode_executions = twap_executions[f"episode_{episode}"]
        episode_total_execution_value = 0
        episode_side = episode_executions[0]["side"]
        total_executed = 0
        for execution in episode_executions:
            
            total_executed += execution["quantity"]
            episode_total_execution_value += execution["price"]*execution["quantity"]
        start_midprice = episode_executions[0]["price"]
        arrival = start_midprice * total_executed
        if episode_side == "sell":
            differences += (arrival - episode_total_execution_value)
            slippage = (arrival - episode_total_execution_value) / arrival
        else:
            differences += (episode_total_execution_value - arrival)
            slippage = (episode_total_execution_value - arrival) / arrival
        slippage_total += slippage

    print(f"Differences: {differences}")
    print((slippage_total/61)*100*100)

def graphInventories(beforetwap, withtwap_buy, withtwap_sell):
    plt.figure(figsize=(12, 8))
    
    # Flatten the lists of lists to get all inventory values
    all_before = []
    all_buy = []
    all_sell = []
    
    # Flatten beforetwap (list of lists across episodes)
    for episode_inventories in beforetwap:
        all_before.extend(episode_inventories)
    
    # Flatten withtwap_buy (list of lists across episodes) 
    for episode_inventories in withtwap_buy:
        all_buy.extend(episode_inventories)
        
    # Flatten withtwap_sell (list of lists across episodes)
    for episode_inventories in withtwap_sell:
        all_sell.extend(episode_inventories)
    
    # Create normalized histograms (density=True gives probability density)
    # weights parameter normalizes to show ratios/proportions that sum to 1
    if all_before:
        weights_before = np.ones(len(all_before)) / len(all_before)
        plt.hist(all_before, bins=30, alpha=0.7, label=f'Before TWAP (n={len(all_before)})', 
                 color='blue', edgecolor='black', weights=weights_before)
    
    if all_buy:
        weights_buy = np.ones(len(all_buy)) / len(all_buy)
        plt.hist(all_buy, bins=30, alpha=0.7, label=f'With TWAP Buy (n={len(all_buy)})', 
                 color='green', edgecolor='black', weights=weights_buy)
    
    if all_sell:
        weights_sell = np.ones(len(all_sell)) / len(all_sell)
        plt.hist(all_sell, bins=30, alpha=0.7, label=f'With TWAP Sell (n={len(all_sell)})', 
                 color='red', edgecolor='black', weights=weights_sell)
    
    # Add median lines
    if all_before:
        plt.axvline(np.median(all_before), color='blue', linestyle='--', linewidth=2, 
                   label=f'Before Median: {np.median(all_before):.1f}')
    if all_buy:
        plt.axvline(np.median(all_buy), color='green', linestyle='--', linewidth=2,
                   label=f'Buy Median: {np.median(all_buy):.1f}')
    if all_sell:
        plt.axvline(np.median(all_sell), color='red', linestyle='--', linewidth=2,
                   label=f'Sell Median: {np.median(all_sell):.1f}')
    
    plt.xlabel('RL Agent Inventory')
    plt.ylabel('Proportion') 
    plt.title('RL Agent Inventory Distribution: Before vs With TWAP (Normalized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Print summary statistics
    print(f"Before TWAP - Count: {len(all_before)}, Median: {np.median(all_before) if all_before else 0:.2f}, Mean: {np.mean(all_before) if all_before else 0:.2f}")
    print(f"With TWAP Buy - Count: {len(all_buy)}, Median: {np.median(all_buy) if all_buy else 0:.2f}, Mean: {np.mean(all_buy) if all_buy else 0:.2f}")
    print(f"With TWAP Sell - Count: {len(all_sell)}, Median: {np.median(all_sell) if all_sell else 0:.2f}, Mean: {np.mean(all_sell) if all_sell else 0:.2f}")
    
    plt.tight_layout()
    plt.show()

def getaveragesplit():
    file = '/Users/alirazajafree/researchprojects/Training Results/slippages/Outputfile/TRAINING_with_twap.sh.o5976730'
    
    buy_slippages = []
    sell_slippages = []
    
    with open(file, 'r') as f:
        lines = f.readlines()
    
    current_episode = None
    current_side = None
    
    for line in lines:
        line = line.strip()
        
        # Check for episode start
        if "Start of episode" in line and "TWAP Time is" in line and "side is" in line:
            # Extract episode number and side
            import re
            match = re.search(r"Start of episode (\d+).*side is (\w+)", line)
            if match:
                current_episode = int(match.group(1))
                current_side = match.group(2)
                print(f"Found episode {current_episode} with side {current_side}")
        
        # Check for SELL slippage
        elif "SELL - Executed:" in line and "Slippage:" in line:
            # Updated regex to capture scientific notation: e.g., -1.23e-5, 4.56E+3, etc.
            match = re.search(r"Slippage: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            if match:
                slippage = float(match.group(1))
                sell_slippages.append(slippage)
                print(f"SELL slippage found: {match.group(1)} = {slippage}")
        
        # Check for BUY slippage  
        elif "BUY - Executed:" in line and "Slippage:" in line:
            # Updated regex to capture scientific notation
            match = re.search(r"Slippage: ([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            if match:
                slippage = float(match.group(1))
                buy_slippages.append(slippage)
                print(f"BUY slippage found: {match.group(1)} = {slippage}")
    
    # Calculate averages
    avg_buy_slippage = np.mean(buy_slippages) if buy_slippages else 0
    avg_sell_slippage = np.mean(sell_slippages) if sell_slippages else 0
    
    print(f"\n=== SLIPPAGE ANALYSIS ===")
    print(f"Buy Episodes: {len(buy_slippages)}")
    print(f"Buy Slippages: {buy_slippages}")
    print(f"Average Buy Slippage: {avg_buy_slippage:.8f} ({avg_buy_slippage*10000:.4f} bps)")
    
    print(f"\nSell Episodes: {len(sell_slippages)}")
    print(f"Sell Slippages: {sell_slippages}")
    print(f"Average Sell Slippage: {avg_sell_slippage:.8f} ({avg_sell_slippage*10000:.4f} bps)")
    
    print(f"\nOverall Average: {(avg_buy_slippage + avg_sell_slippage)/2:.8f} ({((avg_buy_slippage + avg_sell_slippage)/2)*10000:.4f} bps)")
    
    return {
        'buy_slippages': buy_slippages,
        'sell_slippages': sell_slippages,
        'avg_buy_slippage': avg_buy_slippage,
        'avg_sell_slippage': avg_sell_slippage,
        'avg_overall_slippage': (avg_buy_slippage + avg_sell_slippage)/2
    }

def detect_episodes_arr(time_array):
    """Return list of (start_idx, end_idx) episode index ranges."""
    episodes = []
    start_idx = 0
    for i in range(1, len(time_array)):
        if time_array[i] < time_array[i - 1]:  # non-monotonic → new episode
            episodes.append((start_idx, i - 1))
            start_idx = i
    episodes.append((start_idx, len(time_array) - 1))
    return episodes

def plot_rl_twap_action_inv(filepath):
    # Patterns for detecting key lines
    time_pattern = re.compile(r"^(\d+\.\d+)")
    rl_action_pattern = re.compile(r"RL Action:\s*(\(.*\))")
    twap_action_pattern = re.compile(r"TWAP Action:\s*(\(.*\))")
    rl_inv_pattern = re.compile(r"RL Inventory:\s*(-?\d+)")
    twap_inv_pattern = re.compile(r"TWAP Inventory:\s*(-?\d+)")

    # Containers
    rl_times, rl_actions, rl_inventories = [], [], []
    twap_times, twap_actions, twap_inventories = [], [], []

    current_time = None
    current_rl_inv = None
    current_twap_inv = None

    def clean_action_string(s):
        """Remove numpy wrappers like np.float64(100.0) → 100.0"""
        s = re.sub(r"np\.\w+\(([^()]+)\)", r"\1", s)
        return s

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Detect time
            time_match = time_pattern.match(line)
            if time_match:
                current_time = float(time_match.group(1))

            # RL Inventory
            rl_inv_match = rl_inv_pattern.search(line)
            if rl_inv_match:
                current_rl_inv = int(rl_inv_match.group(1))

            # TWAP Inventory
            twap_inv_match = twap_inv_pattern.search(line)
            if twap_inv_match:
                current_twap_inv = int(twap_inv_match.group(1))


            # RL Action
            rl_match = rl_action_pattern.search(line)
            if (rl_match is not None) & (current_rl_inv is not None) & (current_time  is not None):
                action_str = clean_action_string(rl_match.group(1))
                try:
                    action_tuple = ast.literal_eval(action_str)
                    rl_times.append(current_time)
                    rl_actions.append(action_tuple[1][0])  # extract the 12 in (1, (12, 1))
                    rl_inventories.append(current_rl_inv if current_rl_inv is not None else np.nan)
                except Exception as e:
                    print(f"⚠️ Warning: could not parse RL action line:\n  {line}\n  Error: {e}")
                current_time = None  # reset for next block

            # TWAP Action
            twap_match = twap_action_pattern.search(line)
            if (twap_match is not None) & (current_twap_inv is not None) & (current_time  is not None) :
                action_str = clean_action_string(twap_match.group(1))
                try:
                    action_tuple = ast.literal_eval(action_str)
                    twap_times.append(current_time)
                    twap_actions.append(action_tuple[1][0])
                    twap_inventories.append(current_twap_inv if current_twap_inv is not None else np.nan)
                except Exception as e:
                    print(f"⚠️ Warning: could not parse TWAP action line:\n  {line}\n  Error: {e}")
                current_time = None

    rl_array = np.column_stack((rl_times, rl_actions, rl_inventories)) if rl_times else np.empty((0, 3))
    twap_array = np.column_stack((twap_times, twap_actions, twap_inventories)) if twap_times else np.empty((0, 3))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='col')
    fig.suptitle("RL vs TWAP Actions and Inventory over Time", fontsize=14, weight='bold')

    # === RL: Action vs Time ===
    axes[0, 0].scatter(rl_array[:, 0], rl_array[:, 1], color='tab:blue', s=5, label="RL Action")
    axes[0, 0].set_ylabel("Action Value")
    axes[0, 0].set_title("RL: Action vs Time")
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    axes[0, 0].legend()

    # === TWAP: Action vs Time ===
    axes[0, 1].scatter(twap_array[:, 0], twap_array[:, 1], color='tab:orange', s=5, label="TWAP Action")
    axes[0, 1].set_title("TWAP: Action vs Time")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].legend()

    # === RL: Inventory vs Time (faint per episode) ===
    episodes = detect_episodes_arr(rl_array[:, 0])
    for (start, end) in episodes:
        axes[1, 0].plot(rl_array[start:end+1, 0],
                        rl_array[start:end+1, 2],
                        color='tab:green',
                        alpha=0.5)
        axes[1, 1].plot(twap_array[start+1:end-1, 0], twap_array[start+1:end-1, 2], color='tab:red', alpha=0.5)
        # vertical bar to mark episode end in the RL action plot
        end_time = rl_array[end, 0]
        axes[0, 0].axvline(end_time, color='gray', linestyle='--', alpha=0.3)

    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Inventory")
    axes[1, 0].set_title("RL: Inventory vs Time (per episode)")
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # === TWAP: Inventory vs Time ===

    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_title("TWAP: Inventory vs Time")
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # === Separate plot for last episode RL actions ===
    last_start, last_end = episodes[-1]
    plt.figure(figsize=(8, 4))
    plt.scatter(rl_array[last_start:last_end+1, 0],
                rl_array[last_start:last_end+1, 1],
                color='tab:blue', s=8)
    plt.title("RL Actions (Last Episode Only)")
    plt.xlabel("Time")
    plt.ylabel("Action Value")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # === RL action statistics ===
    print("\n=== RL Action Statistics ===")
    times = rl_array[:, 0]
    actions = rl_array[:, 1]
    t_min, t_max = np.min(times), np.max(times)
    bins = np.arange(t_min, t_max + 10, 10)
    bin_indices = np.digitize(times, bins)

    # Compute normalized frequency per bin
    unique_actions = np.unique(actions)
    freq_matrix = np.zeros((len(bins), len(unique_actions)))

    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.any(mask):
            counts = Counter(actions[mask])
            total = sum(counts.values())
            for j, a in enumerate(unique_actions):
                freq_matrix[i, j] = counts.get(a, 0) / total

    # Plot action frequency vs. time
    plt.figure(figsize=(10, 5))
    for j, a in enumerate(unique_actions):
        if a==12: continue
        plt.plot(bins, freq_matrix[:, j], label=f"Action {a}", alpha=0.5)
    plt.title("RL Action Distribution (10s bins)")
    plt.xlabel("Time")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # === Inventory Distribution Comparison (100–250s vs 250–400s) ===
    mask_1 = (rl_array[:, 0] >= 100) & (rl_array[:, 0] < 250)
    mask_2 = (rl_array[:, 0] >= 250) & (rl_array[:, 0] < 400)
    inv_1 = rl_array[mask_1, 2]
    inv_2 = rl_array[mask_2, 2]
    inv_1 = inv_1[~np.isnan(inv_1)]
    inv_2 = inv_2[~np.isnan(inv_2)]
    plt.figure(figsize=(8, 4))
    bins = np.linspace(min(inv_1), max(inv_1), 30)
    plt.hist(inv_1, bins=bins, alpha=0.5, color='tab:green', label='100–250 s', density=True)
    bins = np.linspace(min(inv_2), max(inv_2), 30)
    plt.hist(inv_2, bins=bins, alpha=0.5, color='tab:purple', label='250–400 s', density=True)
    plt.title("RL Inventory Distribution Comparison")
    plt.xlabel("Inventory")
    plt.ylabel("Normalized Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    return rl_array, twap_array

def slippage_calc():
    import numpy as np
    twap_obsv_episodes = np.load("D:\\PhD\\students\\aliraza\\retest_fRL_versus_buytwap_observations.npy", allow_pickle=True)
    starting_midprices = []
    decoded_episodes = {}
    for episode_idx, episode_obsv in enumerate(twap_obsv_episodes):
        episode_data = {
            'episode': episode_idx,
            'timestamps': [],
            'cash': [],
            'inventory': [],
            'lob_data': [],
            'market_volume': [],
            'current_time': [],
            'num_observations': len(episode_obsv)
        }
        for obs_idx, obs in enumerate(episode_obsv):
            episode_data['cash'].append(obs.get('Cash', None))
            episode_data['inventory'].append(obs.get('Inventory', None))
            episode_data['current_time'].append(obs.get('current_time', None))
            episode_data['market_volume'].append(obs.get('market_volume', None))
            lob = obs.get('LOB0', {})
            lob_snapshot = {
                'ask_l1': lob.get('Ask_L1'),
                'bid_l1': lob.get('Bid_L1'),
                'ask_l2': lob.get('Ask_L2'),
                'bid_l2': lob.get('Bid_L2')
            }
            episode_data['lob_data'].append(lob_snapshot)
            if obs_idx == 0:
                starting_midprices.append((lob_snapshot['ask_l1'][0]+lob_snapshot['bid_l1'][0])/2)
    print(starting_midprices)
    file = 'D:\\PhD\\students\\aliraza\\retest_fRL_versus_buytotal_executed.npy'
    exec_sell =np.load(file)
    print(exec_sell)
    file = 'D:\\PhD\\students\\aliraza\\retest_fRL_versus_buyfinal_cash.npy'
    cash_sell = np.load(file)
    exec_price = np.abs(cash_sell- 1e6)/exec_sell
    print(exec_price)
    print((exec_price/starting_midprices - 1).mean()*10000)

# if __name__ == "__main__":
#     # Your example params
#     params = pickle.load(open("D:\\PhD\\calibrated params\\Symmetric_INTC.OQ_ParamsInferredWCutoffEyeMu_sparseInfer_2019-01-02_2019-12-31_CLSLogLin_10", 'rb'))
#
#     # Check symmetry
#     results = check_bid_ask_symmetry(params)
#
#     # Print report
#     print_symmetry_report(results)
#
#     # Also return the boolean result for programmatic use
#     print(f"\nIs symmetric: {results['is_symmetric']}")

if __name__ == "__main__":
    # getaveragesplit()
    # plot_rl_twap_action_inv("D:\\PhD\\students\\aliraza\\twap_fRL_retesting_buy.o6089213") #twap_fRL_retesting_sell.o6083776")
    # a=calcSharpe_fRL("D:\\PhD\\students\\aliraza\\retest_fRL_versus_buy_profit.npy")
    # print(a)
    slippage_calc()
    # Run the analysis
    # df = main()
    # print(calcSharpe())
    # log_label = "/Users/alirazajafree/researchprojects/FinalTWAPTesting/testing_RL/npy_files/"
    # beforetwap = []
    # twapbuy = []
    # twapsell = []
    # # Find all files matching the pattern
    # without_files = glob.glob(log_label + "without_twap_*.npy")
    # for file in without_files:
    #     arr = np.load(file)
    #     beforetwap.append(arr.tolist())

    # buy_files = glob.glob(log_label + "with_twap_*_buy.npy")
    # for file in buy_files:
    #     arr = np.load(file)
    #     twapbuy.append(arr.tolist())

    # sell_files = glob.glob(log_label + "with_twap_*_sell.npy")
    # for file in sell_files:
    #     arr = np.load(file)
    #     twapsell.append(arr.tolist())

    # graphInventories(beforetwap, twapbuy, twapsell)
    # log_label = '/Users/alirazajafree/researchprojects/FinalTWAPTesting/vsNoAgent/'
    # midprices_file = log_label+ "logstest_no_RL,TWAP_start_midprices.npy"
    # executions_file = log_label+"logstest_no_RL,TWAP_twap_executions.npz"
    # print("No RL")
    # calcSlippage(midprices_file, executions_file)

    # # # log_label = "/Users/alirazajafree/researchprojects/FinalTWAPTesting/vsUntrainedRL/Logs/"
    # # # midprices_file = log_label+ "logstest_untrained_RL,TWAP_start_midprices.npy"
    # # # executions_file = log_label+"logstest_untrained_RL,TWAP_twap_executions.npz"
    # # # print("Untrained")
    # # # calcSlippage(midprices_file, executions_file)

    # log_label = '/Users/alirazajafree/researchprojects/FinalTWAPTesting/vsAdversarialAgent/Logs/randomised_starttimes/'
    # midprices_file = log_label+ "with_randomised_starttimestest_ADVERSARIAL_RL,TWAP_randomisedstart_start_midprices.npy"
    # executions_file = log_label+"with_randomised_starttimestest_ADVERSARIAL_RL,TWAP_randomisedstart_twap_executions.npz"
    # print("Trained")
    # calcSlippage_trained(midprices_file, executions_file)

    # print("Sharpe for trained")
    # print(calcSharpe("/Users/alirazajafree/researchprojects/FinalTWAPTesting/vsAdversarialAgent/Logs/randomised_starttimes/with_randomised_starttimestest_ADVERSARIAL_RL,TWAP_randomisedstart_profit.npy"))

    # print("Sharpe for untrained")
    # print(calcSharpe('/Users/alirazajafree/researchprojects/FinalTWAPTesting/vsUntrainedRL/Logs/logstest_untrained_RL,TWAP_profit.npy'))

    
