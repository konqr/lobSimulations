import re
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns
import pickle
import numpy as np
from typing import Dict, Any, Tuple, List

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
    log_filename = "D:\\PhD\\results - icrl\\test_standard.o5707312"  # Change this to your file path

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

def calcSharpe():
    arr = np.load("D:\\PhD\\results - icrl\\logsinv10_symmHP_lowEpochs_standard_profit.npy")
    episode_boundaries = np.where(np.diff(arr[0]) <0)[0]
    start_idxs = episode_boundaries[:-1] + 1
    end_idxs = episode_boundaries[1:]
    log_ret2 = []
    for s, e in zip(start_idxs, end_idxs):
        log_ret2.append(np.log(arr[1][e]/arr[1][s]))
    sharpe=np.mean(log_ret2)/np.std(log_ret2)
    annualized_sharpe = np.sqrt(6.5*12*252)*sharpe
    return sharpe, annualized_sharpe


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
    # Run the analysis
    df = main()
    print(calcSharpe())