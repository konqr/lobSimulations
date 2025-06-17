import re
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import seaborn as sns

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

def plot_analysis(df):
    """
    Create plots for mid-price, inventory PDF, and positions PDF
    """
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Limit Order Book Analysis', fontsize=16)

    # Plot 1: Mid-price over time
    axes[0, 0].plot(df['timestamp'], df['mid_price'], linewidth=1.5, alpha=0.8)
    axes[0, 0].set_title('Mid-Price Over Time')
    axes[0, 0].set_xlabel('Timestamp')
    axes[0, 0].set_ylabel('Mid-Price')
    axes[0, 0].grid(True, alpha=0.3)

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
    print("=== SUMMARY STATISTICS ===")
    print(f"Total records parsed: {len(df)}")
    print(f"Time range: {df['timestamp'].min():.2f} - {df['timestamp'].max():.2f}")
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
    log_filename = "D:\\PhD\\results - icrl\\symmHP_lowEpochs_standard.o5687496"  # Change this to your file path

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
    plt.show()

    # Save the processed data
    output_filename = "processed_lob_data.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nProcessed data saved to: {output_filename}")

    return df

if __name__ == "__main__":
    # Run the analysis
    df = main()