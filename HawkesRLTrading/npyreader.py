import numpy as np
import glob
import matplotlib.pyplot as plt

data = np.load("/Users/alirazajafree/researchprojects/probabilistictests/probabilisticalone.npy")


file_pattern = "/Users/alirazajafree/30JuneCopy/Market Impact/Price path/TWAP/*.npy"
file_list = glob.glob(file_pattern)
data_list = [np.load(f) for f in file_list]


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

def getSharpeNoEpisodeBoundaries(data, window_size=100):
    """
    Calculate Sharpe ratio for single episode data using rolling windows.
    
    Parameters:
    data: numpy array with shape (2, n) where data[0] is time, data[1] is portfolio value
    window_size: size of rolling window for calculating returns
    
    Returns:
    sharpe: Sharpe ratio
    ann_sharpe: Annualized Sharpe ratio
    """
    times = data[0]
    portfolio_values = data[1]
    
    # Method 1: Use rolling windows to calculate returns
    log_returns = []
    
    for i in range(window_size, len(portfolio_values), window_size):
        start_val = portfolio_values[i - window_size]
        end_val = portfolio_values[i]
        if start_val > 0:  # Avoid log(0) or negative values
            log_returns.append(np.log(end_val / start_val))
    
    # If we don't have enough data for rolling windows, use consecutive periods
    if len(log_returns) < 2:
        log_returns = []
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i-1] > 0:
                log_returns.append(np.log(portfolio_values[i] / portfolio_values[i-1]))
    
    log_returns = np.array(log_returns)
    
    if len(log_returns) == 0:
        print("No valid returns calculated")
        return 0, 0
    
    # Calculate Sharpe ratio
    mean_return = np.mean(log_returns)
    std_return = np.std(log_returns)
    
    if std_return == 0:
        sharpe = 0
    else:
        sharpe = mean_return / std_return
    
    # Annualize (assuming your time units and scaling)
    ann_sharpe = sharpe * np.sqrt(6.5 * 12 * 252)
    
    print(f"Single Episode Sharpe: {sharpe}")
    print(f"Single Episode Ann_sharpe: {ann_sharpe}")
    
    return sharpe, ann_sharpe


def getSharpe():
    arr = data
    episode_boundaries = np.where(np.diff(arr[0]) <0)[0]
    start_idxs = episode_boundaries[:-1] + 1
    end_idxs = episode_boundaries[1:]
    log_ret2 = []
    for s, e in zip(start_idxs, end_idxs):
        log_ret2.append(np.log(arr[1][e]/arr[1][s]))
    sharpe = np.mean(log_ret2)/np.std(log_ret2)
    ann_sharpe = sharpe*np.sqrt(6.5*12*252)
    print(f"Sharpe: {sharpe}")
    print(f"Ann_sharpe: {ann_sharpe}")

def aggregatePricePaths():
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
    plt.plot(mean_path, color='orange', linewidth=2, label='Average')
    
    plt.xlabel("Time step")
    plt.ylabel("Percentage change in midprice")
    plt.title("Aggregated Price Paths from Multiple Runs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("aggregated_price_paths.png", dpi=300, bbox_inches='tight')
    plt.show()

# aggregatePricePaths()
getSharpeNoEpisodeBoundaries(data)