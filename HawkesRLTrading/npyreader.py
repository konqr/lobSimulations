import numpy as np
import glob
import matplotlib.pyplot as plt

data = np.load("/Users/alirazajafree/30JuneCopy/logstest_RLAgent_vs_BUY_TWAP_10200q_1s_profit.npy")


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

aggregatePricePaths()
