import numpy as np


data = np.load("/Users/alirazajafree/30JuneCopy/logstest_RLAgent_vs_BUY_TWAP_10200q_1s_profit.npy")

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