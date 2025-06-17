import numpy as np

data = np.load("/Users/alirazajafree/researchprojects/logstest_RLagent_only_profit.npy")

times = data[0]

portfolios = data[1]

boundaryIndices = [0]

for i in range(1, len(times)):
    if(times[i] < times[i-1]):
        boundaryIndices.append(i)
print(boundaryIndices)
logReturns = []

for i in range(len(boundaryIndices)-1):
    ret = 0
    episodicValues = portfolios[boundaryIndices[i]:boundaryIndices[i+1]]
    logReturns.extend(np.diff(np.log(episodicValues)))

logReturns = np.array(logReturns)
std = np.std(logReturns)
mean = np.mean(logReturns)

sharpe = mean/std

print(sharpe)