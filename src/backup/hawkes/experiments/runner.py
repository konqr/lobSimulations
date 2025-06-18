import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt
import os
import sys
#sys.path.append("C:\\Users\\konar\\IdeaProjects\\lobSimulations")#
sys.path.append("/home/konajain/code/lobSimulations")
from src.data.dataLoader import dataLoader #, fit, inference, simulate
import numpy as np
import time
import pickle
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier
stocks = ['SIRI','BAC', 'INTC','CSCO','ORCL','MSFT','AAPL','ABBV', 'PM','IBM','TSLA','CHTR','AMZN', 'GOOG', 'BKNG']
path = '/SAN/fca/Konark_PhD_Experiments/simulated/smallTick/is/'
fnames = os.listdir(path)
spreads = []
labels = []
#fig, ax = plt.subplots()
for fname in fnames:
    if 'demo' in fname:
        with open(path+fname, 'rb') as f:
            T, lob = pickle.load(f)
        ask_t = []
        bid_t = []
        ask_d = []
        bid_d= []
        ask_m_D = []
        bid_m_D = []
        spread = []
        for r in lob:
            #ask_t.append(r['Ask_touch'][0])
            #bid_t.append(r['Bid_touch'][0])
            #ask_d.append(r['Ask_deep'][0])
            #bid_d.append(r['Bid_deep'][0])
            #bid_m_D.append(r['Bid_deep'][0] - 0.01*r['Bid_m_D'])
            #ask_m_D.append(r['Ask_deep'][0] + 0.01*r['Ask_m_D'])
            spread.append(100*(r['Ask_touch'][0] - r['Bid_touch'][0]))
        t = np.append([0], np.array(T[1:])[:,1])
        t = t.astype(float) + 9.5*3600



        count = int((max(t) - min(t))/10)
        spreads.append(spread)
        labels.append(fname.split('demo_')[-1])
dict_res = {}
a , b, k = [] ,[ ],[]
for i in range(len(spreads)):
    s = spreads[i]
    l = labels[i]
    dict_res[l.split('demo_')[-1]] = (np.var(s)/np.mean(s)**2, np.var(s)/np.mean(s))
    if float(l.split('_')[0]) > 11:
        continue
    a += [float(l.split('_')[0])]
    b += [float(l.split('_')[1])]
    k+= [np.mean(s)]


X, y = np.array([a, b]).transpose(), (np.array(k)>1.75).astype(int) + (np.array(k)>5)
clf = DecisionTreeClassifier().fit(X, y)
#plt.figure(figsize=(16,8))
# disp = DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", alpha=0.3)
# scatter=disp.ax_.scatter(X[:, 0], X[:, 1], c=y)
# plt.xscale('log')
# plt.xlabel('$\\alpha$')
# plt.ylabel('$\\beta$')
# plt.ylim(-0.1, 1)
# legend1 = plt.legend(*(scatter.legend_elements()[0],['Large Tick','Medium Tick','Small Tick']),
#                      loc="lower left", title="Classes")
# plt.title('Phase Diagram: $\\alpha$ and $\\beta$')
# plt.tight_layout()
# plt.savefig(path + 'phaseDiag_is2.png')
# plt.show()

# Stock data from the table
stock_data = {
    'SIRI': (0.0102, 0.98),
    'BAC': (0.0103, 0.49),
    'INTC': (0.0130, 0.94),
    'CSCO': (0.0250, 0.60),
    'ORCL': (0.0190, 0.18),
    'MSFT': (0.0230, 0.19),
    'ABBV': (0.0440, 0.46),
    'PM': (0.2750, 0.35),
    'AAPL': (0.1350, 0.59),
    'IBM': (0.2350, 0.59),
    'TSLA': (1.3610, 0.48),
    'CHTR': (1.4100, 0.41),
    'AMZN': (1.9630, 0.41),
    'GOOG': (3.2670, 0.50),
    'BKNG': (3.7410, 0.46)
}

# Create the phase diagram
#plt.figure(figsize=(16,8))
disp = DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", alpha=0.3)
scatter = disp.ax_.scatter(X[:, 0], X[:, 1], c=y)

# Add stock data points as stars
stock_alphas = [alpha for alpha, beta in stock_data.values()]
stock_betas = [beta for alpha, beta in stock_data.values()]
stock_names = list(stock_data.keys())

# Plot stars for stock data
stars = plt.scatter(stock_alphas, stock_betas, marker='*', s=200, c='red',
                    edgecolors='black', linewidth=1, zorder=5, label='Stocks')

# Add labels for each stock
for i, (name, (alpha, beta)) in enumerate(stock_data.items()):
    plt.annotate(name, (alpha, beta), xytext=(5, 5), textcoords='offset points',
                 fontsize=9, fontweight='bold', color='darkred')

plt.xscale('log')
plt.xlabel('$\\alpha$')
plt.ylabel('$\\beta$')
plt.ylim(-0.1, 1)

# Create legends
legend1 = plt.legend(*(scatter.legend_elements()[0], ['Large Tick', 'Medium Tick', 'Small Tick']),
                     loc="lower left", title="Classes")
plt.gca().add_artist(legend1)  # Add the first legend back to the plot

# Add legend for stars
plt.legend(handles=[stars], labels=['Stock Data'], loc="upper right")

plt.title('Phase Diagram: $\\alpha$ and $\\beta$')
plt.tight_layout()
plt.savefig(path + 'phaseDiag_is2.png')
plt.show()