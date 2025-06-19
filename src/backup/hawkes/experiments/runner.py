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
path = 'D:\\PhD\\results - small tick\\sim\\smalltick_is\\' #'/SAN/fca/Konark_PhD_Experiments/simulated/smallTick/is/'
fnames = os.listdir(path)
spreads = []
labels = []
ts = []
mids = []
shapes = []
#fig, ax = plt.subplots()
for fname in fnames:
    if 'smalltickhawkes_' in fname:
        with open(path+fname, 'rb') as f:
            T, lob = pickle.load(f)
        ask_t = []
        bid_t = []
        ask_d = []
        bid_d= []
        ask_m_D = []
        bid_m_D = []
        spread = []
        mid = []
        for r in lob:
            #ask_t.append(r['Ask_touch'][0])
            #bid_t.append(r['Bid_touch'][0])
            #ask_d.append(r['Ask_deep'][0])
            #bid_d.append(r['Bid_deep'][0])
            #bid_m_D.append(r['Bid_deep'][0] - 0.01*r['Bid_m_D'])
            #ask_m_D.append(r['Ask_deep'][0] + 0.01*r['Ask_m_D'])
            spread.append(100*(r['Ask_touch'][0] - r['Bid_touch'][0]))
            mid.append(0.5*(r['Ask_touch'][0] + r['Bid_touch'][0]))
        t = np.append([0], np.array(T[1:])[:,1])
        t = t.astype(float) + 9.5*3600
        volumes = []
        for r in lob[2*len(lob)//5:3*len(lob)//5]:
            volume = [(( - r['Bid_deep'][0] + r['mid']) + 0.01*i, r['Bid_deep'][1]/r['Bid_m_D']) for i in range(int(r['Bid_m_D']))]
            volume += [((-r['Bid_touch'][0] + r['mid']), r['Bid_touch'][1])]
            volume += [((r['Ask_touch'][0] - r['mid']), r['Ask_touch'][1])]
            volume += [((r['Ask_deep'][0] - r['mid']) + 0.01*i, r['Ask_deep'][1]/r['Ask_m_D']) for i in range(int(r['Ask_m_D']))]
            volumes += [volume]
        dict_shape = {}
        for v in volumes:
            dists = np.array(v)[:,0]
            vols =  np.array(v)[:,1]
            for d, vol in zip(dists,vols):
                dict_shape[np.round(d, decimals=2)] = dict_shape.get(np.round(d, decimals=2), 0) + vol
        dist = np.sort(list(dict_shape.keys()))
        vol = np.array([dict_shape[d] for d in dist])
        vol = vol/vol.sum()
        shape = (np.array(dist)[vol > 1e-4], vol[vol > 1e-4])


        count = int((max(t) - min(t))/10)
        ts.append(np.diff(t))
        spreads.append(spread)
        mids.append(mid)
        labels.append(fname.split('smalltickhawkes_')[-1])
        shapes.append(shape)
dict_res = {}
a , b, n, s_bar, eps ,eps2, r_mid , D, shapemax = [] ,[ ], [], [], [], [],[],[], []
for i in range(len(spreads)):
    t = ts[i]
    s = spreads[i]
    l = labels[i]
    mid = mids[i]
    dict_res[l.split('smalltickhawkes_')[-1]] = (np.var(s)/np.mean(s)**2, np.var(s)/np.mean(s))
    if float(l.split('_')[0]) > 11:
        continue
    a += [float(l.split('_')[0])]
    b += [float(l.split('_')[1])]
    n += [float(l.split('_')[2])]
    s_bar+= [np.sum(s[:-1]*t)/np.sum(t)]
    eps += [np.exp((0.78 - np.log(s_bar[-1]))/1.3)]
    eps2 += [np.exp((-2.1035 - np.log(np.var(s)/np.mean(s)))/1.36)] #[(np.var(s)/(np.mean(s)*0.0089))**(-1/1.36)]
    r_mid += [np.mean(np.abs(np.diff(mid)))]
    D+=[np.var(s)/np.mean(s)]
    shape = shapes[i]
    shapemax+=[shape[0][np.argmax(shape[1])]]
print(a,b,n,s_bar, eps, eps2, r_mid, D)
plt.plot(eps, r_mid)
plt.plot(eps, D)
plt.plot(eps, shapemax)
plt.yscale('log')
plt.xscale('log')
plt.show()

def phaseDiag(a,b):
    X, y = np.array([a, b]).transpose(), (np.array(k)>1.75).astype(int) + (np.array(k)>5)
    clf = DecisionTreeClassifier().fit(X, y)

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
    plt.figure(figsize=(16,8))
    disp = DecisionBoundaryDisplay.from_estimator(clf, X, response_method="predict", alpha=0.3)
    scatter = disp.ax_.scatter(X[:, 0], X[:, 1], c=y)

    # Add stock data points as stars
    stock_alphas = [alpha for alpha, beta in stock_data.values()]
    stock_betas = [beta for alpha, beta in stock_data.values()]
    stock_names = list(stock_data.keys())

    # Plot stars for stock data
    LT_stars = plt.scatter(stock_alphas[:6], stock_betas[:6], marker='*', s=200, c='red',
                        edgecolors='black', linewidth=1, zorder=5, label='Large Tick Stocks')
    MT_stars = plt.scatter(stock_alphas[6:10], stock_betas[6:10], marker='*', s=200, c='green',
                           edgecolors='black', linewidth=1, zorder=5, label='Medium Tick Stocks')
    ST_stars = plt.scatter(stock_alphas[10:], stock_betas[10:], marker='*', s=200, c='blue',
                           edgecolors='black', linewidth=1, zorder=5, label='Medium Tick Stocks')
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
                         loc="lower right", title="Classes")
    plt.gca().add_artist(legend1)  # Add the first legend back to the plot

    # Add legend for stars
    plt.legend(handles=[LT_stars, MT_stars, ST_stars], labels=['Calib\'d Params: Large Tick','Calib\'d Params: Med. Tick','Calib\'d Params: Small Tick'], loc="upper right")

    plt.title('Phase Diagram: $\\alpha$ and $\\beta$')
    plt.tight_layout()
    plt.savefig(path + 'phaseDiag_is2.png')
    plt.show()