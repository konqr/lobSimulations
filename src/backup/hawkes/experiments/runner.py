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
import seaborn as sns
import matplotlib.ticker as ticker


def extract_stylized_facts():
    stocks = ['SIRI','BAC', 'INTC','CSCO','ORCL','MSFT','AAPL','ABBV', 'PM','IBM','TSLA','CHTR','AMZN', 'GOOG', 'BKNG']
    path = 'D:\\PhD\\results - small tick\\sim\\smalltick_is\\' #'/SAN/fca/Konark_PhD_Experiments/simulated/smallTick/is/'
    fnames = os.listdir(path)
    spreads = []
    labels = []
    ts = []
    mids = []
    shapes = []
    sparses = []
    #fig, ax = plt.subplots()
    for fname in fnames:
        if 'demo_' in fname:
            print(fname)
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
            #     #ask_t.append(r['Ask_touch'][0])
            #     #bid_t.append(r['Bid_touch'][0])
            #     #ask_d.append(r['Ask_deep'][0])
            #     #bid_d.append(r['Bid_deep'][0])
            #     #bid_m_D.append(r['Bid_deep'][0] - 0.01*r['Bid_m_D'])
            #     #ask_m_D.append(r['Ask_deep'][0] + 0.01*r['Ask_m_D'])
                spread.append(100*(r['Ask_touch'][0] - r['Bid_touch'][0]))
            #     mid.append(0.5*(r['Ask_touch'][0] + r['Bid_touch'][0]))
            t = np.append([0], np.array(T[1:])[:,1])
            t = t.astype(float) + 9.5*3600
            # volumes = []
            # for r in lob[2*len(lob)//5:3*len(lob)//5]:
            #     volume = [(( - r['Bid_deep'][0] + r['mid']) + 0.01*i, r['Bid_deep'][1]/r['Bid_m_D']) for i in range(int(r['Bid_m_D']))]
            #     volume += [((-r['Bid_touch'][0] + r['mid']), r['Bid_touch'][1])]
            #     volume += [((r['Ask_touch'][0] - r['mid']), r['Ask_touch'][1])]
            #     volume += [((r['Ask_deep'][0] - r['mid']) + 0.01*i, r['Ask_deep'][1]/r['Ask_m_D']) for i in range(int(r['Ask_m_D']))]
            #     volumes += [volume]
            # dict_shape = {}
            # shape0s = []
            # for v in volumes:
            #     dict_shape0 = {}
            #     dists = np.array(v)[:,0]
            #     vols =  np.array(v)[:,1]
            #     for d, vol in zip(dists,vols):
            #         dict_shape[np.round(d, decimals=2)] = dict_shape.get(np.round(d, decimals=2), 0) + vol
            #     #     dict_shape0[np.round(d, decimals=2)] = dict_shape0.get(np.round(d, decimals=2), 0) + vol
            #     # dist0 = np.sort(list(dict_shape0.keys()))
            #     # vol0 = np.array([dict_shape0[d] for d in dist0])
            #     # vol0 = vol0/vol0.sum()
            #     # shape0 = (np.array(dist0)[vol0 > 1e-4], vol0[vol0 > 1e-4])
            #     # shape0s.append(shape0)
            # dist = np.sort(list(dict_shape.keys()))
            # vol = np.array([dict_shape[d] for d in dist])
            # vol = vol/vol.sum()
            # shape = (np.array(dist)[vol > 1e-4], vol[vol > 1e-4])
            # #wass distance
            # wass_distances= []
            # size= int(np.max(shape[0])*100) + 1
            # instt_shape_vols = np.zeros(size)
            # shape_vols = np.zeros(size)
            # shape_vols[[int(100*s) for s in shape[0]]] = shape[1]
            # for shape0 in shape0s:
            #     instt_shape_vols[[int(100*s) for s in shape0[0] if int(100*s) < size]] = shape0[1][[i for i,s in zip(np.arange(len(shape0[0])),shape0[0]) if int(100*s) < size]]
            #     instt_shape_vols = instt_shape_vols/instt_shape_vols.sum()
            #     wass_distances.append(np.abs(shape_vols - instt_shape_vols).sum())
            # sparses.append((np.mean(wass_distances), np.var(wass_distances)))
            # count = int((max(t) - min(t))/10)
            ts.append(np.diff(t))
            spreads.append(spread)
            # mids.append(mid)
            labels.append(fname.split('demo_')[-1])
            # shapes.append(shape)
    # pickle.dump(shapes, open(path+'shapes','wb'))
    # dict_res = {}
    a , b, n, s_bar, eps ,eps2, r_mid , D, shapemax = [] ,[ ], [], [], [], [],[],[], []
    for i in range(len(spreads)):
        t = ts[i]
        s = spreads[i]
        l = labels[i]
    #     mid = mids[i]
    #     dict_res[l.split('smalltickhawkes_')[-1]] = (np.var(s)/np.mean(s)**2, np.var(s)/np.mean(s))
    #     if float(l.split('_')[0]) > 11:
    #         continue
        a += [float(l.split('_')[0])]
        b += [float(l.split('_')[1])]
    #     n += [float(l.split('_')[2])]
        s_bar+= [np.sum(s[:-1]*t)/np.sum(t)]
        eps += [np.exp((0.78 - np.log(s_bar[-1]))/1.3)]
    #     eps2 += [np.exp((-2.1035 - np.log(np.var(s)/np.mean(s)))/1.36)] #[(np.var(s)/(np.mean(s)*0.0089))**(-1/1.36)]
    #     r_mid += [np.mean(np.abs(np.diff(mid)))]
    #     D+=[np.var(s)/np.mean(s)]
    #     shape = shapes[i]
    #     shapemax+=[shape[0][np.argmax(shape[1])]]
    # print(a,b,n,s_bar, eps, eps2, r_mid, D, shapemax, sparses)
    # plt.plot(eps, r_mid)
    # plt.plot(eps, D)
    # plt.plot(eps, shapemax)
    # plt.plot(eps, [s[0] for s in sparses])
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()
    return a,b,eps, s_bar

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

def sim_stylizedfacts_plot_reg():
    x_data = np.log(np.array(eps)[2:])/np.log(10)
    y_data = np.log(np.array(D)[2:])/np.log(10)
    slope, intercept = np.polyfit(x_data, y_data, 1)
    print(f"Regression coefficients: slope = {slope:.4f}, intercept = {intercept:.4f}")
    p = sns.regplot(x=x_data, y=y_data,
                    ci=False, line_kws={'color':'purple', 'alpha' : 0.5, 'label' : f'Trend : Slope = {slope:0.2f}'},
                    label = 'Simulated Small/Medium Tick LOBs') # : $(\\alpha,\\beta,\\hat{\\eta})$

    # Your existing annotations
    # for i in range(len(np.array(eps)[2:])):
    #     xy = (5,5)
    #     plt.annotate(f'({a[2+i]:.3f},{b[i]:.2f},{n[i]:.2f})',
    #                  (np.log(np.array(eps)[2:][i])/np.log(10), np.log(np.array(r_mid)[2:][i])/np.log(10)),
    #                  textcoords="offset points", xytext=xy, ha='center', fontsize=8, color='black')

    # Function to convert log10 values back to original scale in scientific notation
    def log_to_scientific_major(x, pos):
        """Convert log10 values to scientific notation for major ticks"""
        return '$10^{'+str(int(x))+'}$'

    def log_to_scientific_minor(x, pos):
        """Convert log10 values to scientific notation for minor ticks"""
        # Calculate the coefficient (2-9) and the power of 10
        power = int(np.floor(x))
        coeff = 10**(x - power)

        # Round coefficient to nearest integer for cleaner display
        coeff_rounded = int(round(coeff))

        if coeff_rounded == 1:
            return '$10^{'+str(power)+'}$'
        else:
            return f'${coeff_rounded} \\times 10^{{{power}}}$'

    # Apply scientific notation formatting to both axes
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(log_to_scientific_major))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(log_to_scientific_major))

    # Set log-like tick locations (powers of 10)
    # Get the current axis limits
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Set major ticks at integer values (which correspond to powers of 10)
    x_ticks = np.arange(np.floor(xlim[0]), np.ceil(xlim[1]) )
    y_ticks = np.arange(np.floor(ylim[0]), np.ceil(ylim[1]) +1)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Add minor ticks for log-like appearance
    def get_log_minor_ticks(major_ticks):
        """Generate minor tick positions for log scale appearance"""
        minor_ticks = []
        for i in range(len(major_ticks)-1):
            base = major_ticks[i]
            # Add ticks at log10(2), log10(3), ..., log10(9) relative to each major tick
            for j in range(2, 10):
                minor_pos = base + np.log10(j)
                if minor_pos < major_ticks[i+1]:
                    minor_ticks.append(minor_pos)
        return minor_ticks

    x_minor_ticks = get_log_minor_ticks(x_ticks)
    y_minor_ticks = get_log_minor_ticks(y_ticks)

    ax.set_xticks(x_minor_ticks, minor=True)
    # ax.set_yticks(y_minor_ticks, minor=True)

    # Add minor tick labels
    ax.xaxis.set_minor_formatter(ticker.FuncFormatter(log_to_scientific_minor))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(log_to_scientific_minor))

    # Style the minor ticks to look like log scale
    ax.tick_params(which='minor', length=3, color='gray', labelsize=8)
    ax.tick_params(which='major', length=6, labelsize=10)

    # Rotate x-axis minor tick labels
    plt.setp(ax.get_xticklabels(minor=True), rotation=90)

    plt.xlabel('Relative Tick Size : $\epsilon$ (in bps) (log)', fontsize=14)
    # plt.ylabel('Mean Price Changes : $ <r_{mid}> $ (\$) (log)', fontsize=14)
    # plt.title('Simulation Study: $ <r_{mid}> $ (\$) vs $\epsilon$ (in bps)', fontsize=14)
    plt.ylabel('$D := \sigma^2 / \mu$ (log)', fontsize=14)
    plt.title('Simulation Study: $ D $ (\$) vs $\epsilon$ (in bps)')
    plt.xlim(xlim)
    plt.ylim((ylim[0], ylim[1]+1e-3))
    plt.legend()
    plt.tight_layout()
    plt.show()

def sim_wass():
    a,b,n,s_bar, eps, eps2, r_mid, D, shapemax, sparses = ([0.0102, 0.013, 0.044, 0.135, 0.235, 0.275, 0.275, 1.361, 1.41, 1.963, 3.267, 3.741, 3.741],
                                                           [0.98, 0.94, 0.46, 0.59, 0.59, 0.35, 0.35, 0.28, 0.41, 0.41, 0.5, 0.36, 0.46] ,
                                                           [0.99, 0.98, 0.79, 0.92, 0.71, 0.75, 0.77, 0.19, 0.15, 0.08, 0.09, 0.03, 0.03] ,
                                                           [np.float64(1.053517887296352), np.float64(1.0819967838919495), np.float64(1.4712345712000399), np.float64(3.084446355419276), np.float64(4.680750954999857), np.float64(2.9671897928082336), np.float64(2.801742380987427), np.float64(3.713616492014545), np.float64(9.553080318593699), np.float64(10.176013789585152), np.float64(20.171962082372858), np.float64(9.217223139545654), np.float64(6.001276233493695)],
                                                           [np.float64(1.7504907992908174), np.float64(1.7149403548546973), np.float64(1.3539124932770736), np.float64(0.7661018636540287), np.float64(0.5558394344035898), np.float64(0.7892855108732292), np.float64(0.8248997101302495), np.float64(0.6641577804703038), np.float64(0.32108513892629487), np.float64(0.3058559475642696), np.float64(0.1806859033508877), np.float64(0.3300476243644297), np.float64(0.4591216290118335)], [np.float64(0.6864834891318858), np.float64(0.5803470281487966), np.float64(0.23937329097607038), np.float64(0.1527279985431357), np.float64(0.1213003120410737), np.float64(0.09205980253878226), np.float64(0.09865002845300119), np.float64(0.053951749587246696), np.float64(0.029205978806834103), np.float64(0.02286663167372488), np.float64(0.016517571776810996), np.float64(0.012944888439698318), np.float64(0.011560497560733383)], [np.float64(0.0005176079819844223), np.float64(0.00045941802080398675), np.float64(0.0005568467266692645), np.float64(0.0005710065023752858), np.float64(0.0007131201302919109), np.float64(0.0006249307531835857), np.float64(0.0005882493264926622), np.float64(0.000959602646814655), np.float64(0.0019276174326801744), np.float64(0.0024863073169604523), np.float64(0.0025274976411937015), np.float64(0.002960719437523365), np.float64(0.0011511907505701066)] ,[np.float64(0.20353758924996754), np.float64(0.255767993901753), np.float64(0.8529402675115613), np.float64(1.571564388036908), np.float64(2.1498558882321483), np.float64(3.1284290083918336), np.float64(2.847667935304688), np.float64(6.470443251487576), np.float64(14.90801962959074), np.float64(20.794408448653297), np.float64(32.36344326980865), np.float64(45.08248563765569), np.float64(52.579140366336475)] ,[np.float64(0.02), np.float64(0.02), np.float64(0.03), np.float64(0.06), np.float64(0.07), np.float64(0.05), np.float64(0.07), np.float64(0.01), np.float64(0.01), np.float64(0.01), np.float64(0.02), np.float64(0.01), np.float64(0.01)] ,[(np.float64(0.23705199049692394), np.float64(0.005531762639857034)), (np.float64(0.3068402517668672), np.float64(0.004502921226459963)), (np.float64(0.32003040946161504), np.float64(0.006801621718988904)), (np.float64(0.3007781673347805), np.float64(0.010614024731556575)), (np.float64(0.3580915445765435), np.float64(0.005608932572486014)), (np.float64(0.38148804422769367), np.float64(0.008864221511888292)), (np.float64(0.32719119267362734), np.float64(0.005750086582192829)), (np.float64(0.5209985745646536), np.float64(0.04847491728282235)), (np.float64(0.44488128015460765), np.float64(0.02164075977110758)), (np.float64(0.49594896107137626), np.float64(0.023775712087363284)), (np.float64(0.5333305581446189), np.float64(0.03447839142965841)), (np.float64(0.720229260548198), np.float64(0.07267812825328017)), (np.float64(0.8695285979946084), np.float64(0.08014820272192642))])

    # eps = [np.float64(1.7504907992908174), np.float64(1.7149403548546973), np.float64(1.3539124932770736), np.float64(0.7661018636540287), np.float64(0.5558394344035898), np.float64(0.7892855108732292), np.float64(0.8248997101302495), np.float64(0.6641577804703038), np.float64(0.32108513892629487), np.float64(0.3058559475642696), np.float64(0.1806859033508877), np.float64(0.3300476243644297), np.float64(0.4591216290118335)]
    plt.errorbar(eps[2:-2], [s[0] for s in sparses[2:-2]], yerr=[np.sqrt(s[1])/2 for s in sparses[2:-2]], marker='o', markersize=8, linestyle='none', capsize=5, color = 'lightcoral', label = 'Simulated Small/Medium Tick LOBs')

    # for i in range(len(stocks)):
    #     xy = (5,5)
    #     if stocks[i] in ['AMZN', 'PM', 'IBM', 'BAC', 'MSFT', 'ORCL']:
    #         xy = (-10, -10)
    #     if stocks[i] in ['INTC']:
    #         continue
    #     plt.annotate(f'{stocks[i]:<4}', (Xs[i], Ys[i]), textcoords="offset points", xytext=xy, ha='center', fontsize=10, color='black')

    #plt.xticks(np.arange(len(Xs)), Xs, rotation=45)
    plt.xscale("log")
    # plt.xscale("log")
    # plt.legend(loc = "lower right")
    plt.ylabel("Average Wasserstein Distance", fontsize=14)
    plt.xlabel('Relative Tick Size : $\epsilon$ (in bps) (log)', fontsize=14)
    plt.title("Simulation Study: Sparsity vs $\epsilon$ (in bps)", fontsize=14)
    # plt.savefig("/SAN/fca/Konark_PhD_Experiments/smallTick/xStock_wassShape_2019.png")
    plt.show()

import matplotlib.cm as cm
a,b,eps,s_bar = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 10.0] , [0.1, 0.25, 0.5, 0.7, 0.9, 0.0, 0.1, 0.25, 0.5, 0.7, 0.9, 0.0, 0.1, 0.25, 0.5, 0.7, 0.9, 0.0, 0.25, 0.9, 0.1, 0.25, 0.5, 0.7, 0.9, 0.0, 0.1, 0.25, 0.5, 0.7, 0.9, 0.0, 0.1, 0.25, 0.5, 0.7, 0.9, 0.0, 0.1, 0.25, 0.5, 0.7, 0.9, 0.7] , [np.float64(1.521971200523886), np.float64(1.5997506067022487), np.float64(1.672371081852595), np.float64(1.7215176234080127), np.float64(1.7546031579962695), np.float64(1.3515479030912225), np.float64(1.4043414173558135), np.float64(1.4840028562387773), np.float64(1.5532704849426477), np.float64(1.583401047844062), np.float64(1.584712263545891), np.float64(1.377683316972155), np.float64(1.3967985609502405), np.float64(1.3686943972838856), np.float64(1.2791282299928723), np.float64(1.241462237705871), np.float64(1.1596029792468017), np.float64(1.3464898226741981), np.float64(1.2180280626916717), np.float64(0.7797025924591643), np.float64(1.2270509599869326), np.float64(1.0525485930410894), np.float64(0.8078439435568344), np.float64(0.5693139181493326), np.float64(0.47832183601184475), np.float64(1.3939730572935158), np.float64(1.1938422235090902), np.float64(0.8847098668082704), np.float64(0.49717717763595937), np.float64(0.33329841765320284), np.float64(0.27548721138630206), np.float64(1.3443406288557755), np.float64(1.1275226457427605), np.float64(0.7018465013543564), np.float64(0.31644037603054), np.float64(0.19896056327730022), np.float64(0.16087116480886185), np.float64(1.3411832914914803), np.float64(1.1500493101878084), np.float64(0.5564554703274981), np.float64(0.18690764569046628), np.float64(0.12028986859808626), np.float64(0.0969957923309993), np.float64(0.027913121049634476)] ,[np.float64(1.2636340820004406), np.float64(1.1843545348165723), np.float64(1.1179367986067619), np.float64(1.076625795189096), np.float64(1.0503090732027203), np.float64(1.4745816321772252), np.float64(1.4029273445021702), np.float64(1.3058236743259104), np.float64(1.2306327797664933), np.float64(1.200277014799721), np.float64(1.1989861087110263), np.float64(1.438319858617688), np.float64(1.4127840503137736), np.float64(1.4506119791820875), np.float64(1.584022648754886), np.float64(1.6467820624951794), np.float64(1.7994822894434015), np.float64(1.4817867196472216), np.float64(1.6880885077515146), np.float64(3.0146855840559486), np.float64(1.6719693678612972), np.float64(2.040961884890777), np.float64(2.8788827698818036), np.float64(4.537246290747047), np.float64(5.69001519577186), np.float64(1.4165079038434125), np.float64(1.7326812692741038), np.float64(2.5580472600836286), np.float64(5.411094964801468), np.float64(9.100525449447211), np.float64(11.657832741877915), np.float64(1.4848670649536184), np.float64(1.8663231395599091), np.float64(3.456486895995809), np.float64(9.735768451302249), np.float64(17.797237398074657), np.float64(23.46000106914796), np.float64(1.4894129351338743), np.float64(1.8189396731739886), np.float64(4.6740155709292655), np.float64(19.303431048934463), np.float64(34.23354942481923), np.float64(45.2867268160343), np.float64(228.66451647544787)]#extract_stylized_facts()
print(a,b,eps,s_bar)
shapes = []
for alpha, beta, e, s in zip(a,b,eps,s_bar):
    try:
        shape = pickle.load(open(f'D:\\PhD\\results - small tick\\sim\\smalltick_is\\shape_{alpha:.2f}_{beta:0.1f}','rb'))
        shapes.append((alpha, beta, e, s, shape[0], shape[1]))
        print(alpha, beta, ' found')
    except:
        print(alpha, beta, ' not found')

colors = cm.viridis(np.linspace(0, 1, len(shapes)))
# a,b,n,s_bar, eps, eps2, r_mid, D, shapemax, sparses = ([0.0102, 0.013, 0.044, 0.135, 0.235, 0.275, 0.275, 1.361, 1.41, 1.963, 3.267, 3.741, 3.741],
#                                                        [0.98, 0.94, 0.46, 0.59, 0.59, 0.35, 0.35, 0.28, 0.41, 0.41, 0.5, 0.36, 0.46] ,
#                                                        [0.99, 0.98, 0.79, 0.92, 0.71, 0.75, 0.77, 0.19, 0.15, 0.08, 0.09, 0.03, 0.03] ,
#                                                        [np.float64(1.053517887296352), np.float64(1.0819967838919495), np.float64(1.4712345712000399), np.float64(3.084446355419276), np.float64(4.680750954999857), np.float64(2.9671897928082336), np.float64(2.801742380987427), np.float64(3.713616492014545), np.float64(9.553080318593699), np.float64(10.176013789585152), np.float64(20.171962082372858), np.float64(9.217223139545654), np.float64(6.001276233493695)],
#                                                        [np.float64(1.7504907992908174), np.float64(1.7149403548546973), np.float64(1.3539124932770736), np.float64(0.7661018636540287), np.float64(0.5558394344035898), np.float64(0.7892855108732292), np.float64(0.8248997101302495), np.float64(0.6641577804703038), np.float64(0.32108513892629487), np.float64(0.3058559475642696), np.float64(0.1806859033508877), np.float64(0.3300476243644297), np.float64(0.4591216290118335)], [np.float64(0.6864834891318858), np.float64(0.5803470281487966), np.float64(0.23937329097607038), np.float64(0.1527279985431357), np.float64(0.1213003120410737), np.float64(0.09205980253878226), np.float64(0.09865002845300119), np.float64(0.053951749587246696), np.float64(0.029205978806834103), np.float64(0.02286663167372488), np.float64(0.016517571776810996), np.float64(0.012944888439698318), np.float64(0.011560497560733383)], [np.float64(0.0005176079819844223), np.float64(0.00045941802080398675), np.float64(0.0005568467266692645), np.float64(0.0005710065023752858), np.float64(0.0007131201302919109), np.float64(0.0006249307531835857), np.float64(0.0005882493264926622), np.float64(0.000959602646814655), np.float64(0.0019276174326801744), np.float64(0.0024863073169604523), np.float64(0.0025274976411937015), np.float64(0.002960719437523365), np.float64(0.0011511907505701066)] ,[np.float64(0.20353758924996754), np.float64(0.255767993901753), np.float64(0.8529402675115613), np.float64(1.571564388036908), np.float64(2.1498558882321483), np.float64(3.1284290083918336), np.float64(2.847667935304688), np.float64(6.470443251487576), np.float64(14.90801962959074), np.float64(20.794408448653297), np.float64(32.36344326980865), np.float64(45.08248563765569), np.float64(52.579140366336475)] ,[np.float64(0.02), np.float64(0.02), np.float64(0.03), np.float64(0.06), np.float64(0.07), np.float64(0.05), np.float64(0.07), np.float64(0.01), np.float64(0.01), np.float64(0.01), np.float64(0.02), np.float64(0.01), np.float64(0.01)] ,[(np.float64(0.23705199049692394), np.float64(0.005531762639857034)), (np.float64(0.3068402517668672), np.float64(0.004502921226459963)), (np.float64(0.32003040946161504), np.float64(0.006801621718988904)), (np.float64(0.3007781673347805), np.float64(0.010614024731556575)), (np.float64(0.3580915445765435), np.float64(0.005608932572486014)), (np.float64(0.38148804422769367), np.float64(0.008864221511888292)), (np.float64(0.32719119267362734), np.float64(0.005750086582192829)), (np.float64(0.5209985745646536), np.float64(0.04847491728282235)), (np.float64(0.44488128015460765), np.float64(0.02164075977110758)), (np.float64(0.49594896107137626), np.float64(0.023775712087363284)), (np.float64(0.5333305581446189), np.float64(0.03447839142965841)), (np.float64(0.720229260548198), np.float64(0.07267812825328017)), (np.float64(0.8695285979946084), np.float64(0.08014820272192642))])
for i, s in enumerate(shapes):
    print(s[0], s[1], s[3])
    d = s[4]
    v = s[5]
    plt.plot(d[v>1e-4],v[v>1e-4], label = f'$\epsilon$ : {s[2]:.2f}',color=colors[i])
plt.yscale("log")
plt.xscale("log")
plt.legend(loc = "upper right", ncol=3)
plt.ylabel("Ratio of total volume : $\overline{q}(x)$", fontsize=16)
plt.xlabel("Distance from mid : $x$ (ticks)", fontsize=16)
plt.title("Simulation Study: Shape of the LOB", fontsize=16)
plt.show()