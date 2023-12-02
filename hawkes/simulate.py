from tick.plot import plot_point_process
from tick.hawkes import SimuHawkes, HawkesKernelPowerLaw
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd


def plot_hawkes_kernel_norms(n_nodes, kernel_norms, show=True, pcolor_kwargs=None,
                             node_names=None, rotate_x_labels=0.):
    """Generic function to plot Hawkes kernel norms.

    Parameters
    ----------
    kernel_object : `Object`
        An object that must have the following API :

        * `kernel_object.n_nodes` : a field that stores the number of nodes
          of the associated Hawkes process (thus the number of kernels is
          this number squared)
        * `kernel_object.get_kernel_norms()` : must return a 2d numpy
          array with the norm of each kernel

    show : `bool`, default=`True`
        if `True`, show the plot. Otherwise an explicit call to the show
        function is necessary. Useful when superposing several plots.

    pcolor_kwargs : `dict`, default=`None`
        Extra pcolor kwargs such as cmap, vmin, vmax

    node_names : `list` of `str`, shape=(n_nodes, ), default=`None`
        node names that will be displayed on axis.
        If `None`, node index will be used.

    rotate_x_labels : `float`, default=`0.`
        Number of degrees to rotate the x-labels clockwise, to prevent
        overlapping.

    Notes
    -----
    Kernels are displayed such that it shows norm of column influence's
    on row.
    """

    if node_names is None:
        node_names = range(n_nodes)
    elif len(node_names) != n_nodes:
        ValueError('node_names must be a list of length {} but has length {}'
                   .format(n_nodes, len(node_names)))

    row_labels = ['${} \\rightarrow$'.format(i) for i in node_names]
    column_labels = ['$\\rightarrow {}$'.format(i) for i in node_names]

    norms = kernel_norms
    fig, ax = plt.subplots()

    if rotate_x_labels != 0.:
        # we want clockwise rotation because x-axis is on top
        rotate_x_labels = -rotate_x_labels
        x_label_alignment = 'right'
    else:
        x_label_alignment = 'center'

    if pcolor_kwargs is None:
        pcolor_kwargs = {}

    if norms.min() >= 0:
        pcolor_kwargs.setdefault("cmap", plt.cm.Blues)
    else:
        # In this case we want a diverging colormap centered on 0
        pcolor_kwargs.setdefault("cmap", plt.cm.RdBu)
        max_abs_norm = np.max(np.abs(norms))
        pcolor_kwargs.setdefault("vmin", -max_abs_norm)
        pcolor_kwargs.setdefault("vmax", max_abs_norm)

    heatmap = ax.pcolor(norms, **pcolor_kwargs)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(norms.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(norms.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False, fontsize=17,
                       rotation=rotate_x_labels, ha=x_label_alignment)
    ax.set_yticklabels(column_labels, minor=False, fontsize=17)

    fig.subplots_adjust(right=0.8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    fig.colorbar(heatmap, cax=cax)

    if show:
        plt.show()

    return fig

def simulate(T , paramsPath , Pis , Pi_Q0):
    """
    :param T: time limit of simulations
    :param paramsPath: path of fitted params
    :param Pis: distribution of order sizes
    :param Pi_Q0: depleted queue size distribution
    """
    hawkes = SimuHawkes(n_nodes=12, end_time=T, verbose=True)
    with open(paramsPath, "rb") as f:
        params = pickle.load(f)
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
                   "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    norms = {}
    for i in range(12):
        for j in range(12):
            kernelParams = params[cols[i] + "->" + cols[j]]
            kernel = HawkesKernelPowerLaw(kernelParams[0]*np.exp(kernelParams[1][0]), 8.1*1e-4, -1*kernelParams[1][1], support = 1000)
            print(cols[i] + "->" + cols[j])
            print(kernel.get_norm())
            norms[cols[i] + "->" + cols[j] ] = kernel.get_norm()
            #print(kernelParams[0]*np.exp(kernelParams[1][0]))
            #print(-1*kernelParams[1][1])
            if abs(kernel.get_norm()) >= 0.1 :
                hawkes.set_kernel(j,i, kernel)
        hawkes.set_baseline(i, params[cols[i]])
    fig = plot_hawkes_kernel_norms(12, np.array(list(norms.values())).reshape((12,12)).T,
                             node_names=["LO_{ask_{+1}}", "CO_{ask_{+1}}",
                                         "LO_{ask_{0}}", "CO_{ask_{0}}", "MO_{ask_{0}}",
                                         "LO_{ask_{-1}}",
                                         "LO_{bid_{+1}}",
                                         "LO_{bid_{0}}", "CO_{bid_{0}}", "MO_{bid_{0}}",
                                         "LO_{bid_{-1}}", "CO_{bid_{-1}}"])
    fig.savefig(paramsPath + "_kernels.png")
    dt = 1e-4
    hawkes.track_intensity(dt)
    hawkes.simulate()
    timestamps = hawkes.timestamps

    fig, ax = plt.subplots(12, 2, figsize=(16, 50))
    plot_point_process(hawkes, n_points=50000, t_min=2, max_jumps=10, ax=ax[:,0])
    plot_point_process(hawkes, n_points=50000, t_min=2, t_max=20, ax=ax[:, 1])
    fig.savefig(paramsPath+".png")

    sizes = {}
    dictTimestamps = {}
    for t, col in zip(timestamps, cols):
        if "co" in col: # handle size of cancel order in createLOB
            size = 0
        else:
            pi = Pis[col]
            cdf = np.cumsum(pi)
            a = np.random.uniform(0, 1, size = len(t))
            size = np.argmax(cdf>=a)-1
        sizes[col]  = size
        dictTimestamps[col] = timestamps

    lob = createLOB(dictTimestamps, sizes, Pi_Q0)
    return lob

def createLOB(dictTimestamps, sizes, Pi_Q0, priceMid0 = 100, spread0 = 10, ticksize = 0.01, numOrdersPerLevel = 10):
    lob = []
    lob_l3 = []
    T = []
    lob0 = {}
    lob0_l3 = {}
    levels = ["Ask_deep", "Ask_touch", "Bid_touch", "Bid_deep"]
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    lob0['Ask_touch'] = (priceMid0 + np.floor(spread0/2)*ticksize, 0)
    lob0['Bid_touch'] = (priceMid0 - np.ceil(spread0/2)*ticksize, 0)
    lob0['Ask_deep'] = (priceMid0 + np.floor(spread0/2)*ticksize + ticksize, 0)
    lob0['Bid_deep'] = (priceMid0 - np.ceil(spread0/2)*ticksize - ticksize, 0)
    for k, Pi in Pi_Q0.iteritems():
        cdf = np.cumsum(Pi)
        a = np.random.uniform(0, 1)
        qSize = np.argmax(cdf>=a)-1
        lob0[k] = (lob0[k][0], qSize)
    for l in levels:
        tmp = (numOrdersPerLevel - 1)*[np.floor(lob0[l][1]/numOrdersPerLevel)]
        lob0_l3[l] = [lob0[l][1] - sum(tmp)] + tmp

    dfs = []
    for event in dictTimestamps.keys():
        sizes_e = sizes[event]
        timestamps_e = dictTimestamps[event]
        dfs += [pd.DataFrame({"event" : len(timestamps_e)*[event], "time": timestamps_e, "size" : sizes_e})]
    dfs = pd.concat(dfs)
    dfs = dfs.sort_values("time")
    lob.append(lob0)
    T.append(0)
    lob_l3.append(lob0_l3)
    for i in range(len(dfs)):
        r = dfs.iloc[i]
        lobNew = lob[i]
        lob_l3New = lob_l3[i]
        T.append(r.time)
        if "Ask" in r.event :
            side = "Ask"
        else:
            side = "Bid"

        if "lo" in r.event:
            if "deep" in r.event:
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] + r.size)
                lob_l3New[side + "_deep"] += [r.size]
            elif "top" in r.event:
                lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] + r.size)
                lob_l3New[side + "_touch"] += [r.size]
            else: #inspread
                direction = 1
                if side == "Ask": direction = -1
                lobNew[side + "_deep"] = lobNew[side + "_touch"]
                lob_l3New[side + "_deep"] = lob_l3New[side + "_touch"]
                lobNew[side + "_touch"] = (lobNew[side + "_touch"][0] + direction*ticksize, r.size)
                lob_l3New[side + "_touch"] = [r.size]

        if "mo" in r.event:
            lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] - r.size)
            if lobNew[side + "_touch"][1] > 0:
                cumsum = np.cumsum(lob_l3New[side + "_touch"])
                idx = np.argmax(cumsum >= r.size)
                tmp = lob_l3New[side + "_touch"][idx:]
                tmp[0] = tmp[0] - cumsum[idx] + r.size
                lob_l3New[side + "_touch"] = tmp
            while lobNew[side + "_touch"][1] <= 0: # queue depletion
                extraVolume = -1*lobNew[side + "_touch"][1]
                lobNew[side + "_touch"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] - extraVolume)
                lob_l3New[side + "_touch"] = lob_l3New[side + "_deep"]
                if lobNew[side + "_touch"][1] > 0:
                    cumsum = np.cumsum(lob_l3New[side + "_touch"])
                    idx = np.argmax(cumsum >= extraVolume)
                    tmp = lob_l3New[side + "_touch"][idx:]
                    tmp[0] = tmp[0] - cumsum[idx] + extraVolume
                    lob_l3New[side + "_touch"] = tmp
                direction = 1
                if side == "Bid": direction = -1
                cdf = np.cumsum(Pi_Q0[side+"_deep"])
                a = np.random.uniform(0, 1)
                qSize = np.argmax(cdf>=a)-1
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0] + direction*ticksize, qSize)
                tmp = (numOrdersPerLevel - 1)*[np.floor(lobNew[side + "_deep"][1]/numOrdersPerLevel)]
                lob_l3New[side + "_deep"] = [lobNew[side + "_deep"][1] - sum(tmp)] + tmp

        if "co" in r.event:

            if "deep" in r.event:
                size = np.random.choice(lob_l3New[side + "_deep"])
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] - size)
                lob_l3New[side + "_deep"].remove(size)
            elif "top" in r.event:
                size = np.random.choice(lob_l3New[side + "_touch"])
                lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] - size)
                lob_l3New[side + "_touch"].remove(size)
            if lobNew[side + "_touch"][1] <= 0: # queue depletion
                lobNew[side + "_touch"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1])
                lob_l3New[side + "_touch"] = lob_l3New[side + "_deep"]
                direction = 1
                if side == "Bid": direction = -1
                cdf = np.cumsum(Pi_Q0[side+"_deep"])
                a = np.random.uniform(0, 1)
                qSize = np.argmax(cdf>=a)-1
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0] + direction*ticksize, qSize)
                tmp = (numOrdersPerLevel - 1)*[np.floor(lobNew[side + "_deep"][1]/numOrdersPerLevel)]
                lob_l3New[side + "_deep"] = [lobNew[side + "_deep"][1] - sum(tmp)] + tmp

            if lobNew[side + "_deep"][1] <= 0: # queue depletion
                direction = 1
                if side == "Bid": direction = -1
                cdf = np.cumsum(Pi_Q0[side+"_deep"])
                a = np.random.uniform(0, 1)
                qSize = np.argmax(cdf>=a)-1
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0] + direction*ticksize, qSize)
                tmp = (numOrdersPerLevel - 1)*[np.floor(lobNew[side + "_deep"][1]/numOrdersPerLevel)]
                lob_l3New[side + "_deep"] = [lobNew[side + "_deep"][1] - sum(tmp)] + tmp

        lob.append(lobNew)

    return lob

