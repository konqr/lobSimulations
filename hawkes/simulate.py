# from tick.plot import plot_point_process
# from tick.hawkes import SimuHawkes, HawkesKernelPowerLaw
# import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import time
import numpy as np

def powerLawKernel(x, alpha = 1., t0 = 1., beta = -2.):
    if x < t0: return 0
    return alpha*(x**beta)

def powerLawCutoff(time, alpha, beta, gamma):
    # alpha = a*beta*(gamma - 1)
    funcEval = alpha/((1 + gamma*time)**beta)
    # funcEval[time < t0] = 0
    return funcEval

def powerLawKernelIntegral(x1, x2, alpha = 1., t0 = 1., beta = -2.):
    return (x2/(1+beta))*powerLawKernel(x2, alpha = alpha, t0=t0, beta=beta) - (x1/(1+beta))*powerLawKernel(x1, alpha = alpha, t0=t0, beta=beta)

def thinningOgata(T, paramsPath, num_nodes = 12, maxJumps = None):
    if maxJumps is None: maxJumps = np.inf
    with open(paramsPath, "rb") as f:
        params = pickle.load(f)
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    baselines = num_nodes*[0]
    mat = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = params[cols[i] + "->" + cols[j]]
            # print(cols[i] + "->" + cols[j])
            # print((kernelParams[0]*np.exp(kernelParams[1][0]) , kernelParams[1][1] , kernelParams[1][2]))
            mat[i][j]  = kernelParams[0]*np.exp(kernelParams[1][0])/((-1 - kernelParams[1][1])*(kernelParams[1][2])**(-1 - kernelParams[1][1]))
        baselines[i] = params[cols[i]]
    # print("spectral radius = ", np.max(np.linalg.eig(mat)[0]))
    s = 0
    numJumps = 0
    n = num_nodes*[0]
    Ts = num_nodes*[()]
    if type(baselines[0]) == float:
        lamb = sum(baselines)
    else:
        lamb = sum(np.array(baselines)[:,0])
    while s <= T:
        lambBar = lamb
        u = np.random.uniform(0,1)
        w = -1*np.log(u)/lambBar
        s += np.max([1e-6,w])
        if type(baselines[0]) == float:
            decays = baselines.copy()
        else:
            hourIndex = np.min([12,int(np.floor(s/1800))])
            decays = np.array(baselines)[:,hourIndex]
        for i in range(len(Ts)):
            taus = Ts[i]
            for tau in taus:
                if s - tau >= 500: continue
                if s - tau < 1e-4: continue
                for j in range(len(Ts)):
                    kernelParams = params[cols[i] + "->" + cols[j]]
                    decay = powerLawKernel(s - tau, alpha = kernelParams[0]*np.exp(kernelParams[1][0]), t0 = kernelParams[1][2], beta = kernelParams[1][1])
                    decays[j] += decay
        decays = [np.max([0, d]) for d in decays]
        lamb = sum(decays)
        D = np.random.uniform(0,1)
        if D*lambBar <= lamb: #accepted
            k = 0
            while D*lambBar >= sum(decays[:k+1]):
                k+=1
            n[k] += 1
            Ts[k] += (s,)
            numJumps += 1
            if numJumps >= maxJumps:
                return n,Ts
    return n, Ts

def thinningOgataIS(T, paramsPath, todPath, num_nodes = 12, maxJumps = None, s = None, n = None, Ts = None, spread=None, beta = 0.7479, avgSpread = 0.0169,lamb= None):
    if maxJumps is None: maxJumps = np.inf
    tryer = 0
    while tryer < 5: # retry on pickle clashes
        try:
            with open(paramsPath, "rb") as f:
                params = pickle.load(f)
            with open(todPath, "rb") as f:
                tod = pickle.load(f)
        except:
            time.sleep(1)
            continue
        tryer +=1
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    baselines = num_nodes*[0]
    mat = np.zeros((num_nodes, num_nodes))
    if s is None: s = 0
    hourIndex = np.min([12,int(np.floor(s/1800))])
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = params.get(cols[i] + "->" + cols[j], None)
            if kernelParams is None: continue
            if np.isnan(kernelParams[1][2]): continue
            # print(cols[i] + "->" + cols[j])
            # print((kernelParams[0]*np.exp(kernelParams[1][0]) , kernelParams[1][1] , kernelParams[1][2]))
            todMult = tod[cols[j]][hourIndex]
            mat[i][j]  = todMult*kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2]) # alpha/(beta -1)*gamma
        baselines[i] = params[cols[i]]
    baselines[5] = ((spread/avgSpread)**beta)*baselines[5]
    baselines[6] = ((spread/avgSpread)**beta)*baselines[6]
    specRad = np.max(np.linalg.eig(mat)[0])
    print("spectral radius = ", specRad)
    specRad = np.max(np.linalg.eig(mat)[0]).real
    if specRad < 1 : specRad = 0.99 #  # dont change actual specRad if already good
    numJumps = 0
    if n is None: n = num_nodes*[0]
    if Ts is None: Ts = num_nodes*[()]
    Ts_new = num_nodes*[()]
    if spread is None: spread = 1
    if lamb is None:
        print(baselines)

        decays = baselines.copy()
        for i in range(len(Ts)):
            todMult = tod[cols[i]][hourIndex]*0.99/specRad
            decays[i] = todMult*decays[i]
        if (type(baselines[0]) == np.float64)or(type(baselines[0]) == float):
            lamb = sum(decays)
        else:
            lamb = sum(np.array(baselines)[:,0])
    while s <= T:
        lambBar = lamb
        print(lambBar)
        u = np.random.uniform(0,1)
        if lambBar == 0:
            s += 0.1 # wait for some time
        else:
            w = np.max([1e-7,-1*np.log(u)/lambBar]) # floor at 0.1 microsec
            s += w

        decays = baselines.copy()
        hourIndex = np.min([12,int(np.floor(s/1800))])

        for i in range(len(Ts)):
            todMult = tod[cols[i]][hourIndex]*0.99/specRad
            decays[i] = todMult*decays[i]
        for i in range(len(Ts)):
            taus = Ts[i]
            idx = np.searchsorted(taus, s - 10)
            for tau in taus[idx:]:
                if s - tau >= 10: continue
                #if s - tau < 1e-4: continue
                for j in range(len(Ts)):
                    kernelParams = params.get(cols[i] + "->" + cols[j], None)
                    todMult = tod[cols[j]][hourIndex]*0.99/specRad
                    if kernelParams is None: continue
                    if np.isnan(kernelParams[1][2]): continue
                    # decay = todMult*powerLawKernel(s - tau, alpha = kernelParams[0]*np.exp(kernelParams[1][0]), t0 = kernelParams[1][2], beta = kernelParams[1][1])
                    # print(decay)
                    decay = todMult*powerLawCutoff(s - tau, kernelParams[0]*kernelParams[1][0], kernelParams[1][1], kernelParams[1][2])
                    decays[j] += decay
        decays = [np.max([0, d]) for d in decays]
        decays[5] = ((spread/avgSpread)**beta)*decays[5]
        decays[6] = ((spread/avgSpread)**beta)*decays[6]
        if 100*np.round(spread, 2) < 2 : decays[5] = decays[6] = 0
        print(decays)
        lamb = sum(decays)
        print(lamb)
        D = np.random.uniform(0,1)
        if D*lambBar <= lamb: #accepted
            print(w)
            k = 0
            while D*lambBar >= sum(decays[:k+1]):
                k+=1
            # instantaneous lamb jumps
            if k in [5,6]:
                spread = spread - 0.01
            newdecays = len(cols)*[0]
            for i in range(len(Ts)):
                kernelParams = params.get(cols[k] + "->" + cols[i], None)
                todMult = tod[cols[i]][hourIndex]*0.99/specRad
                if kernelParams is None: continue
                if np.isnan(kernelParams[1][2]): continue
                # decay = todMult*powerLawKernel(0, alpha = kernelParams[0]*np.exp(kernelParams[1][0]), t0 = kernelParams[1][2], beta = kernelParams[1][1])
                decay = todMult*powerLawCutoff(0, kernelParams[0]*kernelParams[1][0], kernelParams[1][1], kernelParams[1][2])
                # print(decay)
                newdecays[i] += decay
            newdecays = [np.max([0, d]) for d in newdecays]
            newdecays[5] = ((spread/avgSpread)**beta)*newdecays[5]
            newdecays[6] = ((spread/avgSpread)**beta)*newdecays[6]
            if 100*np.round(spread, 2) < 2 : newdecays[5] = newdecays[6] = 0
            lamb += sum(newdecays)
            print(lamb)
            n[k] += 1
            if len(Ts[k]) > 0:
                T_Minus1 = Ts[k][-1]
            else:
                T_Minus1 = 0
            decays = np.array(baselines.copy())
            hourIndex = np.min([12,int(np.floor(s/1800))])
            decays[5] = ((spread/avgSpread)**beta)*decays[5]
            decays[6] = ((spread/avgSpread)**beta)*decays[6]
            decays = decays*(s-T_Minus1)
            tau = decays[k]
            Ts[k] += (s,)
            Ts_new[k] += (s,)
            numJumps += 1
            if numJumps >= maxJumps:
                return s,n,Ts, Ts_new, tau, lamb
    return s,n, Ts, Ts_new, -1, lamb

def simulate(T , paramsPath , todPath, s0 = None, filePathName = None, Pis = None, Pi_Q0 = None, beta = 0.7479, avgSpread = 0.0169, spread0 = 3, price0 = 260):
    """
    :param T: time limit of simulations
    :param paramsPath: path of fitted params
    :param Pis: distribution of order sizes
    :param Pi_Q0: depleted queue size distribution
    """
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    if Pis == None:
        ## AAPL
        Pis = {
            'mo_Bid' : [8.16e-3, [(1, 0.072), (10, 0.04), (50, 0.028), (100, 0.427), (200, 0.051), (500, 0.07)]],
            'lo_top_Bid' : [2.09e-3, [(1, 0.02), (10, 0.069), (50, 0.005), (100, 0.6), (200, 0.036), (500, 0.054)]],
            'lo_deep_Bid' : [2.33e-3, [(1, 0.021), (10, 0.112), (50, 0.015), (100, 0.276), (200, 0.097), (500, 0.172)]]
        }
        Pis["lo_inspread_Bid"] = Pis["lo_top_Bid"]
        if "AMZN.OQ" in paramsPath:
            Pis = {'lo_top_Bid': [0.05365508940934027,
                                  [(1, 0.0432323548960116),
                                   (10, 0.013435798495166417),
                                   (50, 0.0005709704503910774),
                                   (100, 6.663879261160762e-05),
                                   (200, 1.5747668929585514e-05),
                                   (500, 6.09046480638754e-07)]],
                   'lo_deep_Bid': [0.06060130535300165,
                                   [(1, 0.05639948122151968),
                                    (10, 0.05518191451020626),
                                    (50, 0.0003330187521711186),
                                    (100, 4.9983396108122606e-05),
                                    (200, 7.903710504897672e-06),
                                    (500, 1.6657318546374149e-06)]],
                   'lo_inspread_Bid': [0.040813664165380875,
                                       [(1, 0.020322265459336963),
                                        (10, 0.005277804876103806),
                                        (50, 0.0005669113835675268),
                                        (100, 5.090044812498355e-05),
                                        (200, 6.6587761856594516e-06),
                                        (500, 6.606025886842982e-07)]],
                   'mo_Bid': [0.030074069098223778,
                              [(1, 0.05510488974651141),
                               (10, 0.008831314046535154),
                               (50, 0.0019306968599348715),
                               (100, 0.0001651778474078509),
                               (200, 3.460981489505782e-05),
                               (500, 3.7078651677298155e-06)]]}
        elif "TSLA.OQ" in paramsPath:
            Pis = {'lo_top_Bid': [0.01906341526374109,
                                  [(1, 0.013336873454420098),
                                   (10, 0.002499960585631139),
                                   (50, 0.00258837727124698),
                                   (100, 9.016197290658499e-05),
                                   (200, 2.1484763827487066e-05),
                                   (500, 1.4176290754765072e-06)]],
                   'lo_deep_Bid': [0.01834598886408305,
                                   [(1, 0.006567760273974217),
                                    (10, 0.008358970313982568),
                                    (50, 0.0026727677268739785),
                                    (100, 5.1652220786639296e-05),
                                    (200, 1.4287611741998702e-05),
                                    (500, 1.7712494515096627e-06)]],
                   'lo_inspread_Bid': [0.013924826336477625,
                                       [(1, 0.007544542317155571),
                                        (10, 0.005196331116912839),
                                        (50, 0.0010228895870503783),
                                        (100, 8.177891598970127e-05),
                                        (200, 2.1061506356115122e-05),
                                        (500, 3.808005108713909e-06)]],
                   'mo_Bid': [0.015972671818734865,
                              [(1, 0.021494545426040023),
                               (10, 0.006376156312818793),
                               (50, 0.0023541898985074998),
                               (100, 0.00031172774984016764),
                               (200, 8.539722286633611e-05),
                               (500, 1.0538996186294693e-05)]]}
        elif "INTC.OQ" in paramsPath:
            Pis = {'lo_top_Bid': [0.0012032594626957276,
                                  [(1, 0.0004058911201508531),
                                   (10, 0.004937100775363866),
                                   (50, 0.011839578897579354),
                                   (100, 4.2540607105295814e-05),
                                   (200, 1.3215923263800984e-05),
                                   (500, 4.946493652744191e-06)]],
                   'lo_deep_Bid': [0.001446173610413779,
                                   [(1, 0.0009504555025436379),
                                    (10, 0.005970974450434397),
                                    (50, 0.011838812712121737),
                                    (100, 7.747866832159131e-05),
                                    (200, 4.951687702705543e-06),
                                    (500, 1.4416468076059402e-06)]],
                   'lo_inspread_Bid': [0.0005940490977412251,
                                       [(1, 0.0028645165426634195),
                                        (10, 0.0007559432806511614),
                                        (50, 0.0007559432806511614),
                                        (100, 0.0007559432806511614),
                                        (200, 0.0006377821948293843),
                                        (500, 0.00011229632687737945)]],
                   'mo_Bid': [0.0038746648120546903,
                              [(1, 0.005292089328158008),
                               (10, 0.0007267533052621235),
                               (50, 0.0007267533052621235),
                               (100, 0.0007267533052621235),
                               (200, 0.0003865080466058066),
                               (500, 6.402501409270695e-05)]]}
        Pis["mo_Ask"] = Pis["mo_Bid"]
        Pis["lo_top_Ask"] = Pis["lo_top_Bid"]
        Pis["lo_inspread_Ask"] = Pis["lo_inspread_Bid"]
        Pis["lo_deep_Ask"] = Pis["lo_deep_Bid"]
    if Pi_Q0 == None:
        Pi_Q0 = {
            "Ask_touch" : [0.0015, [(1, 0.013), (10, 0.016), (50, 0.004), (100, 0.166), (200, 0.133), (500, 0.04)]],
            "Ask_deep" : [0.0012, [(1, 0.002), (10, 0.004), (50, 0.001), (100, 0.042), (200, 0.046), (500, 0.057), (1000,0.031 )]]
        }
        if "AMZN.OQ" in paramsPath:
            Pi_Q0 = {'Ask_touch': [0.010140090684645622,
                           [(1, 0.03884704158300229),
                            (10, 0.009296778366251864),
                            (50, 0.001379651264534182),
                            (100, 0.0024608509274181133),
                            (200, 0.0006879249113755069),
                            (500, 3.2288238302197284e-05),
                            (1000, 1.305272362958619e-05)]],
                      'Ask_deep': [0.01497727301958245,
                          [(1, 0.0487474866024748),
                           (10, 0.013947978218672049),
                           (50, 0.0009005062923042582),
                           (100, 0.0021614857143755863),
                           (200, 0.0006018244607243744),
                           (500, 3.349174663545714e-05),
                           (1000, 1.2020554944921454e-05)]]}
        elif "TSLA.OQ" in paramsPath:
            Pi_Q0 = {'Ask_touch': [0.0034673497896682984,
                                   [(1, 0.009820667253646062),
                                    (10, 0.0032147821340202986),
                                    (50, 0.0015155315888310396),
                                    (100, 0.0016285648957023635),
                                    (200, 0.0007759488694235814),
                                    (500, 6.953740217580516e-05),
                                    (1000, 2.1420795641788244e-05)]],
                     'Ask_deep': [0.0043116989714031065,
                                  [(1, 0.010456320220449275),
                                   (10, 0.0027367429710106975),
                                   (50, 0.0018515085129190948),
                                   (100, 0.0016343241590795659),
                                   (200, 0.0006310486072334213),
                                   (500, 6.689678826479514e-05),
                                   (1000, 2.4596571677673654e-05)]]}
        elif "INTC.OQ" in paramsPath:
            Pi_Q0 ={'Ask_touch': [0.00033136559119983304,
                                  [(1, 5.5538455639021544e-05),
                                   (10, 3.411738751405578e-05),
                                   (50, 0.0004079465189684229),
                                   (100, 3.910578960810687e-05),
                                   (200, 5.05693218396739e-05),
                                   (500, 6.784776064336188e-05),
                                   (1000, 0.00010653405107868813)]],
                    'Ask_deep': [0.00023264066631313243,
                                 [(1, 2.225694124202009e-06),
                                  (10, 6.82101843855456e-06),
                                  (50, 1.4531995189359232e-05),
                                  (100, 1.4422351707753386e-06),
                                  (200, 2.006026241383471e-06),
                                  (500, 3.5387085979501684e-06),
                                  (1000, 1.0407294782729011e-05)]]}
        Pi_Q0["Bid_touch"] = Pi_Q0["Ask_touch"]
        Pi_Q0["Bid_deep"] = Pi_Q0["Ask_deep"]


    if s0 is None:
        s = 0
    else:
        s = s0
    Ts,lob,lobL3 = [],[],[]
    _, lob0, lob0_l3 = createLOB({}, {}, Pi_Q0, priceMid0 = price0, spread0 = spread0, ticksize = 0.01, numOrdersPerLevel = 5, lob0 = {}, lob0_l3 = {})
    Ts.append(0)
    lob.append(lob0[-1])
    lobL3.append(lob0_l3[-1])
    spread = lob0[0]['Ask_touch'][0] - lob0[0]['Bid_touch'][0]
    n = None
    timestamps = None
    lob0 = lob0[0]
    lob0_l3 = lob0_l3[0]
    lamb = None
    while s <= T:
        s, n, timestamps, timestamps_this, tau, lamb = thinningOgataIS(T, paramsPath, todPath, maxJumps = 1, s = s, n = n, Ts = timestamps, spread=spread, beta = beta, avgSpread = avgSpread, lamb = lamb)
        sizes = {}
        dictTimestamps = {}
        for t, col in zip(timestamps_this, cols):
            if len(t) == 0: continue
            if "co" in col: # handle size of cancel order in createLOB
                size = 0
            else:
                pi = Pis[col] #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
                p = pi[0]
                dd = pi[1]
                pi = np.array([p*(1-p)**k for k in range(1,10000)])
                pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
                for i, p_i in dd:
                    pi[i-1] = p_i + pi[i-1]
                pi = pi/sum(pi)
                cdf = np.cumsum(pi)
                a = np.random.uniform(0, 1, size = len(t))
                if type(a) != float:
                    size =[]
                    for i in a:
                        size.append(np.argmax(cdf>=i)+1)
                else:
                    size = np.argmax(cdf>=a)+1
            sizes[col]  = size
            dictTimestamps[col] = t

        TsTmp, lobTmp, lobL3Tmp = createLOB(dictTimestamps, sizes, Pi_Q0, lob0 = lob0, lob0_l3 = lob0_l3)
        spread = lobTmp[-1]['Ask_touch'][0] - lobTmp[-1]['Bid_touch'][0]
        lob0 = lobTmp[-1]
        lob0_l3 = lobL3Tmp[-1]
        if len(list(dictTimestamps.keys())):
            Ts.append([list(dictTimestamps.keys())[0], TsTmp[-1], tau])
            lob.append(lob0)
            print(lob0)
            lobL3.append(lob0_l3)
        if (filePathName is not None)&(len(T)%100 == 0):
            with open(filePathName , "wb") as f: #"/home/konajain/params/"
                pickle.dump((T, lob, lobL3), f)
    return Ts, lob, lobL3



def createLOB(dictTimestamps, sizes, Pi_Q0, priceMid0 = 260, spread0 = 4, ticksize = 0.01, numOrdersPerLevel = 10, lob0 = {}, lob0_l3 = {}):
    lob = []
    lob_l3 = []
    T = []
    levels = ["Ask_deep", "Ask_touch", "Bid_touch", "Bid_deep"]
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    colsToLevels = {
        "lo_deep_Ask" : "Ask_deep",
        "lo_top_Ask" : "Ask_touch",
        "lo_top_Bid" : "Bid_touch",
        "lo_deep_Bid" : "Bid_deep"
    }
    if len(lob0) == 0:
        lob0['Ask_touch'] = (priceMid0 + np.floor(spread0/2)*ticksize, 0)
        lob0['Bid_touch'] = (priceMid0 - np.ceil(spread0/2)*ticksize, 0)
        lob0['Ask_deep'] = (priceMid0 + np.floor(spread0/2)*ticksize + ticksize, 0)
        lob0['Bid_deep'] = (priceMid0 - np.ceil(spread0/2)*ticksize - ticksize, 0)
        for k, pi in Pi_Q0.items():
            #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
            p = pi[0]
            dd = pi[1]
            pi = np.array([p*(1-p)**k for k in range(1,100000)])
            pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
            for i, p_i in dd:
                pi[i-1] = p_i + pi[i-1]
            pi = pi/sum(pi)
            cdf = np.cumsum(pi)
            a = np.random.uniform(0, 1)
            qSize = np.argmax(cdf>=a) + 1
            lob0[k] = (lob0[k][0], qSize)
        for l in levels:
            tmp = (numOrdersPerLevel - 1)*[np.floor(lob0[l][1]/numOrdersPerLevel)]
            if tmp !=0:
                lob0_l3[l] = [lob0[l][1] - sum(tmp)] + tmp
            else:
                lob0_l3[l] = [lob0[l][1]]
    if len(dictTimestamps) == 0:
        return T, [lob0], [lob0_l3]

    dfs = []
    for event in dictTimestamps.keys():
        sizes_e = sizes[event]
        timestamps_e = dictTimestamps[event]
        dfs += [pd.DataFrame({"event" : len(timestamps_e)*[event], "time": timestamps_e, "size" : sizes_e})]
    dfs = pd.concat(dfs)
    dfs = dfs.sort_values("time")
    print(dfs.head())
    lob.append(lob0.copy())
    T.append(0)
    lob_l3.append(lob0_l3.copy())
    for i in range(len(dfs)):
            r = dfs.iloc[i]
            lobNew = lob[i].copy()
            lob_l3New = lob_l3[i].copy()
            T.append(r.time)
            if "Ask" in r.event :
                side = "Ask"
            else:
                side = "Bid"

            if "lo" in r.event:
                if "deep" in r.event:
                    if np.abs(lobNew[side + "_touch"][0] - lobNew[side + "_deep"][0]) >  2.5*ticksize:
                        direction = 1
                        if side == "Ask": direction = -1
                        lobNew[side + "_deep"] = (np.round(lobNew[side + "_touch"][0] - direction*ticksize, decimals=2), r['size'])
                        lob_l3New[side + "_deep"] = [r['size']]
                    else:
                        lobNew[side + "_deep"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] + r['size'])
                        lob_l3New[side + "_deep"] += [r['size']]
                elif "top" in r.event:
                    lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] + r['size'])
                    lob_l3New[side + "_touch"] += [r['size']]
                else: #inspread
                    direction = 1
                    if side == "Ask": direction = -1
                    lobNew[side + "_deep"] = lobNew[side + "_touch"]
                    lob_l3New[side + "_deep"] = lob_l3New[side + "_touch"].copy()
                    lobNew[side + "_touch"] = (np.round(lobNew[side + "_touch"][0] + direction*ticksize, decimals=2), r['size'])
                    lob_l3New[side + "_touch"] = [r['size']]

            if "mo" in r.event:
                lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] - r['size'])
                if lobNew[side + "_touch"][1] > 0:
                    cumsum = np.cumsum(lob_l3New[side + "_touch"])
                    idx = np.argmax(cumsum >= r['size'])
                    tmp = lob_l3New[side + "_touch"][idx:]
                    offset = 0
                    if idx > 0: offset = cumsum[idx - 1]
                    tmp[0] = tmp[0] + offset - r['size']
                    lob_l3New[side + "_touch"] = tmp.copy()
                while lobNew[side + "_touch"][1] <= 0: # queue depletion
                    extraVolume = -1*lobNew[side + "_touch"][1]
                    lobNew[side + "_touch"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] - extraVolume)
                    lob_l3New[side + "_touch"] = lob_l3New[side + "_deep"].copy()
                    if lobNew[side + "_touch"][1] > 0:
                        if extraVolume > 0:
                            cumsum = np.cumsum(lob_l3New[side + "_touch"])
                            idx = np.argmax(cumsum >= extraVolume)
                            tmp = np.array(lob_l3New[side + "_touch"][idx:])
                            tmp[0] = cumsum[idx] - extraVolume
                            tmp = tmp[tmp>0]
                            lob_l3New[side + "_touch"] = list(tmp).copy()
                    direction = 1
                    if side == "Bid": direction = -1
                    #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
                    pi = Pi_Q0[side+"_deep"]
                    p = pi[0]
                    dd = pi[1]
                    pi = np.array([p*(1-p)**k for k in range(1,100000)])
                    pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
                    for i, p_i in dd:
                        pi[i-1] = p_i + pi[i-1]
                    pi = pi/sum(pi)
                    cdf = np.cumsum(pi)
                    a = np.random.uniform(0, 1)
                    qSize = np.argmax(cdf>=a)+1
                    lobNew[side + "_deep"] = (np.round(lobNew[side + "_deep"][0] + direction*ticksize, decimals=2), qSize)
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
                    lob_l3New[side + "_touch"] = lob_l3New[side + "_deep"].copy()
                    direction = 1
                    if side == "Bid": direction = -1
                    pi = Pi_Q0[side+"_deep"]
                    p = pi[0]
                    dd = pi[1]
                    pi = np.array([p*(1-p)**k for k in range(1,100000)])
                    pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
                    for i, p_i in dd:
                        pi[i-1] = p_i + pi[i-1]
                    pi = pi/sum(pi)
                    cdf = np.cumsum(pi)
                    a = np.random.uniform(0, 1)
                    qSize = np.argmax(cdf>=a)-1
                    lobNew[side + "_deep"] = (np.round(lobNew[side + "_deep"][0] + direction*ticksize, decimals=2), qSize)
                    tmp = (numOrdersPerLevel - 1)*[np.floor(lobNew[side + "_deep"][1]/numOrdersPerLevel)]
                    lob_l3New[side + "_deep"] = [lobNew[side + "_deep"][1] - sum(tmp)] + tmp

                if lobNew[side + "_deep"][1] <= 0: # queue depletion
                    direction = 1
                    if side == "Bid": direction = -1
                    #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
                    pi = Pi_Q0[side+"_deep"]
                    p = pi[0]
                    dd = pi[1]
                    pi = np.array([p*(1-p)**k for k in range(1,100000)])
                    pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
                    for i, p_i in dd:
                        pi[i-1] = p_i + pi[i-1]
                    pi = pi/sum(pi)
                    cdf = np.cumsum(pi)
                    a = np.random.uniform(0, 1)
                    qSize = np.argmax(cdf>=a)+1
                    lobNew[side + "_deep"] = (np.round(lobNew[side + "_deep"][0] + direction*ticksize, decimals=2), qSize)
                    tmp = ((2*numOrdersPerLevel) - 1)*[np.floor(lobNew[side + "_deep"][1]/(2*numOrdersPerLevel))]
                    lob_l3New[side + "_deep"] = [lobNew[side + "_deep"][1] - sum(tmp)] + tmp
            lob.append(lobNew.copy())
            lob_l3.append(lob_l3New.copy())
    return T, lob, lob_l3

def simBacry(paramsPath = "/SAN/fca/Konark_PhD_Experiments/extracted/AAPL.OQ_ParamsInferred_2019-01-02_2019-03-31_EMBacry"):
    with open(paramsPath, "rb") as f:
        params = pickle.load(f)
    from tick.hawkes import SimuHawkes, HawkesKernelPowerLaw, HawkesKernel0
    cols = ['co_top_Ask', 'co_top_Bid', 'lo_top_Ask', 'lo_top_Bid', 'mo_Ask',  'mo_Bid', 'pc_Ask', 'pc_Bid']
    support = 500
    cutoff = 1e-3
    kernels = []
    baselines = []
    for i in range(len(cols)):
        t = []
        for j in range(len(cols)):
            par = params[cols[i]+"->"+cols[j]]
            if par is None:
                t.append(HawkesKernel0())
            else:
                t.append(HawkesKernelPowerLaw(np.exp(par[0]), cutoff, -1*par[1], support))
        kernels.append(t)
        baselines.append(params[cols[i]])
    hawkes = SimuHawkes(kernels=kernels, baseline=baselines,verbose=True)
    hawkes.end_time = 23400
    hawkes.simulate()
    return hawkes.timestamps

def simulateMarketImpactStudy(T , paramsPath , todPath, Pis = None, Pi_Q0 = None, beta = 0.7479, avgSpread = 0.0169, spread0 = 3, price0 = 260, metaQ = 2000, metaSide = "Buy", metaTime = 1800, metaStrategy = ("TWAP", "MO"), chilOrderFreq = 150, orderInitTime = None):
    """
    :param T: time limit of simulations
    :param paramsPath: path of fitted params
    :param Pis: distribution of order sizes
    :param Pi_Q0: depleted queue size distribution
    """
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    if Pis == None:
        ## AAPL
        Pis = {
            'mo_Bid' : [8.16e-3, [(1, 0.072), (10, 0.04), (50, 0.028), (100, 0.427), (200, 0.051), (500, 0.07)]],
            'lo_top_Bid' : [2.09e-3, [(1, 0.02), (10, 0.069), (50, 0.005), (100, 0.6), (200, 0.036), (500, 0.054)]],
            'lo_deep_Bid' : [2.33e-3, [(1, 0.021), (10, 0.112), (50, 0.015), (100, 0.276), (200, 0.097), (500, 0.172)]]
        }
        Pis["lo_inspread_Bid"] = Pis["lo_top_Bid"]
        if "AMZN.OQ" in paramsPath:
            Pis = {'lo_top_Bid': [0.05365508940934027,
                                  [(1, 0.0432323548960116),
                                   (10, 0.013435798495166417),
                                   (50, 0.0005709704503910774),
                                   (100, 6.663879261160762e-05),
                                   (200, 1.5747668929585514e-05),
                                   (500, 6.09046480638754e-07)]],
                   'lo_deep_Bid': [0.06060130535300165,
                                   [(1, 0.05639948122151968),
                                    (10, 0.05518191451020626),
                                    (50, 0.0003330187521711186),
                                    (100, 4.9983396108122606e-05),
                                    (200, 7.903710504897672e-06),
                                    (500, 1.6657318546374149e-06)]],
                   'lo_inspread_Bid': [0.040813664165380875,
                                       [(1, 0.020322265459336963),
                                        (10, 0.005277804876103806),
                                        (50, 0.0005669113835675268),
                                        (100, 5.090044812498355e-05),
                                        (200, 6.6587761856594516e-06),
                                        (500, 6.606025886842982e-07)]],
                   'mo_Bid': [0.030074069098223778,
                              [(1, 0.05510488974651141),
                               (10, 0.008831314046535154),
                               (50, 0.0019306968599348715),
                               (100, 0.0001651778474078509),
                               (200, 3.460981489505782e-05),
                               (500, 3.7078651677298155e-06)]]}
        elif "TSLA.OQ" in paramsPath:
            Pis = {'lo_top_Bid': [0.01906341526374109,
                                  [(1, 0.013336873454420098),
                                   (10, 0.002499960585631139),
                                   (50, 0.00258837727124698),
                                   (100, 9.016197290658499e-05),
                                   (200, 2.1484763827487066e-05),
                                   (500, 1.4176290754765072e-06)]],
                   'lo_deep_Bid': [0.01834598886408305,
                                   [(1, 0.006567760273974217),
                                    (10, 0.008358970313982568),
                                    (50, 0.0026727677268739785),
                                    (100, 5.1652220786639296e-05),
                                    (200, 1.4287611741998702e-05),
                                    (500, 1.7712494515096627e-06)]],
                   'lo_inspread_Bid': [0.013924826336477625,
                                       [(1, 0.007544542317155571),
                                        (10, 0.005196331116912839),
                                        (50, 0.0010228895870503783),
                                        (100, 8.177891598970127e-05),
                                        (200, 2.1061506356115122e-05),
                                        (500, 3.808005108713909e-06)]],
                   'mo_Bid': [0.015972671818734865,
                              [(1, 0.021494545426040023),
                               (10, 0.006376156312818793),
                               (50, 0.0023541898985074998),
                               (100, 0.00031172774984016764),
                               (200, 8.539722286633611e-05),
                               (500, 1.0538996186294693e-05)]]}
        elif "INTC.OQ" in paramsPath:
            Pis = {'lo_top_Bid': [0.0012032594626957276,
                                  [(1, 0.0004058911201508531),
                                   (10, 0.004937100775363866),
                                   (50, 0.011839578897579354),
                                   (100, 4.2540607105295814e-05),
                                   (200, 1.3215923263800984e-05),
                                   (500, 4.946493652744191e-06)]],
                   'lo_deep_Bid': [0.001446173610413779,
                                   [(1, 0.0009504555025436379),
                                    (10, 0.005970974450434397),
                                    (50, 0.011838812712121737),
                                    (100, 7.747866832159131e-05),
                                    (200, 4.951687702705543e-06),
                                    (500, 1.4416468076059402e-06)]],
                   'lo_inspread_Bid': [0.0005940490977412251,
                                       [(1, 0.0028645165426634195),
                                        (10, 0.0007559432806511614),
                                        (50, 0.0007559432806511614),
                                        (100, 0.0007559432806511614),
                                        (200, 0.0006377821948293843),
                                        (500, 0.00011229632687737945)]],
                   'mo_Bid': [0.0038746648120546903,
                              [(1, 0.005292089328158008),
                               (10, 0.0007267533052621235),
                               (50, 0.0007267533052621235),
                               (100, 0.0007267533052621235),
                               (200, 0.0003865080466058066),
                               (500, 6.402501409270695e-05)]]}
        Pis["mo_Ask"] = Pis["mo_Bid"]
        Pis["lo_top_Ask"] = Pis["lo_top_Bid"]
        Pis["lo_inspread_Ask"] = Pis["lo_inspread_Bid"]
        Pis["lo_deep_Ask"] = Pis["lo_deep_Bid"]
    if Pi_Q0 == None:
        Pi_Q0 = {
            "Ask_touch" : [0.0015, [(1, 0.013), (10, 0.016), (50, 0.004), (100, 0.166), (200, 0.133), (500, 0.04)]],
            "Ask_deep" : [0.0012, [(1, 0.002), (10, 0.004), (50, 0.001), (100, 0.042), (200, 0.046), (500, 0.057), (1000,0.031 )]]
        }
        if "AMZN.OQ" in paramsPath:
            Pi_Q0 = {'Ask_touch': [0.010140090684645622,
                                   [(1, 0.03884704158300229),
                                    (10, 0.009296778366251864),
                                    (50, 0.001379651264534182),
                                    (100, 0.0024608509274181133),
                                    (200, 0.0006879249113755069),
                                    (500, 3.2288238302197284e-05),
                                    (1000, 1.305272362958619e-05)]],
                     'Ask_deep': [0.01497727301958245,
                                  [(1, 0.0487474866024748),
                                   (10, 0.013947978218672049),
                                   (50, 0.0009005062923042582),
                                   (100, 0.0021614857143755863),
                                   (200, 0.0006018244607243744),
                                   (500, 3.349174663545714e-05),
                                   (1000, 1.2020554944921454e-05)]]}
        elif "TSLA.OQ" in paramsPath:
            Pi_Q0 = {'Ask_touch': [0.0034673497896682984,
                                   [(1, 0.009820667253646062),
                                    (10, 0.0032147821340202986),
                                    (50, 0.0015155315888310396),
                                    (100, 0.0016285648957023635),
                                    (200, 0.0007759488694235814),
                                    (500, 6.953740217580516e-05),
                                    (1000, 2.1420795641788244e-05)]],
                     'Ask_deep': [0.0043116989714031065,
                                  [(1, 0.010456320220449275),
                                   (10, 0.0027367429710106975),
                                   (50, 0.0018515085129190948),
                                   (100, 0.0016343241590795659),
                                   (200, 0.0006310486072334213),
                                   (500, 6.689678826479514e-05),
                                   (1000, 2.4596571677673654e-05)]]}
        elif "INTC.OQ" in paramsPath:
            Pi_Q0 ={'Ask_touch': [0.00033136559119983304,
                                  [(1, 5.5538455639021544e-05),
                                   (10, 3.411738751405578e-05),
                                   (50, 0.0004079465189684229),
                                   (100, 3.910578960810687e-05),
                                   (200, 5.05693218396739e-05),
                                   (500, 6.784776064336188e-05),
                                   (1000, 0.00010653405107868813)]],
                    'Ask_deep': [0.00023264066631313243,
                                 [(1, 2.225694124202009e-06),
                                  (10, 6.82101843855456e-06),
                                  (50, 1.4531995189359232e-05),
                                  (100, 1.4422351707753386e-06),
                                  (200, 2.006026241383471e-06),
                                  (500, 3.5387085979501684e-06),
                                  (1000, 1.0407294782729011e-05)]]}
        Pi_Q0["Bid_touch"] = Pi_Q0["Ask_touch"]
        Pi_Q0["Bid_deep"] = Pi_Q0["Ask_deep"]

    orderInitTime = orderInitTime or np.random.randint(0, T - metaTime, 1)[0]
    print(orderInitTime)
    newEndTime = orderInitTime + metaTime + 120 # measure impact till until after 2 mins
    if metaStrategy[0] == "TWAP":
        childQ = int(metaQ*chilOrderFreq/metaTime)
        childTimes = orderInitTime + np.arange(int(metaTime/chilOrderFreq))*chilOrderFreq
    if metaStrategy[1] == "MO":
        side = "Bid" if metaSide == "Sell" else "Ask"
        childEvent = "mo_" + side
        child_k = np.where(np.array(cols) == childEvent)[0][0]
    elif "lo" in metaStrategy[1]:
        side = "Ask" if metaSide == "Sell" else "Bid"
        childEvent = metaStrategy[1] + side
        child_k = np.where(np.array(cols) == childEvent)[0][0]
    with open(paramsPath, "rb") as f:
        params = pickle.load(f)
    with open(todPath, "rb") as f:
        tod = pickle.load(f)
    num_nodes = len(cols)

    s = orderInitTime - 100 # memory of more than 100 sec is not going to impact
    Ts,lob,lobL3 = [],[],[]
    _, lob0, lob0_l3 = createLOB({}, {}, Pi_Q0, priceMid0 = price0, spread0 = spread0, ticksize = 0.01, numOrdersPerLevel = 5, lob0 = {}, lob0_l3 = {})
    Ts.append(0)
    lob.append(lob0[-1])
    lobL3.append(lob0_l3[-1])
    spread = lob0[0]['Ask_touch'][0] - lob0[0]['Bid_touch'][0]
    n = None
    timestamps = None
    lob0 = lob0[0]
    lob0_l3 = lob0_l3[0]
    lamb = None
    counter = 0
    currentMetaOrderTime = childTimes[counter]

    while s <= newEndTime:
        if (n is not None )and (timestamps is not None): prev_s, prev_n, prev_timestamps, prev_lamb = s, n.copy(), timestamps.copy(), lamb
        s, n, timestamps, timestamps_this, tau, lamb = thinningOgataIS(T, paramsPath, todPath, maxJumps = 1, s = s, n = n, Ts = timestamps, spread=spread, beta = beta, avgSpread = avgSpread, lamb = lamb)
        metaOrder = False
        if counter < len(childTimes): currentMetaOrderTime = childTimes[counter]
        if (n is not None)&(s >= currentMetaOrderTime)&(counter < len(childTimes)): # reject background and add metaorder
            print("execing meta order")
            metaOrder = True

            s = currentMetaOrderTime
            counter += 1
            n = prev_n
            n[child_k] += 1
            timestamps = prev_timestamps
            timestamps[child_k] += (s,)
            timestamps_this = len(cols)*[()]
            timestamps_this[child_k] += (s,)
            tau = tau

            mat = np.zeros((num_nodes, num_nodes))
            hourIndex = np.min([12,int(np.floor(s/1800))])
            for i in range(num_nodes):
                for j in range(num_nodes):
                    kernelParams = params.get(cols[i] + "->" + cols[j], None)
                    if kernelParams is None: continue
                    if np.isnan(kernelParams[1][2]): continue
                    # print(cols[i] + "->" + cols[j])
                    # print((kernelParams[0]*np.exp(kernelParams[1][0]) , kernelParams[1][1] , kernelParams[1][2]))
                    todMult = tod[cols[j]][hourIndex]
                    mat[i][j]  = todMult*kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2]) # alpha/(beta -1)*gamma
            specRad = np.max(np.linalg.eig(mat)[0]).real
            newdecays = len(cols)*[0]
            for i in range(len(timestamps)):
                kernelParams = params.get(cols[child_k] + "->" + cols[i], None)
                todMult = tod[cols[i]][hourIndex]*0.99/specRad
                if kernelParams is None: continue
                if np.isnan(kernelParams[1][2]): continue
                # decay = todMult*powerLawKernel(0, alpha = kernelParams[0]*np.exp(kernelParams[1][0]), t0 = kernelParams[1][2], beta = kernelParams[1][1])
                decay = todMult*powerLawCutoff(0, kernelParams[0]*kernelParams[1][0], kernelParams[1][1], kernelParams[1][2])
                # print(decay)
                newdecays[i] += decay
            newdecays = [np.max([0, d]) for d in newdecays]
            newdecays[5] = ((spread)**beta)*newdecays[5]
            newdecays[6] = ((spread)**beta)*newdecays[6]
            if 100*np.round(spread, 2) < 2 : newdecays[5] = newdecays[6] = 0
            lamb = prev_lamb + sum(newdecays)

        sizes = {}
        dictTimestamps = {}
        for t, col in zip(timestamps_this, cols):
            if len(t) == 0: continue
            if "co" in col: # handle size of cancel order in createLOB
                size = 0
            elif metaOrder:
                size = childQ
            else:
                pi = Pis[col] #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
                p = pi[0]
                dd = pi[1]
                pi = np.array([p*(1-p)**k for k in range(1,10000)])
                pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
                for i, p_i in dd:
                    pi[i-1] = p_i + pi[i-1]
                pi = pi/sum(pi)
                cdf = np.cumsum(pi)
                a = np.random.uniform(0, 1, size = len(t))
                if type(a) != float:
                    size =[]
                    for i in a:
                        size.append(np.argmax(cdf>=i)+1)
                else:
                    size = np.argmax(cdf>=a)+1
            sizes[col]  = size
            dictTimestamps[col] = t

        TsTmp, lobTmp, lobL3Tmp = createLOB(dictTimestamps, sizes, Pi_Q0, lob0 = lob0, lob0_l3 = lob0_l3)
        spread = lobTmp[-1]['Ask_touch'][0] - lobTmp[-1]['Bid_touch'][0]
        lob0 = lobTmp[-1]
        lob0_l3 = lobL3Tmp[-1]
        if len(list(dictTimestamps.keys())):
            Ts.append([list(dictTimestamps.keys())[0], TsTmp[-1], tau])
            lob.append(lob0)
            lobL3.append(lob0_l3)
        # with open("/SAN/fca/Konark_PhD_Experiments/simulated/AAPL.OQ_ResultsWCutoff_2019-01-02_2019-03-31_CLSLogLin_10_tmp" , "ab") as f: #"/home/konajain/params/"
        #     pickle.dump(([list(dictTimestamps.keys())[0], TsTmp[-1], tau], lob0, lob0_l3), f)
    return orderInitTime, Ts, lob, lobL3
