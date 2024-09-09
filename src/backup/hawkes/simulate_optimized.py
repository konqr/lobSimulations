#%%
import pickle
import numpy as np
import pandas as pd
import time
import numpy as np
import os

def powerLawKernel(x, alpha = 1., t0 = 1., beta = -2.):
    if x < t0: return 0
    return alpha*(x**beta)


def powerLawCutoff(time, alpha, beta, gamma):
    # alpha = a*beta*(gamma - 1)
    return alpha/((1 + gamma*time)**beta)


def powerLawKernelIntegral(x1, x2, alpha = 1., t0 = 1., beta = -2.):
    return (x2/(1+beta))*powerLawKernel(x2, alpha = alpha, t0=t0, beta=beta) - (x1/(1+beta))*powerLawKernel(x1, alpha = alpha, t0=t0, beta=beta)

def expKernel(x, alpha, beta):
    return alpha*np.exp(-x*beta)

num_nodes=12
def preprocessdata(paramsPath: str, todPath: str):
    """Takes in params and todpath and spits out corresponding vectorised numpy arrays
    
    Returns:
    tod: a [12, 13] matrix containing values of f(Q_t), the time multiplier for the 13 different 30 min bins of the trading day.
    params=[kernelparams, baselines]
        kernelparams=params[0]: an array of [12, 12] matrices consisting of mask, alpha0, beta, gamma. the item at arr[i][j] corresponds to the corresponding value from params[cols[i] + "->" + cols[j]]
        So mask=params[0][0], alpha0=params[0][1], beta=params[0][2], gamma=params[0][3]
        baselines=params[1]: a vector of dim=(num_nodes, 1) consisting of baseline intensities
    """
    
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    os.path.exists("")
    with open(todPath, "rb") as f:
        data = pickle.load(f)
    tod=np.zeros(shape=(num_nodes, 13))
    for i in range(num_nodes):
        tod[i]=[data[cols[i]][k] for k in range(13)]
        
    with open(paramsPath, "rb") as f:
        data=pickle.load(f)
    baselines=np.zeros(shape=(num_nodes, 1)) #vectorising baselines
    for i in range(num_nodes):
        baselines[i]=data[cols[i]]
        #baselines[i]=data.pop(cols[i], None)
    
    
    #params=[mask, alpha, beta, gamma] where each is a 12x12 matrix
    mask, alpha, beta, gamma=[np.zeros(shape=(12, 12)) for _ in range(4)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            kernelParams = data.get(cols[i] + "->" + cols[j], None)
            if kernelParams is not None:
                mask[i][j]=kernelParams[0]
                alpha[i][j]=kernelParams[1][0]
                beta[i][j]=kernelParams[1][1]
                gamma[i][j]=kernelParams[1][2]
    kernelparams=[mask, alpha, beta, gamma] 
    params=[kernelparams, baselines] 
    return tod, params

def thinningOgataIS2(T, params, tod, kernel = 'powerlaw', num_nodes=12, maxJumps = None, s = None, n = None, Ts = None, timeseries=None, spread=None, beta = 0.7479, avgSpread = 0.0169,lamb= None, left=None):
    """ 
    Arguments:
    T: timelimit of simulation process
    params=[kernelparams, baselines]
        kernelparams: an array of [12, 12] matrices consisting of mask, alpha0, beta, gamma. the item at arr[i][j] corresponds to the corresponding value from params[cols[i] + "->" + cols[j]]
        baselines: a vector of dim=(num_nodes, 1) consisting of baseline intensities
    tod: a [12, 13] matrix containing values of f(Q_t), the time multiplier for the 13 different 30 min bins of the trading day.
    num_nodes: #of different processes
    timeseries: the sequence of all point-events arranged in the form (t, m) where m is the #of the dimension
    """
    numJumps = 0
    if n is None: n = num_nodes*[0]
    if Ts is None: Ts = num_nodes*[()]
    if spread is None: spread = 1
    """Setting up thinningOgata params"""
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    baselines =params[1].copy()
    mat = np.zeros((num_nodes, num_nodes))
    if s is None: 
        s = 0
    hourIndex = min(12,int(s//1800)) #1800 seconds in 30 minutes
    todmult=tod[:, hourIndex].reshape((12, 1))
    #todMult*kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2]) 
    mat=todmult*params[0][0]*params[0][1]/((params[0][2]-1) *params[0][3])
    baselines[5] = ((spread/avgSpread)**beta)*baselines[5]
    baselines[6] = ((spread/avgSpread)**beta)*baselines[6]
    if (np.sum(mat) != 0)&(not np.isnan(np.sum(mat))):
        specRad = np.max(np.linalg.eig(mat)[0]).real
    else:
        specRad = 1
    if specRad < 1 : specRad = 0.99 #  # dont change actual specRad if already good
    
    """calculating initial values of lamb_bar"""
    if lamb is None:
        decays=(0.99/specRad) * todmult * baselines
        lamb=np.sum(decays) #3.04895025
    if left is None:
        left=0
    if timeseries is None:
        timeseries=[]
        
    """simulation loop"""
    while s<=T:
        """Assign lamb_bar"""
        lamb_bar=lamb 
        #print("lamb_bar: ", lamb_bar)
        """generate random u"""
        u=np.random.uniform(0, 1)
        if lamb_bar==0:
            s+=0.1  # wait for some time
        else:
            w=max(1e-7, -1 * np.log(u)/lamb_bar) # floor at 0.1 microsec
            s+=w  
        """Recalculating baseline lambdas sum with new candidate"""
        hourIndex = min(12,int(s//1800))
        todmult=tod[:, hourIndex].reshape((12, 1)) * (0.99/specRad)
        decays=todmult * baselines
        """Summing cross excitations for previous points"""
        if timeseries==[]:
            pass
        else:
            while left<len(timeseries):
                #print("Iterating: s: ", s, ", timestamp: ", timeseries[left][0])
                if s-timeseries[left][0]>=10:
                    left+=1 
                else:
                    break
            for point in timeseries[left:]: #point is of the form(s, k)
                if kernel == 'powerlaw':
                    kern=powerLawCutoff(time=s-point[0], alpha=params[0][0][point[1]]*params[0][1][point[1]], beta=params[0][2][point[1]], gamma=params[0][3][point[1]])
                elif kernel == 'exp':
                    kern=expKernel(s-point[0], params[0][0][point[1]], params[0][1][point[1]])
                else:
                    raise Exception("kernel must be either 'exp' or 'powerlaw'")
                kern=kern.reshape((12, 1))
                decays+=todmult*kern
        #print(decays.shape) #should be (12, 1)
        decays=np.maximum(decays, 0)
        decays[5] = ((spread/avgSpread)**beta)*decays[5]
        decays[6] = ((spread/avgSpread)**beta)*decays[6]
        if 100*np.round(spread, 2)  < 2 : 
            decays[5] = decays[6] = 0
        lamb = float(sum(decays))
        #print("LAMBDA: ", lamb)
        
        """Testing candidate point"""
        D=np.random.uniform(0, 1)
        #print("Candidate D: ", D)
        if D*lamb_bar<=lamb:
            print(f"S is: {s}")
            """Accepted so assign candidate point to a process by a ratio of intensities"""
            k=0
            total=decays[k]
            while D*lamb_bar >= total:
                k+=1
                total+=decays[k]
            """dimension is cols[k]"""   
            """Update values of lambda for next simulation loop and append point to Ts"""
            if k in [5, 6]:
                spread=spread-0.01
            
            """Precalc next value of lambda_bar"""    
            newdecays=todmult * params[0][0][k].reshape(12, 1)*params[0][1][k].reshape(12, 1)
            newdecays=np.maximum(newdecays, 0)
            newdecays[5] = ((spread/avgSpread)**beta)*newdecays[5]
            newdecays[6] = ((spread/avgSpread)**beta)*newdecays[6]
            if 100*np.round(spread, 2) < 2 : newdecays[5] = newdecays[6] = 0
            lamb+= sum(newdecays)
            lamb=lamb[0]


            if len(Ts[k]) > 0:
                T_Minus1 = Ts[k][-1]
            else:
                T_Minus1 = 0
            decays = np.array(baselines.copy())
            hourIndex = np.min([12,int(np.floor(s/1800))])
            decays[5] = ((spread/avgSpread)**beta)*decays[5]
            decays[6] = ((spread/avgSpread)**beta)*decays[6]
            decays = decays*(s-T_Minus1)
            
            """Updating history and returns"""
            Ts[k]+=(s,)
            tau = decays[k][0]
            numJumps+=1
            n[k]+=1
            timeseries.append((s, k)) #(time, event)
            #print("point added")
            if numJumps>=maxJumps:
                return s, n, Ts, tau, lamb, timeseries, left
    return s, n, Ts, -1, lamb, timeseries, left

def simulate_optimized(T , paramsPath , todPath, s0 = None, filePathName = None, Pis = None, Pi_Q0 = None, beta = 0.7479, avgSpread = 0.0169, spread0 = 3, price0 = 260):
    """
    :param T: time limit of simulations
    :param paramsPath: path of fitted params
    :param Pis: distribution of order sizes
    :param Pi_Q0: depleted queue size distribution
    """
    tod, params=preprocessdata(paramsPath=paramsPath, todPath=todPath)
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
    if Pis == None:
        ## AAPL
        Pis = {'lo_deep_Bid': [0.0028405540014542,
                              [(1, 0.012527718976326372),
                               (10, 0.13008130053050898),
                               (50, 0.01432529704009695),
                               (100, 0.7405118066127269)]],
                'lo_inspread_Bid': [0.001930457915114691,
                                    [(1, 0.03065295587464324),
                                     (10, 0.18510015294680732),
                                     (50, 0.021069809772740915),
                                     (100, 2.594573929265402)]],
                'lo_top_Bid': [0.0028207493506166507,
                               [(1, 0.05839080241927479),
                                (10, 0.17259077005977103),
                                (50, 0.011272365769158578),
                                (100, 2.225254050790496)]],
                'mo_Bid': [0.008810527626617248,
                           [(1, 0.13607245009890734),
                            (10, 0.07035276109045323),
                            (50, 0.041795348623102815),
                            (100, 1.0584893799948996),
                            (200, 0.10656843768185977)]]}

        if "AMZN.OQ" in paramsPath:
            Pis = {'lo_deep_Bid': [0.05958213706845956,
                                   [(1, 0.057446825731119984),
                                    (10, 0.011125136923873007),
                                    (50, 0.01933201666829289),
                                    (100, 0.6348278415533004)]],
                   'lo_inspread_Bid': [0.038562301495905914,
                                       [(1, 0.1168367454898604),
                                        (10, 0.0557382431709154),
                                        (50, 0.3467325997346703),
                                        (100, 0.8833905166426477)]],
                   'lo_top_Bid': [0.05329616963733641,
                                  [(1, 0.19425540200523306),
                                   (10, 0.025295856444822386),
                                   (50, 0.020497438365610236),
                                   (100, 0.4307699176008908)]],
                   'mo_Bid': [0.032949075097457925,
                              [(1, 0.29374511617853377),
                               (10, 0.0718169215529087),
                               (50, 0.055061736880763365),
                               (100, 0.15039996187749854),
                               (200, 0.005369246722650729)]]}
        elif "TSLA.OQ" in paramsPath:
            Pis = {'lo_deep_Bid': [0.01669522098575225,
                                   [(1, 0.06148475865524862),
                                    (10, 0.004654724106600434),
                                    (50, 1.2294987988577017),
                                    (100, 0.8385097528175709)]],
                   'lo_inspread_Bid': [0.011781068254316844,
                                       [(1, 0.0839757385977094),
                                        (10, 0.02316868081220621),
                                        (50, 0.5310721970270249),
                                        (100, 1.4366979487738731)]],
                   'lo_top_Bid': [0.0174892051205745,
                                  [(1, 0.1262749554792728),
                                   (10, 0.020465561130447382),
                                   (50, 0.3133895036151287),
                                   (100, 0.9378616431670065)]],
                   'mo_Bid': [0.0166683116660209,
                              [(1, 0.14858699403417608),
                               (10, 0.06462650434281582),
                               (50, 0.07281400243536842),
                               (100, 0.39822843771940597),
                               (200, 0.031055753303048675)]]}
        elif "INTC.OQ" in paramsPath:
            Pis = {'lo_deep_Bid': [0.001852053034664248,
                                   [(1, 0.002939808343982239),
                                    (10, 0.0008345574724029834),
                                    (50, 0.006765547673248046),
                                    (100, 1.4993234335200165)]],
                   'lo_inspread_Bid': [0.000777255592021559,
                                       [(1, 0.0010509182689977994),
                                        (10, 0.0011177500567861932),
                                        (50, 0.0003950002143943747),
                                        (100, 0.729657046574722)]],
                   'lo_top_Bid': [0.0018096999936874803,
                                  [(1, 0.004068520537671112),
                                   (10, 0.0),
                                   (50, 0.0023684378938113593),
                                   (100, 1.2015631555690254)]],
                   'mo_Bid': [0.0038590215719931324,
                              [(1, 0.06967670505278645),
                               (10, 0.013356445214752877),
                               (50, 0.013653531059951492),
                               (100, 1.5095435525547378),
                               (200, 0.2867609734608456)]]}
        Pis["mo_Ask"] = Pis["mo_Bid"]
        Pis["lo_top_Ask"] = Pis["lo_top_Bid"]
        Pis["lo_inspread_Ask"] = Pis["lo_inspread_Bid"]
        Pis["lo_deep_Ask"] = Pis["lo_deep_Bid"]
    if Pi_Q0 == None:
        Pi_Q0 = {'Ask_touch': [0.0018287411983379015,
                               [(1, 0.007050802017724003),
                                (10, 0.009434048841996959),
                                (100, 0.20149407216104853),
                                (500, 0.054411455742183645),
                                (1000, 0.01605198687975892)]],
                 'Ask_deep': [0.001229380704944344,
                              [(1, 0.0),
                               (10, 0.0005240951083719349),
                               (100, 0.03136813097471952),
                               (500, 0.06869444491232923),
                               (1000, 0.04298980350337664)]]}
        if "AMZN.OQ" in paramsPath:
            Pi_Q0 = {'Ask_touch': [0.010569068336116975,
                                   [(1, 0.11631071538074542),
                                    (10, 0.03942041559910066),
                                    (100, 0.2911463764655624),
                                    (500, 0.0015346534902328998),
                                    (1000, 0.0008246596078224383)]],
                     'Ask_deep': [0.014639327119686312,
                                  [(1, 0.10910251548637026),
                                   (10, 0.03075549949138249),
                                   (100, 0.27677505509194006),
                                   (500, 0.0016563689305610241),
                                   (1000, 0.0012702383211743262)]]}
        elif "TSLA.OQ" in paramsPath:
            Pi_Q0 = {'Ask_touch': [0.0038733508105015736,
                                   [(1, 0.0464140045740905),
                                    (10, 0.017315554389264503),
                                    (100, 0.3691163178826726),
                                    (500, 0.007934978399960931),
                                    (1000, 0.002188238089721264)]],
                     'Ask_deep': [0.0051259834866583965,
                                  [(1, 0.04414388196971091),
                                   (10, 0.02075689833423361),
                                   (100, 0.3733272150737976),
                                   (500, 0.009927773433398753),
                                   (1000, 0.003285049294627093)]]}
        elif "INTC.OQ" in paramsPath:
            Pi_Q0 ={'Ask_touch': [0.0003369108404859812,
                                  [(1, 6.666080692515952e-05),
                                   (10, 0.0),
                                   (100, 0.015325800833422807),
                                   (500, 0.015231414988852629),
                                   (1000, 0.016695461654782216)]],
                    'Ask_deep': [0.00023266882039165479,
                                 [(1, 0.0),
                                  (10, 0.0),
                                  (100, 0.0003916250299040387),
                                  (500, 0.0005987961901425286),
                                  (1000, 0.0010547208579439046)]]}
        Pi_Q0["Bid_touch"] = Pi_Q0["Ask_touch"]
        Pi_Q0["Bid_deep"] = Pi_Q0["Ask_deep"]


    if s0 is None:
        s = 0
    else:
        s = s0
    Ts,lob,lobL3 = [],[],[]
    _, lob0, lob0_l3 = createLOB({}, {}, Pi_Q0, priceMid0 = price0, spread0 = spread0, ticksize = 0.01, numOrdersPerLevel = 5, lob0 = {}, lob0_l3 = {})
    #print("The initial LOB: lob0", lob0, "lob0_l3", lob0_l3)
    Ts.append(0)
    lob.append(lob0[-1])
    lobL3.append(lob0_l3[-1])
    spread = lob0[0]['Ask_touch'][0] - lob0[0]['Bid_touch'][0]
    #print("initial spread: ", spread, "\n")
    n = None
    timestamps = None
    timeseries=None
    lob0 = lob0[0]
    lob0_l3 = lob0_l3[0]
    lamb = None
    trials=1
    left=None
    thinningtime=0
    print("POST INIT Snapshot LOB0: " + str(lob0))
    while s <= T:
        start=time.perf_counter_ns()
        s, n, timestamps, tau, lamb, timeseries, left=thinningOgataIS2(T, params, tod, num_nodes=num_nodes, maxJumps = 1, s = s, n = n, Ts = timestamps, timeseries=timeseries, spread=spread, beta = beta, avgSpread = avgSpread,lamb= lamb, left=left)
        timestamps_this=[()]*num_nodes
        timestamps_this[timeseries[-1][1]]=(timeseries[-1][0],)
        end=time.perf_counter_ns()
        thinningtime+=abs(end-start)
        trials +=1
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
                # pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
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
        #print("Sizes is: ", sizes)
        TsTmp, lobTmp, lobL3Tmp = createLOB(dictTimestamps, sizes, Pi_Q0, lob0 = lob0, lob0_l3 = lob0_l3)
        spread = lobTmp[-1]['Ask_touch'][0] - lobTmp[-1]['Bid_touch'][0]
        lob0 = lobTmp[-1]
        lob0_l3 = lobL3Tmp[-1]
        #print("Snapshot LOB0: ", lob0, "\n")
        if len(list(dictTimestamps.keys())):
            Ts.append([list(dictTimestamps.keys())[0], TsTmp[-1], tau])
            lob.append(lob0)
        if (filePathName is not None)&(len(Ts)%100 == 0):   
            with open(filePathName , "wb") as f: #"/home/konajain/params/"
                pickle.dump((Ts, lob, lobL3), f)
    return Ts, lob, lobL3, thinningtime

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
            # pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
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
        #print("Lob0_l3 post init: ", lob0_l3)
    if len(dictTimestamps) == 0:
        return T, [lob0], [lob0_l3]

    dfs = []
    for event in dictTimestamps.keys():
        sizes_e = sizes[event]
        timestamps_e = dictTimestamps[event]
        dfs += [pd.DataFrame({"event" : len(timestamps_e)*[event], "time": timestamps_e, "size" : sizes_e})]
    dfs = pd.concat(dfs)
    dfs = dfs.sort_values("time")
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
                        print("triggered")
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
                    # pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
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
                    # pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
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
                    # pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
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






