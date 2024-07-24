from hawkes import simulate_optimized
import pickle
import pandas as pd
import time
import numpy as np
import os

num_nodes = 12

def simulate_smallTick(T , paramsPath , todPath, s0 = None, filePathName = None, Pis = None, Pi_Q0 = None, Pi_M0 = None, Pi_eta = None, beta = 0.7479, avgSpread = 0.0169, spread0 = 3, price0 = 260, M_med = 100):
    """
    :param T: time limit of simulations
    :param paramsPath: path of fitted params
    :param Pis: distribution of order sizes
    :param Pi_Q0: depleted queue size distribution
    """
    tod, params= simulate_optimized.preprocessdata(paramsPath=paramsPath, todPath=todPath)
    cols = ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]

    # TODO: move these to a file and read from there.
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
        Pis["co_top_Ask"] = Pis["lo_top_Ask"]
        Pis["co_top_Bid"] = Pis["lo_top_Bid"]
        Pis["lo_inspread_Ask"] = Pis["lo_inspread_Bid"]
        Pis["lo_deep_Ask"] = Pis["lo_deep_Bid"]
        Pis["co_deep_Ask"] = Pis["lo_deep_Ask"]
        Pis["co_deep_Bid"] = Pis["lo_deep_Bid"]
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
    if Pi_M0 == None:
        Pi_M0 = {'m_T': 0.1,
                 'm_D': 0.2}
    if Pi_eta == None:
        Pi_eta = {'eta_T' : 1,
                  'eta_IS' : 1,
                  'eta_T+1': 2}
    if s0 is None:
        s = 0
    else:
        s = s0
    Ts,lob = [],[]
    _, lob0 = createLOB_smallTick({}, {}, Pi_Q0, Pi_M0, Pi_eta, priceMid0 = price0, spread0 = spread0, ticksize = 0.01, lob0 = {}, M_med = M_med)
    #print("The initial LOB: lob0", lob0, "lob0_l3", lob0_l3)
    Ts.append(0)
    lob.append(lob0[-1])
    spread = lob0[0]['Ask_touch'][0] - lob0[0]['Bid_touch'][0]
    print("initial spread: ", spread, "\n")
    n = None
    timestamps = None
    timeseries=None
    lob0 = lob0[0]
    lamb = None
    trials=1
    left=None
    thinningtime=0
    while s <= T:
        start=time.perf_counter_ns()
        s, n, timestamps, tau, lamb, timeseries, left= simulate_optimized.thinningOgataIS2(T, params, tod, num_nodes=num_nodes, maxJumps = 1, s = s, n = n, Ts = timestamps, timeseries=timeseries, spread=spread, beta = beta, avgSpread = avgSpread,lamb= lamb, left=left)
        timestamps_this=[()]*num_nodes
        timestamps_this[timeseries[-1][1]]=(timeseries[-1][0],)
        end=time.perf_counter_ns()
        thinningtime+=abs(end-start)
        trials +=1
        sizes = {}
        dictTimestamps = {}
        for t, col in zip(timestamps_this, cols):
            if len(t) == 0: continue

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
        print("Sizes is: ", sizes)
        TsTmp, lobTmp = createLOB_smallTick(dictTimestamps, sizes, Pi_Q0, Pi_M0, Pi_eta, lob0 = lob0, M_med = M_med)
        spread = lobTmp[-1]['Ask_touch'][0] - lobTmp[-1]['Bid_touch'][0]
        lob0 = lobTmp[-1]
        print("Snapshot LOB0: ", lob0, "\n")
        if len(list(dictTimestamps.keys())):
            Ts.append([list(dictTimestamps.keys())[0], TsTmp[-1], tau])
            lob.append(lob0)
        if (filePathName is not None)&(len(Ts)%100 == 0):
            with open(filePathName , "wb") as f: #"/home/konajain/params/"
                pickle.dump((Ts, lob), f)
    return Ts, lob, thinningtime

def sampleGeometric(pi, maxWidth = 100):
    #geometric
    pi = np.array([pi*(1-pi)**k for k in range(1,maxWidth)])
    pi = pi/sum(pi)
    cdf = np.cumsum(pi)
    a = np.random.uniform(0, 1)
    width = np.argmax(cdf>=a) + 1
    return width

def sampleGeometricWithSpikes(p, dd, maxWidth = 100000):
    pi = np.array([p*(1-p)**k for k in range(1,maxWidth)])
    # pi = pi*(1-sum([d[1] for d in dd]))/sum(pi)
    for i, p_i in dd:
        pi[i-1] = p_i + pi[i-1]
    pi = pi/sum(pi)
    cdf = np.cumsum(pi)
    a = np.random.uniform(0, 1)
    qSize = np.argmax(cdf>=a) + 1
    return qSize

def partition(q, new_width, original_width):
    return np.round(q*new_width/original_width, decimals=0)

def createLOB_smallTick(dictTimestamps, sizes, Pi_Q0, Pi_M0, Pi_eta, priceMid0 = 260, spread0 = 4, ticksize = 0.01, lob0 = {}, M_med = 100):
    lob = []
    T = []
    levels = ["Ask_deep", "Ask_touch", "Bid_touch", "Bid_deep"]

    # Init the state 0
    if len(lob0) == 0:
        # init prices at touch
        lob0['mid'] = priceMid0
        lob0['Ask_touch'] = (priceMid0 + np.floor(spread0/2)*ticksize, 0)
        lob0['Bid_touch'] = (priceMid0 - np.ceil(spread0/2)*ticksize, 0)
        # init touch widths
        for side in ['Ask','Bid']:
            for k, pi in Pi_M0.items():
                #geometric
                lob0[side+'_'+k] = sampleGeometric(pi)
        # init prices at deep
        lob0['Ask_deep'] = (priceMid0 + np.floor(spread0/2)*ticksize + lob0['Ask_m_T']*ticksize, 0)
        lob0['Bid_deep'] = (priceMid0 - np.ceil(spread0/2)*ticksize - lob0['Bid_m_T']*ticksize, 0)
        # init queue sizes
        for k, pi in Pi_Q0.items():
            #geometric + dirac deltas; pi = (p, diracdeltas(i,p_i))
            p = pi[0]
            dd = pi[1]
            qSize = sampleGeometricWithSpikes(p,dd)
            mult = 1
            if 'deep' in k:
                mult = lob0[k.split('_')[0]+'_m_D']
            lob0[k] = (lob0[k][0], qSize*mult)
    if len(dictTimestamps) == 0:
        return T, [lob0]

    # build the LOB
    dfs = []
    for event in dictTimestamps.keys():
        sizes_e = sizes[event]
        timestamps_e = dictTimestamps[event]
        dfs += [pd.DataFrame({"event" : len(timestamps_e)*[event], "time": timestamps_e, "size" : sizes_e})]
    dfs = pd.concat(dfs)
    dfs = dfs.sort_values("time")
    lob.append(lob0.copy())
    T.append(0)
    for i in range(len(dfs)):
        r = dfs.iloc[i]
        lobNew = lob[i].copy()
        T.append(r.time)
        if "Ask" in r.event :
            side = "Ask"
            antiside = 'Bid'
            sgn = 1
        else:
            side = "Bid"
            antiside = 'Ask'
            sgn = -1
        if "lo" in r.event:
            if "deep" in r.event:
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0], lobNew[side + "_deep"][1] + r['size'])
            elif "top" in r.event:
                eta_T_hat = Pi_eta['eta_T']
                # distance from current top to the new order
                eta_T = min([lobNew[side+'_m_T'], sampleGeometric(eta_T_hat)]) - 1
                if eta_T != 0:
                    # order in between top and top +1
                    lobNew[side+'_m_D'] = lobNew[side+'_m_D'] + lobNew[side+'_m_T'] - eta_T
                    lobNew[side+'_m_T'] = eta_T
                    lobNew[side + "_deep"] = (lobNew[side + "_deep"][0] - sgn*ticksize*lob0[side+'_m_T'] + sgn*ticksize*eta_T, lobNew[side + "_deep"][1] + r['size'])
                else:
                    # order at top
                    lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] + r['size'])
            else: # inspread
                eta_IS_hat = Pi_eta['eta_IS']
                # distance from current top to the new order
                spread = np.round(100*(lobNew['Ask_touch'][0] - lobNew['Bid_touch'][0]), decimals=0)
                eta_IS = min([spread - 1, sampleGeometric(eta_IS_hat)])
                totalDepth = lobNew[side+'_m_T'] + lobNew[side+'_m_D'] + spread*0.5
                orig_m_T  = lobNew[side+'_m_T']
                lobNew[side+'_m_T'] = eta_IS
                lobNew[side+'_touch'] = (lobNew[side + "_touch"][0] - sgn*ticksize*eta_IS, r['size'])
                lobNew['mid'] = np.round(0.5*(lobNew[side+'_touch'][0] + lobNew[antiside+'_touch'][0]), decimals=2)
                spread = np.round(100*(lobNew['Ask_touch'][0] - lobNew['Bid_touch'][0]), decimals=0)
                # if totalDepth + eta_IS*0.5 <= M_med:
                #     lobNew[side+'_m_D'] += orig_m_T
                #     lobNew[side + "_deep"] = (lobNew[side + "_deep"][0] - sgn*ticksize*orig_m_T, lobNew[side + "_deep"][1] + lob0[side + "_touch"][1])
                # else:
                #     deletedWidth = min([lobNew[side+'_m_D'] ,lobNew[side+'_m_D'] - M_med + np.round(0.5*spread, decimals=0) + eta_IS + lob0[side+'_m_T']])
                #     deltaQ_D = partition(lobNew[side + "_deep"][1], deletedWidth, lobNew[side+'_m_D'])
                #     lobNew[side+'_m_D'] = M_med - np.round(0.5*spread, decimals=0) - lobNew[side+'_m_T']
                #     lobNew[side + "_deep"] = (lobNew[side + "_touch"][0] + sgn*ticksize*lobNew[side+'_m_T'], lobNew[side + "_deep"][1] + lob0[side + "_touch"][1] - deltaQ_D)
                lobNew[side+'_m_D'] += orig_m_T
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0] - sgn*ticksize*orig_m_T, lobNew[side + "_deep"][1] + lob0[side + "_touch"][1])
        elif 'co_deep' in r.event:
            lobNew[side + "_deep"] = (lobNew[side + "_deep"][0], max([0, lobNew[side + "_deep"][1] - r['size']]))
            spread = np.round(100*(lobNew['Ask_touch'][0] - lobNew['Bid_touch'][0]), decimals=0)
            totalDepth = lobNew[side+'_m_T'] + lobNew[side+'_m_D'] + np.round(0.5*spread, decimals=0)
            if (totalDepth <= M_med) and (lobNew[side + "_deep"][1] <= 0): # go deeper
                width = min([M_med - totalDepth, sampleGeometric(Pi_M0['m_D'])])
                p, dd = Pi_Q0[side+'_deep']
                qSize = sampleGeometricWithSpikes(p,dd)
                lobNew[side + "_deep"] = (lobNew[side + "_deep"][0], width*qSize )
                lobNew[side+'_m_D'] = width
            lobNew['mid'] = np.round(0.5*(lobNew[side+'_touch'][0] + lobNew[antiside+'_touch'][0]), decimals=2)
        else: # mo or co_top
            lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], lobNew[side + "_touch"][1] - r['size'])
            while lobNew[side + "_touch"][1] <= 0: # queue depletion
                if 'co' in r.event: lobNew[side + "_touch"] = (lobNew[side + "_touch"][0], 0) # CO size cant be more than existing queue size
                extraVolume = -1*lobNew[side + "_touch"][1]
                eta_Tp1_hat = Pi_eta['eta_T+1']
                eta_Tp1 = min([lobNew[side+'_m_D'] , sampleGeometric(eta_Tp1_hat)])
                Q_T = partition(lobNew[side + "_deep"][1], eta_Tp1 , lobNew[side+'_m_D'])
                lobNew[side + "_touch"] = (lobNew[side + "_deep"][0], Q_T - extraVolume)
                lobNew[side+'_m_T'] = eta_Tp1
                lobNew[side+'_m_D'] = lobNew[side+'_m_D'] - eta_Tp1
                lobNew[side + "_deep"] = (lobNew[side + "_touch"][0] + ticksize*sgn*lobNew[side+'_m_T'], lobNew[side + "_deep"][1] - Q_T )
                spread = np.round(100*(lobNew['Ask_touch'][0] - lobNew['Bid_touch'][0]), decimals=0)
                totalDepth = lobNew[side+'_m_T'] + lobNew[side+'_m_D'] + np.round(0.5*spread, decimals=0)
                if (totalDepth <= M_med) and (lobNew[side + "_deep"][1] == 0): # go deeper
                    width = min([M_med - totalDepth, sampleGeometric(Pi_M0['m_D'])])
                    p, dd = Pi_Q0[side+'_deep']
                    qSize = sampleGeometricWithSpikes(p,dd)
                    lobNew[side + "_deep"] = (lobNew[side + "_deep"][0] , width*qSize )
                    lobNew[side+'_m_D'] = width
                lobNew['mid'] = np.round(0.5*(lobNew[side+'_touch'][0] + lobNew[antiside+'_touch'][0]), decimals=2)
        for k in ['Ask_touch','Bid_touch','Ask_deep','Bid_deep']:
            lobNew[k] = (np.round(lobNew[k][0],decimals=2), lobNew[k][1])
        lob.append(lobNew.copy())

    return T, lob

def main():
    simulate_smallTick(1000, "C:\\Users\\konar\IdeaProjects\lobSimulations\\fake_ParamsInferredWCutoff_sod_eod_true","C:\\Users\\konar\IdeaProjects\lobSimulations\\fakeData_Params_sod_eod_dictTOD_constt" , beta = 0., avgSpread = .50, spread0 = 110, price0 = 450, M_med = 50)

# main()