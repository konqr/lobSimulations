import numpy as np

class ASLOB():
    def __init__(self, type='poisson', params=None):
        self.MO_rates_type = type
        self.pp_params = {'mo_Ask' : 5, 'mo_Bid': 5}
        self.p_QD = 0.1
        if params is not None:
            self.pp_params = params['pp_params']
            self.p_QD = params['p_QD']

    def thinningOgata(self, T, params, num_nodes = 2, maxJumps = None):
        if maxJumps is None: maxJumps = np.inf

        cols = [ "mo_Ask", "mo_Bid" ]
        baselines = num_nodes*[0]
        for i in range(num_nodes):
            baselines[i] = params[cols[i]]
        s = 0
        numJumps = 0
        n = num_nodes*[0]
        Ts = num_nodes*[()]
        lamb = sum(baselines)
        while s <= T:
            lambBar = lamb
            u = np.random.uniform(0,1)
            w = -1*np.log(u)/lambBar
            s += w
            decays = baselines.copy()
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

    def step(self, action):
        return

    def reset(self):
        return

    def getobservation(self):
        return

    def calculaterewards(self):
        return
