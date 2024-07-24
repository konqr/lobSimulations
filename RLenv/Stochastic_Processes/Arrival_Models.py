from abc import ABC, abstractmethod
import numpy as np
from RLenv.Stochastic_Processes.Stochastic_Models import StochasticModel
from hawkes import simulate_optimized as Simulate
class ArrivalModel(StochasticModel):
    """ArrivalModel models the arrival of orders to the order book. It also generates an initial starting state for the limit order book.
    """
    def __init__(params, seed, T=100):
        super().__init__(params, seed=1, T=100)
    
    @abstractmethod
    def get_nextarrival(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass


class HawkesArrival(ArrivalModel):
    def __init__(self, params, tod, Pis, Pi_Q0, beta, avgSpread, spread0, price0, seed=1, T=100):
        self.Pi_Q0=Pi_Q0
        self.beta=beta
        self.avgSpread=avgSpread
        self.spread0=spread0
        self.price0=price0
        self.T=T
        super().__init__(params, seed=1)
        
    def get_nextarrivaltime(self): #returns a tuple (t, k) where t is timestamp, k is event
        s, n, timestamps, tau, lamb, timeseries, left=Simulate.thinningOgataIS2(T, params, tod, num_nodes=num_nodes, maxJumps = 1, s = s, n = n, Ts = timestamps, timeseries=timeseries, spread=spread, beta = beta, avgSpread = avgSpread,lamb= lamb, left=left)
        
        return timeseries[-1]
    
    
    
    def generate_ordersize(self, loblevel) -> int:
        """
        generate ordersize based on diracdeltas
        loblevel: one of the 4 possible levels in the LOB
        Returns: qsize the size of the order
        """
        pi= self.Pi_Q0[loblevel]
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
        return qSize


    def get_nextarrival(self):
        """
        Returns a tuple (t, k, s) describing the next event where t is the time, k the event, and s the size
        """
        t, k=self.get_nextarrivaltime()
        s=self.generate_ordersize
        return (t, k, s)
    
    def reset(self):
        return super().reset()