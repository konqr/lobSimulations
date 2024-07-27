from abc import ABC, abstractmethod
import numpy as np
from RLenv.Stochastic_Processes.Stochastic_Models import StochasticModel
from hawkes import simulate_optimized as Simulate
from typing import Any, List, Optional, Tuple, ClassVar
class ArrivalModel(StochasticModel):
    """ArrivalModel models the arrival of orders to the order book. It also generates an initial starting state for the limit order book.
    """
    def __init__(params, seed, T=100):
        super().__init__(params, seed=1, T=100)
    
    @abstractmethod
    def get_nextarrival(self):
        pass
    
    @abstractmethod
    def get_nextarrivaltime(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    @abstractmethod
    def generate_ordersize(self):
        pass
    @abstractmethod
    def generate_orders_in_queue(self):
        pass
    
    def reset(self):
        #to be implemented
        pass
class HawkesArrival(ArrivalModel):
    def __init__(self, params, tod, Pis, beta, avgSpread, spread0, price0, seed=1, T=100, Pi_Q0=None):

        if Pi_Q0==None:
            self.Pi_Q0= {'Ask_touch': [0.0018287411983379015,
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
            self.Pi_Q0["Bid_touch"] = self.Pi_Q0["Ask_touch"]
            self.Pi_Q0["Bid_deep"] = self.Pi_Q0["Ask_deep"]
        else:
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

    def generate_orders_in_queue(self, loblevel, numorders=10) -> List[int]:
        """
        generate queuesize of a LOB level
        Arguments:
            loblevel: one of the 4 possible levels in the LOB
            numorders: max number of orders in the queue
        Returns: queue -- a list of orders[ints]
        """
        
        total=self.generate_orders_in_queue(loblevel)
        tmp=(numorders-1)*[np.floor(total/numorders)]
        if tmp!=0:
            queue=[total-sum(tmp)]+tmp
        else:
            queue=total
        return queue
    def get_nextarrival(self):
        """
        Returns a tuple (t, k, s) describing the next event where t is the time, k the event, and s the size
        """
        t, k=self.get_nextarrivaltime()
        s=self.generate_ordersize
        return (t, k, s)
    
    def update_model_state(self, t: float, k: int, s: int):
        print("yes")
        pass
    
    def reset(self):
        super().reset()