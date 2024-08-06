from abc import ABC, abstractmethod
import numpy as np
from RLenv.Stochastic_Processes.Stochastic_Models import StochasticModel
from typing import Any, List, Dict, Optional, Tuple, ClassVar
import logging
from RLenv import logging_config
logger = logging.getLogger(__name__)
class ArrivalModel(StochasticModel):
    """ArrivalModel models the arrival of orders to the order book. It also generates an initial starting state for the limit order book.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params=params)
    
    @abstractmethod
    def get_nextarrival(self):
        pass

    @abstractmethod
    def generate_ordersize(self):
        pass
    @abstractmethod
    def generate_orders_in_queue(self):
        pass
    
    
    
class HawkesArrival(ArrivalModel):
    def __init__(self, params: Dict[str, Any], seed=1):
        """
        T: Simulation time_limit, which should be passed on by the kernel
        Params Dictionary consists of all the necessary parameters for the Hawkes arrival model: 
            parameters:"kernelparams", "tod", "Pis", "beta", "avgSpread", "spread", "price0", "Pi_Q0"
            
        
        """
        self.cols= ["lo_deep_Ask", "co_deep_Ask", "lo_top_Ask","co_top_Ask", "mo_Ask", "lo_inspread_Ask" ,
            "lo_inspread_Bid" , "mo_Bid", "co_top_Bid", "lo_top_Bid", "co_deep_Bid","lo_deep_Bid" ]
        self.coltolevel={0: "Ask_L2",
                         1: "Ask_L2",
                         2: "Ask_L1",
                         3: "Ask_L1",
                         4: "Ask_MO",
                         5: "Ask_inspread",
                         6: "Bid_inspread",
                         7: "Bid_MO",
                         8: "Bid_L1",
                         9: "Bid_L1",
                         10: "Bid_L2",
                         11: "Bid_L2"}
        required_keys=["kernelparams", "tod", "Pis", "beta", "avgSpread", "spread0", "price0", "Pi_Q0"]
        missing_keys = [key for key in required_keys if (key not in params.keys()) or params.get(key) is None]
        if len(missing_keys)>0:
            defaultparams=self.generatefakeparams()
            missing_with_defaults={key: defaultparams[key] for key in missing_keys}
            logger.info(f"Missing Hawkes Arrival model parameters: {', '.join(missing_keys)}. Assuming generating default values: {missing_with_defaults}")
            params.update(missing_with_defaults)
        else:
            pass
        self.num_nodes=len(self.cols)  
        super().__init__(params=params)
        
        #Initializing storage variables for thinning simulation    
        self.todmult=None
        self.baselines=None
        self.s=None   
        self.n = None
        self.Ts=None
        self.timeseries=None
        self.lamb = None
        self.left=None
        self.pointcount=0
        
    def generatefakeparams(self):
        """
        Generates all the necessary default/fake parameters for the Hawkes Simulation model. 
        """
        Pi_Q0= {'Ask_L1': [0.0018287411983379015,
                            [(1, 0.007050802017724003),
                            (10, 0.009434048841996959),
                            (100, 0.20149407216104853),
                            (500, 0.054411455742183645),
                            (1000, 0.01605198687975892)]],
                'Ask_L2': [0.001229380704944344,
                            [(1, 0.0),
                            (10, 0.0005240951083719349),
                            (100, 0.03136813097471952),
                            (500, 0.06869444491232923),
                            (1000, 0.04298980350337664)]],
                'Bid_L1': [0.0018287411983379015,
                            [(1, 0.007050802017724003),
                            (10, 0.009434048841996959),
                            (100, 0.20149407216104853),
                            (500, 0.054411455742183645),
                            (1000, 0.01605198687975892)]],
                'Bid_L2': [0.001229380704944344,
                            [(1, 0.0),
                            (10, 0.0005240951083719349),
                            (100, 0.03136813097471952),
                            (500, 0.06869444491232923),
                            (1000, 0.04298980350337664)]]}
        defaultparams={"Pis": None, "beta": 0.7479, "avgSpread": 0.0169, "spread": 3, "price0": 260, "Pi_Q0": Pi_Q0}
        #generating faketod
        faketod = {}
        for k in self.cols:
            faketod[k] = {}
            for k1 in np.arange(13):
                faketod[k][k1] = 1.0
        tod=np.zeros(shape=(len(self.cols), 13))
        for i in range(len(self.cols)):
            tod[i]=[faketod[self.cols[i]][k] for k in range(13)]
        defaultparams["tod": tod]
        
        #generating fakekernelparams        
        mat = np.zeros((12,12))
        for i in range(12):
            mat[i][i] = .6
        for i in range(12):
            for j in range(12):
                if i == j: continue
                mat[i][j] = np.random.choice([1,-1])*mat[i][i]*np.exp(-.75*np.abs(j-i))
                
        kernelparamsfake = {}
        for i in range(12):
            kernelparamsfake[self.cols[i]] = 0.1*np.random.choice([0.3,0.4,0.5,0.6,0.7])
            for j in range(12):
                maxTOD = np.max(list(faketod[self.cols[j]].values()))
                beta = np.random.choice([1.5,1.6,1.7,1.8,1.9])
                gamma = (1+np.random.rand())*5e3
                alpha = np.abs(mat[i][j])*gamma*(beta-1)/maxTOD
                kernelparamsfake[self.cols[i]+"->"+self.cols[j]] = (np.sign(mat[i][j]), np.array([alpha, beta, gamma]))
        
        
        baselines=np.zeros(shape=(len(self.cols), 1)) #vectorising baselines
        for i in range(len(self.cols)):
            baselines[i]=kernelparamsfake[self.cols[i]]
        
        #kernelparamsfake=[mask, alpha, beta, gamma] where each is a 12x12 matrix
        mask, alpha, beta, gamma=[np.zeros(shape=(12, 12)) for _ in range(4)]
        for i in range(len(self.cols)):
            for j in range(len(self.cols)):
                kernelParams = kernelparamsfake.get(self.cols[i] + "->" + self.cols[j], None)
                mask[i][j]=kernelParams[0]
                alpha[i][j]=kernelParams[1][0]
                beta[i][j]=kernelParams[1][1]
                gamma[i][j]=kernelParams[1][2]
        fakekernelparams=[mask, alpha, beta, gamma] 
        defaultparams["kernelparams": fakekernelparams]
        return defaultparams  
    
    def generate_ordersize(self, loblevel) -> int:
        """
        generate ordersize based on diracdeltas
        loblevel: one of the 4 possible levels in the LOB
        Returns: qsize the size of the order
        """
        try:
            pi= self.Pi_Q0[loblevel]
        except KeyError:
            raise KeyError(f"LOB level {loblevel} not provided in PI_Q0s of arrival model")
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
        
        total=self.generate_ordersize(loblevel)
        tmp=(numorders-1)*[np.floor(total/numorders)]
        if tmp!=0:
            queue=[total-sum(tmp)]+tmp
        else:
            queue=[total]
        return queue
            
        # return queue
        
    
    def thinningOgataIS2(self, T=None):
        """ 
        Simulates a single point via thinningogata
        Arguments:
        T: timelimit of simulation process
        Returns:
        pointcount: counts the number of new points that have been added to the timeseries
        """
        if self.n is None: self.n = self.num_nodes*[0]
        if self.baselines is None: self.baselines=self.num_nodes*[0]
        if self.Ts is None: self.Ts = self.num_nodes*[()]
        if self.spread is None: self.spread = 1
        """Setting up thinningOgata params"""
        self.baselines =self.kernelparams[1].copy()
        mat = np.zeros((self.num_nodes, self.num_nodes))
        if self.s is None: 
            self.s = 0
        hourIndex = min(12,int(self.s//1800)) #1800 seconds in 30 minutes
        self.todmult=self.tod[:, hourIndex].reshape((12, 1))
        #todMult*kernelParams[0]*kernelParams[1][0]/((-1 + kernelParams[1][1])*kernelParams[1][2]) 
        mat=self.todmult*self.kernelparams[0][0]*self.kernelparams[0][1]/((self.kernelparams[0][2]-1) *self.kernelparams[0][3])
        self.baselines[5] = ((self.spread/self.avgSpread)**self.beta)*self.baselines[5]
        self.baselines[6] = ((self.spread/self.avgSpread)**self.beta)*self.baselines[6]
        specRad = np.max(np.linalg.eig(mat)[0])
        #print("spectral radius = ", specRad)
        specRad = np.max(np.linalg.eig(mat)[0]).real
        if specRad < 1 : specRad = 0.99 #  # dont change actual specRad if already good
        
        """calculating initial values of lamb_bar"""
        if self.lamb is None:
            decays=(0.99/specRad) * self.todmult * self.baselines
            self.lamb=np.sum(decays) #3.04895025
        if self.left is None:
            self.left=0
        if self.timeseries is None:
            self.timeseries=[]
            
        """simulate a point"""
        pointcount=0
        while self.s<T:
            """Assign lamb_bar"""
            lamb_bar=self.lamb 
            #print("lamb_bar: ", lamb_bar)
            """generate random u"""
            u=np.random.uniform(0, 1)
            if lamb_bar==0:
                s+=0.1  # wait for some time
            else:
                w=max(1e-7, -1 * np.log(u)/lamb_bar) # floor at 0.1 microsec
                s+=w  
            """Recalculating baseline lambdas sum with new candidate"""
            hourIndex = min(12,int(self.s//1800))
            self.todmult=self.tod[:, hourIndex].reshape((12, 1)) * (0.99/specRad)
            decays=self.todmult * self.baselines
            """Summing cross excitations for previous points"""
            if self.timeseries==[]:
                pass
            else:
                while self.left<len(self.timeseries):
                    #print("Iterating: s: ", s, ", timestamp: ", timeseries[left][0])
                    if self.s-self.timeseries[self.left][0]>=10:
                        self.left+=1 
                    else:
                        break
                for point in self.timeseries[self.left:]: #point is of the form(s, k)
                    kern=powerLawCutoff(time=self.s-point[0], alpha=self.kernelparams[0][0][point[1]]*self.kernelparams[0][1][point[1]], beta=self.kernelparams[0][2][point[1]], gamma=self.kernelparams[0][3][point[1]])
                    kern=kern.reshape((12, 1))
                    decays+=self.todmult*kern
            #print(decays.shape) #should be (12, 1)
            decays=np.maximum(decays, 0)
            decays[5] = ((self.spread/self.avgSpread)**self.beta)*decays[5]
            decays[6] = ((self.spread/self.avgSpread)**self.beta)*decays[6]
            if 100*np.round(self.spread, 2)  < 2 : 
                decays[5] = decays[6] = 0
            self.lamb = float(sum(decays))
            #print("LAMBDA: ", lamb)
            
            """Testing candidate point"""
            D=np.random.uniform(0, 1)
            #print("Candidate D: ", D)
            if D*lamb_bar<=self.lamb:
                pointcount+=1
                """Accepted so assign candidate point to a process by a ratio of intensities"""
                k=0
                total=decays[k]
                while D*lamb_bar >= total:
                    k+=1
                    total+=decays[k]
                """dimension is cols[k]"""   
                """Update values of lambda for next simulation loop and append point to Ts"""
                if k in [5, 6]:
                    self.spread=self.spread-0.01      
                
                """Precalc next value of lambda_bar"""    
                newdecays=self.todmult * self.kernelparams[0][0][k].reshape(12, 1)*self.kernelparams[0][1][k].reshape(12, 1)
                newdecays=np.maximum(newdecays, 0)
                newdecays[5] = ((self.spread/self.avgSpread)**self.beta)*newdecays[5]
                newdecays[6] = ((self.spread/self.avgSpread)**self.beta)*newdecays[6]
                if 100*np.round(self.spread, 2) < 2 : newdecays[5] = newdecays[6] = 0
                self.lamb+= sum(newdecays)
                self.lamb=self.lamb[0]


                if len(self.Ts[k]) > 0:
                    T_Minus1 = self.Ts[k][-1]
                else:
                    T_Minus1 = 0
                decays = np.array(self.baselines.copy())
                decays[5] = ((self.spread/self.avgSpread)**self.beta)*decays[5]
                decays[6] = ((self.spread/self.avgSpread)**self.beta)*decays[6]
                decays = decays*(self.s-T_Minus1)
                
                """Updating history and returns"""
                self.Ts[k]+=(s,)
                tau = decays[k][0]
                self.n[k]+=1
                self.timeseries.append((s, k)) #(time, event)
                self.pointcount+=pointcount
                return pointcount
                
    def orderwrapper(self, time:float, k: int, size: int):
        """
        Takes in an event tuple of (time, event) and wraps it into information compatible with the exchange
        """
        side=None
        if "Ask" in self.cols[k]:
            side="Ask"
        else:
            side="Bid"
        order_type=None
        if "lo" in self.cols[k]:
            order_type="lo"
        elif "mo" in self.cols[k]:
            order_type="mo"
        elif "co" in self.cols[k]:
            order_type="co"
        level=self.coltolevel[k]
        return time, side, order_type, level, size
        
    def orderunwrap(self, time:float, side: str, order_type: str, level: str, size: int):
        """Takes in information about an event from the exchange and wraps it into an event tuple of (s,k)"""
        s=time
        k=-1
        if order_type=="lo":
            if level=="Ask_L1":
                k=2
            elif level=="Ask_L2":
                k=0
            elif level=="Bid_L1":
                k=9
            elif level=="Bid_L2":
                k=11
            elif level=="Bid_inspread":
                k=6
            elif level=="Ask_inspread":
                k=5
            else:
                pass
        elif order_type=="mo":
            if level=="Ask_MO":
                k=4
            elif level=="Bid_MO":
                k=7
            else:
                pass
        elif order_type=="co":
            if level=="Ask_L1":
                k=3
            elif level=="Ask_L2":
                k=1
            elif level=="Bid_L1":
                k=8
            elif level=="Bid_L2":
                k=10
            else:
                pass
        if k==-1:
            raise Exception("'Level' passed incorrectly into order unwrapper")
        return s,k
            
    
    
    def get_nextarrival(self, timelimit):
        """Returns None if no arrivals before the timelimit. Otherwise, 
        Returns: time, side, order_type, level, size"""
        tmp=self.thinningOgataIS2()
        if tmp==0:
            return None
        #Newest point
        t, k=self.timeseries[-1][0], self.timeseries[-1][1]
        size=self.generate_ordersize[self.colstolevel[k]]
        return self.orderwrapper(time=t, k=k, size=size)
        
            
    #Concrete implementations of Abstract Base Class Methods
    def update(self, time:float, side: str, order_type: str, level: str, size: int):
        """
        Update values of lambda for next simulation loop and append point to Ts
        Arguments: spread, (s=time of latest point, k=event, s=size) tuple
        Variables that need updating:
        spread, s, n, Ts, timeseries, left
        
        """
        """Update values of lambda for next simulation loop and append point to Ts"""
        s, k=self.orderunwrap(time=time, side=side, order_type=order_type, level=level, size=size)
        if k in [5, 6]:
            self.spread=self.spread-0.01      
        
        """Precalc next value of lambda_bar"""  
        hourIndex = min(12,int(self.s//1800)) #1800 seconds in 30 minutes
        self.todmult=self.tod[:, hourIndex].reshape((12, 1))  
        newdecays=self.todmult * self.kernelparams[0][0][k].reshape(12, 1)*self.kernelparams[0][1][k].reshape(12, 1)
        newdecays=np.maximum(newdecays, 0)
        newdecays[5] = ((self.spread/self.avgSpread)**self.beta)*newdecays[5]
        newdecays[6] = ((self.spread/self.avgSpread)**self.beta)*newdecays[6]
        if 100*np.round(self.spread, 2) < 2 : newdecays[5] = newdecays[6] = 0
        self.lamb+= sum(newdecays)
        self.lamb=self.lamb[0]


        if len(self.Ts[k]) > 0:
            T_Minus1 = self.Ts[k][-1]
        else:
            T_Minus1 = 0
        decays = np.array(self.baselines.copy())
        decays[5] = ((self.spread/self.avgSpread)**self.beta)*decays[5]
        decays[6] = ((self.spread/self.avgSpread)**self.beta)*decays[6]
        decays = decays*(self.s-T_Minus1)
        
        """Updating history and returns"""
        self.Ts[k]+=(s,)
        tau = decays[k][0]
        self.n[k]+=1
        self.timeseries.append((s, k)) #(time, event)
        #print("point added")
        
        
        return
    
    def reset(self, params={}):
        #Variables to reset
        self.__init__(params=params)
    
    def seed(self):
        return super().seed()

def powerLawCutoff(time, alpha, beta, gamma):
    # alpha = a*beta*(gamma - 1)
    return alpha/((1 + gamma*time)**beta)