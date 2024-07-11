from abc import ABC, abstractmethod
import numpy as np
class StochasticModel(ABC):
    def __init__(self, params, seed, T=100):
        self._params=params
        self.seed=seed
        self.timelimit=T
        
    @abstractmethod
    def simulate(self, T=100, params=None, vars=None):
        pass
    
    @abstractmethod
    def reset(self):
        pass
        
    @property
    @abstractmethod
    def params(self):
        pass
    @params.setter
    def params(self, data):
        self._params = data
        
    
    

print("yes")