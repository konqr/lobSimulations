from abc import ABC, abstractmethod
import numpy as np
class StochasticModel(ABC):
    def __init__(self, params, seed=1):
        self._params=params
        self.seed=seed
        np.random.seed(self.seed)
    
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