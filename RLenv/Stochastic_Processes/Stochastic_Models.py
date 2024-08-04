from abc import ABC, abstractmethod
import numpy as np
from typing import *
class StochasticModel(ABC):
    def __init__(self, params: Dict[str, Any], seed=1):
        #Params is a dictionary of parameters
        for key, value in params.items():
            setattr(self, key, value)
        self.seed=seed
        
    
    @abstractmethod
    def reset(self):
        pass
    @abstractmethod
    def update(self, *args, **kwargs):
        pass
    @abstractmethod
    def seed(self):
        pass
    
    
        
    