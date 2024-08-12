from abc import ABC, abstractmethod
import numpy as np
from typing import *
class StochasticModel(ABC):      
    @abstractmethod
    def reset(self):
        pass
    @abstractmethod
    def update(self, *args, **kwargs):
        pass
    @abstractmethod
    def seed(self):
        pass
    
    
        
    