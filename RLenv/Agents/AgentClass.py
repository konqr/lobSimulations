import numpy as np
from abc import ABC, abstractmethod 
class Agent:
    
    def __init__(self, id: int, startingcash: int=10000, seed=1) -> None:
        self.id=id
        self.cash=startingcash
        self.seed=seed
        self.resetseed(self.seed)
        
    @abstractmethod    
    def get_action(self, observations):
        pass
          
    @abstractmethod        
    def update_state(self, kernelmessage): #update internal agentstate given a kernel message
        pass
    
    @abstractmethod
    def resetseed(self, seed):
        pass
        
    
        
