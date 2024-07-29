import logging
import os
import datetime
import numpy as np
import pandas as pd
from RLenv.SimulationEntities.Entity import Entity
from RLenv.SimulationEntities.TradingAgent import TradingAgent
from RLenv.SimulationEntities.Exchange import Exchange
from RLenv.Messages.Message import Message, WakeupMsg
from RLenv.Messages.AgentMessages import AgentMsg
from RLenv.Messages.ExchangeMessages import ExchangeMsg
from typing import Any, Dict, List, Optional, Tuple, Type

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# trial=Message()
# print("helloworld")
# print(type(trial))
# print(trial.message_id)

class Kernel:
    """
    Simulation Kernel
    
    Arguments:
        agents: list of trading agents
        exchange: which exchange do these agents participate in
        seed: seed of the simulation
        kernel_name: name of the kernel
        stop_time: simulation world time limit in seconds, not wall-time
        wall_time_limit: stop the simulation in wall-time  (in seconds)
        log_to_file: Boolean flag to store record of log
        parameters: simulation parameters, to be implemented
    
    Note that the simulation timefloor is in microseconds
    
    """
    def __init__(self, agents: List[TradingAgent], exchange: Exchange, seed: int=1, kernel_name: str="Alpha", stop_time: int=100, wall_time_limit: int=600, log_to_file: bool=True, parameters: Dict[str, any]=None) -> None:
        self.kernel_name=kernel_name
        self.agents: List[TradingAgent]=agents
        assert len(agents)>0, f"Number of agents must be more than 0" 
        self.exchange: Exchange=exchange
        assert exchange, f"Expected a valid exchange but received None"
        #Create Dictionary of entities by ID and link entity kernels to self
        self.entity_registry: Dict[int, Entity]={x.id: x for x in self.agents}
        self.entity_registry[exchange.id]=self.exchange
        for value in self.entity_registry.values():
            value.kernel=self
        #Global seed:
        self.seed: Optional[int]=seed
        
        #Should the kernel write to a file
        self.log_to_file: bool=log_to_file
        
        
        #Time Variables
        self.start_time: int=0 #Begin simulation time at t=0 microseconds
        self.current_time: int =self.start_time 
        self.stop_time: int=stop_time * 10**6 #Set stop time in microseoncds
        self.wall_time_limit=wall_time_limit #in seconds
        
        #The simulation times of the different entities, used to keep track of whose turn it is
        self.agents_current_times: Dict[int, any] ={j.id: self.start_time for j in self.agents}
        self.exchange_time: int=self.start_time
        
        #Implementation of message queue
        #An item in the queue takes the form of (time, (senderID, recipientID, Message))
        self.queue: List[Tuple[int, Tuple[Optional[int], int, Message]]] =[]
        self.head=0
        
        if parameters:
            for key, value in parameters.items():
                setattr(self, key, value)
        logger.debug("Kernel Initialized")
    
    
    #Functions that run the simulation
    def runner(self):
        """
        Wrapper to run the entire simulation (when not running in the tradingEnv).

        3 Steps:
          - Simulation Begin
          - Simulation Run
          - Simulation Terminate

        Returns:
            An object that contains all the objects at the end of the simulation.
        """
        self.begin()

        self.run()

        return self.terminate()
    
    def begin(self) -> None:
        """This initalizes the simulation once all the agent configuration is completed"""
        pass
    
    def run(self) -> None:
        """
        Start the simulation and processing of the message queue
        Returns:
        Done: Boolean True if the simulation is done, else false
        Results: the raw state of the simulation, containing data which can be formated for using in the GYM environment. 
        """
        
        #While there are still items in the queue or time limit is not up yet and simulation has started, process a message.
        while (self.head < len(self.queue)) and (self.current_time<=self.stop_time) and self.current_time:  
            t, event=self.queue[self.head]
            senderID, recipientID, message=event
            #update new current_time
            self.current_time=t
           
                    
            
            
            
            
            

    
    def terminate(self)-> None:
        """
        Called when the simulation is terminated. Called once the queue is empty, or the gym environement is done, or the simulation reached either of the kernel stop time or wall_time limits.
        Returns: object that contains all relevant information from simulation
        """
    
    def reset(self) -> None:
        """
        Reset simulation
        """
    
    #Functions for communication   
    

        
    def sendmessage(self, senderID, recipientID, message: Message, delay: int=0):
        """
        Called by sender entity to send a message to a recipient
        Arguments:
            senderID: ID of sender Entity
            recipientID: ID of recipient Entity
            message: The '''Message''' class instance to send
            delay: is in microseconds
        """
        
        if senderID in self.entity_registry.keys() and recipientID in self.entity_registry.keys():
            sender=self.entity_registry[senderID]
            recipient=self.entity_registry[recipientID]
            pass
        elif senderID not in self.entity_registry.keys():
            raise KeyError(f"{senderID} is not a valid entity")
        elif recipientID not in self.entity_registry.keys():
            raise KeyError(f"{recipientID} is not a valid entity")
        else:
            pass
        
        #Perform compatibility checks for messages
        if type(message).__name__=="AgentMsg":
            #It is a message from a trader agent to the exchange
            if isinstance(sender, TradingAgent) and isinstance(recipient, Exchange):
                #Typecheck compatible
                pass
            elif not isinstance(sender, TradingAgent):
                raise AssertionError(f"Expects Sender with Entity ID {senderID} to be a Trading Agent. Received '{type(sender).__name__}' instead")
            elif not isinstance(recipient, Exchange):
                raise AssertionError(f"Expects Recipient with Entity ID {senderID} to be an Exchange. Received '{type(recipient).__name__}' instead ")
            else:
                pass
        elif type(message).__name__=="ExchangeMsg": 
            #It is a message from an exchange to a trader agent
            if isinstance(sender, Exchange) and isinstance(recipient, TradingAgent):
                #Typecheck compatible
                pass
            elif not isinstance(recipient, TradingAgent):
                raise AssertionError(f"Expects Recipient with Entity ID {senderID} to be a Trading Agent. Received '{type(recipient).__name__}' instead")
            elif not isinstance(sender, Exchange):
                raise AssertionError(f"Expects Sender with Entity ID {senderID} to be an Exchange. Received '{type(sender).__name__}' instead ")
            else:
                pass
        
        
        item: Tuple[int, Tuple[int, int, Message]]= (self.current_time+delay, (senderID, recipientID, message))
        self.queue.append(item)
    
    def sendbatchmessage(self, senderID: int , recipientIDs: int, message: Message, delay: int=0):
        
        if senderID in self.entity_registry.keys():
            sender=self.entity_registry[senderID]
        else:
            raise KeyError(f"{senderID} is not a valid entity")
            
        recipients=[]   
        for recipientID in recipientIDs:
            if  recipientID in self.entity_registry.keys():
                recipients.append(self.entity_registry[recipientID])
            elif recipientID not in self.entity_registry.keys():
                raise KeyError(f"{recipientID} is not a valid entity")
            else:
                pass
        item: Tuple[int, Tuple[int, List[int], Message]]=(self.current_time+delay, (senderID, recipientIDs, message))
        self.queue.append(item)
        

        
    def set_wakeup(self, agentID: int, requested_time: int=None) -> None:
        """
        Called by an agent to set a specific wake-up call at a certain time in the future. I.e after tau seconds
        """
        agent = TradingAgent.get_entity_by_id(agentID)
        if agent:
            if agentID in self.entity_registry.keys():
                self.agents_current_times[agentID]=requested_time
            else:
                raise KeyError(f"Agent {agentID} is valid but is not registered in Kernel {self.kernel_name} registry")
            
        else:
            raise KeyError(f"No agent exists with ID {agentID}")
                
        
        
    def wakeup(self, agentID: int) -> None:
        #Wake a specific agent up


        if self.current_time==self.agents_current_times[agentID]:
            agent=self.entity_registry[agentID]
            agent.wakeup(self.current_time)
            
        else:
            logger.info(f"Kernel time {self.current_time} does not match Agent {agentID} intended wake-up time {self.agents_current_times[agentID]}")
            raise AssertionError(f"Kernel time {self.current_time} does not match Agent {agentID} intended wake-up time {self.agents_current_times[agentID]}")
        

        
        
    def write_log(self, senderID: int, df_log: pd.DataFrame, filename: Optional[str]):
        """
        Called by any entity at the end of the simulation. If filename is None, the Kernel
        will construct a filename based on the name of the Agent requesting log archival.

        """

        entity = Entity.get_entity_by_id(senderID)
        if entity:
            if df_log:
                if filename is None:
                    filename=f"{entity.__class__.__name__}_{senderID}_{self.name}_log.csv"
                df_log.to_csv(filename)
                print(f"Log written for {entity.__class__.__name__} with ID {senderID} to {filename}")
            else:
                raise ValueError(f"No log found for {entity.__class__.__name__} with ID: {senderID}")
        else:
            raise KeyError(f"No entity found with ID {senderID}")

    def isbatchmessage(self, item: Tuple[int, Tuple[Optional[int], Any, Message]]):
        if type(item[1][1]) is list:
            return True
        else:
            return False
            
            
    def processmessage(self):
        if self.head>=len(self.queue):
            raise IndexError("Queue is empty. Kernel should have been terminated.")
        else:
            item=self.queue[self.head]
            rtn=self.isbatchmessage(item)
            if not rtn:
                #Is a batch message
            else:
                #Is not a batch message
                message=item[1][2]
                recipientID=item[1][1]
                senderID=item[1][1]
                logger.debug(f"Processing message ")