import logging
import os
import datetime
import numpy as np
import pandas as pd
from src.SimulationEntities.Entity import Entity
from src.SimulationEntities.TradingAgent import TradingAgent
from src.SimulationEntities.GymTradingAgent import GymTradingAgent
from src.SimulationEntities.Exchange import Exchange
from src.Messages.Message import *
from src.Messages.AgentMessages import *
from src.Messages.ExchangeMessages import *
from src.Utils.Exceptions import *
from typing import Any, Dict, List, Optional, Tuple

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
    def __init__(self, agents: List[TradingAgent], exchange: Exchange, seed: int=1, kernel_name: str="Alpha", stop_time: int=100, wall_time_limit: int=600, log_to_file: bool=True, **parameters) -> None:
        self.kernel_name=kernel_name
        self.agents: List[TradingAgent]=agents
        self.gymagents= [agent for agent in self.agents if isinstance(agent, GymTradingAgent)]
        assert len(self.gymagents==1), f"This Kernel is currently incompatible with more than one Gym Agent"
        assert len(agents)>0, f"Number of agents must be more than 0" 
        self.exchange: Exchange=exchange
        assert exchange, f"Expected a valid exchange but received None"
        #Create Dictionary of entities by ID and link entity kernels to self
        self.entity_registry: Dict[int, Entity]={x.id: x for x in self.agents}
        self.entity_registry[self.exchange.id]=self.exchange
        for value in self.entity_registry.values():
            value.kernel=self
        #Global seed:
        self.seed: Optional[int]=seed
        
        #Should the kernel write to a file
        self.log_to_file: bool=log_to_file
        
        
        #Time Variables
        self.start_time: int=0 #Begin simulation time at t=0 microseconds
        self.current_time: int =0
        self.stop_time: int=stop_time * 10**6 #Set stop time in microseconds
        self.wall_time_limit=wall_time_limit #in seconds
        self.isrunning=False
        #The simulation times of the different entities, used to keep track of whose turn it is
        self.agents_current_times: Dict[int, any] ={j.id: self.start_time for j in self.agents}
        self.exchange_time: int=self.start_time
        self.agents_action_freq: Dict[int, float]={j.id: j.action_freq for j in self.agents}
        #Implementation of message queue
        #An item in the queue takes the form of (time, (senderID, recipientID, Message))
        self.queue: List[Tuple[int, Tuple[Optional[int], int, Message]]] =[]
        self.head=0
        self.parameters=parameters
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
    
    def getparameter(self, name):
        try:
            rtn=self.parameters[name]
        except:
            raise ValueError(f"Expected parameter -- {name}. Received None")
    
    def initialize_kernel(self) -> None:
        """This initalizes the simulation first by linking all the relevant entity objects together"""
        self.exchange.initialize_exchange( agents=self.agents, kernel=self, Arrival_model=self.getparameter(name="Arrival_model")) 
        
        for agent in self.agents:
            agent.kernel=self 
            agent.exchange=self.exchange
            agent.kernel_start(current_time=0)
    
    def begin(self)-> None:
        """This begins the simulation"""
        self.isrunning=True
    
    def run(self, actions: Optional[Tuple[int, Tuple[int, int]]]=None) -> Dict[str, any]:
        """
        Start the simulation and processing of the message queue. Possibility to add the optional argument agent_actions which corresponds to the actions that the GYM agent can take
        Arguments:
            actions: A Tuple in the form of (AgentID, (Event, Size)) that describes the action to be performed by a specific agent
        Returns:
        Done: Boolean True if the simulation is done, else false
        TimeCode: the time at which the simulation was paused
        Results: the raw state of the simulation, containing data which can be formated for using in the GYM environment. 
        """
        if actions is not None:
            assert self.isrunning==True, f"Kernel cannot take an agent action when it has not begun running"
            agentID=actions[0]
            event=actions[1][0]
            size=actions[1][1]
            assert agentID in self.gymagents and agentID in self.agents, f"Intended action to run does not belong to an experimental gym agent."
            agent: GymTradingAgent=self.entity_registry[agentID]
            order=agent.action_to_order(action=(event, size))
            agent.submitorder(order)
        #While there are still items in the queue or time limit is not up yet and simulation has started, process a message.
        while (self.head < len(self.queue)) and (self.current_time<=self.stop_time):  
            item=self.queue[self.head]
            if self.isbatchmessage(item=item):
                self.processbatchmessage(item=item)
            else:
                interrupt=self.processmessage(item=item)
                if interrupt != None:
                    return {"Done": False, "TimeCode": self.current_time, "Result": self.getresults()}
        
        if self.head==len(self.queue):
            logger.debug("---Kernel Message queue empty. Terminating now ---")
        if self.current_time and (self.current_time > self.stop_time):
            logger.debug("---Kernel Stop Time surpassed---")
        
        return {"Done": True, "TimeCode": self.current_time, "Result": self.getresults}   

    
    def terminate(self)-> None:
        """
        Called when the simulation is terminated. Called once the queue is empty, or the gym environement is done, or the simulation reached either of the kernel stop time or wall_time limits.
        Returns: object that contains all relevant information from simulation
        """
        logger.debug("---Kernel Terminating---")
        for entity in self.entity_registry.values():
            entity.kernel_terminate()
        
        
    
    def reset(self) -> None:
        """
        Reset simulation, used in the gym core environment
        
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
            if senderID==-1:
                #Exception for the kernelID
                pass
            else:
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
        elif senderID==-1:
            pass
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
        
    def processmessage(self, item: Tuple[int, Tuple[int, int, Message]]) -> Optional[bool]:
        message=item[1][2]
        recipientID=item[1][1]
        senderID=item[1][0]
        timesent=item[0]
        logger.debug(f"Processing message with ID {message.message_id} for sender {senderID}")
        if isinstance(message, ExchangeMsg):
            self.current_time=timesent
            if isinstance(message, PartialOrderFill):
                #pass the message back onto the agent
                if recipientID==-1 or recipientID==self.exchange.id:
                    raise UnexpectedMessageType("Partial Order Fills Messages should not be generated for random non-agent orders")
                else:
                    self.agents[recipientID].receivemessage(current_time=self.current_time, senderID=senderID, message=message)
            elif isinstance(message, OrderAutoCancelledMsg):
                #pass message onto agent
                if recipientID==-1 or recipientID==self.exchange.id:
                    raise UnexpectedMessageType("Autocancel Messages should not be generated for random non-agent orders")
                else:
                    self.agents[recipientID].receivemessage(current_time=self.self.current_time, senderID=senderID, message=message)
            elif isinstance(message, OrderExecutedMsg):
                #tells an agent that a previously placed limit order has been executed
                if recipientID==-1 or senderID!=self.exchange.id:
                    raise UnexpectedMessageType("Order Execution Messages should not be generated for random non-agent orders")
                agent: TradingAgent=self.entity_registry[recipientID]
                self.agents_current_times[recipientID]=timesent
                agent.receivemessage(current_time=self.current_time, senderID=senderID, message=message)
                
            elif isinstance(message, WakeAgentMsg):
                #Message sent to agents to tell them to start trading
                if recipientID==-1 or recipientID==self.exchange.id:
                    raise UnexpectedMessageType("WakeAgent Messages should be sent to a valid agentID")
                agent: TradingAgent=self.entity_registry[recipientID]
                if isinstance(agent, GymTradingAgent):
                    #Interrupt the process and 
                    self.agents_current_times[recipientID]=timesent
                    logger.debug(f"Kernel sending wake up message to GYM agent {recipientID}.")
                    self.wakeup(agentID=recipientID)
                    return True
                else:
                    self.agents_current_times[recipientID]=timesent
                    logger.debug(f"Kernel sending wake up message to agent {recipientID}.")
                    self.wakeup(agentID=recipientID)
                    agent.receivemessage(current_time=self.current_time, senderID=senderID, message=message)
            else:
                #SHOULD NEVER HAPPEN
                raise UnexpectedMessageType(f"Unexpected message type received")
                pass
                
        elif isinstance(message, AgentMsg):
            #Process orders from agents and set their wake-ups
            if timesent!=order.time_placed:
                raise TimeSyncError(f"Message time of message {message.message_id} expected to be the same as order placement time{order.time_placed} but is different")
            self.current_time=timesent
            if isinstance(message, LimitOrderMsg):
                order: LimitOrder =message.order
                logger.debug(f"Agent {senderID} placed a Limit Order with ID {order.order_id}")     
                self.exchange.processorder(order=order)
                wakeuptime=self.agents_current_times[senderID]+self.agents_action_freq[senderID]
                self.set_wakeup(agentID=senderID, requested_time=wakeuptime)
            elif isinstance(message, MarketOrderMsg):
                order: MarketOrder=message.order
                logger.debug(f"Agent {senderID} placed a Market Order with ID {order.order_id}")  
                self.exchange.processorder(order=order)
                wakeuptime=self.agents_current_times[senderID]+self.agents_action_freq[senderID]
                self.set_wakeup(agentID=senderID, requested_time=wakeuptime)
            elif isinstance(message, CancelOrderMsg):
                order: CancelOrder=message.order
                logger.debug(f"Agent {senderID} cancelled previous order with ID {order.cancelID}")  
                self.exchange.processorder(order=order)
                wakeuptime=self.agents_current_times[senderID]+self.agents_action_freq[senderID]
                self.set_wakeup(agentID=senderID, requested_time=wakeuptime)
            elif isinstance(message, DoNothing):
                logger.debug(f"Agent {senderID} chose to do nothing at time {self.current_time}")
                wakeuptime=self.agents_current_times[senderID]+self.agents_action_freq[senderID]
                self.set_wakeup(agentID=senderID, requested_time=wakeuptime)
            else: 
                #SHOULD NEVER HAPPEN
                raise UnexpectedMessageType(f"Unexpected message type received")
                pass
        else:
            #SHOULD NEVER HAPPEN
            raise UnexpectedMessageType(f"Unexpected message type received")
            pass
        
    def processbatchmessage(self, item=Tuple[int, Tuple[int, List[int], Message]]):
        message: Message=item[1][2]
        recipientIDs: List[int]=item[1][1]
        senderID: int=item[1][0]
        timesent: float=item[0]
        if isinstance(message,ExchangeMsg):
            if isinstance(OrderExecutedMsg):
                #to be implemented
                pass
            elif isinstance(WakeAgentMsg):
                #to be implemented
                pass
            else:
                raise UnexpectedMessageType(f"Unexpected batch message type received")
                pass
                
            
        
        
        
    
    def set_wakeup(self, agentID: int, requested_time: float=None) -> None:
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
        return isinstance(item[1][1], list)
            
    #Helper functions
    def istruncated(self):
        return self.current_time>self.stop_time
    
    def isterminated(self):
        return [agent.isterminated for agent in self.gymagents]
    
    def getobservations(self):
        pass
    def getinfo(self):
        pass

if __name__ == "__main__":
    print("Kernel compiles")
    