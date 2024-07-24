import numpy as np
from abc import ABC, abstractmethod 
from typing import Any, List, Optional, Tuple, ClassVar
import pandas as pd
import logging
from RLenv.Messages.Message import Message

logger = logging.getLogger(__name__)
class Entity:
    """
    Base Entity class of an object in the simulation. An Entity can either be a trading agent or an exchange

    Attributes:
        id: Must be a unique number (usually autoincremented) belonging to the subclass.
        type: is it a trading agent, exchange or simulation kernel?
        seed: Every entity is given a random seed for stochastic purposes.
        log_events: flag to log or not the events during the simulation 
            Logging format:
                time: time of event
                event_type: label of the event (e.g., Order submitted, order accepted, last trade etc....)
                event: actual event to be logged (co_deep_Ask, lo_deep_Ask...)
                size: size of the order (if applicable)
        log_to_file: flag to write on disk or not the logged events
    """
    _entity_counter: ClassVar[int]=1
    _registry={}
    def __init__(self, type: str = None, seed=1, log_events: bool = True, log_to_file: bool = False) -> None:
        self.id=self.__class__._entity_counter
        type(self)._entity_counter+=1
        self.type=type
        self.name="Entity"+str(self.id)+"_"+str(self.type)+"_"+str(self.id)
        self.log_events=log_events
        self.log_to_file=log_events & log_to_file
        self.seed=seed
        # Simulation attributes
        
        self.kernel=None
        #What time does the entity think it is?
        self.current_time: int = 0 
        
        #Entities will maintain a log of their activities, events will likely be stored in the format of (time, event_type, eventname, size)
        self.log: List[Tuple[int, str, str, int]]
        
        if self.log_to_file:
            self.filename=None
        
        #Finally, add the entity object to the registry
        Entity._registry[self.id]=self
    
    @classmethod
    def get_entity_by_id(cls, ID: int):
        return cls._registry.get(ID)
    
    
    
    def kernel_start(self, start_time: int) -> None:
        assert self.kernel is not None
        logger.debug(
            "Entity {} requesting kernel wakeup at time {}".format(self.id, start_time))
    
    def kernel_terminate(self) -> None:
        if self.log and self.log_to_file:
            df_log=pd.DataFrame(self.log, columns=("EventTime", "Event Type", "Event", "Size"))
            self.write_log(df_log)
    
    
    
    """Methods for communication with the exchange, kernel, or other entities"""
    def sendmessage(self, recipientID: int, message: Message, current_time: int) -> None:
        """
        Sends a message to a specific recipient
        """
        assert self.kernel is not None
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("At {}, entity {} sent: {}".format(self.current_time, self.id, message))
        self.kernel.sendmessage(self.id, recipientID)
        
        
        
    def receivemessage(self, current_time, senderID: int, message: Message):
        """
        Called each time a message destined for this entity reaches the front of the
        kernel's priority queue.
        Arguments:
            current_time: The simulation time at which the kernel is delivering this message -- the entity should treat this as "now".
            sender: The entity that is sending the message
            message: An object guaranteed to inherit from the Message.Message class.
        """
        assert self.kernel is not None
        self.current_time=current_time
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("At {}, entity {} received: {}".format(self.current_time, self.id, message))
        
        
    def wakeup(self, current_time: int) -> None:
        """
        Entities can request a wakeup call at a future simulation time using
        the Entity.set_wakeup(time) method

        This is the method called when the wakeup time arrives.

        Arguments:
            current_time: The simulation time at which the kernel is delivering this
                message -- the entity should treat this as "now".
        """

        assert self.kernel is not None

        self.current_time = current_time

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "At {}, entity {} received wakeup.".format(
                    current_time, self.id
                )
            )
    
    def set_wakeup(self, requested_time: int) -> None:
        """
        Called to receive a future call from the kernel at the point of requested_time
        """
        assert self.kernel is not None
        self.kernel.set_wakeup(self.id, requested_time)
    
           
    """Book-keeping Methods for Internal use"""   
    def _logevent(self, event: Tuple[int, str, str, int]):
        """Adds an event to this entities log"""
        if not self.log_events:
            return
        self.log.append(event)

    def writelog(self, df_log: pd.DataFrame, filename: Optional[str]=None) -> None:
        """
        Called by the entity, usually at the very end of the simulation just before Kernel shuts down to log down any data. The Kernel places the log in a unique
        directory per run, with one filename per agent, also decided by the Kernel using
        agent type, id, etc. If filename is None the Kernel will construct a filename based on the name of
        the Agent requesting log archival.

        Arguments:
            df_log: dataframe that contains all the logged events during the simulation
            filename: Location on disk to write the log to.
        """
        assert self.kernel is not None
        self.kernel.write_log(self.id, df_log, filename)
    
    
          
    @abstractmethod        
    def update_state(self, kernelmessage): #update internal state given a kernel message
        pass
    
        
    
        
