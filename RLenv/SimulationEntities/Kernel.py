import logging
import queue
import os
import datetime
import numpy as np
import pandas as pd
from RLenv.SimulationEntities.Entity import Entity
from RLenv.Messages.Message import Message
from typing import Any, Dict, List, Optional, Tuple, Type
class Kernel:
    def __init__():
        pass
    
    def runner(self):
        """
        Wrapper to run the entire simulation (when not running in the tradingEnv).

        3 Steps:
          - Simulation Initiation
          - Simulation Run
          - Simulation Termination

        Returns:
            An object that contains all the objects at the end of the simulation.
        """
        self.initialize()

        self.run()

        return self.terminate()
    
    def initialize(self) -> None:
        """This initalizes the simulation once all the agent configuration is completed"""
        pass
    
    def run(self) -> None:
        """
        Start the simulation and processing of the message queue
        Returns:
        Done: Boolean True if the simulation is done, else false
        Results: the raw state of the simulation, containing data which can be formated for using in the GYM environment. 
        """
        
        
    def sendmessage(self, senderID, recipientID, message: Message):
        pass
    
    def set_wakeup(self, entityID: int, requested_time: int):
        pass
    def write_log(self, senderID: int, df_log: pd.DataFrame, filename: Optional[str]):
        """
        Called by any entity at the end of the simulation. If filename is None, the Kernel
        will construct a filename based on the name of the Agent requesting log archival.

        """