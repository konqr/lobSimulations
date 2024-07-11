import gymnasium as gym 
from gymnasium import spaces
class tradingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "text"], "render_fps": 4}
    def __init__(self, render_mode):
        """
        
        
        """
        self.observation_space
        self.action_space
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Render mode must be None, human or rgb_array"
        
        
    def updatestate(self, action):
        """
        Update agent and market states
        """
        return None
            
    def step(self, action):
        """
        Wrapper for state step
        Input: Action
        Output: Observations, Rewards, termination, truncation, Logging info+metrics 
        """

        return observations, rewards, dones, infos    
        
        
    def reset(self, seed=None):
        """
        Reset env to default starting state + clear all simulations
        Params: Seed
        Output: Observations, info
        """
        
        #Reset Random seed
        #Sample LOB State
        
        return Observations, info
    
    def render(self):
        """
        Render an environment
        """
        
    def close(self):
        """
        close any active running windows/tasks
        """
    
    def _get_obs(self):
        """
        Private method to return observations on a state
        """
        return 

    def _get_info(self):
        """
        Returns auxiliary info: 
        """
        return
    
    
