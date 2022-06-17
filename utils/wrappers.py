import gym
from gym.spaces import Discrete
import numpy as np

class DiscreteActions(gym.ActionWrapper):
    def __init__(
        self,
        env,
        disc_to_cont= [np.array([-1]), np.array([1])]
        ):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self._action_space = Discrete(len(disc_to_cont))
    
    def action(self, act):
        return self.disc_to_cont[act]

class DiscreteActions3DOF(gym.ActionWrapper):
    def __init__(
        self,
        env,
        disc_to_cont = [[-1, -1], [-1, 0], [-1, +1],
                        [0, -1], [0, 0], [0, +1],
                        [+1, -1], [+1, 0], [+1, +1]]
        ):
        super().__init__(env)
        # Create an action table for all possible 
        # combinations of the values of thrust and
        # gimbaling
        self.disc_to_cont = disc_to_cont
        self._action_space = Discrete(len(disc_to_cont))
    
    def action(self, act):   
        return np.asarray(self.disc_to_cont[act])