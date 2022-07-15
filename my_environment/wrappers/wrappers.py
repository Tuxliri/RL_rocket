__all__ = ['DiscreteActions', 'DiscreteActions3DOF', 'GaudetStateObs']

import gym
from gym.spaces import Discrete, Box
from my_environment.envs.rocket_env import Rocket
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

    def get_keys_to_action(self):
        import pygame
        mapping = {(pygame.K_UP,): 1, (pygame.K_DOWN,): 0}
        return mapping

class DiscreteActions3DOF(gym.ActionWrapper):
    def __init__(
        self,
        env,
        disc_to_cont = [
        [0, -1], [-1, +1],
        [0, +1], [+1, +1]
                    ]
    ):
        super().__init__(env)
        # Create an action table for all possible 
        # combinations of the values of thrust and
        # gimbaling action = [delta, thrust]
        
        self.disc_to_cont = disc_to_cont
        self._action_space = Discrete(len(disc_to_cont))
    
    def action(self, act):   
        return np.asarray(self.disc_to_cont[act])

    def get_keys_to_action(self):
        import pygame
        mapping = {(pygame.K_LEFT,): 1, (pygame.K_RIGHT,): 3,
            (pygame.K_UP,): 2, (pygame.K_MODE,): 0}
        return mapping

class GaudetStateObs(gym.ObservationWrapper):
    def __init__(self, env: Rocket) -> None:
        super().__init__(env)
        self.observation_space = Box(low=-1, high=1, shape=(4,))

    def observation(self, observation):
        x,y,th = observation[0:3]
        vx,vy,vth = observation[3:6]

        r=np.array([x,y])
        v=np.array([vx,vy])

        v_targ, t_go = self.env.unwrapped.compute_vtarg(r,v)
        vx_targ, vy_targ = v_targ
        
        return np.float32([vx-vx_targ, vy-vy_targ, t_go,y])