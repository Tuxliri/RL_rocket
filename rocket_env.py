# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified 
# 3DOF version of the real dynamics
from distutils.log import info
import numpy as np
import gym

from scipy import integrate

from gym import spaces

class Rocket(gym.Env):

    """ Simple environment simulating a 3DOF rocket
        with rotational dynamics and translation along
        two axis """

    def __init__(self) -> None:
        super(Rocket, self).__init__()

        # Initial conditions mean values and +- range
        self.ICMean = np.array([-1e3, 5e3, 30/180*np.pi, 300, 300, 0.05])
        self.ICRange = np.array([100, 500, 10/180*np.pi, 50, 50, 0.01])  # +- range

        # Instantiate the random number generator
        self.rng = np.random.default_rng(12345)

        # Define action and observation spaces
        self.observation_space = spaces.Box(low=-50e3,high=50e3,shape=(6,),dtype=np.float32) # add reasonable lower and upper bounds

        # Two valued vector in the range -1,+1, both for the
        # gimbal angle and the thrust command. It will then be 
        # rescaled to the appropriate ranges in the dynamics
        self.action_space = spaces.Box(low=np.float32([-1, 0]), high=np.float32([1, 1]),shape=(2,), dtype=np.float32)

        # Initial condition space
        self.init_space = spaces.Box(low=self.ICMean-self.ICRange/2,
            high=self.ICMean+self.ICRange/2)

        # distance where rocket can be considered landed
        self.doneDistance = 0.5             # [m]

        # Initialize the state of the system
        self.y = self.ICMean


    def step(self, action):
        observation = []
        reward = 0
        done = False
        info = {}

        reward = 0

        
        # Done if the distance of the rocket to the ground is about 0 
        # (e.g. less than 30cm)
        done = bool(np.linalg.norm(self.y[0:2]) < self.doneDistance)

        return self.y, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        # return the state of the system
        print("x=", self.state[0], "; z=", self.state[1], "; theta=", self.state[2])

    def close(self) -> None:
        pass
    
    def reset(self):
        """ Function defining the reset method of gym
            It returns an initial observation drawn randomly
            from the uniform distribution of the ICs"""

        self.y = self.rng.uniform(low=self.ICMean-self.ICRange/2,
            high=self.ICMean+self.ICRange/2)

        return self.y.astype(np.float32)

    def denormalize(self,action):
        """ Denormalize the action as we've bounded it
            between [-1,+1]. The first element of the 
            array action is the gimbal angle while the
            second is the throttle"""

        return np.array([action[0]*self.maxGimbal,
                action[1]*(self.maxThrust - self.minThrust) + self.minThrust])