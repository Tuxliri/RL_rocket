# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real dynamics
import gym
import numpy as np
from gym import spaces
from gym.error import DependencyNotInstalled

from simulator import Simulator

class Rocket(gym.Env):

    """ Simple environment simulating a 3DOF rocket
        with rotational dynamics and translation along
        two axis """

    def __init__(self, IC = np.float32([-1e3, -5e3, 90/180*np.pi, 300, +300, 0.1]),
        ICRange = np.float32([100, 500, 10/180*np.pi, 50, 50, 0.01])) -> None:
        super(Rocket, self).__init__()

        # Initial conditions mean values and +- range
        self.ICMean = IC
        self.ICRange = ICRange  # +- range

        # Initial condition space
        self.init_space = spaces.Box(low=self.ICMean-self.ICRange/2,
                                     high=self.ICMean+self.ICRange/2)

        self.upperBound = np.maximum(abs(self.init_space.high), abs(self.init_space.low))

        # Maximum simulation time [s]
        self.tMax = 100

        # Maximum rocket angular velocity
        self.omegaMax = np.deg2rad(10)

        # Actuators bounds
        self.maxGimbal = 10
        self.maxThrust = 500

        # Upper and lower bounds of the state
        self.stateLow = np.float32(
            [-1.1*self.upperBound[0],
             -1.1*self.upperBound[1],
             0,
             -1.1*self.upperBound[3],
             0,
             -2*self.omegaMax*np.inf
             ])

        self.stateHigh = np.float32(
            [1.1*self.upperBound[0],
             0,
             2*np.pi,
             1.1*self.upperBound[3],
             self.upperBound[4] + 0.5*9.81*self.tMax**2,
             2*self.omegaMax*np.inf
             ])

        # Define normalizer of the observation space
        self.stateNormalizer = np.maximum(np.maximum(
            np.abs(self.stateLow[0:5]), np.abs(self.stateHigh[0:5])),
            1e-16*np.ones(5))

        self.stateNormalizer = np.append(self.stateNormalizer, np.float32(1))
        

        # Define action and observation spaces
        self.observation_space = spaces.Box(
            low=self.stateLow/self.stateNormalizer, high=self.stateHigh/self.stateNormalizer)

        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(6,))
        
        # Two valued vector in the range -1,+1, both for the
        # gimbal angle and the thrust command. It will then be
        # rescaled to the appropriate ranges in the dynamics
        self.action_space = spaces.Box(low=np.float32(
            [-1, -1]), high=np.float32([1, 1]), shape=(2,))

        # distance where rocket can be considered landed
        self.doneDistance = 0.5             # [m]

        # Environment state variable and simulator object
        self.y = None
        self.RKT = None

        # Renderer variables (pygame)
        self.screen = None
        self.clock = None

    def step(self, action):
        observation = []
        reward = 0
        done = False
        info = {}

        reward = - self.y[1]

        u = self._denormalizeAction(action)

        self.y = self._normalizeState(self.RKT.step())

        # Done if the distance of the rocket to the ground is about 0
        # (e.g. less than 30cm)
        done = self._checkTerminal(self.y.astype(np.float32))

        return self.y.astype(np.float32), float(reward), done, info

    def _checkTerminal(self, state):
        #bool(np.linalg.norm(self.y[0:2]) < self.doneDistance)
        return not bool(self.observation_space.contains(state))

    def render(self, mode='console'):

        try:
            import pygame
            from pygame import gfxdraw

        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run pip install pygame"
            )

        screen_width = 1000
        screen_height = 800

        world_width = 100 + self.stateHigh[0]*2
        scale = screen_width / world_width
        rocketwidth = 10.0
        rocketlen = scale*( 20.0)*100

        if self.y is None:
            return None

        x = self.y

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -rocketwidth / 2, rocketwidth / 2, rocketlen / 2, -rocketlen / 2

        rocketx = x[0]*scale
        rockety = x[1]*scale

        rocket_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2]) # CHECK IF THIS SIGN IN THE ROTATION IS CORRECT
            coord = (coord[0] + rocketx, coord[1] + rockety)
            rocket_coords.append(coord)

        gfxdraw.aapolygon(self.surf, rocket_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, rocket_coords, (202, 152, 101))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if mode == 'console':
            raise NotImplementedError()

        if mode == "human":
            pygame.event.pump()
            self.clock.tick()
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

        pass

    def reset(self):
        """ Function defining the reset method of gym
            It returns an initial observation drawn randomly
            from the uniform distribution of the ICs"""

        # Initialize the state of the system (sample randomly within the IC space)
        initialCondition = self.init_space.sample()
        self.y = self._normalizeState(initialCondition)

        # instantiate the simulator object
        self.RKT = Simulator(initialCondition)

        return self.y.astype(np.float32)

    def _denormalizeAction(self, action):
        """ Denormalize the action as we've bounded it
            between [-1,+1]. The first element of the 
            array action is the gimbal angle while the
            second is the throttle"""
        gimbal = action[0]*self.maxGimbal

        thrust = (action[1]+1)/2*self.maxThrust

        return np.array([gimbal, thrust]).astype(np.float32)

    def _normalizeState(self, state):
        return state / self.stateNormalizer

    def _denormalizeState(self, state):
        return state * self.stateNormalizer


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    initialConditions = np.float32([1,-1e3,np.pi/2,1,1,0.01])
    initialConditionsRange = np.zeros_like(initialConditions)

    RKT = Rocket(initialConditions, initialConditionsRange)
    frames = []
    RKT.reset()
    RKT.render(mode="human")
    done = False
    
    while not done:
        a = RKT.step(np.array([-1.,-1.]))
        RKT.render(mode="human")
        done = a[2]
        frames.append(RKT.render(mode="rgb_array"))


    check_env(RKT)