# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real 6DOF dynamics

import numpy as np
from gym import spaces, Env

from simulator import Simulator
from matplotlib import pyplot as plt

from renderer_utils import blitRotate

class Rocket(Env):

    """ Simple environment simulating a 3DOF rocket
        with rotational dynamics and translation along
        two axis """

    metadata = {"render_modes": [
        "human", "rgb_array", "plot"], "render_fps": 50}

    def __init__(self, IC=np.float32([-1e3, -5e3, 90/180*np.pi, 300, +300, 0.1]),
                 ICRange=np.float32([100, 500, 10/180*np.pi, 50, 50, 0.01]),
                 render_mode="human") -> None:

        super(Rocket, self).__init__()

        # Initial conditions mean values and +- range
        self.ICMean = IC
        self.ICRange = ICRange  # +- range

        # Initial condition space
        self.init_space = spaces.Box(low=self.ICMean-self.ICRange/2,
                                     high=self.ICMean+self.ICRange/2)

        self.upperBound = np.maximum(
            abs(self.init_space.high), abs(self.init_space.low))

        # Maximum simulation time [s]
        self.tMax = 120

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

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,))

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
        self.window_size = 1000  # The size of the PyGame window
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode.
        """

        if render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        # The following line uses the util class Renderer to gather a collection of frames
        # using a method that computes a single frame. We will define _render_frame below.
        # ???WHERE IS THIS Renderer utils function???
        #self.renderer = Renderer(render_mode, self._render_frame)

    def step(self, action):
        observation = []
        reward = 0
        done = False
        info = {}

        reward = - self.y[1]

        u = self._denormalizeAction(action)

        self.y = self.RKT.step()

        # Done if the distance of the rocket to the ground is about 0
        # (e.g. less than 30cm)
        done = self._checkTerminal(self.y.astype(np.float32))

        return self.y.astype(np.float32), float(reward), done, info

    def _checkTerminal(self, state):
        #bool(np.linalg.norm(self.y[0:2]) < self.doneDistance)
        # return (not bool(self.observation_space.contains(state))) or self.RKT.t>self.tMax

        return bool(self.RKT.t > self.tMax or self.y[1] < 0)

    def render(self, mode="human"):
        return self._render_frame(mode)

    def _render_frame(self, mode: str):
        # avoid global pygame dependency. This method is not called with no-render.
        import pygame

        rocket_height = 50

        step_size = (
            self.window_size / self.ICMean[1]
        )  # The number of pixels per each meter

        

        # position of the CoM of the rocket
        agent_location = self.y[0:2] * step_size
        
        """
        Since the 0 in pygame is in the TOP-LEFT corner, while the
        0 in the simulator reference system is in the bottom we need
        to change between these two coordinate frames
        """
        
        agent_location[1] = self.window_size - agent_location[1]
        angleDeg = self.y[2]*180/np.pi - 90
        """
        As the image is vertical when displayed with 0 rotation
        we need to align it with the convention of rocket horizontal
        when the angle is 0
        """

        # Add gridlines?
        
        # load image of rocket
        image = pygame.image.load("rocket.jpg")
        h = image.get_height()
        w = image.get_width()

        if mode == "human":
            assert self.window is not None
            # Draw the image on the screen and update the window
            self.window.fill((255, 255, 255))
            image.set_colorkey((246, 246, 246))

            blitRotate(self.window, image, tuple(agent_location), (w/2, h/2), angleDeg)

            pygame.display.flip()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            # return np.transpose(
            #     np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            # )
            return None

        

    def close(self) -> None:
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

            self.RKT._plotStates()
        pass

    def reset(self):
        """ Function defining the reset method of gym
            It returns an initial observation drawn randomly
            from the uniform distribution of the ICs"""

        # Initialize the state of the system (sample randomly within the IC space)
        initialCondition = self.init_space.sample()
        self.y = initialCondition

        # instantiate the simulator object
        self.RKT = Simulator(initialCondition, 0.1)

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

    initialConditions = np.float32([0, 1e5, np.pi/2, 100, 0, 1])
    initialConditionsRange = np.zeros_like(initialConditions)

    RKT = Rocket(initialConditions, initialConditionsRange)
    frames = []
    RKT.reset()
    RKT.render(mode="human")
    done = False

    while not done:
        action = np.array([-1., -1.])

        obs, rew, done, info = RKT.step(action)
        RKT.render(mode="human")

    RKT.close()

    input()
    check_env(RKT)
