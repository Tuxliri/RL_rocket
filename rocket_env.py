# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real 6DOF dynamics

import numpy as np
from gym import spaces, Env
from gym.wrappers.time_limit import TimeLimit

from simulator import Simulator
from matplotlib import pyplot as plt

from renderer_utils import blitRotate


class Rocket(Env):

    """ Simple environment simulating a 3DOF rocket
        with rotational dynamics and translation along
        two axis """

    metadata = {"render_modes": [
        "human", "rgb_array", "plot"], "render_fps": 50}

    def __init__(
        self,
        IC,
        ICRange,
        timestep=0.5,
        render_mode="human"
    ) -> None:

        super(Rocket, self).__init__()

        # Initial conditions mean values and +- range
        self.ICMean = IC
        self.ICRange = ICRange  # +- range
        self.timestep = timestep

        # Initial condition space
        self.init_space = spaces.Box(low=self.ICMean-self.ICRange/2,
                                     high=self.ICMean+self.ICRange/2)

        # Actuators bounds
        self.maxGimbal = 10
        self.maxThrust = 1e6

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,))

        # Two valued vector in the range -1,+1, both for the
        # gimbal angle and the thrust command. It will then be
        # rescaled to the appropriate ranges in the dynamics
        self.action_space = spaces.Box(low=np.float32(
            [-1, -1]), high=np.float32([1, 1]), shape=(2,))

        # Environment state variable and simulator object
        self.y = None
        self.SIM = None

        # Renderer variables (pygame)
        self.window_size = 900  # The size of the PyGame window
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
        done = False
        info = {}

        u = self._denormalizeAction(action)

        self.y = self.SIM.step(u)

        reward = - self.y[1]

        # Done if the rocket is at ground
        done = self._checkTerminal(self.y.astype(np.float32))

        obs = self.y.astype(np.float32)

        return obs, reward, done, info

    def _checkTerminal(self, state):

        return bool(self.y[1] < 0)

    def render(self, mode="human"):
        return self._render_frame(mode)

    def _render_frame(self, mode: str):
        # avoid global pygame dependency. This method is not called with no-render.
        import pygame

        # The number of pixels per each meter
        step_size = (self.window_size / 100e3)

        # position of the CoM of the rocket
        agent_location = self.y[0:2] * step_size

        agent_location[1] = self.window_size - agent_location[1]
        """
        Since the 0 in pygame is in the TOP-LEFT corner, while the
        0 in the simulator reference system is in the bottom we need
        to change between these two coordinate frames
        """

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

            blitRotate(self.window, image, tuple(
                agent_location), (w/2, h/2), angleDeg)

            pygame.display.flip()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

        else:
            return None

    def close(self) -> None:
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

            self.SIM._plotStates()
        pass

    def reset(self):
        """ Function defining the reset method of gym
            It returns an initial observation drawn randomly
            from the uniform distribution of the ICs"""

        # Initialize the state of the system (sample randomly within the IC space)
        initialCondition = self.init_space.sample()
        self.y = initialCondition

        # instantiate the simulator object
        self.SIM = Simulator(initialCondition, self.timestep)

        return self.y.astype(np.float32)

    def _denormalizeAction(self, action):
        """ Denormalize the action as we've bounded it
            between [-1,+1]. The first element of the 
            array action is the gimbal angle while the
            second is the throttle"""
        gimbal = action[0]*self.maxGimbal

        thrust = (action[1] + 1)/2 * self.maxThrust

        return np.float32([gimbal, thrust])


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    initialConditions = np.float32([0, 10000, np.pi/2-0.05, 0, 0, 0])
    initialConditionsRange = np.zeros_like(initialConditions)

    env = Rocket(initialConditions, initialConditionsRange)
    env = TimeLimit(env, max_episode_steps=200)
    frames = []
    env.reset()
    env.render(mode="human")
    done = False

    while not done:
        action = np.array([0, 1])

        obs, rew, done, info = env.step(action)
        env.render(mode="human")

    tFinal = env.SIM.t

    env.close()

    input()
    check_env(env)
