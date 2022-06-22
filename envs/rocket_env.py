# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real 6DOF dynamics

from turtle import distance
import numpy as np
import gym
from gym import spaces, Env, GoalEnv

from my_environment.utils.simulator import Simulator3DOF

from my_environment.utils.renderer_utils import blitRotate

MAX_SIZE_RENDER = 10e3      # Max size in meters of the rendering window


class Rocket(Env):

    """ Simple environment simulating a 3DOF rocket
        with rotational dynamics and translation along
        two axis """

    metadata = {"render_modes": [
        "human", "rgb_array"], "render_fps" : 5}

    def __init__(
        self,
        IC,
        ICRange,
        timestep=0.1,
        maxTime=300,
        render_mode="human"
    ) -> None:

        super(Rocket, self).__init__()

        # Initial conditions mean values and +- range
        self.ICMean = IC
        self.ICRange = ICRange  # +- range
        self.timestep = timestep
        self.maxTime = maxTime

        # Initial condition space
        self.init_space = spaces.Box(low=self.ICMean-self.ICRange/2,
                                     high=self.ICMean+self.ICRange/2)

        # Actuators bounds
        self.maxGimbal = np.deg2rad(20)     # [rad]
        self.maxThrust = 981e3              # [N]
        

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,))

        # Two valued vector in the range -1,+1, both for the
        # gimbal angle and the thrust command. It will then be
        # rescaled to the appropriate ranges in the dynamics
        self.action_space = spaces.Box(low=np.float32(
            [-1, -1]), high=np.float32([1, 1]), shape=(2,))

        # Environment state variable and simulator object
        self.y = None
        self.SIM = None
        self.action = np.array([0. , 0.])

        # Renderer variables (pygame)
        self.window_size = 900  # The size of the PyGame window
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode.
        """

        self.window = None
        self.clock = None

    def step(self, action):

        u = self._denormalizeAction(action)
        self.action = u

        self.y, __, isterminal, currentTime = self.SIM.step(u)
        obs = self.y.astype(np.float32)

        # Done if the rocket is at ground
        done = bool(isterminal) or currentTime>self.maxTime

        assert done is not bool, "done is not of type bool!"

        reward = 0
        
        if done:
            posNorm = np.linalg.norm(obs[0:2])
            velNorm = np.linalg.norm(obs[3:5])
            angle = obs[2]

            reward = 50 - posNorm - velNorm #- angle NO ANGLE FOR NOW

        info = {'isTruncated' : currentTime>self.maxTime}
        

        return obs, reward, done, info

    def render(self, mode : str="human"):
        import pygame  # import here to avoid pygame dependency with no render

        if (self.window is None) and mode is "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

         
        # The number of pixels per each meter
        MAX_HEIGHT = 1.1 * self.ICMean[1]
        step_size = self.window_size / MAX_HEIGHT

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
        image = pygame.image.load("rocket.png")
        h = image.get_height()
        w = image.get_width()

        # Extend the surface, add arrow
        if self.action[1]>0:
            old_image = image
            image = pygame.Surface(
                (image.get_width(), image.get_height()+20))
            image.fill((255,255,255))
            image.blit(old_image,(0,0))

            start_pos = (w/2,h)
            gimbalRad = self.action[0]

            thrustArrowLength = 20
            end_pos = (
                w/2+thrustArrowLength*np.sin(gimbalRad),
                h+thrustArrowLength*np.cos(gimbalRad)
                )
            pygame.draw.line(image, (255,0,0), start_pos=start_pos, end_pos=end_pos)

        # pygame.transform.scale(image, )

        # Draw on a canvas surface
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        image.set_colorkey((246, 246, 246))

        pygame.font.init()
        font = pygame.font.SysFont(None, 24)

        stringToDisplay1 = f"x: {self.y[0]:5.1f}  y: {self.y[1]:4.1f} Angle: {self.y[2]:4.1f}"
        stringToDisplay2 = f"vx: {self.y[3]:5.1f}  vy: {self.y[4]:4.1f} Time: {self.SIM.t:4.1f} Action: {np.array2string(self.action,precision=2)}"

        img1 = font.render(stringToDisplay1, True, (0,0,0))
        img2 = font.render(stringToDisplay2, True, (0,0,0))
        canvas.blit(img1, (20, 20))
        canvas.blit(img2, (20, 40))

        blitRotate(canvas, image, tuple(
            agent_location), (w/2, h/2), angleDeg)

        # Draw a rectangle at the landing pad
        landing_pad = pygame.Rect(0,0,30,30)
        landing_pad_x = 0

        landing_pad.center=(landing_pad_x,self.window_size)

        pygame.draw.rect(canvas, (255,0,0), landing_pad)

        if mode == "human":
            assert self.window is not None
            # Draw the image on the screen and update the window
            self.window.blit(canvas, (0, 0))

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(1/self.timestep)

            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def plotStates(self, showFig):
        fig1, fig2 = self.SIM._plotStates(showFig)
        return (fig1, fig2)
    
    def close(self, showPlots : bool = False) -> None:
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

        self.plotStates(showPlots)
        pass

    def reset(self):
        """ Function defining the reset method of gym
            It returns an initial observation drawn randomly
            from the uniform distribution of the ICs"""

        # Initialize the state of the system (sample randomly within the IC space)
        initialCondition = self.init_space.sample()
        self.y = initialCondition

        # instantiate the simulator object
        self.SIM = Simulator3DOF(initialCondition, self.timestep)

        return self.y.astype(np.float32)

    def _denormalizeAction(self, action):
        """ Denormalize the action as we've bounded it
            between [-1,+1]. The first element of the 
            array action is the gimbal angle while the
            second is the throttle"""

        assert isinstance(action, (np.ndarray)) and action.shape == (2,),\
            f"Action is of type {type(action)}, shape: {action.shape}"

        gimbal = action[0]*self.maxGimbal

        thrust = (action[1] + 1)/2 * self.maxThrust

        # Add lower bound on thrust with self.minThrust
        return np.float32([gimbal, thrust])

    def plotStates(self, showFig : bool = False):
        fig1, fig2 = self.SIM._plotStates(showFig)
        return (fig1, fig2)


class Rocket1D(GoalEnv, gym.Wrapper):
    def __init__(
        self,
        env: Env,
        rewardType='shaped_terminal',
        goalThreshold=5,
        velocityThreshold=5
        ) -> None:

        super().__init__(env)
        self.env = env
        self.observation_space = spaces.Dict({'observation': spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        ),
            'desired_goal': spaces.Box(
            low=-1.,
            high=1.,
            shape=(2,),
            dtype=np.float32
        ),
            'achieved_goal': spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        )
        }

        )
        self._action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.desired_goal = np.float32([0, 0])
        self.rewardType = rewardType
        self.goalThreshold = goalThreshold
        self.velocityThreshold = velocityThreshold

    def step(self, thrust):

        action = np.float32([0.0, thrust[0]])
        obs, rew, done, info = self.env.step(action)
        obs = self._shrink_obs(obs)

        rew = 0
        if self.rewardType == 'shaped_terminal':
            if done:
                rew = 50 - np.linalg.norm(obs)

        elif self.rewardType == 'sparse_terminal':
            if done:
                rew = float(np.linalg.norm(obs) < self.velocityThreshold)

        elif self.rewardType == 'shaped_landing':
            rew -= (np.linalg.norm(obs) + 0.1)

        elif self.rewardType == 'hovering':
            rew -= np.abs(obs[1])

        if info['isTruncated']:
            rew -= 500
        # Test for PPO only, possibly breaking HER compatibility
        # rew = self.compute_reward(obs, self.desired_goal, {})

        observation = dict({
            'observation': obs,
            'achieved_goal': obs,
            'desired_goal': self.desired_goal
        })

        return observation, rew, done, info

    def reset(self):
        obs_full = self.env.reset()
        obs = self._shrink_obs(obs_full)

        observation = dict({
            'observation': obs,
            'achieved_goal': obs,
            'desired_goal': self.desired_goal
        })

        return observation

    def _shrink_obs(self, obs_original):

        height, velocity = obs_original[1], obs_original[4]
        return np.float32([height, velocity])

    def compute_reward(self, achieved_goal: object, desired_goal: object, info: dict, p: float = 0.5) -> float:

        d = self.goal_distance(achieved_goal, desired_goal)

        if self.rewardType == 'sparse':
            return (d < self.goalThreshold).astype(np.float32)
        else:
            return -d

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
