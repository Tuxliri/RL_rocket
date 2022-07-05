# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real 6DOF dynamics

import importlib
from turtle import distance
import numpy as np
import gym
from gym import spaces, Env, GoalEnv

from my_environment.utils.simulator import Simulator3DOF

from my_environment.utils.renderer_utils import blitRotate
from numpy.typing import ArrayLike
from gym.utils import seeding

class Rocket(Env):

    """ Simple environment simulating a 3DOF rocket
        with rotational dynamics and translation along
        two axis """

    metadata = {"render_modes": [
        "human", "rgb_array"], "render_fps" : 10}

    def __init__(
        self,
        IC = [100, 500, np.pi/2, -10, -50, 0, 50e3],
        ICRange = [0,0,0,0,0,0,0],
        timestep=0.1,
        maxTime=40,
        seed=42
    ) -> None:

        super(Rocket, self).__init__()

        # Initial conditions mean values and +- range
        self.ICMean = np.float32(IC)
        self.ICRange = np.float32(ICRange)  # +- range
        self.timestep = timestep
        self.maxTime = maxTime
        
        # Initial condition space
        self.init_space = spaces.Box(
            low=self.ICMean-self.ICRange/2,
            high=self.ICMean+self.ICRange/2,
            )
        self.seed(seed)

        # Actuators bounds
        self.maxGimbal = np.deg2rad(20)     # [rad]
        self.maxThrust = 981e3              # [N]
        
        """
        Define realistic bounds for episode termination
        they are computed in the reset() method, when
        initial conditions are 
        """
        self.x_bound_right = None
        self.x_bound_left = None
        self.y_bound_up = None
        self.y_bound_down = None

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
        self.infos = []
        self.SIM = None
        self.action = np.array([0. , 0.])

        # Landing parameters
        self.target_r = 50.
        self.a_0 = None # Initial glideslope angle

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
        done = bool(isterminal) or currentTime>=self.maxTime or self._checkBounds(self.y)

        reward = 0
        reward, info = self._compute_reward(currentTime, obs, done)
        
        self.infos.append(info)

        if done and not currentTime>=self.maxTime:
            assert self._checkCrash or self._checkLanding, f"self._checkCrash is {self._checkCrash} and self._checkLanding is f{self._checkLanding}"

        return obs, reward, done, info

    def _compute_reward(self, currentTime, obs, done):
        reward = 0

        info = {
            'stateHistory': self.SIM.states,
            'actionHistory': self.SIM.actions,
            "TimeLimit.truncated": False,
            "EpisodeDone": False,
            }        

        # give a reward if we're going in the correct direction,
        # i.e. if the velocity points in a cone towards the origin bounded by the original glideslope angle
        r = obs[0:2]
        v = obs[3:5]

        r_hat = r/np.linalg.norm(r)
        v_hat = v/np.linalg.norm(v)

        
        glideslope_limit = 0.5*np.pi - self.a_0 +10./180.*np.pi
        if np.arccos(np.dot(v_hat,-r_hat)) <= glideslope_limit:
            reward = +1
            
            if obs[2]-0.5*np.pi < np.pi/6:
                reward+=1
        
        

        # give a bonus reward
        if self._checkCrash(obs) is True:
            reward = -30

        if self._checkLanding(obs) is True:
            reward = 30

        if currentTime>=self.maxTime:
            info["TimeLimit.truncated"] = True

        rewards_log = {
            "reward": reward,
            
        }
        info["rewards_log"] = rewards_log
        
        return reward, info

    def render(self, mode : str="human"):
        import pygame  # import here to avoid pygame dependency with no render

        if (self.window is None) and mode is "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

         
        # The number of pixels per each meter
        MAX_HEIGHT = self.y_bound_up
        step_size = self.window_size / MAX_HEIGHT

        SHIFT_RIGHT = self.window_size/2

        # position of the CoM of the rocket
        agent_location = self.y[0:2] * step_size

        agent_location[0] = agent_location[0] + SHIFT_RIGHT
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
        with importlib.resources.path('my_environment', 'res') as data_path:
            image_path = data_path / "rocket.png"
            image = pygame.image.load(image_path)

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

        # Draw on a canvas surface
        canvas = pygame.Surface((self.window_size, self.window_size))
        backgroundColour = (255, 255, 255)
        canvas.fill(backgroundColour)
        image.set_colorkey((246, 246, 246))

        pygame.font.init()
        font = pygame.font.SysFont(None, 24)
        action = self.action
        action[0] = action[0]*180/np.pi

        stringToDisplay1 = f"x: {self.y[0]:5.1f}  y: {self.y[1]:4.1f} Angle: {self.y[2]*180/np.pi:4.1f}"
        stringToDisplay2 = f"vx: {self.y[3]:5.1f}  vy: {self.y[4]:4.1f} Time: {self.SIM.t:4.1f} Action: {np.array2string(action,precision=2)}"

        img1 = font.render(stringToDisplay1, True, (0,0,0))
        img2 = font.render(stringToDisplay2, True, (0,0,0))
        canvas.blit(img1, (20, 20))
        canvas.blit(img2, (20, 40))

        blitRotate(canvas, image, tuple(
            agent_location), (w/2, h/2), angleDeg)

        # Draw a rectangle at the landing pad
        landing_pad = pygame.Rect(0,0,step_size*self.target_r,30)
        landing_pad_x = 0 + SHIFT_RIGHT

        landing_pad.center=(landing_pad_x,self.window_size)

        pygame.draw.rect(canvas, (255,0,0), landing_pad)

        if mode == "human":
            assert self.window is not None
            # Draw the image on the screen and update the window
            self.window.blit(canvas, dest=(0,0))

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(1/self.timestep)

            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
    
    def close(self) -> None:
        if self.window is not None:
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
        self.y = initialCondition

        self.a_0 = np.arctan2(initialCondition[1],initialCondition[0])

        # Set environment bounds
        self.x_bound_right = 1.5*max(abs(initialCondition[0]),200)
        self.x_bound_left = -self.x_bound_right
        self.y_bound_up = 1.1*abs(initialCondition[1])
        self.y_bound_down = -30

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

    def plotStates(self, showFig : bool = False,
    states = None):
        """
        :param states: list of observations
        """
        import matplotlib.pyplot as plt

        heights = []
        downranges = []
        ths = []
        vxs = []
        vzs = []
        oms = []
        mass = []
        thrusts = []
        gimbals = []
        timesteps = self.SIM.times

        fig1, ax1 = plt.subplots()
        ax1_1 = ax1.twinx()
        
        if states is None:
            states = self.SIM.states

        for state in states:
            downranges.append(state[0])
            heights.append(state[1])
            ths.append(state[2])
            vxs.append(state[3])
            vzs.append(state[4])
            oms.append(state[5])
            mass.append(state[6])

        __, = ax1.plot(timesteps, downranges, label='Downrange (x)')
        __, = ax1.plot(timesteps, heights, label='Height (y)')
        line_theta, = ax1_1.plot(timesteps, ths,'b-')
        
        # __, = ax1.plot(vxs, label='Cross velocity (v_x)')
        __, = ax1.plot(timesteps, vzs, label='Vertical velocity (v_z)')
        
        if not self.SIM.constantMass:
            __, = ax1.plot(timesteps, mass, label='mass')

        if self.infos[-1]["TimeLimit.truncated"]:
            ax1.text(0,0,'Truncated episode')

        elif self.infos[-1]["EpisodeDone"]:
            ax1.text(0,0,'Terminated episode')
        
        else:
            ax1.text(0,0,'NOT-truncated episode')

        ax1.legend()
        ax1_1.set_ylabel('theta',color='b')
        ax1_1.tick_params('y', colors='b')
        ax1.set_xlabel('Time [s]')

        # Plotting actions
        for action in self.SIM.actions:
            gimbals.append(action[0])
            thrusts.append(action[1])
            

        fig2, ax2 = plt.subplots()
        ax2_1 = ax2.twinx()
        
        __, = ax2.plot(timesteps, thrusts, 'bx')
        ax2.set_ylabel('Thrust (kN)', color='b')
        ax2.tick_params('y', colors='b')

        ax2.set_xlabel('Time [s]')

        __, = ax2_1.plot(timesteps, gimbals, 'r.')
        ax2_1.set_ylabel('Gimbals [rad]', color='r')
        ax2_1.tick_params('y', colors='r')

        if showFig:
            plt.show(block=False)

        return (fig1, fig2)

    def _checkBounds(self, state : ArrayLike):
        """
        :param state: state of the rocket, np.ndarray() shape(7,)
        
        Check if the rocket goes outside the side or upper bounds
        of the environment
        """
        outside = False
        x,y = state[0:2]

        if x<=self.x_bound_left or x>=self.x_bound_right:
            outside = True
        
        if y>=self.y_bound_up:
            outside = True

        return outside

    def _checkCrash(self, state):
        x,y = state[0:2]
        vx,vy = state[3:5]

        # Measure the angular deviation from vertical orientation
        theta, vtheta = state[2]-np.pi/2, state[5]

        v = (vx**2 + vy**2)**0.5
        crash = False

        if y >= self.y_bound_up:
            crash = True
        if y <= 1e-3 and v >= 15.0:
            crash = True
        if y <= 1e-3 and abs(x) >= self.target_r:
            crash = True
        # if y <= 1e-3 and abs(theta) >= 10/180*np.pi:
        #     crash = True
        # if y <= 1e-3 and abs(vtheta) >= 10/180*np.pi:
        #     crash = True

        return crash

    def _checkLanding(self, state):
        x,y = state[0:2]
        vx,vy = state[3:5]

        # Measure the angular deviation from vertical orientation
        theta, vtheta = state[2]-np.pi/2, state[5]

        v = (vx**2 + vy**2)**0.5

        if y<1e-3 and v < 15.0 and abs(x)<self.target_r:
            return True

        else:
            return False

    def seed(self, seed: int = 42):
        self.init_space.seed(42)
        return super().seed(seed)

class Rocket1D(gym.Wrapper, GoalEnv):
    def __init__(
        self,
        env: Env,
        reward_type='shaped_terminal',
        goal_threshold=5,
        velocity_threshold=5
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
        self.reward_type = reward_type
        self.goal_threshold = goal_threshold
        self.velocity_threshold = velocity_threshold

    def step(self, thrust):

        action = np.float32([0.0, thrust[0]])
        obs, reward3dof, done, info = self.env.step(action)
        obs = self._shrink_obs(obs)

        rew = 0
        if self.reward_type == 'shaped_terminal':
            if done:
                rew = 50 - np.linalg.norm(obs)

        elif self.reward_type == 'sparse_terminal':
            if done:
                rew = float(np.linalg.norm(obs) < self.velocity_threshold)

        elif self.reward_type == 'shaped_landing':
            rew -= (np.linalg.norm(obs) + 0.1)

        elif self.reward_type == 'hovering':
            rew -= np.abs(obs[1])
        elif self.reward_type == 'test3dof':
            rew = reward3dof

        if info["TimeLimit.truncated"]:
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

        if self.reward_type == 'sparse':
            return (d < self.goal_threshold).astype(np.float32)
        else:
            return -d

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
