# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real 6DOF dynamics

import importlib
import numpy as np
import gym
from gym import spaces, Env, GoalEnv

from my_environment.utils.simulator import Simulator3DOF

from my_environment.utils.renderer_utils import blitRotate
from numpy.typing import ArrayLike

class Rocket(Env):

    """ Simple environment simulating a 3DOF rocket
        with rotational dynamics and translation along
        two axis """

    metadata = {"render.modes": [
        "human", "rgb_array"], "render_fps" : 10}

    def __init__(
        self,
        IC = [100, 500, np.pi/2, -10, -50, 0],
        ICRange = [0,0,0,0,0,0],
        timestep=0.1,
        seed=42
    ) -> None:

        super(Rocket, self).__init__()

        # Initial conditions mean values and +- range
        self.ICMean = np.float32(IC)
        self.ICRange = np.float32(ICRange)  # +- range
        self.timestep = timestep
        self.metadata["render_fps"] = 1/timestep

        # Initial condition space
        self.init_space = spaces.Box(
            low=self.ICMean-self.ICRange/2,
            high=self.ICMean+self.ICRange/2,
            )

        self.seed(seed)
        
        
        # Actuators bounds
        self.max_gimbal = np.deg2rad(20)     # [rad]
        self.max_thrust = 981e3              # [N]
        
        # State normalizer and bounds
        t_free_fall = (-self.ICMean[4]+np.sqrt(self.ICMean[4]**2+2*9.81*self.ICMean[1]))/9.81
        inertia = 6.04e6
        lever_arm = 30.

        self.state_normalizer = np.maximum(np.array([
            1.5*abs(self.ICMean[0]),
            1.5*abs(self.ICMean[1]),
            2*np.pi,
            2*9.81*t_free_fall,
            2*9.81*t_free_fall,
            self.max_thrust*np.sin(self.max_gimbal)*lever_arm/(inertia)*t_free_fall/5.
            ]),
            1
            )

        """
        Define realistic bounds for episode termination
        they are computed in the reset() method, when
        initial conditions are 
        """
        # Set environment bounds
        self.x_bound_right = 0.9*np.maximum(self.state_normalizer[0],100)
        self.x_bound_left = -self.x_bound_right
        self.y_bound_up = 0.9*np.maximum(self.state_normalizer[1],100)
        self.y_bound_down = -30

        # Define observation space
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6,))

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
        self.target_r = 30

        # Renderer variables (pygame)
        self.window_size = 600  # The size of the PyGame window
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode.
        """

        self.window = None
        self.clock = None

    def step(self, action):

        u = self._denormalize_action(action)
        self.action = u

        self.y, __, isterminal, __ = self.SIM.step(u)
        obs = self.y.astype(np.float32)

        # Done if the rocket is at ground
        done = bool(isterminal) or self._checkBounds(obs)

        reward, info = self._compute_reward( obs, done, action)
        
        self.infos.append(info)

        return self._normalize_obs(obs), reward, done, info

    def _normalize_obs(self, obs):
        return obs/self.state_normalizer

    def _denormalize_obs(self,obs):
        return obs*self.state_normalizer
        
    def _compute_reward(self, obs, done, action):
        reward = 0      

        info = {
            'stateHistory': self.SIM.states,
            'actionHistory': self.SIM.actions,
            }        

        # give a reward if we're going in the correct direction,
        # i.e. if the velocity points in a cone towards the origin bounded by the original glideslope angle
        r = obs[0:2]
        v = obs[3:5]
        theta_prime = obs[2]-np.pi/2

        v_targ, __ = self.compute_vtarg(r,v)

        thrust = action[0]

        alfa = -0.01
        beta = -0.05
        eta = 0.01
        gamma = -100
        delta = -20

        theta_lim = np.pi/2
        theta_mgn = np.pi/4

        rew = alfa*np.linalg.norm(v-v_targ) + beta*thrust + eta +\
            gamma*(np.abs(theta_prime)>theta_lim) + delta*np.maximum(0,np.abs(theta_prime)-theta_mgn)
        
        rew_goal = self._reward_goal(obs)

        reward = rew + rew_goal
        
        if done:                
            if self._checkBounds(obs):
                info["Bounds violated"] = True

        rewards_log = {
            "reward": reward,
        }

        info["rewards_log"] = rewards_log
        
        return reward, info

    def _reward_goal(self, obs):
        k = 10
        return k*self._check_landing(obs)
    
    def compute_vtarg(self, r, v):
        tau_1 = 20
        tau_2 = 100
        initial_conditions = self.SIM.states[0]

        v_0 = np.linalg.norm(initial_conditions[3:5])

        if r[1]>15:
            r_hat = r-[0,15]
            v_hat = v-[0,-2]
            tau = tau_1

        else:
            r_hat = [0,15]
            v_hat = v-[0,-1]
            tau = tau_2

        t_go = np.linalg.norm(r_hat)/np.linalg.norm(v_hat)
        v_targ = -v_0*(r_hat/np.linalg.norm(r_hat))*(1-np.exp(-t_go/tau))
        
        return v_targ, t_go

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
        r = self.y[0:2]
        v = self.y[3:5]

        agent_location = r * step_size

        agent_location[0] = agent_location[0] + SHIFT_RIGHT
        agent_location[1] = self.window_size - agent_location[1]
        """
        Since the 0 in pygame is in the TOP-LEFT corner, while the
        0 in the simulator reference system is in the bottom we need
        to change between these two coordinate frames
        """

        angle_draw = self.y[2]*180/np.pi - 90
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

        stringToDisplay1 = f"x: {self.y[0]:5.1f}  y: {self.y[1]:4.1f} Angle: {np.rad2deg(self.y[2]):4.1f}"
        stringToDisplay2 = f"vx: {self.y[3]:5.1f}  vy: {self.y[4]:4.1f} omega: {self.y[5]:4.1f}"
        stringToDisplay3 = f"Time: {self.SIM.t:4.1f} Action: {np.array2string(action,precision=2)}"

        img1 = font.render(stringToDisplay1, True, (0,0,0))
        img2 = font.render(stringToDisplay2, True, (0,0,0))
        img3 = font.render(stringToDisplay3, True, (0,0,0))
        canvas.blit(img1, (20, 20))
        canvas.blit(img2, (20, 40))
        canvas.blit(img3, (20, 60))

        blitRotate(canvas, image, tuple(
            agent_location), (w/2, h/2), angle_draw)

        # Draw the target velocity vector
        v_targ, __ = self.compute_vtarg(r,v)
        
        pygame.draw.line(
            canvas,
            (0,0,0),
            start_pos=tuple(agent_location),
            end_pos=tuple(agent_location+[1,-1]*v_targ*5),
            width=2
            )
        
        # Draw the current velocity vector       
        pygame.draw.line(
            canvas,
            (0,0,255),
            start_pos=tuple(agent_location),
            end_pos=tuple(agent_location+[1,-1]*v*5),
            width=2
            )

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

        initialCondition = self.init_space.sample()
        self.y = initialCondition

        # instantiate the simulator object
        self.SIM = Simulator3DOF(initialCondition, self.timestep)

        return self._normalize_obs(self.y.astype(np.float32))

    def _denormalize_action(self, action : ArrayLike):
        """ Denormalize the action as we've bounded it
            between [-1,+1]. The first element of the 
            array action is the gimbal angle while the
            second is the throttle"""

        gimbal = action[0]*self.max_gimbal

        thrust = (action[1] + 1)/2. * self.max_thrust

        # Add lower bound on thrust with self.minThrust
        return np.float32([gimbal, thrust])

    def _get_obs(self):
        return self._normalize_obs(self.y)

    def plot_states(self, showFig : bool = False,
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

        __, = ax1.plot(timesteps, downranges, label='Downrange (x)')
        __, = ax1.plot(timesteps, heights, label='Height (y)')
        line_theta, = ax1_1.plot(timesteps, np.rad2deg(ths),'b-')
        
        # __, = ax1.plot(vxs, label='Cross velocity (v_x)')
        __, = ax1.plot(timesteps, vzs, label='Vertical velocity (v_z)')

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

    def _check_landing(self, state):
        r = np.linalg.norm(state[0:2])
        v = np.linalg.norm(state[3:5])

        # Measure the angular deviation from vertical orientation
        theta, vtheta = state[2], state[5]
        theta = theta-np.pi/2

        __,y = state[0:2]
        vx, vy = state[3:5]
        glideslope = np.arctan2(np.abs(vy),np.abs(vx))

        v_lim = 2
        r_lim = 5
        glideslope_lim = np.deg2rad(79)
        theta_lim = 0.2
        omega_lim = 0.2

        if y<=1e-3 and v<v_lim and r<r_lim and glideslope<glideslope_lim\
            and np.abs(theta)<theta_lim and np.abs(vtheta)<omega_lim :
            return True
        else:
            return False

    def seed(self, seed: int = 42):
        self.init_space.seed(42)
        return super().seed(seed)

    def _get_normalizer(self):
        return self.state_normalizer

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
