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
        IC = [100, 500, np.pi/2, -10, -50, 0,50e3],
        ICRange = [10,50,0.1,1,10,0.1,1e3],
        timestep=0.1,
        seed=42,
        reward_coeff = {"alfa" : -0.01,
                        "beta" : -1e-8,
                        "eta" : 2,
                        "gamma" : -10,
                        "delta" : -5,
                        "kappa" : 10,
                        }
    ) -> None:

        super(Rocket, self).__init__()

        self.state_names = ['x', 'z', 'theta', 'vx', 'vz', 'omega', 'mass']
        self.action_names = ['gimbal', 'thrust']

        # Initial conditions mean values and +- range
        self.ICMean = np.float32(IC)
        self.ICRange = np.float32(ICRange)  # +- range
        self.timestep = timestep
        self.metadata["render_fps"] = 1/timestep
        self.reward_coefficients = reward_coeff

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
            self.max_thrust*np.sin(self.max_gimbal)*lever_arm/(inertia)*t_free_fall/5.,
            self.ICMean[6]+self.ICRange[6]
            ]),
            1,
            )

        # Set environment bounds
        self.x_bound_right = 0.9*np.maximum(self.state_normalizer[0],100)
        self.x_bound_left = -self.x_bound_right
        self.y_bound_up = 0.9*np.maximum(self.state_normalizer[1],100)
        self.y_bound_down = -30

        # Define observation space
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(7,))

        assert self.observation_space.shape == self.init_space.shape,\
            f"The observation space has shape {self.observation_space.shape} but the init_space has shape {self.init_space.shape}"

        # Two valued vector in the range -1,+1, for the
        # gimbal angle and the thrust command. It will then be
        # rescaled to the appropriate ranges in the dynamics
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

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

    def reset(self):
        """ Function defining the reset method of gym
            It returns an initial observation drawn randomly
            from the uniform distribution of the ICs"""

        initialCondition = self.init_space.sample()
        self.y = initialCondition

        # instantiate the simulator object
        self.SIM = Simulator3DOF(initialCondition, self.timestep)

        return self._normalize_obs(self.y.astype(np.float32))

    def step(self, normalized_action):

        self.action = self._denormalize_action(normalized_action)

        self.y, isterminal, __ = self.SIM.step(self.action)
        state = self.y.astype(np.float32)

        # Done if the rocket is at ground or outside bounds
        done = bool(isterminal) or self._checkBounds(state)

        reward, rewards_dict = self._compute_reward(state, self.action)

        info = {
            "rewards_dict" : rewards_dict,
            "is_done" : done,
            "state_history" : self.SIM.states,
            "action_history" : self.SIM.actions,
            "timesteps" : self.SIM.times,
        }
        
        info["bounds_violation"] = self._checkBounds(state)

        if info['bounds_violation']:
            reward += -10
            
        return self._normalize_obs(state), reward, done, info

    def _compute_reward(self, obs, action):
        reward = 0              

        r = obs[0:2]
        v = obs[3:5]
        zeta = obs[2]-np.pi/2

        v_targ, __ = self.compute_vtarg(r,v)

        thrust = action[1]
         
        # Coefficients
        coeff = self.reward_coefficients

        # Attitude constraints
        zeta_lim = 2*np.pi
        zeta_mgn = np.pi/2        
        
        # Compute each reward term
        rewards_dict = {
            "velocity_tracking" : coeff["alfa"]*np.linalg.norm(v-v_targ),
            "thrust_penalty" : coeff["beta"]*thrust,
            "eta" : coeff["eta"],
            "attitude_constraint" : coeff["gamma"]*float(abs(zeta)>zeta_lim),
            "attitude_hint" : coeff["delta"]*np.maximum(0,abs(zeta)-zeta_mgn),
            "rew_goal": self._reward_goal(obs),
        }

        reward = sum(rewards_dict.values())
        
        return reward, rewards_dict


    def _normalize_obs(self, obs):
        return obs/self.state_normalizer

    def _denormalize_obs(self,obs):
        return obs*self.state_normalizer
        
    def _reward_goal(self, obs):
        k = self.reward_coefficients["kappa"]
        return k*self._check_landing(obs)
    
    def compute_vtarg(self, r, v):
        tau_1 = 20
        tau_2 = 100
        initial_conditions = self.SIM.states[0]

        v_0 = np.linalg.norm(initial_conditions[3:5])

        if r[1]>100:
            r_hat = r-[0,100]
            v_hat = v-[0,-2]
            tau = tau_1

        else:
            r_hat = [0,100]
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
        MAX_SIZE = max(self.y_bound_up, self.x_bound_right)
        step_size = self.window_size / (2*MAX_SIZE)

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

        image = pygame.transform.scale(image, (100*step_size,500*step_size))
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
        landing_pad = pygame.Rect(0,0,2*step_size*self.target_r,10)
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


    def _denormalize_action(self, action : ArrayLike):
        """ Denormalize the action as we've bounded it
            between [-1,+1]. The first element of the 
            array action is the gimbal angle while the
            second is the throttle"""

        gimbal = action[0]*self.max_gimbal

        thrust = (action[1] + 1)/2. * self.max_thrust

        # TODO : Add lower bound on thrust with self.minThrust
        return np.float32([gimbal, thrust])

    def _get_obs(self):
        return self._normalize_obs(self.y)

    def plot_states(self, states = None):
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
        masses = []
        thrusts = []
        gimbals = []
        timesteps = self.SIM.times

        fig1, ax1 = plt.subplots()
        # ax1_1 = ax1.twinx()
        
        if states is None:
            states = self.SIM.states

        for state in states:
            downranges.append(state[0])
            heights.append(state[1])
            ths.append(state[2])
            vxs.append(state[3])
            vzs.append(state[4])
            oms.append(state[5])
            masses.append(state[6]/1e3)

        __, = ax1.plot(timesteps, downranges, label='Downrange (x)')
        __, = ax1.plot(timesteps, heights, label='Height (y)')
        # __, = ax1_1.plot(timesteps, np.rad2deg(ths),'b-')
        
        # __, = ax1.plot(vxs, label='Cross velocity (v_x)')
        __, = ax1.plot(timesteps, vzs, label='Vertical velocity (v_z)')
        __, = ax1.plot(timesteps, masses, label='Mass [T]')

        ax1.legend()
        # ax1_1.set_ylabel('theta [deg]',color='b')
        # ax1_1.tick_params('y', colors='b')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Position/Velocity [m]/[m/s]')

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

        thrust_integral = np.sum(thrusts)/self.max_thrust
        
        return (fig1, fig2, thrust_integral)
        

    def states_to_dataframe(self):
        import pandas as pd
        
        return pd.DataFrame(self.SIM.states, columns=self.state_names)


    def actions_to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.SIM.actions, columns=self.action_names)


    def used_mass(self):
        initial_mass = self.SIM.states[0][6]
        final_mass = self.SIM.states[-1][6]
        return initial_mass-final_mass


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
        zeta = theta-np.pi/2

        __,y = state[0:2]
        vx, vy = state[3:5]
        glideslope = np.arctan2(np.abs(vy),np.abs(vx))

        # Set landing bounds
        v_lim = 15
        r_lim = self.target_r
        # glideslope_lim = np.deg2rad(79)
        zeta_lim = 0.2
        omega_lim = 0.2

        landing_conditions = {
            "zero_height" : y<=1e-3,
            "velocity_limit": v<v_lim,
            "landing_radius" : r<r_lim,
            "attitude_limit" : abs(zeta)<zeta_lim,
            "omega_limit" : abs(vtheta)<omega_lim
        }

        return all(landing_conditions.values())

    def seed(self, seed: int = 42):
        self.init_space.seed(42)
        return super().seed(seed)

    def _get_normalizer(self):
        return self.state_normalizer

    def get_keys_to_action(self):
        import pygame

        mapping = {
            (pygame.K_LEFT,): [1,1],
            (pygame.K_LEFT,pygame.K_UP,): [1,1],
            (pygame.K_RIGHT,): [-1,1],
            (pygame.K_RIGHT,pygame.K_UP,): [-1,1],
            (pygame.K_UP,): [0,1],
            (pygame.K_MODE,): [0,-1],
        }
        return mapping
