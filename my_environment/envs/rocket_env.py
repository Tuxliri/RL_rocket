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
                        "waypoint" : 50,
                        "landing_radius" : 30,
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
        self.vtarg_history = None
        self.atarg_history = None

        # Landing parameters
        self.target_r = reward_coeff["landing_radius"]
        self.waypoint = reward_coeff["waypoint"]

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
        self.vtarg_history = []
        self.atarg_history = []
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
            reward += -50
            
        return self._normalize_obs(state), reward, done, info

    def _compute_reward(self, state, action):
        reward = 0              

        r = state[0:2]
        v = state[3:5]
        zeta = state[2]-np.pi/2
        m = state[6]

        # Get INERTIAL ACCELERATION due to thrust
        a = self.SIM.get_thrust_acceleration()

        a_targ, __ = self.get_atarg(r,v,m)

        thrust = action[1]
         
        # Coefficients
        coeff = self.reward_coefficients

        # Attitude constraints
        zeta_lim = 2*np.pi
        zeta_mgn = np.pi/2        
        
        # Compute each reward term
        rewards_dict = {
            "acceleration_tracking" : coeff["alfa"]*np.linalg.norm(a-a_targ),
            "thrust_penalty" : coeff["beta"]*thrust,
            "eta" : coeff["eta"],
            "attitude_constraint" : coeff["gamma"]*float(abs(zeta)>zeta_lim),
            "attitude_hint" : coeff["delta"]*np.maximum(0,abs(zeta)-zeta_mgn),
            "rew_goal": self._reward_goal(state),
        }

        reward = sum(rewards_dict.values())
        
        return reward, rewards_dict


    def _normalize_obs(self, obs):
        return obs/self.state_normalizer


    def _denormalize_obs(self,obs):
        return obs*self.state_normalizer


    def _reward_goal(self, obs):
        k = self.reward_coefficients["kappa"]
        landing_rews = self._check_landing(obs)
        return k*landing_rews[0] + sum(np.maximum(100-np.array(landing_rews[1:2])**1.2,0))
    

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
        m = self.y[6]

        a = np.array(self.SIM.get_thrust_acceleration())

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

        # Draw the target acceleration vector
        a_targ, __ = self._compute_atarg(r,v,m)
        accel_arrow_length = 5
        pygame.draw.line(
            canvas,
            (0,0,0),
            start_pos=tuple(agent_location),
            end_pos=tuple(agent_location+[1,-1]*a_targ*accel_arrow_length),
            width=2
            )
        
        # Draw the current thrust acceleration vector       
        pygame.draw.line(
            canvas,
            (0,0,255),
            start_pos=tuple(agent_location),
            end_pos=tuple(agent_location+[1,-1]*a*accel_arrow_length),
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

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
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


    def states_to_dataframe(self):
        import pandas as pd
        
        return pd.DataFrame(self.SIM.states, columns=self.state_names)


    def actions_to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.SIM.actions, columns=self.action_names)


    def vtarg_to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.vtarg_history, columns=['v_x', 'v_y'])

    def used_mass(self):
        initial_mass = self.SIM.states[0][6]
        final_mass = self.SIM.states[-1][6]
        return initial_mass-final_mass
    
    
    def _compute_atarg(self,r,v,mass):
        
        def __compute_t_go(r,v) -> float:
            # In order to compute the t_go the following depressed
            # quartic equation has to be solved:
            # $g^2t_{go}^4-4||\mathbf{v}||^2t_{go}^2-24\mathbf{r}^t\mathbf{v}t_{go}-36||\mathbf{r}||^2=0

            solutions = np.roots([
                g[1]**2,
                0,
                -4*np.linalg.norm(v)**2,
                -24*np.dot(r,v),
                -36*np.linalg.norm(r)**2,
                ])

            real_positive_roots = [n for n in solutions if (n.imag == 0 and n.real>0)][0].real

            # Check that we have only one real solution
            #assert len(real_positive_roots) == 1, 'Multiple real solutions to t_go equation'

            return real_positive_roots

        g = [0,-9.81] # Gravitational vector

        # Determine the time to go
        t_go = __compute_t_go(r,v)

        def saturation(q,U) -> np.ndarray:
            # Saturation function of vector q w.r.t magnitude U
            q_norm = np.linalg.norm(q)
            if q_norm<=U:
                return q
            else:
                return q*U/q_norm

        # Compute the saturated optimal target velocity
        a_targ = saturation(
            -6*r/t_go**2 - 4*v/t_go - g,
            self.max_thrust/mass
            )

        self.atarg_history.append(np.concatenate((a_targ,[t_go])))

        return a_targ, t_go

    def get_atarg(self,r,v,m):

        return self._compute_atarg(r,v,m)

    def atarg_to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.atarg_history, columns=["ax", "ay", "t_go"])


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

        # Set landing bounds
        v_lim = 15
        r_lim = self.target_r
        zeta_lim = 0.2
        omega_lim = 0.2

        landing_conditions = {
            "zero_height" : y<=1e-3,
            "velocity_limit": v<v_lim,
            "landing_radius" : r<r_lim,
            "attitude_limit" : abs(zeta)<zeta_lim,
            "omega_limit" : abs(vtheta)<omega_lim
        }
        if not landing_conditions["zero_height"]:
            v=0
            r=0
            
        return all(landing_conditions.values()), v, r

    def seed(self, seed: int = 42):
        self.init_space.seed(seed)
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
