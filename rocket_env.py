# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real 6DOF dynamics

import numpy as np
import gym
from gym import spaces, Env, GoalEnv
from gym.wrappers.time_limit import TimeLimit

from simulator import Simulator3DOF
from matplotlib import pyplot as plt

from renderer_utils import blitRotate

from stable_baselines3 import DDPG, HerReplayBuffer, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

MAX_SIZE_RENDER = 10e3      # Max size in meters of the rendering window

class Rocket(Env):

    """ Simple environment simulating a 3DOF rocket
        with rotational dynamics and translation along
        two axis """

    metadata = {"render_modes": [
        "human", "rgb_array", "plot"], "render_fps": 40}

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
        self.maxGimbal = np.deg2rad(20)     # [rad]
        self.maxThrust = 981e3              # [N]
        self.minThrust = 0.4*981e3          # [N]
        self.dryMass = 25.6e3               # [kg]

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

        # The following line uses the util class Renderer to gather a collection of frames
        # using a method that computes a single frame. We will define _render_frame below.
        # ???WHERE IS THIS Renderer utils function???
        #self.renderer = Renderer(render_mode, self._render_frame)

    def step(self, action):

        u = self._denormalizeAction(action)

        self.y, info = self.SIM.step(u)

        reward = - self.y[1]

        # Done if the rocket is at ground
        done = self._checkTerminal(self.y.astype(np.float32))

        assert done is not bool, "done is not of type bool!"

        obs = self.y.astype(np.float32)

        return obs, reward, done, info

    def _checkTerminal(self, state):
        """
        massCheck : check that the current stage mass is greater than the dryMass
        heightCheck : check that we have not reached ground
        """

        massCheck = (self.y[6] <= self.dryMass)
        heightCheck = (self.y[1] <= 0)

        return bool(heightCheck or massCheck)

    def render(self, mode="human"):
        if (self.window is None) and mode is "human":
            import pygame  # import here to avoid pygame dependency with no render

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        return self._render_frame(mode)

    def _render_frame(self, mode: str):
        # avoid global pygame dependency. This method is not called with no-render.
        import pygame

        # The number of pixels per each meter
        step_size = self.window_size / MAX_SIZE_RENDER

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
            return None
            """ np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            ) """

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
        self.SIM = Simulator3DOF(initialCondition, self.timestep)

        return self.y.astype(np.float32)

    def _denormalizeAction(self, action):
        """ Denormalize the action as we've bounded it
            between [-1,+1]. The first element of the 
            array action is the gimbal angle while the
            second is the throttle"""
        
        assert isinstance(action, (np.ndarray)) and action.shape==(2,),\
            f"Action is of type {type(action)}, shape: {action.shape}"

        gimbal = action[0]*self.maxGimbal

        thrust = (action[1] + 1)/2 * self.maxThrust

        return np.float32([gimbal, thrust])

class Rocket1D(gym.Wrapper, GoalEnv):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self.env = env
        self.observation_space = spaces.Dict({'observation' : spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
            ), 
            'desired_goal' : spaces.Box(
            low=-1.,
            high=1.,
            shape=(2,),
            dtype=np.float32
            ), 
            'achieved_goal' : spaces.Box(
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

    def step(self, thrust):
        
        action = np.float32([0.0, thrust[0]])
        obs, _, done, info = self.env.step(action)
        height, velocity = obs[1], obs[4]

        rew = 0
        #rew = - (velocity**2)

        if done is True:
            rew = -10*(velocity**2) - 100*(height**2)

        """
        Return the height and vertical velocity
        of the rocket as the only observations
        available 
        """
        observation = dict({
            'observation' : np.float32([height, velocity]),
            'achieved_goal' : np.float32([height, velocity]),
            'desired_goal' : self.desired_goal
        })
        return observation, rew, done, info

    def reset(self):
        obs = self.env.reset()
        height, velocity = obs[1], obs[4]

        observation = dict({
            'observation' : np.float32([height, velocity]),
            'achieved_goal' : np.float32([height, velocity]),
            'desired_goal' : self.desired_goal
        })

        return observation

    def compute_reward(self, achieved_goal: object, desired_goal: object, info) -> float:

    
        rew = -np.linalg.norm((np.array(achieved_goal)-np.array(desired_goal))**2,axis=0).sum()
        return rew

def showAgent(env, model):
    # Show the trained agent
    obs = env.reset()
    env.render(mode="human")
    done = False
    rewards = []
    thrusts = []

    while not done:
        
        thrust, _states = model.predict(obs)
        obs, rew, done, info = env.step(thrust)
        thrusts.append(thrust)
        env.render(mode="human")

    fig, ax = plt.subplots()
    ax.plot(thrusts)
    plt.show()

    env.SIM._plotStates()

    return None

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    initialConditions = np.float32([500, 3e3, np.pi/2 , 0, -300, 0, 30e3])
    initialConditionsRange = np.zeros_like(initialConditions)

    env = Rocket(initialConditions, initialConditionsRange, 0.1)
    env = TimeLimit(env, max_episode_steps=400)
    env = Rocket1D(env)  


    model = TD3(
        'MultiInputPolicy',
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=8,
            goal_selection_strategy='future',
            online_sampling=True,
            max_episode_length=400,
        ),
        tensorboard_log="RL_tests/my_environment/logs/HER",
        verbose=1,
        )


    # Show the random agent 
    
    showAgent(env, model)
    
    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Train the agent
    model.learn(total_timesteps=1e4)

    # Save the agent
    model.save("TD3_goddard")
    # del model  # delete trained model to demonstrate loading

    #model = TD3.load("TD3_goddard")
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    showAgent(env, model)

    env.close()
    input()
