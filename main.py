# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real 6DOF dynamics

import numpy as np
from genericpath import exists

from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers.flatten_observation import FlattenObservation

from gym.wrappers.time_limit import TimeLimit

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from my_environment.envs.rocket_env import Rocket, Rocket1D
import gym

def showAgent(env, model, plotStates=False):
    # Show the trained agent
    obs = env.reset()
    env.render(mode="human")
    done = False

    while not done:
        thrust, _states = model.predict(obs)
        obs, rew, done, info = env.step(thrust)
        env.render(mode="human")

    env.close()

    env.plotStates(plotStates)

    return None



def make1Drocket(
    initialConditions = np.float32([500, 500, np.pi/2, 0, -50, 0, 50e3]),
    initialConditionsRange = np.float32([0,0,0,0,0,0,0])
):
    from my_environment.utils.wrappers import DiscreteActions    
    
    env = Rocket(initialConditions, initialConditionsRange,
                 timestep = 0.1, render_mode="None")

    env = Rocket1D(env, goalThreshold=100, rewardType='shaped_terminal')
    env = TimeLimit(env, max_episode_steps=40000)
    env = FlattenObservation(FilterObservation(env, ['observation']))
    env = DiscreteActions(env)

    return env

def makerocket(
    initialConditions = np.float32([100, 500, np.pi/2, -10, -50, 0, 50e3]),
    initialConditionsRange = np.float32([0,0,0,0,0,0,0])
):
    from my_environment.utils.wrappers import DiscreteActions3DOF
     
    env = Rocket(initialConditions, initialConditionsRange,
                 timestep = 0.1, render_mode="None")

    # env.continuous = True
    env = DiscreteActions3DOF(env)

    return env