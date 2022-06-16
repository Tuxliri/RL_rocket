# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real 6DOF dynamics

import os
import sys
from datetime import datetime
import time

import numpy as np
from genericpath import exists

from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers.flatten_observation import FlattenObservation

from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure


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


class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(FigureRecorderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:

        showFig = False
        # [0] needed as the method returns a list containing the tuple of figures
        states_fig, thrust_fig = self.training_env.env_method("plotStates", showFig)[
            0]

        # Close the figure after logging it

        self.logger.record("States", Figure(states_fig, close=True),
                           exclude=("stdout", "log", "json", "csv"))
        self.logger.record("Thrust", Figure(thrust_fig, close=True),
                           exclude=("stdout", "log", "json", "csv"))

        return super()._on_step()


def make1Drocket(
    initialConditions = np.float32([500, 500, np.pi/2, 0, -50, 0, 50e3]),
    initialConditionsRange = np.float32([0,0,0,0,0,0,0])
):
    from my_environment.utils.wrappers import DiscreteActions

    
    
    env = Rocket(initialConditions, initialConditionsRange,
                 timestep = 0.1, render_mode="None", maxTime=20)

    env = Rocket1D(env, goalThreshold=100, rewardType='shaped_terminal')
    env = TimeLimit(env, max_episode_steps=40000)
    env = FlattenObservation(FilterObservation(env, ['observation']))
    env = DiscreteActions(env)

    return env