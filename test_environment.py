import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from my_environment.wrappers.wrappers import RewardAnnealing
import gym
from gym.utils.play import PlayPlot
from gym.utils.play import play
from my_environment.wrappers import DiscreteActions3DOF
import pygame
from gym.wrappers import Monitor, TimeLimit

from main import config
config["max_ep_timesteps"] = int(config["max_time"]/config["timestep"])


def make_env(config):
    env = gym.make(
    config["env_id"],
    IC=config["initial_conditions"],
    ICRange=config["initial_conditions_range"],
    timestep=config["timestep"],
    seed=config["RANDOM_SEED"],
    reward_coeff=config["reward_coefficients"]
    )
    # Anneal the reward (remove v_targ following reward)
    #env = RewardAnnealing(env)

    # Define a new custom action space with only three actions:
    # - no thrust
    # - max thrust gimbaled right
    # - max thrust gimbaled left
    # - max thrust downwards
    # env = DiscreteActions3DOF(env)
    env = TimeLimit(env, max_episode_steps=config["max_ep_timesteps"])
    return env

env=make_env(config)
    
# env = gym.make(env_id)
policy = PPO('MlpPolicy', env)

# Plot the reward received
def callback(obs_t, obs_tp1, action, rew, done, info):
    return [rew,]

plotter = PlayPlot(callback, 30 * 5, ["reward"])
 
play(env)#, callback=plotter.callback,fps=30)