import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from my_environment.utils.imitation_kickstarter import imitationKickstarter
import gym
from gym.utils.play import PlayPlot
from gym.utils.play import play
from my_environment.wrappers import DiscreteActions3DOF, GaudetStateObs
import pygame
from gym.wrappers import Monitor, TimeLimit
env_id = 'my_environment/Falcon3DOF-v0'
# env_id = 'LunarLander-v2'

config = {
    "env_id" : "my_environment/Falcon3DOF-v0",
    "policy_type": "MlpPolicy",
    "total_timesteps": int(1.5e5),
    "timestep" : 0.05,
    "max_time" : 40,
    "RANDOM_SEED" : 42,
    "initial_conditions" : [0, 500, np.pi/2, 0, -50, 0],
    "initial_conditions_range" : [0,50,0,0,0,0]
}
config["max_ep_timesteps"] = int(config["max_time"]/config["timestep"])


def make_env(config):
    env = gym.make(
    config["env_id"],
    IC=config["initial_conditions"],
    ICRange=config["initial_conditions_range"],
    timestep=config["timestep"],
    seed=config["RANDOM_SEED"]
    )
    
    # Define a new custom action space with only three actions:
    # - no thrust
    # - max thrust gimbaled right
    # - max thrust gimbaled left
    # - max thrust downwards
    env = GaudetStateObs(env)
    env = DiscreteActions3DOF(env)
    env = TimeLimit(env, max_episode_steps=config["max_ep_timesteps"])
    return env

env=make_env(config)
    
# env = gym.make(env_id)
policy = PPO('MlpPolicy', env)

# Plot the reward received


class ReturnCallback():
    def __init__(self) -> None:
        self.rewards = []
        self.returns = []
        self.t = 0
        self.gamma = 0.99
        pass

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
      self.rewards.append(rew)
      self.t+=1
      assert not any(np.abs(obs_t))>1, 'observation outside normalization bounds'

      # Extract the values of each
      rewards_list = list(info["rewards_dict"].values())
      # rewards_keys = list(info["rewards_dict"].keys())
      return [rew,]

    def plotReturns(self, title="Episodic Returns"):
        plt.bar(myCallback.callback)
        plt.title(title)
        plt.show()


myCallback = ReturnCallback()
mapping = {(pygame.K_UP,): 2, (pygame.K_DOWN,): 0, (pygame.K_LEFT,): 1, (pygame.K_RIGHT,): 3}
plotter = PlayPlot(myCallback.callback, 30 * 5,
      ['rew_goal'])
 
play(env, callback=plotter.callback,fps=1/0.05)

behavioural_cloner = imitationKickstarter(env, policy=policy.policy)

trajs = behavioural_cloner.play()

model = behavioural_cloner.train(n_epochs=100)

evaluate_policy(model, env, render=True)
