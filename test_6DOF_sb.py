"""
Script to test functionality of the 6DOF environment
"""
from my_environment.envs import Rocket6DOF
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import RecordVideo

# Import the initial conditions from the setup file
from configuration_file import config

# Instantiate the environment
env = Rocket6DOF(IC=config["INITIAL_CONDITIONS"])

# Test usage with stable_baselines_3 model
model = PPO('MlpPolicy', env, verbose=0)

# Use a separate environement for evaluation
eval_env = RecordVideo(Rocket6DOF(IC=config["INITIAL_CONDITIONS"]),'6DOF_videos',video_length=500)

import time

start_time = time.time()

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5,render=True)
finish_time = time.time()

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
print(f"time to record the episodes: {finish_time-start_time}")


