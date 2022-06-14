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
from stable_baselines3 import A2C, DDPG, SAC, TD3, PPO, DQN, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.logger import Figure
from tensorboard import program

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from rocket_env import Rocket, Rocket1D

def showAgent(env, model):
    # Show the trained agent
    obs = env.reset()
    env.render(mode="human")
    done = False

    while not done:
        thrust, _states = model.predict(obs)
        obs, rew, done, info = env.step(thrust)
        env.render(mode="human")

    env.plotStates(False)

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


def make1Drocket():
    from wrappers import DiscreteActions

    initialConditions = np.float32([500, 3e3, np.pi/2, 0, -300, 0, 30e3])
    initialConditionsRange = np.zeros_like(initialConditions)

    env = Rocket(initialConditions, initialConditionsRange,
                 timestep = 0.1, render_mode="None")
    env = Rocket1D(env, distanceThreshold=100)
    env = TimeLimit(env, max_episode_steps=4000)
    env = FlattenObservation(FilterObservation(env, ['observation']))
    env = DiscreteActions(env)

    return env


if __name__ == "__main__":

    # Register the environment
    from gym.envs.registration import register
    register(
        'Falcon-v0',
        entry_point='main:make1Drocket',
        max_episode_steps=400
    )

    # Choose the folder to store tensorboard logs
    TENSORBOARD_LOGS_DIR = "RL_tests/my_environment/logs"
    

    # TEST vec env speed
    # By default, we use a DummyVecEnv as it is usually faster (cf doc)
    num_cpu = 8
    env_id = "Falcon-v0"
    vec_env = make_vec_env(env_id, n_envs=num_cpu)

    model = A2C('MlpPolicy', vec_env, verbose=0)


    if vec_env.observation_space.dtype.name == 'float32' :
        model = DQN(
        'MlpPolicy',
        env,
        tensorboard_log=TENSORBOARD_LOGS_DIR,
        verbose=1,
    )

    else:
        model = TD3(
            'MultiInputPolicy',
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
                online_sampling=True,
                max_episode_length=400
            ),
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=256,
        )

    # Start tensorboard server
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', TENSORBOARD_LOGS_DIR])
    url = tb.launch()
    print(f"Tensorboard listening on {url}")

    # Train the agent
    TRAINING_TIMESTEPS = 5e6

    model.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        callback=EveryNTimesteps(
            n_steps=TRAINING_TIMESTEPS/10, callback=FigureRecorderCallback())
    )

    # Load the model
    showAgent(vec_env, model)

    # Save the agent in the 'models' folder
    date = datetime.now()
    pathname = os.path.dirname(sys.argv[0])
    savefolder = os.path.join(pathname, 'models')

    if not exists(savefolder):
        os.mkdir(savefolder)

    filename = "PPO_" + date.strftime("%Y-%m-%d_%H-%M")

    model.save(os.path.join(savefolder, filename))

