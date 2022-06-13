# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real 6DOF dynamics

import os
import sys
from datetime import datetime

import numpy as np
from genericpath import exists
from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers.flatten_observation import FlattenObservation
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import A2C, DDPG, SAC, TD3, HerReplayBuffer, PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.logger import Figure
from tensorboard import program

from rocket_env import Rocket, Rocket1D


def showAgent(env, model):
    # Show the trained agent
    # env = model.get_env()

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
    initialConditions = np.float32([500, 3e3, np.pi/2, 0, -300, 0, 30e3])
    initialConditionsRange = np.zeros_like(initialConditions)

    env = Rocket(initialConditions, initialConditionsRange,
                 0.1, render_mode="None")
    env = Rocket1D(env, distanceThreshold=100)
    env = TimeLimit(env, max_episode_steps=400)

    return env


if __name__ == "__main__":

    from gym.envs.registration import register
    from wrappers import DiscreteActions
    register(
        'Falcon-v0',
        entry_point='main:make1Drocket',
        max_episode_steps=400
    )

    # Choose the folder to store tensorboard logs
    TENSORBOARD_LOGS_DIR = "RL_tests/my_environment/logs"

    env = make1Drocket()
    env = FlattenObservation(FilterObservation(env, ['observation']))
    env = DiscreteActions(env)

    import pygame
    from gym.utils.play import play
    mapping = {(pygame.K_DOWN,): 0, (pygame.K_UP,): 1}
    play(env, keys_to_action=mapping)

    if env.observation_space.dtype.name == 'float32' :
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
    TRAINING_TIMESTEPS = 5e5

    model.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        callback=EveryNTimesteps(
            n_steps=TRAINING_TIMESTEPS/10, callback=FigureRecorderCallback())
    )

    # Save the agent in the 'models' folder
    date = datetime.now()
    pathname = os.path.dirname(sys.argv[0])
    savefolder = os.path.join(pathname, 'models')

    if not exists(savefolder):
        os.mkdir(savefolder)

    filename = "PPO_" + date.strftime("%Y-%m-%d_%H-%M")

    model.save(os.path.join(savefolder, filename))

    # Load the model
    showAgent(env, model)
