__all__ = [
    "DiscreteActions",
    "DiscreteActions3DOF",
    "GaudetStateObs",
    "RewardAnnealing",
    "RecordVideoFigure",
]

import os
from typing import Callable
import gym
from gym.spaces import Discrete, Box
from gym.wrappers import RecordVideo
from matplotlib import pyplot as plt

from my_environment.envs.rocket_env import Rocket
import numpy as np
from gym import logger
import wandb
import pandas as pd


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, disc_to_cont=[np.array([-1]), np.array([1])]):
        super().__init__(env)
        self.disc_to_cont = disc_to_cont
        self._action_space = Discrete(len(disc_to_cont))

    def action(self, act):
        return self.disc_to_cont[act]

    def get_keys_to_action(self):
        import pygame

        mapping = {(pygame.K_UP,): 1, (pygame.K_DOWN,): 0}
        return mapping


class DiscreteActions3DOF(gym.ActionWrapper):
    def __init__(self, env, disc_to_cont=[[0, -1], [-1, +1], [0, +1], [+1, +1]]):
        super().__init__(env)
        # Create an action table for all possible
        # combinations of the values of thrust and
        # gimbaling action = [delta, thrust]

        self.disc_to_cont = disc_to_cont
        self._action_space = Discrete(len(disc_to_cont))

    def action(self, act):
        return np.asarray(self.disc_to_cont[act])

    def get_keys_to_action(self):
        import pygame

        mapping = {
            (pygame.K_LEFT,): 1,
            (pygame.K_RIGHT,): 3,
            (pygame.K_UP,): 2,
            (pygame.K_MODE,): 0,
        }
        return mapping


class GaudetStateObs(gym.ObservationWrapper):
    def __init__(self, env: Rocket) -> None:
        super().__init__(env)
        self.observation_space = Box(low=-1, high=1, shape=(4,))

    def observation(self, observation):
        x, y, th = observation[0:3]
        vx, vy, vth = observation[3:6]

        r = np.array([x, y])
        v = np.array([vx, vy])

        v_targ, t_go = self.env.unwrapped.compute_vtarg(r, v)
        vx_targ, vy_targ = v_targ

        return np.float32([vx - vx_targ, vy - vy_targ, t_go, y])

class RewardAnnealing(gym.Wrapper):
    def __init__(self, env: gym.Env, thrust_penalty : float = 0.01) -> None:
        super().__init__(env)
        self.xi = thrust_penalty

    def step(self, action):
        obs, __, done, info = super().step(action)
        rewards_dict = info["rewards_dict"]

        rew = -self.xi*float(action[1]>0.1) + rewards_dict["attitude_hint"] +\
            rewards_dict["attitude_constraint"] + rewards_dict["rew_goal"]

        return obs, rew, done, info

class RecordVideoFigure(RecordVideo):
    def __init__(
        self,
        env,
        video_folder: str,
        image_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
    ):
        super().__init__(
            env, video_folder, episode_trigger, step_trigger, video_length, name_prefix
        )

        self.image_folder = os.path.abspath(image_folder)
        # Create output folder if needed
        if os.path.isdir(self.image_folder):
            logger.warn(
                f"Overwriting existing images at {self.image_folder} folder (try specifying a different `image_folder` for the `RecordVideoFigure` wrapper if this is not desired)"
            )
        os.makedirs(self.image_folder, exist_ok=True)
        self.rewards_info = []

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        if not self.is_vector_env:
            self.rewards_info.append(infos["rewards_dict"])

        if self.episode_trigger(self.episode_id):
            if not self.is_vector_env:
                if dones:
                    fig_states, fig_actions, thrust_integral = self.env.unwrapped.plot_states()
                    plt.close()
                    plt.close()

                    fig_rew = pd.DataFrame(self.rewards_info).plot()
                    plt.close()

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "states": fig_states,
                                "actions": fig_actions,
                                "rewards": fig_rew,
                                "landing_success": infos["rewards_dict"]["rew_goal"],
                                "thrust_integral" : thrust_integral,
                            }
                        )
                    else:
                        self.save_figure(fig_states, "states_figure")
                        self.save_figure(fig_actions, "actions_figure")

            elif dones[0]:
                fig_states, fig_actions = self.env.unwrapped.plot_states()
                plt.close()
                plt.close()

                fig_rew = pd.DataFrame(self.rewards_info).plot()

                if wandb.run is not None:
                    wandb.log(
                        {
                            "states": fig_states,
                            "actions": fig_actions,
                            "rewards": fig_rew,
                        }
                    )
                else:
                    self.save_figure(fig_states, "states_figure")
                    self.save_figure(fig_actions, "actions_figure")

            pass

        return observations, rewards, dones, infos

    def reset(self, **kwargs):
        self.rewards_info = []
        return super().reset(**kwargs)

    def save_figure(self, figure: plt.Figure, prefix):
        figure_name = f"{prefix}-step-{self.step_id}"
        if self.episode_trigger:
            figure_name = f"{prefix}-step-{self.episode_id}"

        base_path = os.path.join(self.image_folder, figure_name)

        figure.savefig(base_path)

        return None
