import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.logger import Figure

class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(FigureRecorderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_rollout_end(self) -> None:
        showFig = False
        # [0] needed as the method returns a list containing the tuple of figures
        states_fig, action_fig = self.training_env.env_method("plotStates", showFig)[
            0]

        # Close the figure after logging it

        self.logger.record("States", Figure(states_fig, close=True),
                           exclude=("stdout", "log", "json", "csv"))
        self.logger.record("Thrust", Figure(action_fig, close=True),
                           exclude=("stdout", "log", "json", "csv"))

        return super()._on_rollout_end()

