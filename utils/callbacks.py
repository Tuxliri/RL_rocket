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
        states_fig, thrust_fig = self.training_env.env_method("plotStates", showFig)[
            0]

        # Close the figure after logging it

        self.logger.record("States", Figure(states_fig, close=True),
                           exclude=("stdout", "log", "json", "csv"))
        self.logger.record("Thrust", Figure(thrust_fig, close=True),
                           exclude=("stdout", "log", "json", "csv"))

        return super()._on_rollout_end()

class RecordTrajectoryCallback():
    def __init__(self) -> None:
        self.trajectories = []  # List of recorded trajectories containig tuples (obs, acts)
        self.trajectoryObs = []
        self.trajectoryAct = []
        self.trajectoryInfos = []

        pass

    def callback(
        self,
        obs_t,
        obs_tp1,
        action,
        rew,
        done: bool,
        info: dict,
    ):
        self.trajectoryObs.append(obs_t)
        self.trajectoryAct.append(action)
        self.trajectoryInfos.append(info)

        if done:
            self.trajectoryObs.append(obs_tp1)
            
            self.trajectories.append(
                (np.array(self.trajectoryObs),
                np.array(self.trajectoryAct),
                self.trajectoryInfos,
                True) # Terminal flag
                )
            self.trajectoryObs, self.trajectoryAct, self.trajectoryInfos = [], [], []
        pass

    def returnTrajectories(self):
        from imitation.data.types import Trajectory

        imitationTrajectories = []

        for traj in self.trajectories:
            observations, actions, infos, isterminal = traj
            
            assert observations.shape[0] == actions.shape[0]+1,\
                f"There needs to be {actions.shape[0]+1} observations"\
                    " but there are {observations.shape[0]}"
                    
            imitationTrajectories.append(Trajectory(
                observations,
                actions,
                infos,
                terminal=isterminal)
                )

        return imitationTrajectories
