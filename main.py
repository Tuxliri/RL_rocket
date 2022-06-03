# This is the gym environment to test the RL algorithms
# on the rocket landing control problem. It is a simplified
# 3DOF version of the real 6DOF dynamics

import numpy as np

from gym.wrappers.time_limit import TimeLimit

from matplotlib import pyplot as plt

from tensorboard import program

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from rocket_env import Rocket, Rocket1D

from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback
from stable_baselines3.common.logger import Figure

def showAgent(env, model):
    # Show the trained agent
    obs = env.reset()
    env.render(mode="human")
    done = False
    rewards = []
    thrusts = []

    while not done:
        #thrust =1
        #action = env.action_space.sample()
        thrust, _states = model.predict(obs)
        obs, rew, done, info = env.step(thrust)
        env.render(mode="human")

    env.SIM._plotStates()

    return None

class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(FigureRecorderCallback, self).__init__(verbose)

    def _on_step(self) -> bool:

        showFig = False
        # [0] needed as the method returns a list containing the tuple of figures
        states_fig, thrust_fig = self.training_env.env_method("plotStates", showFig)[0]

        # Close the figure after logging it
        
        self.logger.record("States", Figure(states_fig, close=True), exclude=("stdout", "log", "json", "csv"))
        self.logger.record("Thrust", Figure(thrust_fig, close=True), exclude=("stdout", "log", "json", "csv"))

        
        return super()._on_step()

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    initialConditions = np.float32([500, 3e3, np.pi/2 , 0, -300, 0, 30e3])
    initialConditionsRange = np.zeros_like(initialConditions)

    env = Rocket(initialConditions, initialConditionsRange, 0.1)
    env = TimeLimit(env, max_episode_steps=400)
    env = Rocket1D(env)  

    # Choose the folder to store tensorboard logs 
    TENSORBOARD_LOGS_DIR = "RL_tests/my_environment/logs"

    model = PPO(
        'MlpPolicy',
        env,
        tensorboard_log=TENSORBOARD_LOGS_DIR,
        verbose=1,
        )

    # Start tensorboard server
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', TENSORBOARD_LOGS_DIR])
    url = tb.launch()
    print(f"Tensorboard listening on {url}")
    
    # Show the random agent 
    
    showAgent(env, model)
     
    # Train the agent
    TRAINING_TIMESTEPS = 5e5
    model.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        callback=EveryNTimesteps(n_steps=TRAINING_TIMESTEPS/10, callback=FigureRecorderCallback())
    )

    # Save the agent
    model.save("PPO_goddard")
    del model  # delete trained model to demonstrate loading

    model = PPO.load("PPO_goddard")
    showAgent(env, model)

    env.close()
    input()