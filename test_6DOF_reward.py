"""
Script to test the reward shaping
"""
from my_environment.envs import Rocket6DOF
from gym.utils.play import play, PlayPlot

# Import the initial conditions from the setup file
from configuration_file import config

# Instantiate the environment
env = Rocket6DOF(
    IC=config["INITIAL_CONDITIONS"],
    ICRange=config["IC_RANGE"],
    reward_coeff=config["reward_coefficients"]
    )

# Define a callback to plot the reward
def callback(obs_t, obs_tp1, action, rew, done, info):
        return [rew,]

plotter = PlayPlot(callback, 30 * 20, ["reward"])


play(env,callback=plotter.callback)
