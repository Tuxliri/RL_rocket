"""
Script to test functionality of the 6DOF environment
"""
from stable_baselines3.common.env_checker import check_env
from my_environment.envs import Rocket6DOF

# Import the initial conditions from the setup file
from configuration_file import config

# Instantiate the environment
env = Rocket6DOF(IC=config["INITIAL_CONDITIONS"])

# Check for the environment compatibility with gym and sb3
# check_env(env, skip_render_check=False)

null_action = [0.,0.,-1]

# Initialize the environment
done = False
obs = env.reset()
env.render(mode='human')

while not done:
    obs,rew,done,info = env.step(null_action)
    env.render(mode='human')

fig=env.get_trajectory_plotly()
env.close()

fig.show()