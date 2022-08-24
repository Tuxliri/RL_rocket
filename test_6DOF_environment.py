"""
Script to test functionality of the 6DOF environment
"""
from my_environment.envs import Rocket6DOF
from gym.wrappers import RecordVideo
# Import the initial conditions from the setup file
from configuration_file import config

# Instantiate the environment
env = Rocket6DOF(
    IC=config["INITIAL_CONDITIONS"],
    ICRange=config["IC_RANGE"]
    )
# env = RecordVideo(env,video_folder='video_6DOF_new_rot',name_prefix='new_rotation')

# [delta_y, delta_z, thrust]
null_action = [0.,0.,-1]
non_null_action = [1.,1.,-0.5]

# Initialize the environment
done = False
obs = env.reset()
env.render(mode='human')

while not done:
    obs,rew,done,info = env.step(null_action)
    env.render(mode='human')

fig=env.get_attitude_trajectory()
env.close()

fig.show()