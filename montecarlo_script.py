from asyncore import write
import numpy as np
from stable_baselines3.common.evaluation import  evaluate_policy
from stable_baselines3.common import results_plotter
from config import config
from stable_baselines3 import PPO
from main import make_env
import csv

# open the file in the write mode
f = open('results_montecarlo.csv', 'w')
writer = csv.writer(f)

header = ['final_position','final_velocity']

# write the header
writer.writerow(header)

def test_callback(locals, globals):
    if locals['dones'][0]:
        x,z,theta,vx,vz,omega,mass=locals['info']['state_history'][-1]
        r = np.linalg.norm([x,z])
        v = np.linalg.norm([vx,vz])
        writer.writerow([r,v])
    return None

env = make_env()

model = PPO.load('model', env)
evaluate_policy(
    model=model,
    env=env,
    n_eval_episodes=100,
    callback=test_callback,
    )

# close the file
f.close()

import pandas as pd
pd.options.plotting.backend = "plotly"

result_dataframe = pd.read_csv('results_montecarlo.csv')

fig_position = result_dataframe['final_position'].plot.hist()
fig_velocity = result_dataframe['final_velocity'].plot.hist()
print("The final position has mean:{mu} and standard deviation: {sigma}".format(
    mu=result_dataframe['final_position'].mean(),
    sigma=result_dataframe['final_position'].std()
    )
    )
print("The final velocity has mean:{mu} and standard deviation: {sigma}".format(
    mu=result_dataframe['final_velocity'].mean(),
    sigma=result_dataframe['final_velocity'].std()
    )
    )
fig_position.show()
fig_velocity.show()