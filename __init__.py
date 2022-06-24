from gym.envs.registration import register
import numpy as np

from gym.wrappers.filter_observation import FilterObservation
from gym.wrappers.flatten_observation import FlattenObservation

from my_environment.envs.rocket_env import Rocket, Rocket1D

# Register the environment
register(
    id='Falcon-v0',
    entry_point='my_environment.main:make1Drocket',
)

register(
    id='Falcon3DOF-v0',
    entry_point='my_environment.main:makerocket'
)

def makerocket(
    initialConditions = [100, 500, np.pi/2, -10, -50, 0, 50e3],
    initialConditionsRange = [0,0,0,0,0,0,0],
    timestep = 0.1
):
    from my_environment.utils.wrappers import DiscreteActions3DOF
     
    env = Rocket(
        np.float32(initialConditions),
        np.float32(initialConditionsRange),
        timestep = timestep
        )

    env = DiscreteActions3DOF(env)

    return env

def make1Drocket(
    initialConditions = [0, 500, np.pi/2, 0, -50, 0, 50e3],
    initialConditionsRange = [0,0,0,0,0,0,0],
    timestep = 0.1
):
    from my_environment.utils.wrappers import DiscreteActions    
    
    env = Rocket(
        np.float32(initialConditions),
        np.float32(initialConditionsRange),
        timestep
        )

    env = Rocket1D(env, goalThreshold=100, rewardType='shaped_terminal')
    env = FlattenObservation(FilterObservation(env, ['observation']))
    env = DiscreteActions(env)

    return env

