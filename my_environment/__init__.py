from gym.envs.registration import register

# Register the environment
register(
    id='my_environment/Falcon1DOF-v0',
    entry_point='my_environment.envs:Rocket1D',
)

register(
    id='my_environment/Falcon3DOF-v0',
    entry_point='my_environment.envs:Rocket'
)
