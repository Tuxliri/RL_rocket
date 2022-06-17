from gym.envs.registration import register

# Register the environment
register(
    id='Falcon-v0',
    entry_point='my_environment.main:make1Drocket',
    max_episode_steps=4000
)

register(
    id='Falcon3DOF-v0',
    entry_point='my_environment.main:makerocket',
    max_episode_steps=4000
)
