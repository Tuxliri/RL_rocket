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
        thrusts.append(thrust)
        env.render(mode="human")

    fig, ax = plt.subplots()
    ax.plot(thrusts)
    plt.show(block=False)

    env.SIM._plotStates()

    return None

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
    
    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Train the agent
    model.learn(total_timesteps=8e5)
    # Save the agent
    model.save("PPO_goddard")
    del model  # delete trained model to demonstrate loading

    model = PPO.load("PPO_goddard")
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    showAgent(env, model)

    env.close()
    input()