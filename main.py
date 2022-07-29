import gym
import wandb
import numpy as np
import stable_baselines3
import ast

from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from tensorboard import program
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

import my_environment
from my_environment.wrappers.wrappers import DiscreteActions3DOF, RecordVideoFigure
from gym.wrappers import TimeLimit

def make_net_arch(arch):
    return ast.literal_eval(arch)

if __name__ == "__main__":

    # Choose the folder to store tensorboard logs
    TENSORBOARD_LOGS_DIR = "./logs"

    config = {
        "env_id" : "my_environment/Falcon3DOFHER-v0",
        "policy_type": "MultiInputPolicy",
        "total_timesteps": int(8e6),
        "timestep" : 0.05,
        "max_time" : 40,
        "RANDOM_SEED" : 42,
        "initial_conditions" : [50, 500, np.pi/2, 0, -50, 0],
        "initial_conditions_range" : [5,50,0,0,0,0],
        "online_sampling" : True,
        "goal_selection_strategy" : 'future',
        "batch_size" : 1024,
        "net_arch": "[512, 512, 512]",
        "eval_freq" : 50e3
    }

    config["max_ep_timesteps"] = int(config["max_time"]/config["timestep"])
    

    run = wandb.init(
        project="RL_rocket",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )   

    def make_env():
        env = gym.make(
        config["env_id"],
        IC=config["initial_conditions"],
        ICRange=config["initial_conditions_range"],
        timestep=config["timestep"],
        seed=config["RANDOM_SEED"],
        )
        
        # Define a new custom action space with only three actions:
        # - no thrust
        # - max thrust gimbaled right
        # - max thrust gimbaled left
        # - max thrust downwards
   
        env = DiscreteActions3DOF(env)
        env = TimeLimit(env, max_episode_steps=config["max_ep_timesteps"])
        env = Monitor(
            env,
            allow_early_resets=True,
            filename="logs_PPO",
            )
        return env

    env = make_env()
    
    model = DQN(
        config["policy_type"],
        env,
        replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy=config["goal_selection_strategy"],
            online_sampling=config["online_sampling"],
            handle_timeout_termination=True,
        ),
        tensorboard_log=f"runs/{run.id}",
        seed=config["RANDOM_SEED"],
        verbose=1,
        batch_size= config["batch_size"],
        policy_kwargs=dict(net_arch=make_net_arch(config["net_arch"])),
        )

    def make_eval_env():
        training_env = make_env()
        return RecordVideoFigure(training_env, video_folder=f"videos/{run.id}",
        image_folder=f"images/{run.id}", episode_trigger= lambda x: x%5==0 )

  
    eval_env = DummyVecEnv([make_eval_env])
    
    callbacksList = [
        EvalCallback(
            eval_env,
            eval_freq = config["eval_freq"],
            n_eval_episodes = 5,
            render=False,
            deterministic=True,
            ),
        WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
            ),
        ]

    

    # Train the model
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacksList
    )
    
    run.finish()