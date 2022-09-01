import os
import my_environment
import gym
import wandb

from gym.wrappers import TimeLimit
from configuration_file import env_config, sb3_config

from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from my_environment.wrappers import EpisodeAnalyzer
from wandb.integration.sb3 import WandbCallback

def make_env():
    kwargs = env_config
    env = gym.make("my_environment/Falcon6DOF-v0",**kwargs)
    env = TimeLimit(env, max_episode_steps=sb3_config["max_ep_timesteps"])
    env = Monitor(env)

    
    
    return env

def start_training():

    run = wandb.init(
        project="test_runs",
        config={**env_config, **sb3_config},
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )   

    env = make_env()

    model = PPO(
        sb3_config["policy_type"],
        env,
        tensorboard_log=f"runs/{run.id}",
        verbose=2,
        seed=env_config["seed"],
        ent_coef=0.01,
        )

    def make_eval_env():
        training_env = make_env()
        return EpisodeAnalyzer(training_env)
        # return EpisodeAnalyzer6DOF(training_env,video_folder=f"videos_6DOF/{run.id}",
            # episode_trigger=lambda x: x%5==0)
    
    eval_env = DummyVecEnv([make_eval_env])

    callbacksList = [
        EvalCallback(
            eval_env,
            eval_freq = sb3_config["eval_freq"],
            n_eval_episodes = 5,
            render=False,
            deterministic=True,
            verbose=2,
            log_path='evaluation_logs'
            ),
        WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
            ),
        ]

    # Train the model
    model.learn(
        total_timesteps=sb3_config["total_timesteps"],
        callback=callbacksList
    )

    savepath = os.getcwd()
    model.save(savepath)

    run.finish()

    return None

if __name__=="__main__":
    start_training()