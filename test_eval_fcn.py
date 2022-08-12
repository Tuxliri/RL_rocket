import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from main import config, make_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

class CallbackEvaluation():
    def __init__(self) -> None:
        self.successes_counter = 0
        pass

    def dummy_callback(self, locals_dict, globals_dict):
        if locals_dict['dones'].any():
            for info in locals_dict['infos']:
                self.successes_counter += sum(int(info["rewards_dict"]['rew_goal']>0))

        return None


def dummy_callback(a,b):
    print(a)
    print(b)

    return None

def main_fcn():
    env = make_env()
    path = 'MODEL_1'
    model = PPO.load(path=path,env=env)

    CALLBACK = CallbackEvaluation()

    eval_env_list = [make_env,make_env,make_env,make_env,make_env]
    eval_env = DummyVecEnv(eval_env_list)
    eval_env.seed()

    rewards_list, ep_len_list = evaluate_policy(
        model, 
        eval_env, 
        callback=CALLBACK.dummy_callback, 
        return_episode_rewards=True,
        n_eval_episodes=len(eval_env_list),
        render=True
        )
    
    return model

if __name__=='__main__':
    main_fcn()