import rocket_env as rkt
from stable_baselines3.common.env_checker import check_env

FALCON = rkt.Rocket()

check_env(FALCON)