from imitation.data.types import Trajectory
from imitation.algorithms import bc, dagger
import numpy as np
from stable_baselines3.common import policies
import gym
from gym.utils.play import play
from my_environment.utils.callbacks import RecordTrajectoryCallback

from my_environment.envs.rocket_env import Rocket

class imitationKickstarter():
    def __init__(self, env : gym.Env = None, obs=None, actions=None,
    policy : policies.BasePolicy = None) -> None:
        self.env = env
        self.obs = obs
        self.actions = actions
        self.policy = policy
        self.trajectories = []
        pass

    def play(self, keys_action_map = None):
        assert self.env is not None, 'You need to provide an environment'

        myCallback = RecordTrajectoryCallback()

        play(env=self.env,fps=10,callback=myCallback.callback,keys_to_action=keys_action_map)

        self.trajectories = myCallback.returnTrajectories()

        self.env.close()
        return self.trajectories

    def train(self,n_epochs=1) -> policies.BasePolicy:
        # Train the policy using the trajectories stored in the 
        # self.obs and self.actions attributes
        bc_trainer = bc.BC(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            policy=self.policy,
            demonstrations=self.trajectories
            )

        self.policy = bc_trainer.train(n_epochs=n_epochs)
        
        return bc_trainer.policy

    

def heuristic(env : Rocket, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.
    Args:
        env: The environment
        s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the angle
            s[3] is the horizontal speed
            s[4] is the vertical speed
            s[5] is the angular speed
    Returns:
        a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """
    vx ,vy = s[3:5]
    th = s[2]

    s = s / np.maximum(np.abs(env.ICMean), np.ones_like(env.ICMean))

    angle_targ = s[0] * 0.5 + vx * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    
    angle_targ += np.pi/2
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - th) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (vy) * 0.5


    # if env.continuous:
    #     a = np.array([hover_todo * 50 - 1, angle_todo * 5])
    #     a = np.clip(a, -1, +1)
    # else:
    a = 3

    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 5 # Fire straight down
    elif angle_todo < -0.05:
        a = 8 # Fire to rotate counterclockwise
    elif angle_todo > +0.05:
        a = 2 # Fire to rotate clockwise
        
    return a

