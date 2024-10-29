import numpy as np

config = {
    "env_id" : "my_environment/Falcon3DOF-v0",
    "policy_type": "MlpPolicy",
    "total_timesteps": int(2e6),
    "timestep" : 0.1,
    "max_time" : 100,
    "RANDOM_SEED" : 42,
    "initial_conditions" : [50, 500, np.pi, 0, -50, 0, 41e3],
    "initial_conditions_range" : [5,50,0,0,0,0,1e3],
    "reward_coefficients" : {
                            "alfa" : -0.01, 
                            "beta" : -1e-7,
                            "delta" : -5,
                            "gamma" : -10,
                            "eta" : 0.05,
                            "kappa" : 10,
                            "waypoint": 50,
                            "landing_radius" : 30,
                            },
}
