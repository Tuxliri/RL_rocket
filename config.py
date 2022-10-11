import numpy as np

config = {
    "env_id" : "my_environment/Falcon3DOF-v0",
    "policy_type": "MlpPolicy",
    "total_timesteps": int(6e6),
    "timestep" : 0.1,
    "max_time" : 100,
    "RANDOM_SEED" : 42,
    "initial_conditions" : [-1600, 2000, np.pi*3/4, 180, -90, 0, 41e3],
    "initial_conditions_range" : [0,0,0,0,0,0,0],
    "reward_coefficients" : {
                            "alfa" : -0.01, 
                            "beta" : -1e-8,
                            "delta" : -5,
                            "eta" : 0.1,
                            "gamma" : -10,
                            "kappa" : 10,
                            "xi" : 0.004,
                            "landing_radius" : 50,
                            "w_r_f" : 0.1,
                            "w_v_f" : 0.8,
                            "max_r_f": 100,
                            "max_v_f": 50,
                            },
}