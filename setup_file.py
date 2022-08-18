config = {
    "env_id" : "my_environment/Falcon3DOF-v0",
    "policy_type": "MlpPolicy",
    "total_timesteps": int(5e3),
    "timestep" : 0.05,
    "max_time" : 150,
    "RANDOM_SEED" : 42,
    "INITIAL_CONDITIONS" : [500, 100, 100,
                            0, -10, -10,
                            1, 0, 0, 0,
                            0,0,0,
                            50e3],
    "IC_RANGE" : [50,10,10,
                1,1,1,
                0,0,0,0,
                0,0,0,
                1e3],
    "reward_coefficients" : {
                            "alfa" : -0.01, 
                            "beta" : 0,
                            "delta" : -5,
                            "eta" : 0.2,
                            "gamma" : -10,
                            "kappa" : 10,
                            "xi" : 0.004,
                            },
    "landing_params" : {
            "waypoint" : 50,
            "landing_radius" : 30,
            "maximum_velocity" : 10,
            "attitude_lim" : [1,0,0,0], #TODO: COMPUTE THE CORRECT VALUES!!
            "omega_lim" : [0.2, 0.2, 0.2]
        }
}