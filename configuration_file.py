from numpy import pi

env_config = {
    "timestep" : 0.05,
    "seed" : 42,
    "IC" : [500, 100, 100,
                            0, 0, 0,
                            1, 0, 0, 0,
                            0,0,0.5,
                            45e3],
    "ICRange" : [0,0,0,
                0,0,0,
                0,0,0,0,
                0,0,0,
                0],
    "reward_coeff" : {
                            "alfa" : -0.01, 
                            "beta" : 0,
                            "delta" : -5,
                            "eta" : 0.2,
                            "gamma" : -10,
                            "kappa" : 10,
                            "xi" : 0.004,
                            },
    "trajectory_limits": {"attitude_limit": [1.5, 1.5, 2*pi]},
    "landing_params" : {
            "waypoint" : 50,
            "landing_radius" : 30,
            "maximum_velocity" : 10,
            "landing_attitude_limit" : [10/180*pi, 10/180*pi, 2*pi], # [Yaw, Pitch, Roll] in RAD,
                                                # rotations order zyx
                                                # VISUALIZATION:
                                                # https://bit.ly/3CoEdvH 
            "omega_lim" : [0.2, 0.2, 0.2]
        }
}

sb3_config = {
    "env_id" : "my_environment/Falcon3DOF-v0",
    "policy_type": "MlpPolicy",
    "total_timesteps": int(5e3),
    "max_time" : 150,
}