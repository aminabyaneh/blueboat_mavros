# config for the environment, TODO: convert to JSON
env_cfg = {
    # sim
    "dt": 0.01,
    "max_visualize_fps": 20,
    "render": True,
    "show_fps": False,

    # camera
    "camera_pos": (0, 0, 10),
    "camera_lookat": (0.0, 0.0, 0),
    "camera_fov": 40.0,
    "visualize_target": True,
    "visualize_camera": False,

    # envs
    "num_envs": 2,
    "num_visualize_envs": 1,
    "env_spacing": (15.0, 15.0),
    "grid_size": (4.0, 2.0),
    "num_actions": 2, # velocity: magnitude + angle
    "episode_length_seconds": 5.0,
    "target_color": (1.0, 0.0, 0.0),
    "image_observation": True,

    # obstacles
    "obstacle_static": True,
    "num_obstacles": 5,
    "obstacle_height": 0.2,
    "obstacle_z_pos": 0.0,
    "obstacle_max_radius": 0.1,
    "obstacle_min_radius": 0.05,

    "obstacle_max_pos": (4.0, 2.0),
    "obstacle_min_pos": (-4.0, -2.0),
    "obstacle_base_velocity": 1.0,
    "obstacle_resample_velocity_steps": 500,

    # boat shell (visualization)
    "boat_shell_start_pose": (-5.5, 0.0, 0.1, 0.0, 0.0, 0.0),
    "boat_shell_scale": 0.0006,
    "boat_shell_color": (0.0, 0.0, 1.0),

    # boat core (physics)
    "boat_core_start_pose": (-5.5, 0.0, 0.0, 0.0, 0.0, 0.0),
    "boat_core_radius": 0.3,
    "boat_core_height": 0.1,

    "boat_max_lin_speed": 2.0,
    "boat_max_ang_speed": 2.0,
    "action_clamp": 2.0,

    "at_target_threshold": 0.5,
    "collision_margin": 0.05,
}

# config for the observations
obs_cfg = {
    "num_obs": 23,
}

# config for the command
command_cfg = {
    "num_commands": 3,
}

# config for the command
# NOTE: "angular" this is not used, but can be added to penalize angular velocity
reward_cfg = {
    "reward_scales": {
        "target": 2.0,
        "smooth": -0.0001,
    },
}