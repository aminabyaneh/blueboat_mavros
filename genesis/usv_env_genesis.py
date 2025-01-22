"""
usv_env_genesis.py

BoatEnv is an instanciable class for simulating an environment with boats and obstacles using the Genesis simulation framework.

Author: Amin Abyaneh
Year: 2025

TODO:
- Implement boat appearance using the commented code.
- Add functionality for goal-conditioned setup using command_cfg.
- Implement the _resample_commands method.
- Add more complex reward functions: speed, fuel consumption, etc.
- Implement the _is_collision method.
"""

import torch
import math
import random

import genesis as gs
from scipy.spatial.transform import Rotation as R

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# config for the environment
env_cfg = {
    # sim
    "dt": 0.01,
    "max_visualize_fps": 30,
    "total_sim_duration": 5000,
    "device": gs.cuda,

    # camera
    "camera_pos": (0, 0, 10),
    "camera_lookat": (0.0, 0.0, 0),
    "camera_fov": 40.0,
    "visualize_target": True,

    # envs
    "num_envs": 12,
    "env_spacing": (15.0, 15.0),

    "num_actions": 3,
    "episode_length_seconds": 30.0,

    # obstacles
    "num_obstacles": 10,
    "obstacle_height": 0.1,
    "obstacle_z_pos": 0.0,
    "obstacle_max_radius": 0.3,
    "obstacle_min_radius": 0.1,

    "obstacle_max_pos": (4.0, 2.0),
    "obstacle_min_pos": (-4.0, -2.0),
    "obstacle_velocity_max": 1.0,
    "obstacle_velocity_min": -1.0,
    "obstacle_resample_velocity_steps": 200,


    # boat shell (visualization)
    "boat_shell_start_pos": (-5.0, 0.0, 0.0),
    "boat_shell_start_quat": (0.707, 0.707, 0.0, 0.0),
    "boat_shell_scale": 0.25,
    "boat_shell_color": (0.0, 0.0, 1.0),

    # boat core (physics)
    "boat_core_start_pos": (-5.5, 0.0, 0.0),
    "boat_core_start_quat": (0.0, 0.0, 0.0, 1.0),
    "boat_core_size": (1.0, 0.5, 0.15),

    "boat_max_speed": 1.0,
    "at_target_threshold": 0.05,
}

# config for the observations
obs_cfg = {
    "num_obs": env_cfg["num_obstacles"] * (2 + 2 + 1), # obstacle position + velocity + radius
}

# config for the command
command_cfg = {
    "num_commands": 3,
}

class BoatEnv:
    def __init__(self,
                 num_envs,
                 env_cfg,
                 obs_cfg,
                 command_cfg,
                 show_viewer=False,
                 device=device):

        self.device = torch.device(device)

        self.num_envs = num_envs if num_envs > 1 else env_cfg["num_envs"]
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.dt = env_cfg["dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg

        self.command_cfg = command_cfg # TODO: to be used for goal conditioned setup

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_fps"],
                camera_pos=env_cfg["camera_pos"],
                camera_lookat=env_cfg["camera_lookat"],
                camera_fov=env_cfg["camera_fov"],
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add random obstacles, initial placement
        dynamic_obstacles = []

        self.num_obstacles = env_cfg["num_obstacles"]
        for _ in range(self.num_obstacles):
            while True:
                # random radius
                radius = random.uniform(env_cfg["obstacle_min_radius"], env_cfg["obstacle_max_radius"])

                # random positions
                position = (random.uniform(env_cfg["obstacle_min_pos"][0], env_cfg["obstacle_max_pos"][0]),
                            random.uniform(env_cfg["obstacle_min_pos"][1], env_cfg["obstacle_max_pos"][1]),
                            env_cfg["obstacle_z_pos"])

                # check for initial collision
                if not self._is_collision(position, radius, dynamic_obstacles):
                    break

            obstacle = self.scene.add_entity(gs.morphs.Cylinder(pos=position, radius=radius,
                                                           height=env_cfg["obstacle_height"]))
            dynamic_obstacles.append({"obstacle": obstacle, "position": position, "radius": radius})


        # add target marker
        self.target = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="meshes/sphere.obj",
                scale=0.2,
                pos=(5.0, 0.0, 0.0),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
        )

        # TODO: add the boat shell for visualization
        # self.boat_shell_pos_init = torch.tensor(env_cfg["boat_start_pos"], device=self.device, dtype=gs.tc_float)
        # self.boat_shell_quat_init = torch.tensor([0.707, 0.707, 0.0, 0.0], device=self.device, dtype=gs.tc_float)

        # self.boat = self.scene.add_entity(
        #     morph=gs.morphs.Mesh(
        #         file="meshes/boat/boat.obj",
        #         scale=env_cfg["boat_scale"],
        #         collision=False,
        #     ),

        #     surface=gs.surfaces.Rough(
        #         diffuse_texture=gs.textures.ColorTexture(
        #             color=self.env_cfg["boat_color"],
        #         ),
        #     ),
        # )

        # add boat core for physics
        self.boat_core_pos_init = torch.tensor(env_cfg["boat_core_start_pos"], device=self.device, dtype=gs.tc_float)
        self.boat_core_quat_init = torch.tensor(env_cfg["boat_core_start_quat"], device=self.device, dtype=gs.tc_float)

        self.boat_core = self.scene.add_entity(
            gs.morphs.Box(pos=env_cfg["boat_core_start_pos"],
                quat=env_cfg["boat_core_start_quat"],
                size=env_cfg["boat_core_size"],
                collision=True)
        )

        # build scene
        self.scene.build(n_envs=self.num_envs, env_spacing=env_cfg["env_spacing"])

        # initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        # targets
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # add for the boat later
        self.base_pos = torch.tensor(env_cfg["boat_core_start_pos"],
                                     device=self.device,
                                     dtype=gs.tc_float).repeat(self.num_envs, 1)

        self.base_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.obstacle_pos = torch.zeros((self.num_envs, self.num_obstacles, 3), device=self.device, dtype=gs.tc_float)
        self.obstacle_size = torch.zeros((self.num_envs, self.num_obstacles, 3), device=self.device, dtype=gs.tc_float)
        self.obstacle_vel = torch.zeros((self.num_envs, self.num_obstacles, 3), device=self.device, dtype=gs.tc_float)
        self.last_obstacle_pos = torch.zeros_like(self.obstacle_pos)

        self.extras = dict()  # extra information for logging


    # def _resample_commands(self, envs_idx):
    #     self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
    #     self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
    #     self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)
    #     if self.target is not None:
    #         self.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    def _at_target(self):
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        )
        return at_target

    def step(self, actions):
        # apply actions
        self.actions[:] = actions

        # step the simulation
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.drone.get_pos()
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[:] = self.drone.get_quat()

        # resample commands
        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        # check termination and reset
        self.crash_condition = (
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"])
            | (torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"])
            | (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 1]) > self.env_cfg["termination_if_y_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
            | (self.base_pos[:, 2] < self.env_cfg["termination_if_close_to_ground"])
        )
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                torch.clip(self.rel_pos * self.obs_scales["rel_pos"], -1, 1),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1, 1),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1, 1),
                self.last_actions,
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.last_base_pos[envs_idx] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.drone.set_pos(self.base_pos[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.drone.set_quat(self.base_quat[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.drone.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_yaw(self):
        yaw = self.base_euler[:, 2]
        yaw = torch.where(yaw > 180, yaw - 360, yaw) / 180 * 3.14159  # use rad for yaw_reward
        yaw_rew = torch.exp(self.reward_cfg["yaw_lambda"] * torch.abs(yaw))
        return yaw_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.base_ang_vel / 3.14159, dim=1)
        return angular_rew

    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew

    # ------------ misc functions----------------
    def _is_collision(pos, radius, existing_obstacles):
        for obstacle_dict in existing_obstacles:
            obs_pos = obstacle_dict["position"]
            obs_radius = obstacle_dict["radius"]

            if (abs(pos[0] - obs_pos[0]) < (radius + obs_radius) and
                abs(pos[1] - obs_pos[1]) < (radius + obs_radius)):
                return True

        return False

    def _convert_rotation(input_rotation, input_type='quat', output_type='euler'):
        if input_type == output_type:
            return input_rotation

        if input_type == 'quat':
            r = R.from_quat(input_rotation)
        elif input_type == 'euler':
            r = R.from_euler('zyx', input_rotation, degrees=True)

        if output_type == 'euler':
            return r.as_euler('xyz', degrees=False)
        elif output_type == 'quat':
            return r.as_quat()

        return None