import torch
import random

import genesis as gs

from typing import List
from pynput import keyboard


class SimConfig:
    def __init__(self):

        # generic simulation config
        self.dt = 0.01
        self.sim_duration = 5000 # num steps
        self.device = gs.cuda

        # parallel envs
        self.num_envs = 50
        self.env_spacing = (15.0, 15.0)

        # camera config
        self.camera_pos = (0, 0, 10)
        self.camera_lookat = (0.0, 0.0, 0)
        self.camera_fov = 40.0

        # obstacle config
        self.num_obstacles = 10
        self.obstacle_height = 0.1
        self.obstacle_z_pos = 0.0

        self.obstacle_max_radius = 0.3
        self.obstacle_min_radius = 0.1

        self.grid_size = (4.0, 2.0)
        self.obstacle_max_pos = (self.grid_size[0], self.grid_size[1])
        self.obstacle_min_pos = (-self.grid_size[0], -self.grid_size[1])

        self.velocity = 2.0
        self.velocity_max = self.velocity
        self.velocity_min = -self.velocity
        self.resample_velocity_steps = 200

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.linear_increment = 2.0
        self.angular_increment = 2.0

sim_config = SimConfig()

# initialize the physics engine
gs.init(backend=gs.cuda)

# create a scene
scene = gs.Scene(
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        camera_pos=sim_config.camera_pos,
        camera_lookat=sim_config.camera_lookat,
        camera_fov=sim_config.camera_fov,
    ),
    rigid_options=gs.options.RigidOptions(
        dt=sim_config.dt,
    ),
    # vis_options=gs.options.VisOptions(n_rendered_envs=1),
    show_FPS=False,
)

# add simulation plane
plane = scene.add_entity(gs.morphs.Plane())

# add random obstacles, initial placement
dynamic_obstacles: List = []
def is_collision(pos, radius, existing_obstacles):
    for obstacle_dict in existing_obstacles:
        obs_pos = obstacle_dict["pos"]
        obs_radius = obstacle_dict["radius"]

        if (abs(pos[0] - obs_pos[0]) < (radius + obs_radius) and
            abs(pos[1] - obs_pos[1]) < (radius + obs_radius)):
            return True

    return False

for obstacle_idx in range(sim_config.num_obstacles):
    while True:
        # random radius
        radius = random.uniform(sim_config.obstacle_min_radius, sim_config.obstacle_max_radius)

        # random positions
        pos = (random.uniform(sim_config.obstacle_min_pos[0], sim_config.obstacle_max_pos[0]),
               random.uniform(sim_config.obstacle_min_pos[1], sim_config.obstacle_max_pos[1]),
               sim_config.obstacle_z_pos)

        if not is_collision(pos, radius, dynamic_obstacles):
            break

    obstacle = scene.add_entity(gs.morphs.Cylinder(pos=pos, radius=radius, height=sim_config.obstacle_height))
    dynamic_obstacles.append({"obstacle": obstacle, "pos": pos, "radius": radius})


# add blue box covering the entire field
blue_box = scene.add_entity(
    morph=gs.morphs.Box(
        pos=(0.0, 0.0, sim_config.obstacle_height / 8),
        size=(sim_config.grid_size[0] * 2 + 4.0, sim_config.grid_size[1] * 2 + 1.0, sim_config.obstacle_height / 4),
        fixed=True,
        collision=False,
    ),
    surface=gs.surfaces.Rough(
        diffuse_texture=gs.textures.ColorTexture(
            color=(0.0, 0.5, 1.0),
        ),
    ),
)


# add target
target = scene.add_entity(
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

# add boat cover
boat_cover = scene.add_entity(
    morph=gs.morphs.Mesh(
        file="assets/boat.stl",
        scale=0.0008,
        collision=False,
    ),
    surface=gs.surfaces.Rough(
        diffuse_texture=gs.textures.ColorTexture(
            color=(0.0, 0.0, 1.0),
        ),
    ),
)

# add boat collision mesh
boat_collision = scene.add_entity(
    gs.morphs.Cylinder(pos=[-5.4, 0, 0],
                       radius=0.4,
                       height=0.1,
                       collision=True,
                       visualization=False),
    surface=gs.surfaces.Rough(
        diffuse_texture=gs.textures.ColorTexture(
            color=(0.0, 1.0, 0.0),
        ),
    ),
)

# define initial velocities

linear_velocity = sim_config.linear_velocity
angular_velocity = sim_config.angular_velocity

linear_increment = sim_config.linear_increment
angular_increment = sim_config.angular_increment

# build the scene
if sim_config.num_envs == 1:
    scene.build()
else:
    scene.build(n_envs=sim_config.num_envs, env_spacing=sim_config.env_spacing)

# set the boat position

# keyboard handle
def on_press(key):
    global linear_velocity, angular_velocity

    if key == keyboard.Key.up:
        linear_velocity = linear_increment
    if key == keyboard.Key.down:
        linear_velocity = -linear_increment

    if key == keyboard.Key.left:
        angular_velocity = angular_increment
    if key == keyboard.Key.right:
        angular_velocity = -angular_increment

def on_release(key):
    global linear_velocity, angular_velocity

    if key == keyboard.Key.up:
        linear_velocity = 0
    if key == keyboard.Key.down:
        linear_velocity = 0

    if key == keyboard.Key.left:
        angular_velocity = 0
    if key == keyboard.Key.right:
        angular_velocity = 0

# keyboard initiate
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# set the boat cover
boat_cover_pos = gs.tensor([[-5.5, 0.0, 0.1, 0.0, 0.0, 0.0]])
boat_cover_pos = boat_cover_pos.repeat(sim_config.num_envs, 1)
boat_cover.set_dofs_position(boat_cover_pos)

# simulation main loop
for simulation_step in range(sim_config.sim_duration):

    # set the velocities to the boat
    current_angle = boat_collision.get_dofs_position()[0, 5]
    boat_velocity = gs.tensor([[linear_velocity * torch.cos(current_angle),
                                linear_velocity * torch.sin(current_angle),
                                0.0, 0.0, 0.0, angular_velocity]])
    boat_velocity = boat_velocity.repeat(sim_config.num_envs, 1)
    boat_collision.set_dofs_velocity(boat_velocity)

    # place the cover nicely
    boat_cover_pos[:, 0] = boat_collision.get_dofs_position()[:, 0] # offset the boat cover
    boat_cover_pos[:, 1] = boat_collision.get_dofs_position()[:, 1]
    boat_cover_pos[:, 5] = boat_collision.get_dofs_position()[:, 5]
    boat_cover.set_dofs_position(boat_cover_pos)

    if simulation_step % sim_config.resample_velocity_steps == 0:
        for obstacle_dict in dynamic_obstacles:

            velocity_range = sim_config.velocity_max - sim_config.velocity_min
            random_velocity = torch.rand(sim_config.num_envs, 6)

            # scale and shift the first two columns to match the velocity range
            random_velocity[:, :2] = random_velocity[:, :2] * velocity_range + sim_config.velocity_min
            random_velocity[:, 2:] = 0.0

            # convert to gs.tensor
            random_velocity = gs.tensor(random_velocity.tolist())
            obstacle_dict["velocity"] = random_velocity

    for obstacle_dict in dynamic_obstacles:
        obstacle = obstacle_dict["obstacle"]
        velocity = obstacle_dict["velocity"]
        position = obstacle.get_dofs_position()

        # check if the obstacle is about to leave the grid
        mask_x = (position[:, 0] + velocity[:, 0] * sim_config.dt > sim_config.obstacle_max_pos[0]) | \
                 (position[:, 0] + velocity[:, 0] * sim_config.dt < sim_config.obstacle_min_pos[0])

        mask_y = (position[:, 1] + velocity[:, 1] * sim_config.dt > sim_config.obstacle_max_pos[1]) | \
                 (position[:, 1] + velocity[:, 1] * sim_config.dt < sim_config.obstacle_min_pos[1])

        velocity[mask_x, 0] = -velocity[mask_x, 0]
        velocity[mask_y, 1] = -velocity[mask_y, 1]

        obstacle.set_dofs_velocity(velocity)


    # check if the boat is about to leave the grid
    boat_position = boat_collision.get_dofs_position()
    boat_velocity = boat_collision.get_dofs_velocity()

    mask_x = (boat_position[:, 0] + boat_velocity[:, 0] * sim_config.dt > (sim_config.grid_size[0] + 1.5)) | \
             (boat_position[:, 0] + boat_velocity[:, 0] * sim_config.dt < -(sim_config.grid_size[0] + 1.5))
    boat_velocity[mask_x, 0] = 0.0

    mask_y = (boat_position[:, 1] + boat_velocity[:, 1] * sim_config.dt > sim_config.obstacle_max_pos[1]) | \
             (boat_position[:, 1] + boat_velocity[:, 1] * sim_config.dt < sim_config.obstacle_min_pos[1])
    boat_velocity[mask_y, 1] = 0.0

    boat_collision.set_dofs_velocity(boat_velocity)

    # step the simulation
    scene.step()
