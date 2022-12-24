import os 
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import torch
import random 
import numpy as np

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Holybro x500 Drone")

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# set PhysX-specific parameters
sim_params.physx.use_gpu = True

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "assets"

# load holybro asset
holybro_asset_file = "x500/x500.urdf"
asset_options = gymapi.AssetOptions()

asset_options.fix_base_link = False
asset_options.flip_visual_attachments = False
holybro = gym.load_asset(sim, asset_root, holybro_asset_file, asset_options)

asset_options.fix_base_link = True
marker_asset = gym.create_sphere(sim, 0.1, asset_options)

# configure holybro dofs
dof_props = gym.get_asset_dof_properties(holybro)
num_dofs = gym.get_asset_dof_count(holybro)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

dof_lower_limits = []
dof_upper_limits = []
for i in range(num_dofs):
    dof_lower_limits.append(dof_props['lower'][i])
    dof_upper_limits.append(dof_props['upper'][i])

dof_lower_limits = to_torch(dof_lower_limits, device=device)
dof_upper_limits = to_torch(dof_upper_limits, device=device)
dof_ranges = dof_upper_limits - dof_lower_limits

# configure env grid
num_envs = 10
num_per_row = int(math.sqrt(num_envs))
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

default_pose = gymapi.Transform()
default_pose.p = gymapi.Vec3(0.0, 1.0, 5.0)
# default_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

default_marker_pose = gymapi.Transform()
default_marker_pose.p.z = 1.0

envs = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env instance
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)

    actor_handle = gym.create_actor(env, holybro, default_pose, "drone", i, 1, 0)

    dof_props = gym.get_actor_dof_properties(env, actor_handle)
    #dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
    dof_props['stiffness'].fill(1000.0)
    dof_props['damping'].fill(0.0)
    gym.set_actor_dof_properties(env, actor_handle, dof_props)

    # gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)
    default_marker_pose.p.z = random.randint(1,5)
    marker_handle = gym.create_actor(env, marker_asset, default_marker_pose, "marker", i, 1, 1)
    gym.set_rigid_body_color(env, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

    envs.append(env)

# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# aquire root state tensor descriptor
_root_tensor = gym.acquire_actor_root_state_tensor(sim) # drone, marker, drone, marker....
root_tensor = gymtorch.wrap_tensor(_root_tensor).view(num_envs, 2, 13)

drone_states = root_tensor[:, 0, :]
drone_positions = drone_states[:, 0:3]

marker_states = root_tensor[:, 1, :]
marker_positions = marker_states[:, 0:3]


# acquire dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

# wrap it in a PyTorch Tensor and create convenient views

# main simulation loop
while not gym.query_viewer_has_closed(viewer):
    
    # step the physics
    gym.simulate(sim)
    gym.refresh_actor_root_state_tensor(sim)
    print(marker_positions)
    
    gym.refresh_dof_state_tensor(sim)

    # Deploy actions
    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
   
# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
