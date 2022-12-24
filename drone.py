import os 
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from base.vec_task import VecTask

import math
import torch
import random 
import numpy as np

class drone():

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self.cfg = cfg

        num_obs = 13 # 0:13 - root stae
        num_acts = 4 # thrust to each rotor
        self.cfg["env"]["numObservations"]= num_obs
        self.cfg["env"]["numActions"] = num_acts

        self.up_axis = "z"
        self.up_axis_idx = 2
        self.dt = 1/60.0

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        dofs_per_env = 4

        # get gym GPU state tensors
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) # drone, marker, drone, marker....
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(_root_tensor).view(self.num_envs, 2, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(_dof_states) #.view(self.num_envs, dofs_per_env,2)

        self.drone_states = vec_root_tensor[:, 0, :]
        self.drone_positions = self.drone_states[:, 0:3]
        self.drone_quats = self.drone_states[:, 3:7]
        self.drone_linvels = self.drone_states[:, 7:10]
        self.drone_angvels = self.drone_states[:, 10:13]

        self.marker_states = vec_root_tensor[:, 1, :]
        self.marker_positions = self.marker_states[:, 0:3]

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_drone_states = vec_root_tensor.clone()  # for reset_idx
        self.initial_dof_states = vec_dof_tensor.clone() # for reset_idx

        # To-Do: fix these values
        max_thrust = 2
        self.thrust_lower_limits = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.thrust_upper_limits = max_thrust * torch.ones(4, device=self.device, dtype=torch.float32)

        self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2)) # drone and target

    def create_sim(self):
        # configure sim
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 2
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        # set torch device
        self.device = args.sim_device if args.use_gpu_pipeline else 'cpu'

        # set PhysX-specific parameters
        self.sim_params.physx.use_gpu = True

        self.sim = super().create_sim(self.compute_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
    
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % num_envs)

        # load holybro asset
        asset_root = "assets"
        holybro_asset_file = "x500/x500.urdf"
        asset_options = gymapi.AssetOptions()

        # drone
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = False
        holybro = self.gym.load_asset(self.sim, asset_root, holybro_asset_file, asset_options)

        # sphere
        asset_options.fix_base_link = True
        marker_asset = self.gym.create_sphere(self.sim, 0.1, asset_options)

        # configure holybro dofs
        dof_props = self.gym.get_asset_dof_properties(holybro)
        self.num_dofs = self.gym.get_asset_dof_count(holybro)
        #dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

        self.dof_lower_limits = []
        self.dof_upper_limits = []
        for i in range(self.num_dofs):
            self.dof_lower_limits.append(dof_props['lower'][i])
            self.dof_upper_limits.append(dof_props['upper'][i])

        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)
        self.dof_ranges = self.dof_upper_limits - self.dof_lower_limits
        
        self.envs = []

        default_pose = gymapi.Transform()   
        default_pose.p = gymapi.Vec3(0.0, 1.0, 5.0)
        # default_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        default_marker_pose = gymapi.Transform()
        default_marker_pose.p.z = 1.0

        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row
            )

            actor_handle = self.gym.create_actor(env, holybro, default_pose, "drone", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            #dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
            dof_props['stiffness'].fill(1000.0)
            dof_props['damping'].fill(0.0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

            # gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)
            default_marker_pose.p.z = random.randint(1,5)
            marker_handle = self.gym.create_actor(env, marker_asset, default_marker_pose, "marker", i, 1, 1)
            self.gym.set_rigid_body_color(env, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

            self.envs.append(env)
        
        self.init_data()
        
    def init_data(self):
        default_pose = gymapi.Transform()   
        default_pose.p = gymapi.Vec3(0.0, 1.0, 5.0)
        # default_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        default_marker_pose = gymapi.Transform()
        default_marker_pose.p.z = 1.0

    def set_targets(self, env_ids):
        # randomly set marker positions
        num_sets = len(env_ids)
        # set target position randomly with x, y in (-5, 5) and z in (1, 2)
        self.target_root_positions[env_ids, 0:2] = (torch.rand(num_sets, 2, device=self.device) * 10) - 5
        self.target_root_positions[env_ids, 2] = torch.rand(num_sets, device=self.device) + 1
        self.marker_positions[env_ids] = self.target_root_positions[env_ids]
        # copter "position" is at the bottom of the legs, so shift the target up so it visually aligns better
        self.marker_positions[env_ids, 2] += 0.4
        actor_indices = self.all_actor_indices[env_ids, 1].flatten()  # targets

        return actor_indices

    def reset_idx(self, env_ids):
        # set rotor speeds
        self.dof_velocities[:, 1] = -50
        self.dof_velocities[:, 3] = 50

        num_resets = len(env_ids)

        target_actor_indices = self.set_targets(env_ids)

        actor_indices = self.all_actor_indices[env_ids, 0].flatten()

        self.drone_states[env_ids] = self.initial_drone_states[env_ids]
        self.drone_states[env_ids, 0] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        self.drone_states[env_ids, 1] += torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()
        self.drone_states[env_ids, 2] += torch_rand_float(-0.2, 1.5, (num_resets, 1), self.device).flatten()

        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        return torch.unique(torch.cat([target_actor_indices, actor_indices]))


        


