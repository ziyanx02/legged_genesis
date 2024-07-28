# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

import genesis as gs

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
# from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

from PIL import Image
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, sim_device, headless)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
    
    def _render_headless(self):
        if self._recording and len(self._recorded_frames) < self.cfg.camera.num_upload_frames:
            robot_pos = np.array(self.root_states[0, :3].cpu())
            self._floating_camera.set_pose(pos=robot_pos + np.array(self.cfg.camera.pos), lookat=robot_pos)
            frame, _, _ = self._floating_camera.render()
            self._recorded_frames.append(frame)

    def get_recorded_frames(self):
        if len(self._recorded_frames) == self.cfg.camera.num_upload_frames:
            frames = self._recorded_frames
            self._recorded_frames = []
            self._recording = False
            return frames
        else:
            return None
    
    def start_recording(self):
        self._recorded_frames = []
        self._recording = True
        
    def render(self, img_path=None, sync_frame_time=True):

        img_np, _, _ = self.camera.render(rgb=True, depth=True, segmentation=True) # type: np.ndarray    
        img = Image.fromarray(img_np)
        self.images.append(img_np)
        
        if len(self.images) % 200 == 0:
            # self.images[0].save('./videos/view.gif', 'GIF', append_images=self.images[1:], save_all=True, duration=200, loop=0)
            clip = ImageSequenceClip(self.images, fps=50)
            clip.write_videofile(self.cfg.viewer.video_path, codec='libx264')

        if img_path is not None: img.save(img_path)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame

        for dec in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.robot.control_dofs_force(self.torques, dofs_idx_local=self.dofs_idx_local_full)            
            self.scene.step()

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_buffers()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.cfg.viewer.debug:
            self._draw_debug_vis()
        
        self._render_headless()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1]) > self.cfg.asset.terminate_if_pitch_greater_than, torch.abs(self.rpy[:,0]) > self.cfg.asset.terminate_if_roll_greater_than)
        self.reset_buf |= self.time_out_buf
        if self.cfg.asset.terminate_if_height_lower_than is not None:
            self.height_illegal_buf = self.base_pos[:, self.up_axis_idx] < self.cfg.asset.terminate_if_height_lower_than
            self.reset_buf |= self.height_illegal_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return None, None
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_scene(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly

        if self.headless == False:
            # subscribe to keyboard shortcuts
            viewer_options = gs.options.ViewerOptions(
                res=(1080, 720),
                max_FPS=int(1 / self.cfg.sim.dt),
                camera_pos=self.cfg.viewer.pos,
                camera_lookat=self.cfg.viewer.lookat,
                camera_fov=self.cfg.viewer.fov,
            )
        else:
            viewer_options = None
            # self.gym.subscribe_viewer_keyboard_event(
            #     self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            # self.gym.subscribe_viewer_keyboard_event(
            #     self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                substeps=self.cfg.sim.substeps,
                dt=self.cfg.sim.dt,
            ),
            viewer_options=viewer_options,
            rigid_options=gs.options.RigidOptions(
                dt=self.cfg.sim.dt, # TODO: what's this
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-0.1, -0.1, -0.1),
                upper_bound=(1.1, 1.1, 1.1),
                grid_density=64,
            ),
            vis_options=gs.options.VisOptions(
                geom_type=self.cfg.viewer.geom_type,
            ),
            show_FPS=False,
        )
        self._create_terrain()
        self._create_robot()
        self._set_camera()

    def build_scenes(self):
        """ Build parallel envs and randomizations """

        self.scene.build(n_envs=self.num_envs)

        self._get_env_origins()

        # TODO: apply randomizations initialized in self._create_robot()

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    # def _process_dof_props(self, props, env_id):
    #     """ Callback allowing to store/change/randomize the DOF properties of each environment.
    #         Called During environment creation.
    #         Base behavior: stores position, velocity and torques limits defined in the URDF

    #     Args:
    #         props (numpy.array): Properties of each DOF of the asset
    #         env_id (int): Environment id

    #     Returns:
    #         [numpy.array]: Modified DOF properties
    #     """
    #     if env_id==0:
    #         self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
    #         self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
    #         self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
    #         for i in range(len(props)):
    #             self.dof_pos_limits[i, 0] = props["lower"][i].item()
    #             self.dof_pos_limits[i, 1] = props["upper"][i].item()
    #             self.dof_vel_limits[i] = props["velocity"][i].item()
    #             self.torque_limits[i] = props["effort"][i].item()
    #             # soft limits
    #             m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
    #             r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
    #             self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
    #             self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
    #     return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % int(self.cfg.domain_rand.push_interval_s / self.dt) == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        # TODO: add randomization
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = 0.

        dof_vel = torch.zeros([self.dof_vel[env_ids].shape[0], 6 + self.num_dof], device=self.device, requires_grad=False)
        self.robot.set_dofs_position(self.dof_pos[env_ids], dofs_idx_local=self.dofs_idx_local_full, envs_idx=env_ids)
        self.robot.set_dofs_velocity(dof_vel, envs_idx=env_ids)

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """

        # TODO: add randomization
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, :2] += 1 * (torch.rand(self.root_states[env_ids, :2].shape, device=self.device) - 0.5)
        self.base_pos[env_ids] = self.root_states[env_ids, :3]
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = self.root_states[env_ids, 7:10]
        self.base_ang_vel[env_ids] = self.root_states[env_ids, 10:]

        self.robot.set_pos(self.base_pos[env_ids], envs_idx=env_ids)
        self.robot.set_quat(self.base_quat[env_ids], envs_idx=env_ids)

        return
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        push_vel = self.robot.get_dofs_velocity() # (num_envs, num_dof) [0:3] ~ base_link_vel
        push_vel[:, :2] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.robot.set_dofs_velocity(push_vel)

        return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[12+3*self.num_actions:self.num_obs] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """

        self.dofs_idx_local_full = torch.arange(6, 6 + self.num_dof).to(self.device)

        gravity_vec = torch.tensor([0., 0., 0.], device=self.device)
        gravity_vec[self.up_axis_idx] = -1
        self.gravity_vec = gravity_vec.repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        
        # initialize buffers storing robots' state info
        self._update_buffers()

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.measured_height_points = 0
        
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.bool, device=self.device, requires_grad=False)

        # buffers won't change during simulation
        self.dof_pos_limits = torch.cat([limit.unsqueeze(1) for limit in self.robot.get_dofs_limit()], dim=1)[6:]
        self.torque_limits = self.robot.get_dofs_force_range()[1][6:]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

    def _update_buffers(self):
        """ Prepares and update buffers inducing the current state of robot after each step
        """
        self.base_pos = self.robot.get_pos()      # (num_envs, 3)
        base_quat = self.robot.get_quat()         # (num_envs, 4)
        self.base_quat = torch.cat([base_quat[:, 1:], base_quat[:, :1]], dim=1)
        self.rpy = get_euler_xyz(self.base_quat[:])
        self.root_states = torch.cat([self.base_pos, self.base_quat, self.robot.get_vel(), self.robot.get_ang()], dim=1)

        self.dof_pos = self.robot.get_dofs_position()[:, 6:] # (num_envs, num_joints)
        self.dof_vel = self.robot.get_dofs_velocity()[:, 6:] # (num_envs, num_joints)

        self.contact_forces = torch.tensor(self.robot.get_links_net_contact_force(), device=self.device) # (num_envs, num_body, 3)
        
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_robot(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """

        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        self.init_state = self.cfg.init_state
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.quat + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = torch.tensor(base_init_state_list, device=self.device, requires_grad=False)

        # TODO: prepare friction randomization
        # TODO: prepare kv & kp or stiffness & damping randomization
        # TODO: prepare mass randomization

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file = asset_path,
                pos = self.cfg.init_state.pos,
                scale = 1.0,
            ),
            visualize_contact=self.cfg.viewer.visualize_contact
        )

        self.num_body = len(self.robot.links)
        self.num_dof = len(self.robot.joints) - 1 # remove joint_base

        self.body_names = []
        for link in self.robot.links:
            self.body_names.append(link.name)

        self.dof_names = []
        for joint in self.robot.joints:
            self.dof_names.append(joint.name)
        
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        dof_id = 0
        for name in self.dof_names[1:]:
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[dof_id] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[dof_id] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[dof_id] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[dof_id] = 0.
                self.d_gains[dof_id] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
            dof_id += 1
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)

        feet_names = []
        for name in self.cfg.asset.foot_name:
            feet_names.extend([s for s in self.body_names if name in s])
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])
        
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)

        for i in range(len(self.body_names)):
            for j in range(len(feet_names)):
                if feet_names[j] == self.body_names[i]:
                    self.feet_indices[j] = i

            for j in range(len(penalized_contact_names)):
                if penalized_contact_names[j] == self.body_names[i]:
                    self.penalised_contact_indices[j] = i

            for j in range(len(termination_contact_names)):
                if termination_contact_names[j] == self.body_names[i]:
                    self.termination_contact_indices[j] = i

    def _create_terrain(self):
        """ Create terrains:
            Add entity gs.morphs.Terrain with selected terrain type into the scene
        """
        terrain_type = self.cfg.terrain.terrain_type
        if terrain_type == "height_filed":
            raise NotImplementedError
            # TODO: refactor Terrain
            self.terrain = Terrain(self.cfg.terrain)
            height_field_raw = self.terrain.height_field_raw
        else:
            height_field_raw = None
        
        """ Currently only support using subterrains to get terrain_origin """

        # y 
        # ^ |-------------num_rows------------|
        # | |-terrain_length-|                |
        # | +----------------+----------------+----
        # | |                |                |   |
        # | |                |                |   |
        # | |  terrain_type  |  terrain_type  |   num_cols
        # | |     [0, 0]     |     [0, 1]     |   terrain_width
        # | |                |                |   |
        # | |                |                |   |
        # | +----------------+----------------+----
        # O---------------------------------------------> x

        self.horizontal_scale = self.cfg.terrain.horizontal_scale
        self.vertical_scale = self.cfg.terrain.vertical_scale

        # create terrain
        self.terrain = self.scene.add_entity(
            morph=gs.morphs.Terrain(
                n_subterrains = (self.cfg.terrain.num_rows, self.cfg.terrain.num_cols),
                subterrain_size = (self.cfg.terrain.terrain_length, self.cfg.terrain.terrain_width),
                horizontal_scale = self.horizontal_scale,
                vertical_scale = self.vertical_scale,
                subterrain_types=terrain_type,
                height_field=height_field_raw,
            ),
        )

        # get terrain origins
        rows = 0.5 + torch.arange(0, self.cfg.terrain.num_rows, 1, device="cuda", requires_grad=False).unsqueeze(1).repeat(1, self.cfg.terrain.num_cols).unsqueeze(-1)
        cols = 0.5 + torch.arange(0, self.cfg.terrain.num_cols, 1, device="cuda", requires_grad=False).unsqueeze(0).repeat(self.cfg.terrain.num_rows, 1).unsqueeze(-1)

        xys = torch.cat([rows * self.cfg.terrain.terrain_length, cols * self.cfg.terrain.terrain_width], dim=1).reshape(-1, 2)
        self.height_field_raw = torch.tensor(self.terrain.metadata["height_field"], device=self.device, requires_grad=False)
        xy_indices = torch.ceil(xys / self.horizontal_scale).to(torch.long)
        zs = self.height_field_raw[xy_indices[:, 0], xy_indices[:, 1]].unsqueeze(-1) * self.vertical_scale
        
        self.terrain_origins = torch.cat([xys, zs], dim=1)

    def _set_camera(self):
        """ Set camera position and direction
        """
        self.camera = self.scene.add_camera(
            pos=(self.cfg.viewer.pos[0], self.cfg.viewer.pos[1], self.cfg.viewer.pos[2]),
            lookat=(self.cfg.viewer.lookat[0], self.cfg.viewer.lookat[1], self.cfg.viewer.lookat[2]),
            # res=(500, 500),
            fov=self.cfg.viewer.fov,
            GUI=False,
        )
        self.images = []

        self._floating_camera = self.scene.add_camera(
            pos=np.array(self.cfg.camera.pos),
            lookat=np.array([0, 0, 0]),
            # res=self.cfg.camera.res,
            fov=self.cfg.camera.fov,
            GUI=False,
        )

        self._recording = False
        self._recorded_frames = []

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        # TODO: env_origins should be related to curriculum
        if self.cfg.terrain.terrain_type == "height_field":
            raise NotImplementedError
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            # curriculum not implemented
            self.custom_origins = False
            terrain_choices = torch.randint(0, self.terrain_origins.shape[0], (self.num_envs,), device=self.device, dtype=torch.long)
            self.env_origins = self.terrain_origins[terrain_choices]

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.cfg.sim.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        # if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
        #     self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self, env_id=0):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        self.scene.clear_debug_objects()
        self.scene.draw_debug_spheres(poss=self.measured_height_points[env_id].reshape(-1, 3), radius=0.02, color=(0, 0, 1, 0.7))

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if env_ids:
            raise NotImplementedError # check return
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        # points += self.terrain.cfg.border_size
        points = (points/self.horizontal_scale).long()
        px = points[:, :, 0]
        py = points[:, :, 1]
        px = torch.clip(px, 0, self.height_field_raw.shape[0]-2)
        py = torch.clip(py, 0, self.height_field_raw.shape[1]-2)

        heights1 = self.height_field_raw[px, py]
        heights2 = self.height_field_raw[px+1, py]
        heights3 = self.height_field_raw[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        self.measured_height_points = torch.cat([self.horizontal_scale * px.unsqueeze(-1), self.horizontal_scale * py.unsqueeze(-1), self.vertical_scale * heights.unsqueeze(-1)], dim=-1)

        return heights.view(self.num_envs, -1) * self.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return NotImplementedError

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
