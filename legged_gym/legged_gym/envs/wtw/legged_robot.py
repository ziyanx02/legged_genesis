# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from typing import Dict

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.wtw.legged_robot_config import LeggedRobotCfgWTW
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict

class LeggedRobotWTW(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfgWTW, sim_params, sim_device, headless, record):
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
        super().__init__(cfg, sim_params, sim_device, headless, record)

        self.cfg = cfg
        self.rew_buf_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_neg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()

        self._update_buffers()
        
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.cfg.viewer.debug:
            self._draw_debug_vis()
        
        self._render_headless()

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        self._randomize_rigids(env_ids)
        self._randomize_controls(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        train_env_ids = env_ids
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][train_env_ids]) / self.max_episode_length_s
            self.episode_sums[key][train_env_ids] = 0.

        if self.cfg.commands.command_curriculum:
            self.extras["env_bins"] = torch.Tensor(self.env_command_bins)[:self.num_envs]
            self.extras["episode"]["min_command_duration"] = torch.min(self.commands[:, 8])
            self.extras["episode"]["max_command_duration"] = torch.max(self.commands[:, 8])
            self.extras["episode"]["min_command_bound"] = torch.min(self.commands[:, 7])
            self.extras["episode"]["max_command_bound"] = torch.max(self.commands[:, 7])
            self.extras["episode"]["min_command_offset"] = torch.min(self.commands[:, 6])
            self.extras["episode"]["max_command_offset"] = torch.max(self.commands[:, 6])
            self.extras["episode"]["min_command_phase"] = torch.min(self.commands[:, 5])
            self.extras["episode"]["max_command_phase"] = torch.max(self.commands[:, 5])
            self.extras["episode"]["min_command_freq"] = torch.min(self.commands[:, 4])
            self.extras["episode"]["max_command_freq"] = torch.max(self.commands[:, 4])
            self.extras["episode"]["min_command_x_vel"] = torch.min(self.commands[:, 0])
            self.extras["episode"]["max_command_x_vel"] = torch.max(self.commands[:, 0])
            self.extras["episode"]["min_command_y_vel"] = torch.min(self.commands[:, 1])
            self.extras["episode"]["max_command_y_vel"] = torch.max(self.commands[:, 1])
            self.extras["episode"]["min_command_yaw_vel"] = torch.min(self.commands[:, 2])
            self.extras["episode"]["max_command_yaw_vel"] = torch.max(self.commands[:, 2])
            if self.cfg.commands.num_commands > 9:
                self.extras["episode"]["min_command_swing_height"] = torch.min(self.commands[:, 9])
                self.extras["episode"]["max_command_swing_height"] = torch.max(self.commands[:, 9])
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras["episode"][f"command_area_{category}"] = np.sum(curriculum.weights) / curriculum.weights.shape[0]

            self.extras["episode"]["min_action"] = torch.min(self.actions)
            self.extras["episode"]["max_action"] = torch.max(self.actions)

            self.extras["curriculum/distribution"] = {}
            for curriculum, category in zip(self.curricula, self.category_names):
                self.extras[f"curriculum/distribution"][f"weights_{category}"] = curriculum.weights
                self.extras[f"curriculum/distribution"][f"grid_{category}"] = curriculum.grid
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf[:self.num_envs]

        self.gait_indices[env_ids] = 0

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
        self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew

        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.projected_gravity,
                                  (self.dof_pos[:, :self.num_actions] - self.default_dof_pos[:, :self.num_actions]) * self.obs_scales.dof_pos,
                                  self.dof_vel[:, :self.num_actions] * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)

        if self.cfg.env.observe_command:
            self.obs_buf = torch.cat((self.projected_gravity,
                                      self.commands * self.commands_scale,
                                      (self.dof_pos[:, :self.num_actions] - self.default_dof_pos[:, :self.num_actions]) * self.obs_scales.dof_pos,
                                      self.dof_vel[:, :self.num_actions] * self.obs_scales.dof_vel,
                                      self.actions
                                      ), dim=-1)

        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf,
                                      self.clock_inputs), dim=-1)

        if self.cfg.env.observe_vel:
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                      self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_ang_vel:
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_lin_vel:
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                      self.obs_buf), dim=-1)

        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            # heading_error = torch.clip(0.5 * wrap_to_pi(heading), -1., 1.).unsqueeze(1)
            self.obs_buf = torch.cat((self.obs_buf,
                                      heading), dim=-1)

        if self.cfg.env.observe_contact_states:
            self.obs_buf = torch.cat((self.obs_buf, (self.contact_forces[:, self.feet_indices, 2] > 1.).view(
                self.num_envs,
                -1) * 1.0), dim=1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # build privileged obs

        self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.friction_coeffs[:, 0].unsqueeze(
                                                     1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.friction_coeffs[:, 0].unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_ground_friction:
            self.ground_friction_coeffs = self._get_ground_frictions(range(self.num_envs))
            ground_friction_coeffs_scale, ground_friction_coeffs_shift = get_scale_shift(
                self.cfg.normalization.ground_friction_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.ground_friction_coeffs.unsqueeze(
                                                     1) - ground_friction_coeffs_shift) * ground_friction_coeffs_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.ground_friction_coeffs.unsqueeze(
                                                          1) - friction_coeffs_shift) * friction_coeffs_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.restitutions[:, 0].unsqueeze(
                                                     1) - restitutions_shift) * restitutions_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.restitutions[:, 0].unsqueeze(
                                                          1) - restitutions_shift) * restitutions_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_com_displacement:
            com_displacements_scale, com_displacements_shift = get_scale_shift(
                self.cfg.normalization.com_displacement_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.com_displacements - com_displacements_shift) * com_displacements_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_strength:
            motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.normalization.motor_strength_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (
                                                              self.motor_strengths - motor_strengths_shift) * motor_strengths_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_motor_offset:
            motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.normalization.motor_offset_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (
                                                         self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                      (
                                                              self.motor_offsets - motor_offset_shift) * motor_offset_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_body_height:
            body_height_scale, body_height_shift = get_scale_shift(self.cfg.normalization.body_height_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 ((self.root_states[:self.num_envs, 2]).view(
                                                     self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.root_states[:self.num_envs, 2]).view(
                                                          self.num_envs, -1) - body_height_shift) * body_height_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_body_velocity:
            body_velocity_scale, body_velocity_shift = get_scale_shift(self.cfg.normalization.body_velocity_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 ((self.base_lin_vel).view(self.num_envs,
                                                                           -1) - body_velocity_shift) * body_velocity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      ((self.base_lin_vel).view(self.num_envs,
                                                                                -1) - body_velocity_shift) * body_velocity_scale),
                                                     dim=1)
        if self.cfg.env.priv_observe_gravity:
            gravity_scale, gravity_shift = get_scale_shift(self.cfg.normalization.gravity_range)
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 (self.gravities - gravity_shift) / gravity_scale),
                                                dim=1)
            self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf,
                                                      (self.gravities - gravity_shift) / gravity_scale), dim=1)

        if self.cfg.env.priv_observe_clock_inputs:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.clock_inputs), dim=-1)

        if self.cfg.env.priv_observe_desired_contact_states:
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                                 self.desired_contact_states), dim=-1)

        assert self.privileged_obs_buf.shape[
                   1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # resample commands
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._step_contact_targets()

        # measure terrain heights
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(torch.arange(self.num_envs, device=self.device), self.cfg)

        # push robots
        push_interval = int(self.cfg.domain_rand.push_interval_s / self.dt)
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % push_interval == 0):
            self._push_robots()
        
        # randomize dof properties
        rand_interval = int(self.cfg.domain_rand.rand_interval_s / self.dt)
        env_ids = (self.episode_length_buf % rand_interval == 0).nonzero(as_tuple=False).flatten()
        # TODO: _randomize_dof_props
        # TODO: _randomize_rigid_body_props
        # TODO: refresh_actor_rigid_shape_props

    def _init_command_distribution(self, env_ids):
        # new style curriculum
        self.category_names = ['nominal']
        if self.cfg.commands.gaitwise_curricula:
            self.category_names = ['pronk', 'trot', 'pace', 'bound']

        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from .curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                               x_vel=(self.cfg.commands.limit_vel_x[0],
                                                      self.cfg.commands.limit_vel_x[1],
                                                      self.cfg.commands.num_bins_vel_x),
                                               y_vel=(self.cfg.commands.limit_vel_y[0],
                                                      self.cfg.commands.limit_vel_y[1],
                                                      self.cfg.commands.num_bins_vel_y),
                                               yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                        self.cfg.commands.limit_vel_yaw[1],
                                                        self.cfg.commands.num_bins_vel_yaw),
                                               body_height=(self.cfg.commands.limit_body_height[0],
                                                            self.cfg.commands.limit_body_height[1],
                                                            self.cfg.commands.num_bins_body_height),
                                               gait_frequency=(self.cfg.commands.limit_gait_frequency[0],
                                                               self.cfg.commands.limit_gait_frequency[1],
                                                               self.cfg.commands.num_bins_gait_frequency),
                                               gait_phase=(self.cfg.commands.limit_gait_phase[0],
                                                           self.cfg.commands.limit_gait_phase[1],
                                                           self.cfg.commands.num_bins_gait_phase),
                                               gait_offset=(self.cfg.commands.limit_gait_offset[0],
                                                            self.cfg.commands.limit_gait_offset[1],
                                                            self.cfg.commands.num_bins_gait_offset),
                                               gait_bounds=(self.cfg.commands.limit_gait_bound[0],
                                                            self.cfg.commands.limit_gait_bound[1],
                                                            self.cfg.commands.num_bins_gait_bound),
                                               gait_duration=(self.cfg.commands.limit_gait_duration[0],
                                                              self.cfg.commands.limit_gait_duration[1],
                                                              self.cfg.commands.num_bins_gait_duration),
                                               footswing_height=(self.cfg.commands.limit_footswing_height[0],
                                                                 self.cfg.commands.limit_footswing_height[1],
                                                                 self.cfg.commands.num_bins_footswing_height),
                                               body_pitch=(self.cfg.commands.limit_body_pitch[0],
                                                           self.cfg.commands.limit_body_pitch[1],
                                                           self.cfg.commands.num_bins_body_pitch),
                                               body_roll=(self.cfg.commands.limit_body_roll[0],
                                                          self.cfg.commands.limit_body_roll[1],
                                                          self.cfg.commands.num_bins_body_roll),
                                               stance_width=(self.cfg.commands.limit_stance_width[0],
                                                             self.cfg.commands.limit_stance_width[1],
                                                             self.cfg.commands.num_bins_stance_width),
                                               stance_length=(self.cfg.commands.limit_stance_length[0],
                                                                self.cfg.commands.limit_stance_length[1],
                                                                self.cfg.commands.num_bins_stance_length),
                                               aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0],
                                                           self.cfg.commands.limit_aux_reward_coef[1],
                                                           self.cfg.commands.num_bins_aux_reward_coef),
                                               )]

        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold,
                                      binary_phases=self.cfg.commands.binary_phases)
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int32)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int32)
        low = np.array(
            [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
             self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.body_height_cmd[0],
             self.cfg.commands.gait_frequency_cmd_range[0],
             self.cfg.commands.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
             self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
             self.cfg.commands.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
             self.cfg.commands.body_roll_range[0],self.cfg.commands.stance_width_range[0],
             self.cfg.commands.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0], ])
        high = np.array(
            [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
             self.cfg.commands.ang_vel_yaw[1], self.cfg.commands.body_height_cmd[1],
             self.cfg.commands.gait_frequency_cmd_range[1],
             self.cfg.commands.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
             self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1],
             self.cfg.commands.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
             self.cfg.commands.body_roll_range[1],self.cfg.commands.stance_width_range[1],
             self.cfg.commands.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1], ])
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _resample_commands(self, env_ids):

        if len(env_ids) == 0: return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.max_episode_length, timesteps)

        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force", "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds, local_range=np.array( [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in
                            range(len(self.category_names))]

        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.cfg.commands.num_commands]).to(
                self.device)

        if self.cfg.commands.num_commands > 5:
            if self.cfg.commands.gaitwise_curricula:
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":  # pronking
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":  # trotting
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":  # pacing
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":  # bounding
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            elif self.cfg.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[torch.logical_and(0.34 <= random_env_floats, random_env_floats < 0.67)]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[pronking_envs, 6] = (self.commands[pronking_envs, 6] / 2 - 0.25) % 1
                self.commands[pronking_envs, 7] = (self.commands[pronking_envs, 7] / 2 - 0.25) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

            if self.cfg.commands.binary_phases:
                self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
                self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
                self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1

        # setting the smaller commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.

    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.commands[:, 4]
            phases = self.commands[:, 5]
            offsets = self.commands[:, 6]
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + bounds,
                            self.gait_indices + phases]

            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                            0.5 / (1 - durations[swing_idxs]))

            # if self.cfg.commands.durations_warp_clock_inputs:

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
            self.desired_contact_states[:, 2] = smoothing_multiplier_RL
            self.desired_contact_states[:, 3] = smoothing_multiplier_RR

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            self.joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type

        if control_type == "actuator_net":
            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
            self.joint_vel = self.dof_vel
            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
        elif control_type == "P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos + self.motor_offsets) \
                    - self.d_gains * self.Kd_factors * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.ones(self.num_actions) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(self.num_actions) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                               torch.zeros(self.num_actions),
                               ), dim=0)

        if self.cfg.env.observe_command:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.gravity * noise_level,
                                   torch.zeros(self.cfg.commands.num_commands),
                                   torch.ones(self.num_actions) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                                   torch.ones(self.num_actions) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                                   torch.zeros(self.num_actions),
                                   ), dim=0)
        if self.cfg.env.observe_two_prev_actions:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(self.num_actions)
                                   ), dim=0)
        if self.cfg.env.observe_timing_parameter:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1)
                                   ), dim=0)
        if self.cfg.env.observe_clock_inputs:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(4)
                                   ), dim=0)
        if self.cfg.env.observe_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   torch.ones(3) * noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel,
                                   noise_vec
                                   ), dim=0)

        if self.cfg.env.observe_only_lin_vel:
            noise_vec = torch.cat((torch.ones(3) * noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel,
                                   noise_vec
                                   ), dim=0)

        if self.cfg.env.observe_yaw:
            noise_vec = torch.cat((noise_vec,
                                   torch.zeros(1),
                                   ), dim=0)

        if self.cfg.env.observe_contact_states:
            noise_vec = torch.cat((noise_vec,
                                   torch.ones(4) * noise_scales.contact_states * noise_level,
                                   ), dim=0)


        noise_vec = noise_vec.to(self.device)

        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """

        super()._init_buffers()
        self._init_custom_buffers()

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.commands_value = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros_like(self.commands_value)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                            self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                            self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
                                            self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
                                           self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd],
                                           device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )

        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool,
                                             device=self.device,
                                             requires_grad=False)

        # joint positions offsets and PD gains
        if self.cfg.control.control_type == "actuator_net":
            actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1.pt'
            actuator_network = torch.jit.load(actuator_path).to(self.device)

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                torques = actuator_network(xs.view(self.num_envs * 12, 6))
                return torques.view(self.num_envs, 12)

            self.actuator_network = eval_actuator_network

            self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)

    def _update_buffers(self):

        super()._update_buffers()

        self.links_pos = self.robot.get_links_pos()   # (num_envs, num_body, 3)
        links_quat = self.robot.get_links_quat()      # (num_envs, num_body, 4)
        self.links_quat = torch.cat([links_quat[:, :, 1:], links_quat[:, :, :1]], dim=2)
        self.links_vel = self.robot.get_links_vel()
        self.links_ang = self.robot.get_links_ang()
        self.rigid_body_state = torch.cat([self.links_pos, self.links_quat, self.robot.get_links_vel(), self.robot.get_links_ang()], dim=2)

        self.prev_base_pos = self.base_pos.clone()

    def _init_custom_buffers(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        # if custom initialization values were passed in, set them here
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]
        # if self.initial_dynamics_dict is not None:
        #     for k, v in self.initial_dynamics_dict.items():
        #         if k in dynamics_params:
        #             setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from .rewards import WTWRewards
        reward_containers = {"WTWRewards": WTWRewards}
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.cfg.sim.dt
        self.obs_scales = self.cfg.obs_scales
        self.reward_scales = class_to_dict(self.cfg.reward_scales)
        self.command_ranges = class_to_dict(self.cfg.commands)
        self.curriculum_thresholds = class_to_dict(self.cfg.curriculum_thresholds)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
