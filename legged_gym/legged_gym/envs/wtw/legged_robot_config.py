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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class LeggedRobotCfgWTW(LeggedRobotCfg):
    class env:
        num_envs = 4096 
        num_observations = 70
        num_scalar_observations = 70
        num_privileged_obs = 2 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        privileged_future_horizon = 1
        num_actions = 12
        num_observation_history = 30
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        observe_vel = False
        observe_only_ang_vel = False
        observe_only_lin_vel = False
        observe_yaw = False
        observe_contact_states = False
        observe_command = True
        observe_height_command = False
        observe_gait_commands = True
        observe_timing_parameter = False
        observe_clock_inputs = True
        observe_two_prev_actions = True
        observe_imu = False

        priv_observe_friction = True
        priv_observe_friction_indep = False
        priv_observe_ground_friction = False
        priv_observe_ground_friction_per_foot = False
        priv_observe_restitution = True
        priv_observe_base_mass = False
        priv_observe_com_displacement = False
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_joint_friction = True
        priv_observe_Kp_factor = False
        priv_observe_Kd_factor = False
        priv_observe_contact_forces = False
        priv_observe_contact_states = False
        priv_observe_body_velocity = False
        priv_observe_foot_height = False
        priv_observe_body_height = False
        priv_observe_gravity = False
        priv_observe_terrain_type = False
        priv_observe_clock_inputs = False
        priv_observe_doubletime_clock_inputs = False
        priv_observe_halftime_clock_inputs = False
        priv_observe_desired_contact_states = False
        priv_observe_dummy_variable = False

    class terrain:
        terrain_type = 'plane' # "plane" or "flat_terrain"
        horizontal_scale = 0.25 # [m]
        vertical_scale = 0.005 # [m]
        # border_size = 25 # [m]
        curriculum = False
        # static_friction = 1.0
        # dynamic_friction = 1.0
        # restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,] # 0.8mx0.8m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4,]
        # selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        # max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 20.
        terrain_width = 20.
        num_rows= 1 # number of terrain rows (levels)
        num_cols = 1 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        # slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:

        command_curriculum = True
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        exclusive_command_sampling = False
        num_commands = 15
        resampling_time = 10.  # time before command are changed[s]
        subsample_gait = False
        gait_interval_s = 10.  # time between resampling gait params
        vel_interval_s = 10.
        jump_interval_s = 20.  # time between jumps
        jump_duration_s = 0.1  # duration of jump
        jump_height = 0.3
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False
        observe_accel = False
        distributional_commands = True
        curriculum_type = "RewardThresholdCurriculum"
        lipschitz_threshold = 0.9

        num_lin_vel_bins = 30
        lin_vel_step = 0.3
        num_ang_vel_bins = 30
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100

        lin_vel_x = [-0.6, 0.6]
        lin_vel_y = [-0.6, 0.6]
        ang_vel_yaw = [-1.0, 1.0]
        body_height_cmd = [-0.25, 0.15]
        impulse_height_commands = False

        limit_vel_x = [-5.0, 5.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-5.0, 5.0]
        limit_body_height = [-0.25, 0.15]
        limit_gait_phase = [0.0, 1.0]
        limit_gait_offset = [0.0, 1.0]
        limit_gait_bound = [0.0, 1.0]
        limit_gait_duration = [0.5, 0.5]
        limit_gait_frequency = [2.0, 4.0]
        limit_footswing_height = [0.03, 0.35]
        limit_body_pitch = [-0.4, 0.4]
        limit_body_roll = [-0.0, 0.0]
        limit_aux_reward_coef = [0.0, 0.01]
        limit_compliance = [0.0, 0.01]
        limit_stance_width = [0.10, 0.45]
        limit_stance_length = [0.35, 0.45]

        num_bins_vel_x = 21
        num_bins_vel_y = 1
        num_bins_vel_yaw = 21
        num_bins_body_height = 1
        num_bins_gait_frequency = 1
        num_bins_gait_phase = 1
        num_bins_gait_offset = 1
        num_bins_gait_bound = 1
        num_bins_gait_duration = 1
        num_bins_footswing_height = 1
        num_bins_body_roll = 1
        num_bins_body_pitch = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1
        num_bins_aux_reward_coef = 1
        num_bins_compliance = 1
        num_bins_compliance = 1

        heading = [-3.14, 3.14]

        gait_phase_cmd_range = [0.0, 1.0]
        gait_offset_cmd_range = [0.0, 1.0]
        gait_bound_cmd_range = [0.0, 1.0]
        gait_frequency_cmd_range = [2.0, 4.0]
        gait_duration_cmd_range = [0.5, 0.5]
        footswing_height_range = [0.03, 0.35]
        body_pitch_range = [-0.4, 0.4]
        body_roll_range = [-0.0, 0.0]
        stance_width_range = [0.10, 0.45]
        stance_length_range = [0.35, 0.45]
        aux_reward_coef_range = [0.0, 0.01]

        exclusive_phase_offset = False
        binary_phases = True
        pacing_offset = False
        balance_gait_distribution = True
        gaitwise_curricula = True

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        quat = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        hip_scale_reduction = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = [] # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        terminate_if_height_lower_than = 0
        terminate_if_roll_greater_than = 0.8
        terminate_if_pitch_greater_than = 1.0
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        rand_interval_s = 10
        randomize_rigids_after_start = True
        randomize_friction = True
        friction_range = [0.5, 1.25]  # increase range
        randomize_restitution = False
        restitution_range = [0, 1.0]
        randomize_base_mass = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        added_mass_range = [-1., 1.]
        randomize_com_displacement = False
        # add link masses, increase range, randomize inertia, randomize joint properties
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = False
        Kd_factor_range = [0.5, 1.5]
        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0
        randomize_gravity = False
        gravity_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_lag_timesteps = True
        lag_timesteps = 6

    class curriculum_thresholds:
        tracking_lin_vel = 0.8  # closer to 1 is tighter
        tracking_ang_vel = 0.7
        tracking_contacts_shaped_force = 0.9  # closer to 1 is tighter
        tracking_contacts_shaped_vel = 0.9
    
    class rewards:
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.02
        reward_container_name = "WTWRewards"
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_lat = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_long = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_sigma_yaw = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.30
        max_contact_force = 100.  # forces above this value are penalized
        use_terminal_body_height = False
        terminal_body_height = 0.20
        use_terminal_foot_height = False
        terminal_foot_height = -0.005
        use_terminal_roll_pitch = False
        terminal_body_ori = 0.5
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.
        footswing_height = 0.09

    class reward_scales:
        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -0.02
        ang_vel_xy = -0.001
        orientation = 0.
        orientation_control = -5.0
        torques = -0.0001
        dof_vel = -1e-4
        dof_acc = -2.5e-7
        base_height = 0.
        feet_air_time = 0.0
        collision = -5.
        feet_stumble = -0.0
        action_rate = -0.01
        stand_still = -0.
        tracking_lin_vel_lat = 0.
        tracking_lin_vel_long = 0.
        tracking_contacts = 0.
        tracking_contacts_shaped = 0.
        tracking_contacts_shaped_force = 4.
        tracking_contacts_shaped_vel = 4.
        jump = 10.0
        energy = 0.0
        energy_expenditure = 0.0
        survival = 0.0
        dof_pos_limits = -10.
        feet_contact_forces = 0.
        feet_slip = -0.04
        feet_clearance_cmd_linear = -30.
        dof_pos = 0.
        action_smoothness_1 = -0.1
        action_smoothness_2 = -0.1
        base_motion = 0.
        feet_impact_vel = 0.0
        raibert_heuristic = -10.0

    class normalization:
        clip_observations = 100.
        clip_actions = 10.

        friction_range = [0, 1]
        ground_friction_range = [0, 1]
        restitution_range = [0, 1.0]
        added_mass_range = [-1., 3.]
        com_displacement_range = [-0.1, 0.1]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        joint_friction_range = [0.0, 0.7]
        contact_force_range = [0.0, 50.0]
        contact_state_range = [0.0, 1.0]
        body_velocity_range = [-6.0, 6.0]
        foot_height_range = [0.0, 0.15]
        body_height_range = [0.0, 0.60]
        gravity_range = [-1.0, 1.0]
        motion = [-0.01, 0.01]

    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        imu = 0.1
        height_measurements = 5.0
        friction_measurements = 1.0
        body_height_cmd = 2.0
        gait_phase_cmd = 1.0
        gait_freq_cmd = 1.0
        footswing_height_cmd = 0.15
        body_pitch_cmd = 0.3
        body_roll_cmd = 0.3
        aux_reward_cmd = 1.0
        compliance_cmd = 1.0
        stance_width_cmd = 1.0
        stance_length_cmd = 1.0
        segmentation_image = 1.0
        rgb_image = 1.0
        depth_image = 1.0

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

    class noise_scales:
        dof_pos = 0.01
        dof_vel = 1.5
        lin_vel = 0.1
        ang_vel = 0.2
        imu = 0.1
        gravity = 0.05
        contact_states = 0.05
        height_measurements = 0.1
        friction_measurements = 0.0
        segmentation_image = 0.0
        rgb_image = 0.0
        depth_image = 0.0

    # viewer camera:
    class viewer:
        pos = [-5., -5., 5.]  # [m]
        lookat = [0., 0., 1.]  # [m]
        fov = 40
        geom_type = 'visual' # ['visual', 'collision', 'collision_sdf']
        visualize_contact = False
        debug = False
    
    class camera:
        pos = [0., -1., 1.]  # [m]
        fov = 80
        num_upload_frames = 150

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

class LeggedRobotCfgPPOWTW(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        runner_class_name = "OnPolicyRunner"
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        log_interval = 10
        save_interval = 100
        record_interval = 50
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt