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

class A1RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_observations = 48
        episode_length_s = 20 # episode length in seconds

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        quat = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FR_hip_joint': 0.,  # [rad]
            'FL_hip_joint': 0.,   # [rad]
            'RR_hip_joint': 0.,   # [rad]
            'RL_hip_joint': 0.,   # [rad]

            'FR_thigh_joint': 0.8,     # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]
            'RL_thigh_joint': 1.,   # [rad]

            'FR_calf_joint': -1.5,  # [rad]
            'FL_calf_joint': -1.5,   # [rad]
            'RR_calf_joint': -1.5,    # [rad]
            'RL_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = ["calf",]
        penalize_contacts_on = ["thigh"]
        terminate_after_contacts_on = ["base"]
        terminate_if_height_lower_than = None

    class terrain:
        # terrain_type = [
        #     ['random_uniform_terrain',],
        #     ['pyramid_sloped_terrain',],
        # ]
        terrain_type = 'plane'
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        curriculum = False
        measure_heights = False
        measured_points_x = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,] # 0.8mx0.8m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4,]
        # selected = False # select a unique terrain type and pass all arguments
        # terrain_kwargs = None # Dict of arguments for selected terrain
        # max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 6.
        terrain_width = 12.
        num_rows = 2 # number of terrain rows (levels)
        num_cols = 1 # number of terrain cols (types)

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.27
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            base_height = -100.0

class A1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_a1'

  