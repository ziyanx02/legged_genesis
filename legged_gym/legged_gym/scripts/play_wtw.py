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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
import faulthandler

import numpy as np
import torch
import time

def get_load_path(root, checkpoint=-1, model_name_include="model"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def play(args):
    args.task = "go2_wtw"
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 2
    env_cfg.terrain.terrain_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.commands.resampling_time = 1e6
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_Kp_factor = False
    env_cfg.domain_rand.randomize_Kd_factor = False
    env_cfg.domain_rand.randomize_gravity = False
    env_cfg.domain_rand.push_robots = False

    command_list = [
        #|----vel----|body_h|gait_f|----gait----|gait_d|foot_h|-body_pr-|-stance_wl-|
        [  0, 0, 0,     0,      3,   0.5, 0, 0,   0.5,   0.1,    0, 0,    0.3, 0.4,  ],
        [  0, 0, 0,     0,      3,   0.5, 0, 0,   0.5,   0.1,    0, 0,    0.3, 0.4,  ],
        [  0, 0, 0,    -0.15,   3,   0.5, 0, 0,   0.5,   0.1,    0, 0,    0.3, 0.4,  ],
        [  0, 0, 0,     0.15,   3,   0.5, 0, 0,   0.5,   0.1,    0, 0,    0.3, 0.4,  ],
        [  0, 0, 0,     0,      3,   0.5, 0, 0,   0.5,   0.1,    0, 0,    0.1, 0.35, ],
        [  0, 0, 0,     0,      3,   0.5, 0, 0,   0.5,   0.1,    0, 0,    0.45,0.45, ],
        [  2, 0, 0,     0,      3,   0.5, 0, 0,   0.5,   0.1,    0, 0,    0.3, 0.4,  ],
        [  0, 0, 1,     0,      3,   0.5, 0, 0,   0.5,   0.1,    0, 0,    0.3, 0.4,  ],
        [  0, 0, 0,     0,      3,   0, 0.5, 0,   0.5,   0.1,    0, 0,    0.3, 0.4,  ],
        [  0, 0, 0,     0,      3,   0, 0, 0.5,   0.5,   0.1,    0, 0,    0.3, 0.4,  ],
        [  0, 0, 0,     0,      3,   0, 0, 0,     0.5,   0.1,    0, 0,    0.3, 0.4,  ],
    ]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=log_pth)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # if RECORD_FRAMES:
    #     env.start_recording()

    for command in command_list:
        env.commands = torch.tensor(command, device=env.device, requires_grad=False)[None, :].repeat(2, 1)
        start_time = time.time()
        while time.time() - start_time < 2:
            actions = policy(obs.detach())
            # print(env.commands)
            # print(obs.shape)
            # print("gravity", obs[0, :3].cpu())
            # print("command", obs[0, 3:17].cpu())
            # print("dof_pos", obs[0, 17:29].cpu())
            # print("dof_vel", obs[0, 29:41].cpu())
            # print("last_action", obs[0, 41:53].cpu())
            # print("last_last_action", obs[0, 53:65].cpu())
            # print("clock_input", obs[0, 65:69].cpu()) # see func clock_input
            # exit()
            obs, _, _, _, _ = env.step(actions.detach())

def clock_input(t=None, command=None):

    if command == None:
        command = torch.tensor([0, 0, 0, 0, 3, 0.5, 0, 0, 0.5, 0.1, 0, 0, 0.3, 0.4,])
    if t == None:
        start = time.time()
        time.sleep(0.5)
        end = time.time()
        t = end - start
    else:
        t -= int(t)

    frequencies = command[4]
    phases = command[5]
    offsets = command[6]
    bounds = command[7]

    gait_indices = torch.remainder(t * frequencies, 1.0)
    foot_indices = torch.tensor([phases + offsets + bounds, offsets, bounds, phases]) + gait_indices

    clock_inputs = torch.sin(2 * np.pi * foot_indices)

    return clock_inputs

if __name__ == '__main__':
    faulthandler.enable()
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    args = get_args()
    play(args)
