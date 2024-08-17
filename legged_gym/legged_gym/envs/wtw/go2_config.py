from legged_gym.envs.wtw.legged_robot_config import LeggedRobotCfgWTW, LeggedRobotCfgPPOWTW

class GO2CfgWTW( LeggedRobotCfgWTW ):
    class init_state( LeggedRobotCfgWTW.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.,   # [rad]
            'FR_hip_joint': 0.,  # [rad]
            'RL_hip_joint': 0.,   # [rad]
            'RR_hip_joint': 0.,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfgWTW.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfgWTW.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = ["calf",]
        penalize_contacts_on = ["thigh"]
        terminate_after_contacts_on = ["base"]
  
class GO2CfgPPOWTW( LeggedRobotCfgPPOWTW ):
    class algorithm( LeggedRobotCfgPPOWTW.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPOWTW.runner ):
        run_name = ''
        experiment_name = 'go2'

  
