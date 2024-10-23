import os

from onmi.isaac.lab.actuators import ImplicitActuatorCfg

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation import ArticulationCfg

usd_path = os.path.join(os.environ["EXP_PATH"], "exts/pointfoot/pointfoot/assets/usd/PF_P441A/PF_441A.usd")

POINTFOOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            retain_acceleration=False,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0918,
            "knee_L_Joint": -0.057,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0918,
            "knee_R_Joint": -0.057,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "abad_L_Joint",
                "hip_L_Joint",
                "knee_L_Joint",
                "abad_R_Joint",
                "hip_R_Joint",
                "knee_R_Joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "abad_L_Joint": 60.0,
                "hip_L_Joint": 60.0,
                "knee_L_Joint": 60.0,
                "abad_R_Joint": 60.0,
                "hip_R_Joint": 60.0,
                "knee_R_Joint": 60.0,
            },
            damping={
                "abad_L_Joint": 5.0,
                "hip_L_Joint": 5.0,
                "knee_L_Joint": 5.0,
                "abad_R_Joint": 5.0,
                "hip_R_Joint": 5.0,
                "knee_R_Joint": 5.0,
            },
        ),
    },
)
