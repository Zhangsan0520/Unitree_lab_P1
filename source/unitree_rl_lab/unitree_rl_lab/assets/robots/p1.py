from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg


P1_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path="/tmp/IsaacLab/unitree_rl_lab/robot.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.654),
        joint_pos={
            ".*hip_pitch.*": 0.0,
            ".*knee_pitch.*": 0.05,
            ".*ankle_pitch.*": 0.0,
            ".*hip_roll.*": 0.0,
            ".*hip_yaw.*": 0.0,
            ".*ankle_roll.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_roll.*"],
            effort_limit_sim=120.0,
            velocity_limit_sim=13.30,
            stiffness=180.0,
            damping=10.0,
            armature=0.06,
        ),
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_pitch.*"],
            effort_limit_sim=120.0,
            velocity_limit_sim=13.30,
            stiffness=180.0,
            damping=10.0,
            armature=0.06,
        ),
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_yaw.*"],
            effort_limit_sim=60.0,
            velocity_limit_sim=16.02,
            stiffness=180.0,
            damping=10.0,
            armature=0.0264,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=[".*knee_pitch.*"],
            effort_limit_sim=60.0,
            velocity_limit_sim=16.02,
            stiffness=180.0,
            damping=10.0,
            armature=0.0264,
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle_pitch.*"],
            effort_limit_sim=34.0,
            velocity_limit_sim=8.69,
            stiffness=187.03872,
            damping=9.072,
            armature=0.03888,
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle_roll.*"],
            effort_limit_sim=34.0,
            velocity_limit_sim=8.69,
            stiffness=187.03872,
            damping=9.072,
            armature=0.03888,
        ),
    },
    joint_sdk_names=[
        "hip_roll_l_joint",
        "hip_pitch_l_joint",
        "hip_yaw_l_joint",
        "knee_pitch_l_joint",
        "ankle_pitch_l_joint",
        "ankle_roll_l_joint",
        "hip_roll_r_joint",
        "hip_pitch_r_joint",
        "hip_yaw_r_joint",
        "knee_pitch_r_joint",
        "ankle_pitch_r_joint",
        "ankle_roll_r_joint",
    ],
)

P1_CFG.spawn.replace_asset(
    meshes_dir="/work/Projects/unitree_rl_lab/unitree_usd/p1/meshes",
    urdf_path="/work/Projects/unitree_rl_lab/unitree_usd/p1/urdf/p1.urdf",
)
