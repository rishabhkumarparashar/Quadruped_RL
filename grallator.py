import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Put your exported Isaac Sim USD here after importing grallator_isaac_lab.urdf.
# You can override this from the terminal using:
#   export GRALLATOR_USD_PATH=/absolute/path/to/grallator.usd
GRALLATOR_USD_PATH = os.environ.get(
    "GRALLATOR_USD_PATH",
    os.path.join(os.path.dirname(__file__), "grallator.usd"),
)

GRALLATOR_DEFAULT_JOINT_POS = {
    # Native STANDBY keyframe pose from grallator_final.xml.
    # Isaac-friendly names:
    # hip   = original collar / ab-ad joint
    # thigh = original hip joint
    # calf  = original knee joint

    "FR_hip_joint": 0.0,
    "FR_thigh_joint": -0.69,
    "FR_calf_joint": -1.40,

    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.69,
    "FL_calf_joint": 1.40,

    "RR_hip_joint": 0.0,
    "RR_thigh_joint": -0.69,
    "RR_calf_joint": -1.40,

    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.69,
    "RL_calf_joint": 1.40,
}

GRALLATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GRALLATOR_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Tune z after the Isaac Sim import test. If the body drops too far, increase it slightly.
        pos=(0.0, 0.0, 0.199411),
        joint_pos=GRALLATOR_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=60.0,
            saturation_effort=60.0,
            velocity_limit=25.0,
            stiffness=50.0,
            damping=2.0,
            friction=0.0,
        ),
    },
)
