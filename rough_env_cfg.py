from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.grallator.grallator import GRALLATOR_CFG


@configclass
class GrallatorRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot asset
        self.scene.robot = GRALLATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Grallator base body is named trunk in the Isaac-friendly URDF.
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"

        # Conservative terrain settings. Use rough only after flat walking works.
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.08)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.005, 0.04)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Actions are residual joint-position targets. Keep small for first training.
        self.actions.joint_pos.scale = 0.20

        # Command curriculum: start easy. Expand after the robot walks.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Events/randomization: disable pushes initially; keep mild mass randomization.
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Rewards: feet are cleanly named FR_foot, FL_foot, RR_foot, RL_foot.
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.05
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.3
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01

        # Termination: fall when trunk/base contacts ground.
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"


@configclass
class GrallatorRoughEnvCfg_PLAY(GrallatorRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
