import math
from dataclasses import MISSING

from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from pointfoot.tasks.locomotion import mdp

##################
# Scene Definition
##################


@configclass
class PFSceneCfg(InteractiveSceneCfg):
    """Configuration for the test scene"""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # sky light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # pointfoot robot
    robot: ArticulationCfg = MISSING

    # height sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
    height_scanner = None

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
        track_pose=True,
    )


##############
# MDP settings
##############


@configclass
class CommandsCfg:
    """Command terms for the MDP"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        debug_vis=True,
        resampling_time_range=(10.0, 10.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-1, 1), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.4, use_default_offset=True)


@configclass
class ObservarionsCfg:
    """Observation specifications for the MDP"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group"""

        # robot base measurements
        # base_lin_vel = ObsTerm(func=mdp.base_lib_vel, noise=GaussianNoise(mean=0.0, std=0.1), scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.2), scale=0.25)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.05), scale=1.0)

        # robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoise(mean=0.0, std=0.01), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoise(mean=0.0, std=1.5), scale=0.05)

        # last action
        last_action = ObsTerm(func=mdp.last_action, scale=1.0)

        # velocity command
        vel_command = ObsTerm(
            func=mdp.scaled_generated_commands,
            params={
                "command_name": "base_velocity",
                "scale_factor": [2.0, 0.25, 1.0],
                "clip_heading": True,
            },
        )
        # height measurement
        # heights = ObsTerm(func=mdp.height_scan,
        #                   params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        #                             noise=GaussianNoise(mean=0.0, std=0.01),
        #                     )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Configuration for events"""

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    # reset
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi / 6, math.pi / 6)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP"""

    # rewards
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=10.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    rew_no_fly = RewTerm(
        func=mdp.no_fly,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*contact_[LR]_Link")},
    )

    # penalizations
    pen_feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*contact_[LR]_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*contact_[LR]_Link"),
        },
    )
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-50.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*knee_[LR]_Link"), "threshold": 1.0},
    )
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    pen_joint_powers = RewTerm(func=mdp.joint_powers_l1, weight=-5e-4)
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    pen_base_height = RewTerm(func=mdp.base_height_l2, params={"target_height": 0.62}, weight=-10.0)
    pen_feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link"), "threshold": 1.0},
        weight=-0.01,
    )
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    pen_applied_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.1)
    pen_unbalance_feet_air_time = RewTerm(
        func=mdp.unbalance_feet_air_time,
        weight=-300.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*foot_[LR]_Link",
            ),
        },
    )
    pen_unbalance_feet_height = RewTerm(
        func=mdp.unbalance_feet_height,
        weight=-60.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*foot_[LR]_Link",
            ),
        },
    )
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-100.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=".*foot_[LR]_Link",
            ),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


########################
# Environment definition
########################


@configclass
class PFEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the test environment"""

    # Scene settings
    scene: PFSceneCfg = PFSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservarionsCfg = ObservarionsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization"""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        # simulation settings
        self.sim.dt = 0.005
        self.seed = 42
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
