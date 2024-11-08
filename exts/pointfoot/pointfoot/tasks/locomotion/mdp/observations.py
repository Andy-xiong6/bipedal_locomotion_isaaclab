from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.string import resolve_matching_names

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_contact_bools(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Feet contact booleans. The foot is in contact when the force sensor exceeds the threshold"""

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    # check which contact forces exceed the threshold
    return torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1) > threshold


def joint_pos_rel_debug(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Get joint positions relative to the default positions based on joint names."""
    asset: Articulation = env.scene[asset_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    joint_names_all = asset.data.joint_names
    print("joint_pos: ", asset.data.joint_pos[:, asset_cfg.joint_ids])
    print("default_joint_pos: ", asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    print(asset_cfg.joint_ids)
    print(joint_names_all)

    return joint_pos - default_joint_pos
