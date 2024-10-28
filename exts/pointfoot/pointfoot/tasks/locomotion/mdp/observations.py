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

    # 如果 joint_names 为空，则使用 asset 的所有 joint_names
    joint_names_all = asset.data.joint_names
    joint_names_to_use = asset_cfg.joint_names if asset_cfg.joint_names else joint_names_all

    # 通过 joint_names_to_use 中的名称在 joint_names_all 中查找对应索引
    matched_indices = [joint_names_all.index(name) for name in joint_names_to_use if name in joint_names_all]

    # 打印匹配的关节名称和索引
    matched_names = [joint_names_all[i] for i in matched_indices]
    print("Matched joint names:", matched_names)
    print("Matched indices:", matched_indices)

    # 使用匹配的索引访问 joint_pos
    joint_pos = asset.data.joint_pos[:, matched_indices]
    default_joint_pos = asset.data.default_joint_pos[:, matched_indices]

    print("Joint positions:", joint_pos)
    print("Default joint positions:", default_joint_pos)

    return joint_pos - default_joint_pos
