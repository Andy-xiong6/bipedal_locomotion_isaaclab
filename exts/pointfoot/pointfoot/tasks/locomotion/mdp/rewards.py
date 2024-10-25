from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)

def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward if only one foot is in contact with the ground."""
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]

    contacts = latest_contact_forces > 0.1
    single_contact = torch.sum(contacts.float(), dim=1) == 1
    
    return 1. * single_contact

def unbalance_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, balance_threshold: float = 0.1) -> torch.Tensor:
    """Penalize if the feet air time variance exceeds the balance threshold."""
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    return torch.var(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], dim=-1)
   
def unbalance_feet_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the variance of feet maximum height using sensor positions."""
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    feet_positions = contact_sensor.data.pos_w[:, sensor_cfg.body_ids]
    
    if feet_positions is None:
        return torch.zeros(env.num_envs)
    
    feet_heights = feet_positions[:, :, 2]
    max_feet_heights = torch.max(feet_heights, dim=-1)[0]
    height_variance = torch.var(max_feet_heights, dim=-1)
    return height_variance

def feet_distance(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, min_feet_distance: float = 0.1) -> torch.Tensor:
    """Penalize if the distance between feet is below a minimum threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_positions = contact_sensor.data.pos_w[:, sensor_cfg.body_ids]
    
    if feet_positions is None:
        return torch.zeros(env.num_envs)
    
    reward = 0
    num_feet = feet_positions.shape[1]

    for i in range(num_feet - 1):
        for j in range(i + 1, num_feet):
            feet_distance = torch.norm(feet_positions[:, i, :2] - feet_positions[:, j, :2], dim=-1)
            reward += torch.clip(min_feet_distance - feet_distance, 0, 1)
    return reward