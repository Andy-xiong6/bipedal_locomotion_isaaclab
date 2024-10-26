from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_contact_bools(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Feet contact booleans. The foot is in contact when the force sensor exceeds the threshold"""

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    # check which contact forces exceed the threshold
    return torch.norm(net_contact_forces[:, sensor_cfg.body_ids], dim=-1) > threshold

def scaled_generated_commands(env: ManagerBasedRLEnv, command_name: str, scale_factor: list, clip_heading: bool = False) -> torch.Tensor:
    """Retrieve, scale, and optionally clip the generated command by individual factors."""
    
    # 获取命令
    command = env.command_manager.get_command(command_name)
    scale_tensor = torch.tensor(scale_factor, device=command.device)
    
    # 检查 scale_factor 和 command 的维度是否匹配
    assert scale_tensor.shape == command.shape[1:], "scale_factor must match command dimensions"
    
    # 应用缩放
    scaled_command = command * scale_tensor
    
    # 根据 clip_heading 选择性剪裁 heading 分量
    if clip_heading:
        scaled_command[:, 2] = torch.clamp(scaled_command[:, 2], -1, 1)
    
    return scaled_command