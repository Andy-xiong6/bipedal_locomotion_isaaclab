import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.rsl_rl_ppo_cfg import PointFootPPORunnerCfg, WheelFootPPORunnerCfg
from bipedal_locomotion.tasks.locomotion.agents.rsl_rl_ppo_mlp_cfg import PFFlatPPORunnerMlpCfg, PFStairPPORunnerMlpCfg, WFFlatPPORunnerMlpCfg, WFRoughPPORunnerMlpCfg, WFStairPPORunnerMlpCfg

from . import pointfoot_env_cfg, wheelfoot_env_cfg

##
# Create PPO runners for RSL-RL
##

pf_blind_flat_runner_cfg = PointFootPPORunnerCfg()
pf_blind_flat_runner_cfg.experiment_name = "pf_blind_flat"

pf_blind_rough_runner_cfg = PointFootPPORunnerCfg()
pf_blind_rough_runner_cfg.experiment_name = "pf_blind_rough"

pf_blind_stairs_runner_cfg = PointFootPPORunnerCfg()
pf_blind_stairs_runner_cfg.experiment_name = "pf_blind_stairs"

pf_mlp_blind_flat_runner_cfg = PFFlatPPORunnerMlpCfg()
pf_mlp_blind_flat_runner_cfg.experiment_name = "pf_mlp_blind_flat"

pf_mlp_stair_runner_cfg = PFStairPPORunnerMlpCfg()
pf_mlp_stair_runner_cfg.experiment_name = "pf_mlp_stairs"

wf_blind_flat_runner_cfg = WheelFootPPORunnerCfg()
wf_blind_flat_runner_cfg.experiment_name = "wf_blind_flat"

wf_mlp_blind_flat_runner_cfg = WheelFootPPORunnerCfg()
wf_mlp_blind_flat_runner_cfg.experiment_name = "wf_mlp_blind_flat"

wf_blind_rough_runner_cfg = WheelFootPPORunnerCfg()
wf_blind_rough_runner_cfg.experiment_name = "wf_blind_rough"

wf_mlp_blind_rough_runner_cfg = WheelFootPPORunnerCfg()
wf_mlp_blind_rough_runner_cfg.experiment_name = "wf_mlp_blind_rough"

wf_blind_stair_runner_cfg = WheelFootPPORunnerCfg()
wf_blind_stair_runner_cfg.experiment_name = "wf_blind_stair"

wf_mlp_blind_stair_runner_cfg = WheelFootPPORunnerCfg()
wf_mlp_blind_stair_runner_cfg.experiment_name = "wf_mlp_blind_stair"

wf_mlp_flat_runner_cfg = WFFlatPPORunnerMlpCfg()
wf_mlp_flat_runner_cfg.experiment_name = "wf_mlp_flat"

wf_mlp_rough_runner_cfg = WFRoughPPORunnerMlpCfg()
wf_mlp_rough_runner_cfg.experiment_name = "wf_mlp_rough"

wf_stair_runner_cfg = WFStairPPORunnerMlpCfg()
wf_stair_runner_cfg.experiment_name = "wf_mlp_stair"



##
# Register Gym environments
##

############################
# PF Blind Flat Environment
############################

gym.register(
    id="Isaac-PF-Blind-Flat-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Flat-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pf_blind_flat_runner_cfg,
    },
)


#############################
# PF Blind Flat Environment v1
#############################

gym.register(
    id="Isaac-PF-Blind-Flat-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": pf_mlp_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Flat-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pf_mlp_blind_flat_runner_cfg,
    },
)


#############################
# PF Blind Rough Environment
#############################

gym.register(
    id="Isaac-PF-Blind-Rough-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Rough-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg,
    },
)


#############################
# PF Blind Rough Environment v1
#############################

gym.register(
    id="Isaac-PF-Blind-Rough-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Rough-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg,
    },
)


##############################
# PF Blind Stair Environment
##############################

gym.register(
    id="Isaac-PF-Blind-Stairs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindStairEnvCfg,
        "rsl_rl_cfg_entry_point": pf_blind_stairs_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Stairs-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pf_blind_stairs_runner_cfg,
    },
)


#############################
# PF Blind Stair Environment v1
#############################

gym.register(
    id="Isaac-PF-Blind-Stair-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindStairEnvCfg,
        "rsl_rl_cfg_entry_point": pf_mlp_stair_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Stair-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pf_mlp_stair_runner_cfg,
    },
)


#############################
# WF Blind Flat Environment
#############################

gym.register(
    id="Isaac-WF-Blind-Flat-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": wf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-WF-Blind-Flat-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_blind_flat_runner_cfg,
    },
)


#############################
# WF Blind Flat Environment v1
#############################

gym.register(
    id="Isaac-WF-Blind-Flat-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": wf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-WF-Blind-Flat-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_blind_flat_runner_cfg,
    },
)


#############################
# WF Blind Rough Environment
#############################

gym.register(
    id="Isaac-WF-Blind-Rough-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": wf_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-WF-Blind-Rough-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_blind_rough_runner_cfg,
    },
)


#############################
# WF Blind Rough Environment v1
#############################

gym.register(
    id="Isaac-WF-Blind-Rough-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": wf_mlp_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-WF-Blind-Rough-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_mlp_rough_runner_cfg,
    },
)


#############################
# WF Blind Stair Environment
#############################

gym.register(
    id="Isaac-WF-Blind-Stair-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindStairEnvCfg,
        "rsl_rl_cfg_entry_point": wf_stair_runner_cfg,
    },
)

gym.register(
    id="Isaac-WF-Blind-Stair-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_stair_runner_cfg,
    },
)


#############################
# WF Blind Stair Environment v1
#############################

gym.register(
    id="Isaac-WF-Blind-Stair-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindStairEnvCfg,
        "rsl_rl_cfg_entry_point": wf_mlp_blind_stair_runner_cfg,
    },
)

gym.register(
    id="Isaac-WF-Blind-Stair-Play-v1",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFBlindStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_mlp_blind_stair_runner_cfg,
    },
)


#############################
# WF Flat Environment 
#############################

gym.register(
    id="Isaac-WF-Flat",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFFlatEnvCfg,
        "rsl_rl_cfg_entry_point": wf_mlp_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-WF-Flat-PLAY",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_mlp_flat_runner_cfg,
    },
)


#############################
# WF Rough Environment 
#############################

gym.register(
    id="Isaac-WF-Rough",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFRoughEnvCfg,
        "rsl_rl_cfg_entry_point": wf_mlp_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-WF-Stair-Rough",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_mlp_rough_runner_cfg,
    },
)


#############################
# WF Stair Environment 
#############################

gym.register(
    id="Isaac-WF-Stair",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFStairEnvCfg,
        "rsl_rl_cfg_entry_point": wf_stair_runner_cfg,
    },
)

gym.register(
    id="Isaac-WF-Stair-Play",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wheelfoot_env_cfg.WFStairEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_stair_runner_cfg,
    },
)





