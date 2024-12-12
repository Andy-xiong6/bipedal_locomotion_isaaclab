import gymnasium as gym

from pointfoot.tasks.locomotion.agents.rsl_rl_ppo_cfg import PointFootPPORunnerCfg

from . import pointfoot_env_cfg

##
# Create PPO runners for RSL-RL
##

pf_blind_flat_runner_cfg = PointFootPPORunnerCfg()
pf_blind_flat_runner_cfg.experiment_name = "pf_blind_flat"

pf_blind_rough_runner_cfg = PointFootPPORunnerCfg()
pf_blind_rough_runner_cfg.experiment_name = "pf_blind_rough"

pf_blind_stairs_runner_cfg = PointFootPPORunnerCfg()
pf_blind_stairs_runner_cfg.experiment_name = "pf_blind_stairs"


test_blind_rough_runner_cfg = PointFootPPORunnerCfg()
test_blind_rough_runner_cfg.experiment_name = "test_blind_rough"
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

##############################
# PF Blind Stairs Environment
##############################

gym.register(
    id="Isaac-PF-Blind-Stairs-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindStairsEnvCfg,
        "rsl_rl_cfg_entry_point": pf_blind_stairs_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Stairs-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.PFBlindStairsEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": pf_blind_stairs_runner_cfg,
    },
)

##############################
# PF Rough Test Environment
##############################

gym.register(
    id="Isaac-PF-Blind-Rough-test",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.TestBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Rough-Play-test",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.TestBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": test_blind_rough_runner_cfg,
    },
)

##############################
# PF Flat Test Environment
##############################

gym.register(
    id="Isaac-PF-Blind-Flat-test",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.TestBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": pf_blind_rough_runner_cfg,
    },
)

gym.register(
    id="Isaac-PF-Blind-Flat-Play-test",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pointfoot_env_cfg.TestBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": test_blind_rough_runner_cfg,
    },
)