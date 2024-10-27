# Pointfoot Isaac Lab Extension

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.2.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository is used to simulate pointfoot robots, such as limxdynamics TRON1.

**Keywords:** extension, isaaclab, locomotion, pointfoot

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/Andy-xiong6/pointfoot-IsaacLabExtension.git

# Option 2: SSH
git clone git@github.com:Andy-xiong6/pointfoot-IsaacLabExtension.git
```

```bash
# Enter the repository
conda activate isaaclab
cd pointfoot-IsaacLabExtension
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e exts/pointfoot
```

## Training the pointfoot agent
- Use the `scripts/rsl_rl/train.py` script to train the robot directly, specifying the task:

```bash
python scripts/rsl_rl/train.py --task=Isaac-PF-Blind-Flat-v0 --headless
```

- It is recommend to use `start.sh` script to train the robot, specifying the task in the script:

```bash
bash ./start.sh
```

- The following arguments can be used to customize the training:
    * --headless: Run the simulation in headless mode
    * --num_envs: Number of parallel environments to run
    * --max_iterations: Maximum number of training iterations
    * --save_interval: Interval to save the model
    * --seed: Seed for the random number generator

## Playing the trained model
- To play a trained model:

```bash
python scripts/rsl_rl/play.py --task=Isaac-PF-Blind-Flat-Play-v0 --checkpoint_path=path/to/checkpoint --num_envs=32
```

- The following arguments can be used to customize the playing:
    * --num_envs: Number of parallel environments to run
    * --headless: Run the simulation in headless mode
    * --checkpoint_path: Path to the checkpoint to load

## Running exported model in mujoco (sim2sim)
- After playing the model, the policy has already been saved. You can export the policy to mujoco environment and run it in mujoco by using the `rl-deploy-with-python` repo [@limxdynamics/rl-deploy-with-python](https://github.com/limxdynamics/rl-deploy-with-python).

- Follwing the instructions to install it properly and replace the `model/pointfoot/{Robot Type}/policy/policy.onnx` by your trained policy.onnx.

## Running exported model in real robot (sim2real)
- TODO

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/exts/pointfoot"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
