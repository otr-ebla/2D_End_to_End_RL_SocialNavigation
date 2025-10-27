# 2D_End_to_End_RL_SocialNavigation
Social Navigation with Reinforcement Learning. A 2D environment for indoor mobile robot navigation.

## Overview

This repository provides a ready-to-use 2D simulation environment that can be
used as the training ground for reinforcement learning algorithms focused on
social navigation. The environment features:

* A detailed building floor plan with a cross-shaped corridor spine, wide
  hallways, and individual rooms connected through explicit door openings.
* A mobile robot modelled as a disc that receives a goal position and 2D LiDAR
  measurements to perceive the scene.
* Moving humans, also represented as discs, following looping waypoint paths
  that the robot must avoid while heading towards the goal.

## Installation

```bash
pip install -r requirements.txt
```

`gymnasium` is required to interface with the environment and `matplotlib` is
only necessary when calling `env.render()`.

## Usage Example

```python
import numpy as np
from social_nav_env import SocialNavigationEnv, EnvironmentConfig

env = SocialNavigationEnv(EnvironmentConfig())
obs, _ = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.render()
```

The observation vector concatenates the robot position, heading, goal position
relative to the robot, and the simulated LiDAR distances. Actions are the
normalized linear and angular velocity commands applied to the robot.

## Manual keyboard control

You can teleoperate the differential-drive robot with the arrow keys by using
the provided helper:

```python
from social_nav_env import EnvironmentConfig, SocialNavigationEnv, keyboard_control

env = SocialNavigationEnv(EnvironmentConfig())
keyboard_control(env, linear_step=0.15, angular_step=0.25)
```

While the session is running:

* **Up/Down** increase or decrease the forward velocity command.
* **Left/Right** increase or decrease the angular velocity command.
* **Space** instantly cancels both commands.
* Press **r** to reset the environment and **q** to exit.

Keep a Matplotlib window open to monitor the robot as it navigates through the
larger corridor network and rooms with doors.

## Updating your local repository

If you have an older checkout of this project and want to pull in the latest
environment code and assets, run the following steps from the repository root:

```bash
git fetch origin
git checkout work
git pull --ff-only origin work
```

These commands download the newest commits from the default `work` branch and
fast-forward your local branch so it matches the current version of the
environment, including the expanded layout and keyboard teleoperation helper.
