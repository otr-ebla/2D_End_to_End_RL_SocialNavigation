# 2D_End_to_End_RL_SocialNavigation
Social Navigation with Reinforcement Learning. A 2D environment for indoor mobile robot navigation.

## Overview

This repository provides a ready-to-use 2D simulation environment that can be
used as the training ground for reinforcement learning algorithms focused on
social navigation. The environment features:

* A simplified building floor plan with corridors and rooms represented by
  static walls.
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
