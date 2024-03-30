#! /bin/bash

set -ex

# python3 eval_trajectory.py
python3 eval_reward_surface.py
python3 eval_boost.py
