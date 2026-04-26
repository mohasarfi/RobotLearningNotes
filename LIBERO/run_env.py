"""Step through one LIBERO episode with random actions."""
import os
os.environ["MUJOCO_GL"] = "egl"  # headless rendering on Linux

from lerobot.envs.libero import LiberoEnv
import numpy as np

# LIBERO suites: libero_spatial, libero_object, libero_goal, libero_10, libero_90
# We pick libero_spatial — 10 tasks about spatial reasoning, the smallest suite.
env = LiberoEnv(task="libero_spatial", task_id=0, seed=0)

obs, info = env.reset(seed=0)
print(f"obs keys: {list(obs.keys())}")
for k, v in obs.items():
    print(f"  {k}: shape={getattr(v, 'shape', None)}")
print(f"action space: {env.action_space}")

for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

print("done — libero env works")