"""Step through one LIBERO episode with random actions."""
import os
os.environ["MUJOCO_GL"] = "egl"  # headless rendering on Linux

from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv

# 1. Get the task suite (a benchmark object that holds all 10 spatial tasks)
task_suite_name = "libero_spatial"
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[task_suite_name]()

# 2. Build the LeRobot wrapper around task 0 of that suite
env = LiberoEnv(
    task_suite=task_suite,
    task_id=0,
    task_suite_name=task_suite_name,
)

# 3. Reset and step
obs, info = env.reset(seed=0)
print(f"obs keys: {list(obs.keys())}")
for k, v in obs.items():
    shape = getattr(v, "shape", None)
    print(f"  {k}: shape={shape}")
print(f"action space: {env.action_space}")

for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

print("done — libero env works")