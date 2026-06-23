import gymnasium as gym
import metaworld
import numpy as np

TASK = "reach-v3"
SEED = 0

# Build the env using Meta-World's MT1 single-task setup
mt1 = metaworld.MT1(TASK, seed=SEED)
env = mt1.train_classes[TASK]()
env.set_task(mt1.train_tasks[0])

obs, info = env.reset(seed=SEED)
print(f"obs shape: {obs.shape}")
print(f"action space: {env.action_space}")

for step in range(50):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

print("done — env works")