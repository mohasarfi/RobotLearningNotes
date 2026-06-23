"""Record a short video of random actions in Meta-World."""
import imageio
import metaworld
import numpy as np

TASK = "reach-v3"
SEED = 0
N_STEPS = 100
OUTPUT = "metaworld_test/random_policy.mp4"

mt1 = metaworld.MT1(TASK, seed=SEED)
env = mt1.train_classes[TASK](render_mode="rgb_array")
env.set_task(mt1.train_tasks[0])

obs, info = env.reset(seed=SEED)
frames = []

for step in range(N_STEPS):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    frames.append(frame)
    if terminated or truncated:
        obs, info = env.reset()

print(f"captured {len(frames)} frames, shape: {frames[0].shape}")
imageio.mimsave(OUTPUT, frames, fps=30)
print(f"saved video to {OUTPUT}")