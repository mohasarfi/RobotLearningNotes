"""Collect expert demos with both state observations and camera frames."""
import os
import numpy as np
import metaworld
from metaworld.policies import SawyerReachV3Policy

TASK = "reach-v3"
SEED = 0
N_EPISODES = 50
IMG_SIZE = 128
OUT_PATH = os.path.join(os.path.dirname(__file__), "demos_reach_img.npz")

mt1 = metaworld.MT1(TASK, seed=SEED)
env = mt1.train_classes[TASK](render_mode="rgb_array")
expert = SawyerReachV3Policy()

obs_list, action_list, episode_id_list, img_list = [], [], [], []
successes = 0

for ep in range(N_EPISODES):
    task_spec = mt1.train_tasks[ep % len(mt1.train_tasks)]
    env.set_task(task_spec)
    obs, info = env.reset(seed=SEED + ep)

    ep_success = False
    for step in range(150):
        frame = env.render()  # (480, 480, 3) uint8

        action = expert.get_action(obs)

        obs_list.append(obs.astype(np.float32))
        action_list.append(action.astype(np.float32))
        episode_id_list.append(ep)
        img_list.append(frame)

        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("success", 0.0) > 0.5:
            ep_success = True
            break

    successes += int(ep_success)

env.close()

imgs = np.stack(img_list)  # (N, 480, 480, 3)
print(f"collected {N_EPISODES} episodes, {successes} succeeded ({100*successes/N_EPISODES:.0f}%)")
print(f"total frames: {len(obs_list)}, raw img shape: {imgs.shape}")

# Downsample to IMG_SIZE x IMG_SIZE using strided slicing (no PIL dependency)
# 480 / 128 is not integer, so we use simple area-based resize via reshape trick:
# crop center 384x384 (divisible by 128) then reshape-mean to 128x128
crop = (480 - 384) // 2
imgs_cropped = imgs[:, crop:crop+384, crop:crop+384, :]  # (N, 384, 384, 3)
factor = 384 // IMG_SIZE  # 3
imgs_resized = imgs_cropped.reshape(
    len(imgs), IMG_SIZE, factor, IMG_SIZE, factor, 3
).mean(axis=(2, 4)).astype(np.uint8)  # (N, 128, 128, 3)

print(f"resized img shape: {imgs_resized.shape}")

np.savez_compressed(
    OUT_PATH,
    obs=np.array(obs_list),
    action=np.array(action_list),
    episode_id=np.array(episode_id_list),
    image=imgs_resized,
)
print(f"saved to {OUT_PATH}")
