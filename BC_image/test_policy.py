"""Roll out the image-based BC policy in Meta-World and measure success rate."""
import numpy as np
import torch
import metaworld
from train import ImageBCPolicy

TASK = "reach-v3"
SEED = 0
N_EPISODES = 20
IMG_SIZE = 128

ckpt = torch.load("BC_image/policy_img.pt", weights_only=True)
model = ImageBCPolicy(action_dim=ckpt["action_dim"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

act_mean, act_std = ckpt["act_mean"], ckpt["act_std"]

mt1 = metaworld.MT1(TASK, seed=SEED)
env = mt1.train_classes[TASK](render_mode="rgb_array")

successes = 0
for ep in range(N_EPISODES):
    task_spec = mt1.train_tasks[ep % len(mt1.train_tasks)]
    env.set_task(task_spec)
    obs, info = env.reset(seed=SEED + ep)

    for step in range(150):
        frame = env.render()  # (480, 480, 3) uint8

        # Same downsampling as collection: center-crop 384x384, resize to 128x128
        crop = (480 - 384) // 2
        frame = frame[crop:crop+384, crop:crop+384, :]
        factor = 384 // IMG_SIZE
        frame = frame.reshape(IMG_SIZE, factor, IMG_SIZE, factor, 3).mean(axis=(1, 3)).astype(np.uint8)

        img_t = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, 128, 128)
        img_t = img_t.unsqueeze(0)  # (1, 3, 128, 128)

        with torch.no_grad():
            pred_n = model(img_t)
        action = (pred_n.squeeze(0) * act_std + act_mean).numpy()
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("success", 0.0) > 0.5:
            successes += 1
            break

env.close()
print(f"success rate: {successes}/{N_EPISODES} ({100*successes/N_EPISODES:.0f}%)")
