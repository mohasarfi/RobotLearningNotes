"""Roll out the trained BC policy in Meta-World and measure success rate."""
import numpy as np
import torch
import torch.nn as nn
import metaworld
from train import MLP

# Load checkpoints
ckpt = torch.load("BC/policy.pt", weights_only=True)
model = MLP(ckpt["in_dim"], ckpt["out_dim"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

obs_mean, obs_std = ckpt["obs_mean"], ckpt["obs_std"]
act_mean, act_std = ckpt["act_mean"], ckpt["act_std"]

# Setup env
TASK = "reach-v3"
SEED = 0
N_EPISODES = 20

mt1 = metaworld.MT1(TASK, seed=SEED)
env = mt1.train_classes[TASK]()

successes = 0
for ep in range(N_EPISODES):
    task_spec = mt1.train_tasks[ep % len(mt1.train_tasks)]
    env.set_task(task_spec)
    obs, info = env.reset(seed=SEED + ep)

    for step in range(150):
        # normalize → predict → denormalize
        obs_t = torch.tensor(obs, dtype=torch.float32)
        obs_n = (obs_t - obs_mean) / obs_std
        with torch.no_grad():
            pred_n = model(obs_n)
        action = (pred_n * act_std + act_mean).numpy()
        action = np.clip(action, -1.0, 1.0)  # env expects [-1, 1]

        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("success", 0.0) > 0.5:
            successes += 1
            break

print(f"success rate: {successes}/{N_EPISODES} ({100*successes/N_EPISODES:.0f}%)")