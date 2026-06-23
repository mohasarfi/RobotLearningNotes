"""Collect demos from Meta-World's scripted expert on the reach task."""
import os
import numpy as np
import metaworld
from metaworld.policies import SawyerReachV3Policy

TASK = "reach-v3"
SEED = 0
N_EPISODES = 50
OUT_PATH = os.path.join(os.path.dirname(__file__), "demos_reach.npz")

# Build the env
mt1 = metaworld.MT1(TASK, seed=SEED) # MT1 Contains only 1 task! but with many variants (different goal positions and objects in the environment)
env = mt1.train_classes[TASK]()
expert = SawyerReachV3Policy()

obs_list, action_list, episode_id_list = [], [], []
successes = 0

for ep in range(N_EPISODES):
    # Each episode samples a new task variant from the suite
    task_spec = mt1.train_tasks[ep % len(mt1.train_tasks)] # N_EPISODES may be larger than the number of tasks, so we wrap around! --> We might have a variant of the same task multiple times in our dataset
    env.set_task(task_spec)
    obs, info = env.reset(seed=SEED + ep)

    ep_success = False
    for step in range(150):  # Meta-World episodes are max 500 steps; 150 is plenty for reach
        action = expert.get_action(obs)
        obs_list.append(obs.astype(np.float32))
        action_list.append(action.astype(np.float32))
        episode_id_list.append(ep)

        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("success", 0.0) > 0.5:
            ep_success = True
            break

    successes += int(ep_success)

print(f"collected {N_EPISODES} episodes, {successes} succeeded ({100*successes/N_EPISODES:.0f}%)")
print(f"total frames: {len(obs_list)}")

np.savez(
    OUT_PATH,
    obs=np.array(obs_list),
    action=np.array(action_list),
    episode_id=np.array(episode_id_list),
)
print(f"saved to {OUT_PATH}")