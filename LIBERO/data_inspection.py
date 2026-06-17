"""Load one episode from the LIBERO dataset and look at it."""
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# The LIBERO dataset preprocessed for LeRobot, hosted on HF Hub.
# First run will stream/download a chunk of it.
REPO_ID = "HuggingFaceVLA/libero"

print(f"loading {REPO_ID}...")
dataset = LeRobotDataset(REPO_ID)

# Top-level info
print(f"\n=== dataset overview ===")
print(f"total frames:    {dataset.num_frames}")
print(f"total episodes:  {dataset.num_episodes}")
print(f"fps:             {dataset.fps}")

# Features schema — what's in each frame
print(f"\n=== features ===")
for name, info in dataset.features.items():
    shape = info.get("shape", "—")
    dtype = info.get("dtype", "—")
    print(f"  {name:<50} shape={shape}  dtype={dtype}")

# Look at one frame
print(f"\n=== one frame (index 0) ===")
frame = dataset[0]
for key, value in frame.items():
    if hasattr(value, "shape"):
        print(f"  {key:<50} shape={tuple(value.shape)}  dtype={value.dtype}")
    else:
        print(f"  {key:<50} value={value!r}")