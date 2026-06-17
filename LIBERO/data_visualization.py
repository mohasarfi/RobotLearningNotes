"""Save the first frame of a few episodes as PNGs to look at."""
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision.utils import save_image

dataset = LeRobotDataset("HuggingFaceVLA/libero")

print("dataset attributes:", [a for a in dir(dataset) if "episode" in a.lower()])

# Look at the first frame of episodes 0, 100, 500, 1000
for ep in [0, 100, 500, 1000]:
    # find the first frame of this episode by scanning the underlying table
    # hf_dataset is the underlying HuggingFace Dataset object
    episode_indices = dataset.hf_dataset["episode_index"]
    ep_start = next(i for i, e in enumerate(episode_indices) if e == ep)
    frame = dataset[ep_start]

    img = frame["observation.images.image"]
    save_image(img, f"LIBERO/ep{ep:04d}_view.png")

    print(f"episode {ep:4d}: task='{frame['task']}'")