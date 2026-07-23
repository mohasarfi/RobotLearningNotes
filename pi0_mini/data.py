"""Data pipeline for pi0-mini: LIBERO episodes -> chunked training batches.

One training example is (x, y):
  x = two camera images + robot state + a language instruction
  y = an H-step action chunk, shape (H, action_dim)

LeRobot builds the chunk for us via `delta_timestamps` and, crucially, hands
back an `action_is_pad` mask (shape (H,)) that is True wherever the chunk ran
off the end of the episode. We must exclude those steps from the loss.

"""

from dataclasses import dataclass, field

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class DataConfig:
    repo_id: str = "HuggingFaceVLA/libero"
    fps: int = 10                       # LIBERO is 10 fps
    chunk_size: int = 10                # H: 1.0 s of actions at 10 fps

    # Feature keys as they exist in the LeRobotDataset schema.
    image_keys: tuple[str, ...] = ("observation.images.image",       # agentview
                                   "observation.images.image2")      # wrist
    state_key: str = "observation.state"
    action_key: str = "action"

    # Load only a handful of episodes while developing on the Mac. None = all.
    episodes: list[int] | None = field(default_factory=lambda: list(range(4)))

    @property
    def delta_timestamps(self) -> dict[str, list[float]]:
        """Ask LeRobot for an H-step action chunk starting at the current frame.

        Times are in SECONDS relative to now: [0, 1/fps, 2/fps, ...].
        State and images use the default (current frame only), so we don't list
        them here.
        """
        dt = 1.0 / self.fps
        return {self.action_key: [i * dt for i in range(self.chunk_size)]}


def build_dataset(cfg: DataConfig) -> LeRobotDataset:
    """Construct the LeRobotDataset that yields chunked examples."""
    return LeRobotDataset(
        cfg.repo_id,
        episodes=cfg.episodes,
        delta_timestamps=cfg.delta_timestamps,
    )


if __name__ == "__main__":
    # Quick sanity check that construction works and one item has the right keys.
    cfg = DataConfig()
    ds = build_dataset(cfg)
    print(ds)
    item = ds[0]
    for k, v in item.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else v
        print(f"{k:32s} {type(v).__name__:12s} {shape}")
