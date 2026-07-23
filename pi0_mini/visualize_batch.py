"""D1 gate: prove you know exactly what one training example is.

Run:  python -m pi0_mini.visualize_batch

It answers, from real output:
  1. What every tensor in one example is (key, type, shape, dtype).
  2. What an action chunk looks like at an episode boundary (the is_pad mask).
  3. Whether the action is delta or absolute (from the per-dim stats).
And renders the two camera views with the instruction overlaid.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt

from pi0_mini.data import DataConfig, build_dataset


def describe_example(item: dict) -> None:
    print("\n=== one training example ===")
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:32s} {str(tuple(v.shape)):16s} {v.dtype}")
        else:
            print(f"  {k:32s} {v!r}")


def find_boundary_example(ds, action_key="action") -> int:
    """Return the index of an example whose action chunk crosses an episode end
    (i.e. has at least one padded step). Falls back to 0 if none found."""
    pad_key = f"{action_key}_is_pad"
    for i in range(len(ds)):
        if bool(ds[i][pad_key].any()):
            return i
    return 0


def show_boundary(ds, idx: int, action_key="action") -> None:
    item = ds[idx]
    pad = item[f"{action_key}_is_pad"]
    print(f"\n=== episode-boundary example (idx={idx}) ===")
    print("  action_is_pad:", pad.tolist())
    print("  -> True = this step ran off the episode end; LeRobot clamped it to")
    print("     the last valid frame. Exclude these from the loss with the mask.")
    # Show that padded rows are identical (repeat-last), confirming the fill rule.
    print("  last 3 action rows:\n", item[action_key][-3:].numpy().round(3))


def report_action_semantics(ds, cfg: DataConfig) -> None:
    """Delta vs absolute: collect current-frame actions and look at their range.
    Delta/velocity actions center near 0; absolute positions do not."""
    xs = np.stack([ds[i][cfg.action_key][0].numpy() for i in range(min(len(ds), 500))])
    print("\n=== action semantics (first step of each chunk) ===")
    print("  per-dim mean:", xs.mean(0).round(3))
    print("  per-dim std :", xs.std(0).round(3))
    print("  per-dim min :", xs.min(0).round(3))
    print("  per-dim max :", xs.max(0).round(3))
    print("  -> centered near 0 => delta/velocity control; otherwise absolute.")


def render_cameras(item: dict, cfg: DataConfig, out="pi0_mini/example_frame.png") -> None:
    fig, axes = plt.subplots(1, len(cfg.image_keys), figsize=(8, 4))
    for ax, key in zip(np.atleast_1d(axes), cfg.image_keys):
        img = item[key]
        if isinstance(img, torch.Tensor):      # CHW float [0,1] -> HWC
            img = img.permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(key.split(".")[-1], fontsize=9)
        ax.axis("off")
    fig.suptitle(item.get("task", ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    cfg = DataConfig()
    ds = build_dataset(cfg)
    print(ds)

    describe_example(ds[0])
    report_action_semantics(ds, cfg)
    show_boundary(ds, find_boundary_example(ds, cfg.action_key), cfg.action_key)
    render_cameras(ds[0], cfg)
