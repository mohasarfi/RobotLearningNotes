"""
Image-based behavioral cloning: CNN encoder + MLP head.

Same BC loop as state-based, but observation is now a 128x128 RGB image.
The CNN compresses the image into a compact embedding, then the MLP
maps that embedding to actions.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "demos_reach_img.npz")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "policy_img.pt")


class CNNEncoder(nn.Module):
    """Simple 4-layer CNN: 128x128 → 64 → 32 → 16 → 8, then flatten."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 128→64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64→32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 32→16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 16→8
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 8 * 8, embed_dim)

    def forward(self, x):
        # x: (B, 3, 128, 128)
        h = self.convs(x)          # (B, 256, 8, 8)
        h = h.reshape(h.size(0), -1)  # (B, 256*8*8)
        return self.fc(h)          # (B, embed_dim)


class ImageBCPolicy(nn.Module):
    def __init__(self, action_dim, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, img):
        emb = self.encoder(img)
        return self.mlp(emb)


if __name__ == "__main__":
    data = np.load(DATA_PATH)
    images = torch.tensor(data["image"], dtype=torch.float32)  # (N, 128, 128, 3)
    actions = torch.tensor(data["action"], dtype=torch.float32)  # (N, 4)

    # HWC → CHW and normalize pixels to [0, 1]
    images = images.permute(0, 3, 1, 2) / 255.0  # (N, 3, 128, 128)

    # Normalize actions (same approach as state-based BC)
    act_mean, act_std = actions.mean(dim=0), actions.std(dim=0).clamp(min=1e-6)
    act_norm = (actions - act_mean) / act_std

    print(f"images: {tuple(images.shape)}")
    print(f"actions: {tuple(actions.shape)}")

    model = ImageBCPolicy(action_dim=actions.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    loader = DataLoader(
        TensorDataset(images, act_norm),
        batch_size=32,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    N_EPOCHS = 100
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        for batch_img, batch_act in loader:
            pred = model(batch_img)
            loss = loss_fn(pred, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch {epoch:3d}  loss={avg_loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "act_mean": act_mean, "act_std": act_std,
        "action_dim": actions.shape[1],
    }, SAVE_PATH)
    print(f"saved to {SAVE_PATH}")
