"""
Behavioral cloning on expert demonstartion.
for each training step:
    sample a batch of trnsitions (state,action)
    predict actions from observations with the network
    loss = mse(predicted_actions, expert_actions)
    backprop
"""

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset # DataLoader is a class that provides an iterable over a dataset, with support for automatic batching, sampling, shuffling and multiprocess data loading.

data = np.load("./MetaWorld/demos_reach.npz")
obs = torch.tensor(data["obs"], dtype=torch.float32)  # shpe will be (N, obs_dim)n  ,39
actions = torch.tensor(data["action"], dtype=torch.float32) # (N,4) --> 4: (dx,dy,dz,gripper)

# Sanity check
print(f"obs shape:     {tuple(obs.shape)}")
print(f"actions shape: {tuple(actions.shape)}")

# Normalize the observations and actions to have zero mean and unit variance
obs_mean, obs_std = obs.mean(dim=0), obs.std(dim=0).clamp(min=1e-6)
act_mean, act_std = actions.mean(dim=0), actions.std(dim=0).clamp(min=1e-6)
obs_norm = (obs - obs_mean) / obs_std
act_norm = (actions - act_mean) / act_std

# neural network architecture - simple 3 layer mlp for now
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self,x):
        return self.net(x)

model = MLP(in_dim=obs.shape[1], out_dim=actions.shape[1])
print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")


# Train 
loader = DataLoader(TensorDataset(obs_norm, act_norm), batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

N_EPOCHS = 50 
for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    for batch_obs, batch_act in loader:
        pred = model(batch_obs)
        loss = loss_fn(pred, batch_act)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
        print(f"epoch {epoch:3d}  loss={epoch_loss/len(loader):.4f}")


# Save the model

torch.save({
    "model_state_dict": model.state_dict(),
    "obs_mean": obs_mean, "obs_std": obs_std,
    "act_mean": act_mean, "act_std": act_std,
    "in_dim": obs.shape[1], "out_dim": actions.shape[1],
}, "BC/policy.pt")
print("saved to BC/policy.pt")