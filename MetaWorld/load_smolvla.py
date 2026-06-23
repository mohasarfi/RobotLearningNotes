"""Load a pretrained SmolVLA and run one forward pass with a dummy observation."""
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

POLICY_ID = "lerobot/smolvla_base"

print(f"loading {POLICY_ID}...")
policy = SmolVLAPolicy.from_pretrained(POLICY_ID)
policy.eval()

print(f"policy loaded: {type(policy).__name__}")
print(f"input features: {list(policy.config.input_features.keys())}")
print(f"output features: {list(policy.config.output_features.keys())}")

n_params = sum(p.numel() for p in policy.parameters())
print(f"params: {n_params/1e6:.1f}M")