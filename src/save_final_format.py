"""Convert the final CACo checkpoint to the paper's release naming format.

Produces files matching the Cornell CACo checkpoint release naming:
  {backbone}_{method}_geo_{dataset_size}_{epochs}.ckpt  (Lightning wrapper)
  {backbone}_{method}_geo_{dataset_size}_{epochs}.pth   (encoder_q state_dict)

The .pth file is saved in the exact format expected by the eurosat_eval_pytorch
notebook: a plain state_dict that loads into
  nn.Sequential(*list(resnet18().children())[:-1], nn.Flatten())
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# Required so torch.load can deserialize our custom callbacks from the ckpt
from train_caco import MetricLogger, FinalModelSaver  # noqa: F401

CKPT_DIR = Path('/users/vsmokash/deep-learning/CACo/checkpoints/resnet18-clean_10k_geography-caco-ep400-bs256-q4096')

# Input: prefer the callback-saved final .ckpt, fall back to the last periodic one
candidates = [
    CKPT_DIR / 'caco_final.ckpt',
    CKPT_DIR / 'caco-epoch=0399.ckpt',
    CKPT_DIR / 'caco-epoch=0299.ckpt',
]
src_ckpt = None
for c in candidates:
    if c.exists():
        src_ckpt = c
        break
if src_ckpt is None:
    raise FileNotFoundError('No source checkpoint found')

print(f'Source checkpoint: {src_ckpt}')

# Output naming (matches Cornell's release convention)
BACKBONE = 'resnet18'
METHOD = 'caco'
DATASET_SIZE = '10k'
EPOCHS = 400
MODEL_NAME = f'{BACKBONE}_{METHOD}_geo_{DATASET_SIZE}_{EPOCHS}'

out_ckpt = CKPT_DIR / f'{MODEL_NAME}.ckpt'
out_pth = CKPT_DIR / f'{MODEL_NAME}.pth'

# Load PL checkpoint
ckpt = torch.load(str(src_ckpt), map_location='cpu')
print(f'  Loaded — epoch: {ckpt.get("epoch")}, global_step: {ckpt.get("global_step")}')

# 1) Lightning .ckpt: just copy verbatim (full trainer state, loadable by MocoV2.load_from_checkpoint)
torch.save(ckpt, str(out_ckpt))
print(f'Lightning checkpoint saved: {out_ckpt}')

# 2) PyTorch .pth: only encoder_q state_dict, keys stripped of the "encoder_q." prefix
#    so it loads directly into nn.Sequential(*resnet18_children[:-1], Flatten())
state_dict = ckpt['state_dict']
encoder_q_sd = {
    k[len('encoder_q.'):]: v
    for k, v in state_dict.items()
    if k.startswith('encoder_q.')
}
torch.save(encoder_q_sd, str(out_pth))
print(f'Encoder state_dict saved:    {out_pth}')
print(f'  {len(encoder_q_sd)} tensors')

# Sanity check: round-trip load into the expected architecture
from torchvision.models import resnet18
from torch import nn
model = nn.Sequential(*list(resnet18().children())[:-1], nn.Flatten())
model.load_state_dict(torch.load(str(out_pth)))
print(f'Round-trip load into resnet18 backbone: OK ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)')

import os
print(f'\nFile sizes:')
print(f'  {out_ckpt.name}: {os.path.getsize(out_ckpt)/1024/1024:.1f} MB')
print(f'  {out_pth.name}: {os.path.getsize(out_pth)/1024/1024:.1f} MB')
