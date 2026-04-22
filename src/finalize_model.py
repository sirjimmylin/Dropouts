"""Extract final model from the latest periodic checkpoint."""
import sys
import torch
from pathlib import Path

# Import MetricLogger and FinalModelSaver so torch.load can find them
sys.path.insert(0, str(Path(__file__).parent))
from train_caco import MetricLogger, FinalModelSaver

CKPT_DIR = Path('/users/vsmokash/deep-learning/CACo/checkpoints/resnet18-clean_10k_geography-caco-ep400-bs256-q4096')
SRC_CKPT = CKPT_DIR / 'caco-epoch=0299.ckpt'
FINAL_CKPT = CKPT_DIR / 'caco_final.ckpt'
ENCODER_PATH = CKPT_DIR / 'caco_encoder_final.pth'

print(f'Loading {SRC_CKPT}...')
ckpt = torch.load(str(SRC_CKPT), map_location='cpu')
print(f'  Epoch: {ckpt.get("epoch")}')
print(f'  Global step: {ckpt.get("global_step")}')

state_dict = ckpt['state_dict']
print(f'  Total params: {len(state_dict)}')

# Split state dict by component
encoder_q = {k.replace('encoder_q.', ''): v for k, v in state_dict.items() if k.startswith('encoder_q.')}
encoder_k = {k.replace('encoder_k.', ''): v for k, v in state_dict.items() if k.startswith('encoder_k.')}
heads_q = {k.replace('heads_q.', ''): v for k, v in state_dict.items() if k.startswith('heads_q.')}
heads_k = {k.replace('heads_k.', ''): v for k, v in state_dict.items() if k.startswith('heads_k.')}

print(f'  encoder_q: {len(encoder_q)} tensors')
print(f'  encoder_k: {len(encoder_k)} tensors')
print(f'  heads_q: {len(heads_q)} tensors')
print(f'  heads_k: {len(heads_k)} tensors')

# Save full checkpoint as final
torch.save(ckpt, str(FINAL_CKPT))
print(f'\nFinal checkpoint saved: {FINAL_CKPT}')

# Save encoder weights for downstream tasks
torch.save({
    'encoder_q': encoder_q,
    'encoder_k': encoder_k,
    'heads_q': heads_q,
    'heads_k': heads_k,
    'hparams': ckpt.get('hyper_parameters', {}),
    'epoch': ckpt.get('epoch'),
    'note': 'Extracted from epoch 299 periodic checkpoint. Training was killed at epoch ~305 of 400 by SLURM time limit.',
}, str(ENCODER_PATH))
print(f'Encoder weights saved: {ENCODER_PATH}')

import os
print(f'\nFinal checkpoint size: {os.path.getsize(FINAL_CKPT)/1024/1024:.1f} MB')
print(f'Encoder weights size:  {os.path.getsize(ENCODER_PATH)/1024/1024:.1f} MB')
