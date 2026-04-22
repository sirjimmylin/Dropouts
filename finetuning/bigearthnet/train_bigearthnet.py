import sys
import os
sys.path.insert(0, os.path.expanduser('~/CACo/src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
from torchvision import transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from itertools import chain
from urllib import request

from utils.pretrained_checkpoints import saved_torch_ckpts
from datasets.bigearthnet_dataset import Bigearthnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

CKPT_DIR   = os.path.expanduser('~/CACo/checkpoints')
DATA_DIR   = os.path.expanduser('~/scratch/ben_fresh')
TRAIN_FRAC = 0.1
NUM_EPOCHS = 100
BATCH_SIZE = 256
LR         = 1e-5
MILESTONES = [60, 80]
FREEZE_BACKBONE = True

os.makedirs(CKPT_DIR, exist_ok=True)

CKPT_URL = 'https://research.cs.cornell.edu/caco/checkpoints/'

def download_checkpoint(ckpt_name):
    dest = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.isfile(dest):
        print(f'Downloading {ckpt_name} ...')
        request.urlretrieve(CKPT_URL + ckpt_name, dest)
        print('Done.')
    else:
        print(f'Already exists: {ckpt_name}')

for key in ['r18_100k_caco', 'r18_100k_seco']:
    download_checkpoint(saved_torch_ckpts[key])

# Authors did NOT use normalization — just resize and toTensor
transform = T.Compose([
    T.Resize((128, 128), interpolation=Image.BICUBIC),
    T.ToTensor()
])

print('Loading datasets...')
full_train  = Bigearthnet(root=DATA_DIR, split='train', transform=transform)
val_dataset = Bigearthnet(root=DATA_DIR, split='val',   transform=transform)

n_train = int(len(full_train) * TRAIN_FRAC)
np.random.seed(42)
train_indices = np.random.choice(len(full_train), n_train, replace=False)
train_dataset = Subset(full_train, train_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, drop_last=True)

print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)}')


class BigEarthNetClassifier(nn.Module):
    def __init__(self, backbone, in_features=512, num_classes=19,
                 hidden_dim=1024, dropout=0.2, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone
        # MLP head matching SSLEvaluator
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        with torch.set_grad_enabled(not self.freeze_backbone):
            feats = self.backbone(x)
        return self.head(feats)


def load_backbone(backbone_type, ckpt_key=None):
    model    = resnet18(pretrained=False)
    backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    if backbone_type == 'imagenet':
        model    = resnet18(pretrained=True)
        backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    elif backbone_type == 'pretrain':
        ckpt_path  = os.path.join(CKPT_DIR, saved_torch_ckpts[ckpt_key])
        state_dict = torch.load(ckpt_path, map_location='cpu')
        result     = backbone.load_state_dict(state_dict, strict=False)
        print(f'Loaded: {saved_torch_ckpts[ckpt_key]}')
        print(f'Missing: {result.missing_keys} | Unexpected: {result.unexpected_keys}')
    return backbone


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    model.backbone.eval()  # keep backbone in eval mode like authors
    total_loss, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total += len(y)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        preds = torch.sigmoid(model(x.to(device))).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.numpy())
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return average_precision_score(all_targets, all_preds, average='micro') * 100.0


def run_experiment(name, backbone_type, ckpt_key=None):
    print(f'\n========== {name} ==========')
    backbone  = load_backbone(backbone_type, ckpt_key).to(device)
    model     = BigEarthNetClassifier(backbone, freeze_backbone=FREEZE_BACKBONE).to(device)

    # Authors chain backbone + head params when not frozen
    if not FREEZE_BACKBONE:
        params = chain(model.backbone.parameters(), model.head.parameters())
    else:
        params = model.head.parameters()

    optimizer = optim.Adam(params, lr=LR)
    criterion = nn.MultiLabelSoftMarginLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    best_map = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        if epoch % 10 == 0:
            val_map = evaluate(model, val_loader)
            if val_map > best_map:
                best_map = val_map
            print(f'  Epoch {epoch:3d}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | Val mAP: {val_map:.2f}')

    print(f'  => Best Val mAP: {best_map:.2f}')
    return best_map


torch.manual_seed(42)
experiments = [
    ('Random Init', 'random',   None),
    ('ImageNet',    'imagenet', None),
    ('SeCo 100k',   'pretrain', 'r18_100k_seco'),
    ('CACo 100k',   'pretrain', 'r18_100k_caco'),
]

results = {}
for name, btype, ckpt_key in experiments:
    results[name] = run_experiment(name, btype, ckpt_key)

paper_10pct = {
    'Random Init': 42.87,
    'ImageNet':    65.43,
    'SeCo 100k':   65.80,
    'CACo 100k':   67.89,
}

print(f"\n{'Method':<20} {'Ours (mAP)':>12} {'Paper (mAP)':>12} {'Diff':>10}")
print('-' * 56)
for name in paper_10pct:
    ours  = results.get(name, 0)
    paper = paper_10pct[name]
    print(f'{name:<20} {ours:>12.2f} {paper:>12.2f} {ours-paper:>+10.2f}')

methods    = list(paper_10pct.keys())
our_vals   = [results.get(m, 0) for m in methods]
paper_vals = [paper_10pct[m]    for m in methods]
x     = np.arange(len(methods))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, our_vals,   width, label='Our Results',   color='steelblue')
bars2 = ax.bar(x + width/2, paper_vals, width, label='Paper Results', color='coral')
ax.set_ylabel('mAP')
ax.set_title('BigEarthNet: Our Results vs Paper (ResNet-18, 100k, 10% data)')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(30, 80)
ax.legend()
ax.bar_label(bars1, fmt='%.1f', padding=3)
ax.bar_label(bars2, fmt='%.1f', padding=3)
plt.tight_layout()
plt.savefig(os.path.expanduser('~/CACo/finetune/bigearthnet_results.png'), dpi=150)
print('Plot saved.')
