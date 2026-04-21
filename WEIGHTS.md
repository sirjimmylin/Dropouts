# Pretrained Weights

Weights for the three self-supervised pretraining variants (CACo, SeCo, TeCo) live on Google Drive.

**📁 [Drive folder — all three `.pth` files](https://drive.google.com/drive/folders/1--bZxzBoeUYKOH7TfrxEeI06rUeX1Qj5?usp=sharing)**

## Files

| File | Method | Size | Description |
|------|--------|------|-------------|
| `resnet18_caco_geo_10k_400.pth` | CACo | 43 MB | Change-Aware Contrastive — samples positive pairs from temporally nearby views with large scene change |
| `resnet18_seco_geo_10k_400.pth` | SeCo | 43 MB | Seasonal Contrastive — positives drawn across seasons of the same location |
| `resnet18_teco_geo_10k_400.pth` | TeCo | 43 MB | Temporal Contrastive — positives drawn from the same location at different times |

All three share identical architecture (ResNet-18 backbone) and training config; the only difference is the positive-pair sampling strategy, so the three files are directly comparable for downstream evaluation.

## Loading the weights

Each `.pth` is a plain `state_dict` for the encoder (MoCo `encoder_q`, projection head stripped) and loads directly into the standard ResNet-18 feature extractor:

```python
import torch
from torch import nn
from torchvision.models import resnet18

def load_pretrained_backbone(pth_path: str) -> nn.Module:
    backbone = nn.Sequential(
        *list(resnet18().children())[:-1],  # drop final FC
        nn.Flatten(),
    )
    backbone.load_state_dict(torch.load(pth_path, map_location="cpu"))
    return backbone  # outputs 512-dim features

# e.g.
backbone = load_pretrained_backbone("resnet18_caco_geo_10k_400.pth")
backbone.eval()
feats = backbone(torch.randn(4, 3, 224, 224))  # -> (4, 512)
```

Attach any classifier/segmentation head on top of the 512-d features for fine-tuning.

## Training configuration

All three variants were pretrained with identical hyperparameters (see `launch_training.sh` / `launch_seco.sh` / `launch_teco.sh`):

| Setting | Value |
|---|---|
| Backbone | ResNet-18 |
| Dataset | `clean_10k_geography` (10k Sentinel-2 locations × multiple timestamps; ~1.8 GB unpacked) |
| Batch size | 256 |
| Embedding dim | 128 |
| MoCo queue size | 4096 |
| Encoder momentum | 0.999 |
| Softmax temperature | 0.07 |
| Learning rate | 0.03 (cosine schedule, drops at epochs 250, 350) |
| Epochs | 400 |
| Hardware | 1× GPU (Oscar) |

## Reproducing from scratch

```bash
# 1. Env (see CACo_upstream_README.md for full details)
bash install.sh

# 2. Data: caco10k.tar.gz (1.4 GB) → data/clean_10k_geography/
# Download script inside src/sample_training.sh, or:
#   wget https://research.cs.cornell.edu/caco/data/caco/caco10k.tar.gz
#   tar -xf caco10k.tar.gz -C data/

# 3. Launch (one of)
bash launch_training.sh   # CACo
bash launch_seco.sh       # SeCo
bash launch_teco.sh       # TeCo
```

Outputs land under `checkpoints/resnet18-clean_10k_geography-<method>-ep400-bs256-q4096/`. The `src/save_final_format.py` script converts the final Lightning checkpoint to the `resnet18_<method>_geo_10k_400.pth` format above.
