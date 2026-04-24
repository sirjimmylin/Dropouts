# Dropouts

## Paper
[Change-Aware Sampling and Constrastive Learning for Satellite Images](https://openaccess.thecvf.com/content/CVPR2023/papers/Mall_Change-Aware_Sampling_and_Contrastive_Learning_for_Satellite_Images_CVPR_2023_paper.pdf)


## Reference GitHub
[Change-Aware Sampling and Contrastive Learning for Satellite Images](https://github.com/utkarshmall13/CACo)

## Part 2 (Check-in #2) implementation in this repo

This repo now includes:
- Vendored reference code at `caco/`
- EuroSAT downstream trainer: `caco/src/main_eurosat_part2.py`
- BigEarthNet downstream trainer: `caco/src/main_bigearthnet.py`
- Dataset preparation script: `scripts/prepare_datasets.py`
- Batch experiment runner: `scripts/run_checkin2_part2.sh`
- Result table generator: `scripts/generate_checkin2_tables.py`

### 1) Environment setup
```bash
bash scripts/setup_part2_env.sh
```

### 2) Dataset setup
Download EuroSAT + BigEarthNet metadata/splits:
```bash
python scripts/prepare_datasets.py \
  --download_eurosat \
  --download_bigearthnet_metadata
```

Optional: extract a **small BigEarthNet subset** (for smoke runs when full dataset is too large):
```bash
python scripts/prepare_datasets.py \
  --download_bigearthnet_metadata \
  --subset_train 512 \
  --subset_val 128 \
  --subset_test 128 \
  --activate_subset_splits
```

Optional: download the full BigEarthNet-S2 v1.0 archive (~66 GiB compressed):
```bash
python scripts/prepare_datasets.py --download_bigearthnet_archive
```

### 3) Run EuroSAT experiments
If your teammate checkpoint is not already in CACo format, normalize it once first:
```bash
python scripts/normalize_pretrain_checkpoint.py \
  --input_ckpt /path/to/teammate_pretrain.ckpt \
  --output_ckpt checkpoints/teammate_resnet18_caco_compatible.pt \
  --base_encoder resnet18
```

Linear probe:
```bash
python caco/src/main_eurosat_part2.py \
  --data_dir data/eurosat/EuroSAT_RGB \
  --backbone_type pretrain \
  --base_encoder resnet18 \
  --ckpt_path /path/to/caco_pretrain.ckpt \
  --train_mode linear
```

Fine-tune:
```bash
python caco/src/main_eurosat_part2.py \
  --data_dir data/eurosat/EuroSAT_RGB \
  --backbone_type pretrain \
  --base_encoder resnet18 \
  --ckpt_path /path/to/caco_pretrain.ckpt \
  --train_mode finetune
```

### 4) Run BigEarthNet experiments
Linear probe (10% split):
```bash
python caco/src/main_bigearthnet.py \
  --data_dir data/bigearthnet \
  --backbone_type pretrain \
  --base_encoder resnet18 \
  --ckpt_path /path/to/caco_pretrain.ckpt \
  --train_mode linear \
  --train_frac 0.1
```

Fine-tune (10% split):
```bash
python caco/src/main_bigearthnet.py \
  --data_dir data/bigearthnet \
  --backbone_type pretrain \
  --base_encoder resnet18 \
  --ckpt_path /path/to/caco_pretrain.ckpt \
  --train_mode finetune \
  --train_frac 0.1
```

### 5) Run all Check-in #2 Part 2 experiments
```bash
CKPT_PATH=/path/to/caco_pretrain.ckpt bash scripts/run_checkin2_part2.sh
```
When `INITS` includes `pretrain`, this script automatically writes a normalized checkpoint
under `results/.../normalized_checkpoints/` and uses it for both EuroSAT and BigEarthNet runs.

Optional environment variables for batch runs:
```bash
SEEDS="42 43 44" \
BEN_TRAIN_FRACS="0.1 1.0" \
CKPT_PATH=/path/to/caco_pretrain.ckpt \
bash scripts/run_checkin2_part2.sh
```

Run baselines now (no teammate checkpoint yet):
```bash
INITS="random imagenet" \
SEEDS="42 43 44" \
BEN_TRAIN_FRACS="0.1 1.0" \
bash scripts/run_checkin2_part2.sh
```

### 6) Generate/refresh comparison tables from existing runs
```bash
python scripts/generate_checkin2_tables.py --results_root results/checkin2_part2
```

Outputs are saved under `results/` as:
- `metrics.csv` (epoch-wise metrics)
- `best.pt` (best validation checkpoint)
- `summary.json` (final run summary)
- `tables/run_metrics.csv` (one row per run)
- `tables/aggregated_metrics.csv` (seed-aggregated metrics)
- `tables/paper_vs_ours.csv` (paper target comparisons)
- `tables/claim_checks.csv` (claim + hypothesis checks)
