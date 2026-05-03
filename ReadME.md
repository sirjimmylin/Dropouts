# Dropouts — Critical Replication of CACo

CSCI 1470 (Deep Learning) final project, Brown University. We replicate
**Change-Aware Sampling and Contrastive Learning for Satellite Images**
(Mall, Hariharan, Bala, CVPR 2023) and run a hypothesis-driven ablation
that isolates the contribution of long-term temporal contrast vs. the
change-awareness mechanism.

**Team:** Devraj Raghuvanshi, Jimmy Lin, Paneri Patel, Vinayak Mokashi.

- Paper (original): [`paper.pdf`](paper.pdf)
- Poster: [`poster/poster.jpg`](poster/poster.jpg)
- Pretrained checkpoints (CACo / SeCo / TeCo, ResNet-18, 10k, 400 ep): see [`WEIGHTS.md`](WEIGHTS.md)

## Headline result

At 10k pretraining scale, long-term temporal contrast (TeCo) alone
captures the majority of CACo's gain over SeCo on EuroSAT linear probe;
the change-awareness mechanism on top adds only +0.6 points and is
*counterproductive* on UC Merced (a distribution-shift dataset the
paper does not consider).

| Method | EuroSAT (10k pretrain) | UC Merced (10k pretrain) |
|---|---|---|
| Random Init | 53.61 | 29.51 |
| SeCo        | 67.64 | 55.19 |
| TeCo        | 72.38 | **67.62** |
| CACo        | **73.02** | 65.23 |

Pipeline calibration against the paper's published 100k checkpoints
reproduces EuroSAT to within ~1 point (CACo 93.85 vs. 93.08 reported).
Full numbers and per-epoch histories are in [`eval_results/`](eval_results/).

## Repository layout

```
.
├── paper.pdf                 # original CACo paper
├── poster/poster.jpg         # final poster (4:3 horizontal JPG)
├── ProjectOverview.md        # course-provided project brief
├── WEIGHTS.md                # Drive links for the three pretrained checkpoints
├── requirements.txt          # pretraining environment
├── requirements-notebook-local.txt   # downstream-eval environment
├── install.sh                # creates the conda env
│
├── src/                      # pretraining + evaluation code
│   ├── train_caco.py         # SeCo / TeCo / CACo (gated by --data_mode)
│   ├── main_pretrain.py
│   ├── eval_eurosat_all.py   # produces Table 1 EuroSAT rows
│   ├── eval_ucmerced_all.py  # produces Table 1 UC Merced rows
│   ├── finalize_model.py     # Lightning-ckpt → plain ResNet-18 .pth
│   ├── save_final_format.py
│   ├── datasets/             # EuroSAT, UC Merced, BigEarthNet, SeCo
│   ├── models/               # MoCo v2 module, SSL finetuner, segmentation
│   └── utils/                # transforms, histogram callback, etc.
│
├── scripts/
│   ├── launch/               # SLURM launchers for the three pretrains + resume
│   ├── prepare_datasets.py   # data download / extraction helpers
│   ├── normalize_pretrain_checkpoint.py
│   ├── generate_checkin2_tables.py
│   ├── setup_local_notebook_env.sh
│   ├── setup_part2_env.sh
│   └── run_checkin2_part2.sh
│
├── finetuning/               # downstream finetune (vs. linear-probe)
│   ├── eurosat/              # eurosat_finetune.{py,ipynb}
│   └── bigearthnet/          # train_bigearthnet.py + slurm out
│
├── notebooks/                # exploratory / colab linear-probe notebooks
├── caco/                     # unmodified upstream CACo reference (read-only mirror)
│
├── eval_results/             # numerical artifacts behind the report
│   ├── eurosat/{metrics_summary.json, linear_probe/*_history.json}
│   └── ucmerced/{metrics_summary.json, linear_probe/*_history.json}
├── training_logs/            # per-epoch metrics.csv for each pretrain
└── media/histogram.png       # representative GMM change-ratio histogram
```

## Reproducing the headline numbers

The launchers in `scripts/launch/` hard-code paths to the working clone
on Brown CCV (`/users/vsmokash/deep-learning/CACo/...`). They are
preserved as snapshots of the exact commands that produced the
checkpoints. To re-run elsewhere:

```bash
# 1. Set up the environment
bash install.sh
conda activate caco

# 2. Download data (Sentinel-2 10k clean_geography from the paper authors,
#    plus EuroSAT_RGB.zip and UCMerced_LandUse.zip — see scripts/prepare_datasets.py)
python scripts/prepare_datasets.py

# 3. Pretrain (one of: caco | seco | teco)
python src/train_caco.py \
  --data_dir data/clean_10k_geography \
  --data_mode caco \
  --base_encoder resnet18 \
  --batch_size 256 --num_negatives 4096 --emb_dim 128 \
  --learning_rate 0.03 --max_epochs 400 --schedule 250 350

# 4. Convert Lightning checkpoint → plain encoder .pth
python src/finalize_model.py --ckpt checkpoints/.../caco_final.ckpt

# 5. Linear probe (produces the Table 1 numbers + JSONs in eval_results/)
python src/eval_eurosat_all.py
python src/eval_ucmerced_all.py
```

For the 100k pipeline-calibration numbers, point the eval scripts at
the upstream CACo / SeCo checkpoints from
`research.cs.cornell.edu/caco/checkpoints/`.

## Notes on scope

- We pretrain at 10k for 400 epochs (vs. the paper's 100k for 1000 epochs)
  due to compute on Brown CCV. Absolute accuracy is therefore lower than
  Table 1 of the paper; the *relative* CACo ≥ TeCo > SeCo > Random ordering
  is what we replicate.
- BigEarthNet is reported only at 100k pretraining (using the authors'
  released checkpoints). 10k-pretrain BigEarthNet was scoped out.
- OSCD change detection (the task on which the paper claims its largest
  gains) is left as future work.
