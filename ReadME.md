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

## How the artifacts were produced

The exact commands run on Brown CCV are preserved as launcher snapshots
in `scripts/launch/` (one per pretraining objective, plus a resume
script). Pretraining was driven by `src/train_caco.py` with
`--data_mode {seco,teco,caco}`; downstream linear probes were run with
`src/eval_eurosat_all.py` and `src/eval_ucmerced_all.py`; BigEarthNet
finetuning lives in `finetuning/bigearthnet/`. The 100k pipeline-
calibration row in Table 1 used the upstream checkpoints from
`research.cs.cornell.edu/caco/checkpoints/`, registered in
`src/utils/pretrained_checkpoints.py`.

## Notes on scope

- We pretrain at 10k for 400 epochs (vs. the paper's 100k for 1000 epochs)
  due to compute on Brown CCV. Absolute accuracy is therefore lower than
  Table 1 of the paper; the *relative* CACo ≥ TeCo > SeCo > Random ordering
  is what we replicate.
- BigEarthNet is reported only at 100k pretraining (using the authors'
  released checkpoints). 10k-pretrain BigEarthNet was scoped out.
- OSCD change detection (the task on which the paper claims its largest
  gains) is left as future work.
