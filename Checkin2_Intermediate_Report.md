# Check-in #2 Reflection / Intermediate Report (CACo Replication)

This file is the **Check-in #2 reflection** requested in `ProjectOverview.md` (the check-in reflection and intermediate report are the same deliverable for this checkpoint).

## Introduction
Our project is a critical replication of **Change-Aware Sampling and Contrastive Learning for Satellite Images (CACo, CVPR 2023)**.  
We implemented and ran downstream evaluation on **EuroSAT** and **BigEarthNet** using three initialization settings (`random`, `imagenet`, `pretrain`) and two training modes (`linear`, `finetune`) to test paper claims and hypothesis-driven ablations.

## Paper Method / Architecture (from the CACo paper)
The paper builds on a **MoCo v2 contrastive learning** framework for satellite imagery:
1. **Backbone encoders:** ResNet-18 and ResNet-50 (reported in Tables 1 and 2).
2. **Dual encoder setup:** query encoder + momentum-updated key encoder.
3. **Projection heads + InfoNCE objective:** contrastive learning with a memory queue of negatives (paper uses 16,384 negatives in the reported setup).
4. **Temporal sampling structure:** short-term pairs (within-set temporal variation) and long-term pairs (across-set variation, with a 4-year gap in the paper setup).
5. **Change-aware component (CACo):** estimate change using a ratio of long-term to short-term feature differences, smooth it over training, then cluster (GMM) to condition whether long-term pairs should be pulled together or pushed apart.
6. **Downstream evaluation:** linear probe and fine-tune on EuroSAT (top-1 acc) and BigEarthNet (micro-mAP).

## Hypothesis-Driven Ablations (written before experiments)
1. **H1 (low-label benefit):** CACo pretraining should provide a larger gain in low-label BigEarthNet (`train_frac=0.1`) than full-label (`train_frac=1.0`).
2. **H2 (fine-tune gap behavior):** Fine-tuning should narrow initialization gaps, while CACo remains strongest.

## Insights (Concrete Results)
Metrics below come from `results/checkin2_part2/tables/aggregated_metrics.csv` (seed 42).

| Dataset | Mode | Init | Setting | Best metric |
| --- | --- | --- | --- | --- |
| EuroSAT | Linear | Random | 1.0 | 41.185 acc |
| EuroSAT | Linear | ImageNet | 1.0 | 86.167 acc |
| EuroSAT | Linear | Pretrain | 1.0 | 83.074 acc |
| EuroSAT | Finetune | Random | 1.0 | 77.167 acc |
| EuroSAT | Finetune | ImageNet | 1.0 | 94.778 acc |
| EuroSAT | Finetune | Pretrain | 1.0 | 95.574 acc |
| BigEarthNet | Linear | Random | 0.1 | 12.617 mAP |
| BigEarthNet | Linear | ImageNet | 0.1 | 19.580 mAP |
| BigEarthNet | Linear | Pretrain | 0.1 | 18.289 mAP |
| BigEarthNet | Linear | Random | 1.0 | 26.080 mAP |
| BigEarthNet | Linear | ImageNet | 1.0 | 35.311 mAP |
| BigEarthNet | Linear | Pretrain | 1.0 | 18.346 mAP |

Claim/hypothesis checks from `results/checkin2_part2/tables/claim_checks.csv`:
1. **Claim 1 (EuroSAT linear: pretrain > baselines):** **Fail** (`83.074 < 86.167`).
2. **Claim 2 (BigEarthNet linear: pretrain > baselines):** **Fail** at both 0.1 and 1.0.
3. **H1:** **Pass** by relative criterion in the script (`gain@0.1=-1.291` vs `gain@1.0=-16.965`), but gains are still negative.
4. **H2:** **Fail** overall (pretrain not consistently strongest after fine-tuning, especially on BigEarthNet).

## Challenges
1. **BigEarthNet data alignment and completeness:** split files and locally available patch contents were inconsistent, causing missing-file crashes and requiring filtering to valid patch directories.
2. **Checkpoint/tooling compatibility:** modern PyTorch loading behavior and checkpoint format mismatches required adapting checkpoint loading and using a compatible `.pth` pretrained checkpoint.
3. **Compute/data scale gap vs paper:** current BigEarthNet runs use a very small local subset, so these numbers are preliminary and not yet directly comparable to full-scale paper results.

## Plan
1. Expand BigEarthNet data coverage (prefer full extraction) and rerun key experiments with multiple seeds.
2. Complete the required **new dataset / stress-test** component (dataset not used in paper), e.g., RESISC45.
3. Re-run the final matrix after data expansion and regenerate:
   - `run_metrics.csv`
   - `aggregated_metrics.csv`
   - `paper_vs_ours.csv`
   - `claim_checks.csv`
4. Add explicit “why numbers differ” analysis (data scale, protocol differences, compute budget, checkpoint source).

## Check-in #2 Requirement Coverage (from ProjectOverview.md)
1. **Introduction / Challenges / Insights / Plan sections present:** ✅
2. **Concrete quantitative results included:** ✅
3. **Implementation largely complete and experiments running:** ✅ (EuroSAT + BigEarthNet pipelines running, result tables generated)
4. **Hypothesis-driven ablations defined and tested:** ✅
5. **Two claim-verification checks included:** ✅
6. **Remaining gap:** required **new dataset stress-test** not yet finished (planned next).

## Public References Used
1. Paper: Mall et al., *Change-Aware Sampling and Contrastive Learning for Satellite Images* (CVPR 2023).
2. Reference implementation: https://github.com/utkarshmall13/CACo
