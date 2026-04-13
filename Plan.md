# CSCI 1470 Final Project Plan (Option 1: Critical Replication)
 
 ## Problem + Approach
 Replicate **Change-Aware Sampling and Contrastive Learning for 
Satellite Images (CACo, CVPR 2023)**, then extend it with 
rigorous evaluation and stress tests that satisfy CSCI 1470 
Option 1 requirements.
 
 Your split is:
 - **Part 1 (groupmates):** pre-train CACo on 10k and save 
checkpoint.
 - **Part 2 (you):** use the reference implementation to run 
downstream training/fine-tuning on **EuroSAT** and 
**BigEarthNet**, compare against paper claims, and package 
results for Check-in #2 and final report.
 
 ---
 
 ## Non-negotiable course requirements to bake into the plan
 From `ProjectOverview.md` (Option 1 + Check-in #2):
 1. Evaluate on **dataset(s) not used in the paper**.
 2. Include a **non-trivial extension/stress test**.
 3. Verify at least **two quantitative claims** from the paper.
 4. Include at least **two hypothesis-driven ablations** (written
 before experiments; discuss at check-in #2).
 5. Check-in #2 reflection must include: **Introduction, 
Challenges, Insights (with concrete results), Plan**.
 6. By check-in #2 you should be near implementation completion 
and actively running experiments.
 
 ---
 
 ## Ground-truth targets from paper/supplement (for comparison tables)
 
 ### EuroSAT (linear probe, top-1 accuracy; main paper Table 1)
 - CACo 100k: **93.08 (R18)**, **94.48 (R50)**
 - CACo 1m: **94.72 (R18)**, **95.90 (R50)**
 
 ### BigEarthNet (linear probe, mAP; main paper Table 2)
 - CACo 100k: **67.89 / 69.43 (R18, 10% / 100%)**
 - CACo 100k: **71.55 / 73.63 (R50, 10% / 100%)**
 - CACo 1m: **68.64 / 70.41 (R18, 10% / 100%)**
 - CACo 1m: **73.40 / 74.98 (R50, 10% / 100%)**
 
 ### EuroSAT (fine-tuning, top-1; supplemental Table 2)
 - CACo 100k: **97.02 (R18)**, **97.17 (R50)**
 - CACo 1m: **97.47 (R18)**, **97.77 (R50)**
 
 ### Land-cover training details (supplement)
 - Loss: cross-entropy (EuroSAT), multi-label soft margin 
(BigEarthNet)
 - Optimizer: Adam
 - LR: **1e-3 linear probe**, **1e-5 fine-tuning**
 - Epochs: 100, LR drops at epochs 60 and 80
 - Batch size: **32 (EuroSAT)**, **1024 (BigEarthNet)**
 
 ---
 
 ## Reference repo reality check (important)
 - `src/main_eurosat.py` exists for EuroSAT linear evaluation.
 - BigEarthNet dataloaders/models exist 
(`bigearthnet_datamodule.py`, `ssl_finetuner.py`) but there is 
**no ready-made `main_bigearthnet.py` script** in the reference 
repo.
 - So Part 2 should include adding/adapting a BigEarthNet 
training entrypoint and consistent logging/evaluation.
 
 ---
 
 ## Full project execution plan
 
 ## Phase A — Reproducibility foundation (shared)
 1. Lock environment close to reference versions 
(Python/Torch/PyTorch Lightning).
 2. Define fixed run naming + output locations for:
    - checkpoints
    - logs (TensorBoard/CSV)
    - result tables
 3. Fix random seeds and document hardware (for variance/context 
in report).
 4. Define one canonical metrics table format for all runs.
 
 ## Phase B — Part 1 pretraining handoff (groupmates)
 1. Train CACo 10k and export checkpoint path + config.
 2. Handoff contract to Part 2:
    - checkpoint file path
    - backbone type (R18/R50)
    - data_mode + key hparams
    - training log summary
 
 ## Phase C — Part 2 downstream evaluation (your focus for Check-in #2)
 
 ### C1. EuroSAT
 1. Run **linear probe** with:
    - random init
    - ImageNet init
    - CACo-10k pretrained checkpoint
 2. Run **fine-tuning** version (unfreeze backbone) with same 
three initializations.
 3. Record:
    - final val accuracy
    - best epoch accuracy
    - seed-wise mean/std (if possible)
 
 ### C2. BigEarthNet
 1. Implement/adapt `main_bigearthnet.py` using existing:
    - `BigearthnetDataModule`
    - `SSLFineTuner`
 2. Support both **10%** and **100%** training fractions.
 3. Run linear probe + fine-tune for:
    - random
    - ImageNet
    - CACo-10k
 4. Record micro mAP and ensure metric matches paper convention.
 
 ### C3. Comparison packaging
 1. Build a “Paper vs Ours” comparison table for EuroSAT and 
BigEarthNet.
 2. Add a “why numbers differ” notes column:
    - 10k pretrain vs paper’s 100k/1m
    - compute budget
    - implementation/version differences
 
 ## Phase D — Option 1 required science components
 
 ### D1. Claim verification (at least two quantitative claims)
 Recommended claims to verify:
 1. **CACo > SeCo/MoCo/ImageNet** on EuroSAT linear probing.
 2. **CACo > baselines** on BigEarthNet mAP for both 10% and 100%
 label settings.
 
 ### D2. Hypothesis-driven ablations (at least two)
 Write these before running experiments:
 1. **H1:** CACo pretraining gives larger gain in low-label 
regime (BigEarthNet 10%) than full-label regime.
 2. **H2:** Fine-tuning narrows absolute gaps, but CACo remains 
the strongest initialization.
 
 ### D3. Stress test + new dataset requirement
 To satisfy course rubric, add one dataset **not used in paper** 
(main or supplement).  
 Recommended: **RESISC45** (or UC Merced / So2Sat if easier with 
your pipeline).
 
 Stress-test idea:
 - Domain shift test (different sensor/resolution/style) with 
linear probe and/or fine-tune.
 - Report whether CACo gains persist or collapse under shift.
 
 ## Phase E — Check-in #2 package
 Produce a 1-page reflection with:
 1. **Introduction:** project framing from proposal.
 2. **Challenges:** concrete blockers (data prep, metric parity, 
missing BigEarthNet script).
 3. **Insights:** preliminary EuroSAT/BigEarthNet numbers and 
early claim checks.
 4. **Plan:** what remains, risk mitigation, and what you are 
changing.
 
 Include in repo:
 - current results table(s)
 - scripts used so far
 - clear “next experiments” checklist
 
 ## Phase F — Final report + poster readiness
 1. Add “What the Paper Doesn’t Tell You” section:
    - matched claims
    - failed claims
    - best explanations for divergence
 2. Include qualitative + quantitative results, limitations, 
ethics, reflection.
 3. Ensure all code for reproduction is in GitHub with run 
commands.
 
 ---
 
 ## Adversarial checks (quality gate before reporting numbers)
 Use this checklist each time you publish results:
 1. **Split leakage check:** no train/val/test contamination.
 2. **Metric parity check:** EuroSAT uses top-1; BigEarthNet uses
 micro mAP.
 3. **Protocol parity check:** linear probe truly freezes 
encoder; fine-tune unfreezes.
 4. **Label mapping check:** BigEarthNet 19-class grouped-label 
mapping is consistent.
 5. **Preprocess parity check:** image resize/bands exactly as 
intended (RGB only if matching paper setup).
 6. **Checkpoint identity check:** correct backbone/checkpoint 
loaded (no silent mismatch).
 7. **Seed variance check:** at least 2–3 seeds for key claims 
when feasible.
 8. **Compute-fairness note:** clearly mark where setup differs 
from paper (10k vs 100k/1m).
 9. **Failure-accounting check:** include unsuccessful runs and 
plausible causes, not only best run.
 
 ---
 
 ## Immediate plan for “Project Check-in #2 Part 2” (next 2implementation sprint)
 1. Finalize EuroSAT linear + fine-tune pipeline from checkpoint.
 2. Implement BigEarthNet training entrypoint and run at least:
    - 10% subset, linear probe (all 3 initializations)
    - 10% subset, fine-tune (all 3 initializations)
 3. Generate first comparison table against paper.
 4. Draft check-in reflection with concrete preliminary numbers +
 blockers + next steps.