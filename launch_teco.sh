#!/bin/bash
# TeCo full training launcher — mirrors the CACo/SeCo setup for fair comparison
module load anaconda3/2023.09-0-aqbc 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate /users/vsmokash/deep-learning/CACo/envs/caco

cd /users/vsmokash/deep-learning/CACo/src

echo "Starting TeCo pretraining at: $(date)"

python train_caco.py \
  --data_dir ../data/clean_10k_geography \
  --data_mode teco \
  --base_encoder resnet18 \
  --batch_size 256 \
  --num_negatives 4096 \
  --emb_dim 128 \
  --encoder_momentum 0.999 \
  --softmax_temperature 0.07 \
  --learning_rate 0.03 \
  --max_epochs 400 \
  --schedule 250 350 \
  --num_workers 2 \
  --gpus 1 \
  --save_dir ../checkpoints \
  --log_dir ../training_logs \
  -d "teco_10k_final"

echo "End time: $(date)"
