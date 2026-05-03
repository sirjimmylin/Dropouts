#!/bin/bash
# CACo pretraining on 10k dataset - NVIDIA L40S (46GB VRAM)
# Estimated runtime: ~5.5 hours (800 epochs × ~25s/epoch)

# Setup environment
module load anaconda3/2023.09-0-aqbc 2>/dev/null
eval "$(conda shell.bash hook)"
conda activate /users/vsmokash/deep-learning/CACo/envs/caco

cd /users/vsmokash/deep-learning/CACo/src

# Create output directories
mkdir -p ../checkpoints
mkdir -p ../training_logs

# Training configuration
DATA_DIR="../data/clean_10k_geography"
ENCODER="resnet18"
BATCH_SIZE=256
NUM_NEGATIVES=4096
EMB_DIM=128
LR=0.03
MOMENTUM=0.999
TEMPERATURE=0.07
MAX_EPOCHS=800
SCHEDULE="500 700"
NUM_WORKERS=2
DESCRIPTION="caco_10k_r18_bs256_q4096_ep800"

echo "============================================"
echo "CACo Pretraining - 10k Dataset"
echo "============================================"
echo "Encoder:        $ENCODER"
echo "Batch size:     $BATCH_SIZE"
echo "Queue size:     $NUM_NEGATIVES"
echo "Learning rate:  $LR"
echo "Epochs:         $MAX_EPOCHS"
echo "LR schedule:    $SCHEDULE"
echo "Workers:        $NUM_WORKERS"
echo "Description:    $DESCRIPTION"
echo "Start time:     $(date)"
echo "============================================"

python main_pretrain.py \
    --data_dir "$DATA_DIR" \
    --base_encoder "$ENCODER" \
    --batch_size $BATCH_SIZE \
    --num_negatives $NUM_NEGATIVES \
    --emb_dim $EMB_DIM \
    --encoder_momentum $MOMENTUM \
    --softmax_temperature $TEMPERATURE \
    --learning_rate $LR \
    --data_mode caco \
    --max_epochs $MAX_EPOCHS \
    --schedule $SCHEDULE \
    --num_workers $NUM_WORKERS \
    --gpus 1 \
    -d "$DESCRIPTION" \
    2>&1 | tee ../training_logs/training_$(date +%Y%m%d_%H%M%S).log

echo "============================================"
echo "Training completed at: $(date)"
echo "============================================"
