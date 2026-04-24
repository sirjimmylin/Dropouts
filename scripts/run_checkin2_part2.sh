#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

EUROSAT_DIR="${EUROSAT_DIR:-$ROOT_DIR/data/eurosat/EuroSAT_RGB}"
BIGEARTH_DIR="${BIGEARTH_DIR:-$ROOT_DIR/data/bigearthnet}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/checkin2_part2}"
BASE_ENCODER="${BASE_ENCODER:-resnet18}"
SEEDS="${SEEDS:-42}"
INITS="${INITS:-random imagenet pretrain}"
BEN_TRAIN_FRACS="${BEN_TRAIN_FRACS:-0.1}"
BEN_VAL_FRAC="${BEN_VAL_FRAC:-1.0}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CKPT_PATH="${CKPT_PATH:-}"
NORMALIZED_CKPT_PATH=""

if [[ " $INITS " == *" pretrain "* ]]; then
  if [[ -z "$CKPT_PATH" ]]; then
    echo "INITS includes 'pretrain', but CKPT_PATH is empty."
    echo "Set CKPT_PATH to your CACo pretrain checkpoint (.ckpt/.pt),"
    echo "or run without pretrain (e.g., INITS=\"random imagenet\")."
    exit 1
  fi
  if [[ ! -f "$CKPT_PATH" ]]; then
    echo "CKPT_PATH does not exist: $CKPT_PATH"
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"

if [[ " $INITS " == *" pretrain "* ]]; then
  CKPT_FILE="$(basename "$CKPT_PATH")"
  CKPT_STEM="${CKPT_FILE%.*}"
  NORMALIZED_CKPT_PATH="$OUTPUT_DIR/normalized_checkpoints/${CKPT_STEM}_${BASE_ENCODER}_caco_compatible.pt"
  echo "Normalizing pretrain checkpoint for $BASE_ENCODER -> $NORMALIZED_CKPT_PATH"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/normalize_pretrain_checkpoint.py" \
    --input_ckpt "$CKPT_PATH" \
    --output_ckpt "$NORMALIZED_CKPT_PATH" \
    --base_encoder "$BASE_ENCODER" \
    --overwrite
fi

echo "=== EuroSAT: linear + finetune, inits [$INITS] ==="
for CUR_SEED in $SEEDS; do
  for TRAIN_MODE in linear finetune; do
    for BACKBONE_TYPE in $INITS; do
      RUN_NAME="eurosat_${BASE_ENCODER}_${BACKBONE_TYPE}_${TRAIN_MODE}_seed${CUR_SEED}"
      CMD=(
        "$PYTHON_BIN" "$ROOT_DIR/caco/src/main_eurosat_part2.py"
        --data_dir "$EUROSAT_DIR"
        --base_encoder "$BASE_ENCODER"
        --backbone_type "$BACKBONE_TYPE"
        --train_mode "$TRAIN_MODE"
        --max_epochs "$MAX_EPOCHS"
        --num_workers "$NUM_WORKERS"
        --seed "$CUR_SEED"
        --run_name "$RUN_NAME"
        --output_dir "$OUTPUT_DIR/eurosat"
      )
      if [[ "$BACKBONE_TYPE" == "pretrain" ]]; then
        CMD+=(--ckpt_path "$NORMALIZED_CKPT_PATH")
      fi
      "${CMD[@]}"
    done
  done
done

echo "=== BigEarthNet: train fractions [$BEN_TRAIN_FRACS], linear + finetune, inits [$INITS] ==="
for TRAIN_FRAC in $BEN_TRAIN_FRACS; do
  FRAC_TAG="${TRAIN_FRAC/./p}"
  for CUR_SEED in $SEEDS; do
    for TRAIN_MODE in linear finetune; do
      for BACKBONE_TYPE in $INITS; do
        RUN_NAME="bigearthnet_${BASE_ENCODER}_${BACKBONE_TYPE}_${TRAIN_MODE}_trainfrac${FRAC_TAG}_seed${CUR_SEED}"
        CMD=(
          "$PYTHON_BIN" "$ROOT_DIR/caco/src/main_bigearthnet.py"
          --data_dir "$BIGEARTH_DIR"
          --base_encoder "$BASE_ENCODER"
          --backbone_type "$BACKBONE_TYPE"
          --train_mode "$TRAIN_MODE"
          --train_frac "$TRAIN_FRAC"
          --val_frac "$BEN_VAL_FRAC"
          --max_epochs "$MAX_EPOCHS"
          --num_workers "$NUM_WORKERS"
          --seed "$CUR_SEED"
          --run_name "$RUN_NAME"
          --output_dir "$OUTPUT_DIR/bigearthnet"
        )
        if [[ "$BACKBONE_TYPE" == "pretrain" ]]; then
          CMD+=(--ckpt_path "$NORMALIZED_CKPT_PATH")
        fi
        "${CMD[@]}"
      done
    done
  done
done

"$PYTHON_BIN" "$ROOT_DIR/scripts/generate_checkin2_tables.py" --results_root "$OUTPUT_DIR"

echo "All Check-in #2 Part 2 runs completed. Tables are in $OUTPUT_DIR/tables."
