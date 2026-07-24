#!/bin/bash

# Centralized training baseline: single node with all training data (C=1).
# Paper protocol: R=30 rounds x E=4 local epochs, batch size 64, 5 random seeds.

source .venv/bin/activate

DATASET="${1:-aramis}"  # aramis | swat

ROUNDS=30
EPOCHS=4
BATCH_SIZE=64
NUM_CLIENTS=1
RESULTS_DIR="results/${DATASET}_centralized"
NUM_SEEDS=5

MODELS=("vae" "wgan_gp" "fedsw_tsad" "ddpm")

for MODEL in "${MODELS[@]}"; do
  for ((SEED=1; SEED<=NUM_SEEDS; SEED++)); do
    echo "Running model: $MODEL, dataset: $DATASET, seed: $SEED"
    python3 main_independent.py \
      --model_name "$MODEL" \
      --dataset_name "$DATASET" \
      --experiment_id "$SEED" \
      --seed "$SEED" \
      --num_clients "$NUM_CLIENTS" \
      --num_rounds "$ROUNDS" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --results_dir "$RESULTS_DIR"
  done
done
