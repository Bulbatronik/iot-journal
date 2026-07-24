#!/bin/bash

# Federated training: C=5 clients with full, encoder/critic (enc), or
# decoder/generator (dec) parameter sharing.
# Paper protocol: R=30 rounds x E=4 local epochs, batch size 64, 5 random seeds.

source .venv/bin/activate

DATASET="${1:-aramis}"  # aramis | swat

ROUNDS=30
EPOCHS=4
BATCH_SIZE=64
NUM_CLIENTS=5
RESULTS_DIR="results/${DATASET}_federated"
NUM_SEEDS=5

MODELS=("vae" "wgan_gp" "fedsw_tsad" "ddpm")
FED_TYPES=("full" "enc" "dec")

for MODEL in "${MODELS[@]}"; do
  for FED_TYPE in "${FED_TYPES[@]}"; do
    for ((SEED=1; SEED<=NUM_SEEDS; SEED++)); do
      echo "Running model: $MODEL, federation: $FED_TYPE, dataset: $DATASET, seed: $SEED"
      python3 main_federated.py \
        --model_name "$MODEL" \
        --dataset_name "$DATASET" \
        --fed_type "$FED_TYPE" \
        --experiment_id "$SEED" \
        --seed "$SEED" \
        --num_clients "$NUM_CLIENTS" \
        --num_rounds "$ROUNDS" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --results_dir "$RESULTS_DIR"
    done
  done
done
