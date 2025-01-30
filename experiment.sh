#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

split_scheme=time_after_2016 # time_before_2004 time_mid
pretrained_model_path=sweep/fmow/best_model.pth
sweep_dir=sweep/fmow

# Learn Distribution Graph
python examples/learn_graph.py \
    --split_scheme $split_scheme \
    --pretrained_model_path $pretrained_model_path

# Save the model predictions
python examples/prune.py \
    --sweep_dir $sweep_dir \
    --split_scheme $split_scheme

# Ensemble Pruning
for algo_z in ERM uniform_prior laplacian DRO SRM; do
    python examples/prune.py \
        --features \
        --sweep_dir $sweep_dir \
        --split_scheme $split_scheme \
        --algo_z $algo_z
done