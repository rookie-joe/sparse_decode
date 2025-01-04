#!/bin/bash
# HOME_DIR="/home/lujianqiao/"

# Set parameters
BS_DS="AI-MO/aimo-validation-aime"
SAVE_DS='tworookieman/sparse_decode'
MODEL_DIR="/DATA/disk2/lujianqiao/models/QwQ-32B-Preview"
#MODEL_DIR="/DATA/disk2/lujianqiao/models/Qwen2.5-0.5B-Instruct"
SAVE_DS_BRANCH="model_QwQ"
#SAVE_DS_BRANCH="model_test"
N_SAMPLES=90
MAX_TOKENS=16384

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch with accelerate
accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    -m distributed_inference \
    --model_path="$MODEL_DIR" \
    --bs_ds="$BS_DS" \
    --save_ds="$SAVE_DS" \
    --save_ds_branch="$SAVE_DS_BRANCH" \
    --n_samples="$N_SAMPLES" \
    --max_tokens="$MAX_TOKENS" \