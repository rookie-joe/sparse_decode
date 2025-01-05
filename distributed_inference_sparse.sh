#!/bin/bash
# HOME_DIR="/home/lujianqiao/"

# Set parameters
BS_DS="AI-MO/aimo-validation-aime"
SAVE_DS='tworookieman/sparse_decode'
MODEL_DIR="/DATA/disk2/lujianqiao/models/QwQ-32B-Preview"
#MODEL_DIR="/DATA/disk2/lujianqiao/models/Qwen2.5-0.5B-Instruct"


#sparse decode
sparse_decode_method="quest"
sparse_decode_config="methods/quest/config/1_4_sparse.json"

SAVE_DS_BRANCH="model_QwQ_${sparse_decode_method}"
#SAVE_DS_BRANCH="model_test_${sparse_decode_method}"
N_SAMPLES=900
MAX_TOKENS=16384

# Set distributed communication port
MASTER_PORT=29512
MASTER_ADDR="localhost"
NCCL_DEBUG=INFO

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Launch with accelerate
accelerate launch \
    --multi_gpu \
    --main_process_port=$MASTER_PORT \
    --num_processes=4 \
    -m distributed_inference \
    --model_path="$MODEL_DIR" \
    --bs_ds="$BS_DS" \
    --save_ds="$SAVE_DS" \
    --save_ds_branch="$SAVE_DS_BRANCH" \
    --n_samples="$N_SAMPLES" \
    --max_tokens="$MAX_TOKENS" \
    --sparse_decode_method="$sparse_decode_method" \
    --sparse_decode_config="$sparse_decode_config" \
