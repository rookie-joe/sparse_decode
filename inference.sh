#!/bin/bash
HOME_DIR="/home/lujianqiao/"

# Set parameters
BS_DS="AI-MO/aimo-validation-aime"
SAVE_DS='rookiemango/sparse_decode'
MODEL_DIR="/DATA/disk2/lujianqiao/models/QwQ-32B-Preview/"
SAVE_DS_BRANCH="model_QwQ"
N_SAMPLES=1000000
MAX_TOKENS=16384


# Run bootstrapping
cd ${HOME_DIR}/autoformalizer_v2


python3 -m inference \
    --model_path="$MODEL_DIR" \
    --bs_ds="$BS_DS" \
    --save_ds="$SAVE_DS" \
    --save_ds_branch="$SAVE_DS_BRANCH" \
    --n_samples="$N_SAMPLES" \
    --max_tokens="$MAX_TOKENS" \
