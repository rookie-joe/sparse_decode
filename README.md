# Inference Script Usage Guide

This guide explains how to use the inference script for model evaluation on datasets.

## Prerequisites

- Python 3.x
- Required packages: `fire`, `datasets`
- Access to Hugging Face Hub
- Sufficient disk space for model and dataset storage

## Configuration

Edit the following parameters in `inference.sh` according to your needs:

```bash
BS_DS="AI-MO/aimo-validation-aime"          # Source dataset on HuggingFace Hub
MODEL_DIR="/path/to/your/model/"            # Local path to your model
SAVE_DS_BRANCH="your-model-name"            # Branch name for your model results
N_SAMPLES=1000000                           # Number of samples to process
MAX_TOKENS=16384                            # Maximum tokens for generation
```

### Important Notes:

1. **DO NOT change SAVE_DS**: Keep it as `rookiemango/sparse_decode` to maintain all results in one repository
2. **Branch Naming**: Use `SAVE_DS_BRANCH` to identify your specific model results (e.g., "model_QwQ", "model_XYZ")
3. **Dataset Access**: Ensure you have access rights to both source dataset and the shared results repository
4. **Model Path**: `MODEL_DIR` should point to your local model directory
5. **Storage**: Ensure sufficient storage for `N_SAMPLES` and model outputs

## Running the Script

1. Make the script executable:
   ```bash
   chmod +x inference.sh
   ```

2. Run the script:
   ```bash
   ./inference.sh
   ```

## Output

The script will:
1. Sample data from the source dataset
2. Generate model outputs
3. Calculate accuracy metrics
4. Push results to the shared repository under your specified branch

Results will be available on HuggingFace Hub at: `https://huggingface.co/datasets/rookiemango/sparse_decode/tree/{SAVE_DS_BRANCH}`

## Troubleshooting

- Ensure you're logged into HuggingFace Hub
- Check write permissions for the shared repository
- Verify model path exists and is accessible
- Monitor disk space during large sample runs