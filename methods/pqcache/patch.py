import json
import logging
from typing import Dict

from infer_utils.modeling_qwen2 import Qwen2FlashAttention2, Qwen2ForCausalLM
from methods.pqcache.attention import PQCacheFlashAttention

logger = logging.getLogger(__name__)


def load_pqcache_config(config_path: str) -> Dict:
    """
    Load PQCache configuration parameters from a JSON file.

    Args:
        config_path (str): Path to the JSON configuration file

    Returns:
        dict: Configuration parameters with defaults if not specified

    Raises:
        FileNotFoundError: If config file is not found
        JSONDecodeError: If config file contains invalid JSON
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Parse configuration parameters with defaults
        pq_params = {
            "pq_m": config.get("pq_m", 8),  # Number of subvectors
            "pq_k": config.get("pq_k", 256),  # Number of centroids
            "top_k": config.get("top_k", 64),  # Number of tokens to retrieve
            "local_window": config.get(
                "local_window", 128
            ),  # Size of recent context window
            "global_window": config.get(
                "global_window", 128
            ),  # Size of early context window
        }

        # Log the configuration being used
        logger.info(f"Loaded PQCache configuration: {pq_params}")
        return pq_params

    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using default values.")
        default_config = {
            "pq_m": 8,
            "pq_k": 256,
            "top_k": 64,
            "local_window": 128,
            "global_window": 128,
        }
        logger.info(f"Using default configuration: {default_config}")
        return default_config

    except json.JSONDecodeError:
        logger.warning(
            f"Invalid JSON in config file: {config_path}. Using default values."
        )
        default_config = {
            "pq_m": 8,
            "pq_k": 256,
            "top_k": 64,
            "local_window": 128,
            "global_window": 128,
        }
        logger.info(f"Using default configuration: {default_config}")
        return default_config

    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def patch_model(
    model: Qwen2ForCausalLM,
    max_new_tokens: int,
    sparse_decode_config: str,
    verbose: bool = False,
) -> None:
    """
    Apply PQCache patch to the model using configuration from a JSON file.

    Args:
        model (Qwen2ForCausalLM): The model to patch
        max_new_tokens (int): Maximum number of new tokens to generate
        sparse_decode_config (str): Path to the sparse decoding configuration file
        verbose (bool, optional): Enable verbose output. Defaults to False.
    """
    logger.info(f"Loading PQCache configuration from: {sparse_decode_config}")

    try:
        config = load_pqcache_config(config_path=sparse_decode_config)

        # Extract all configuration parameters with defaults
        pq_m = config.get("pq_m", 8)
        pq_k = config.get("pq_k", 256)
        top_k = config.get("top_k", 64)
        local_window = config.get("local_window", 128)
        global_window = config.get("global_window", 128)

        logger.info(
            f"Applying PQCache patch with parameters:\n"
            f"  max_new_tokens: {max_new_tokens}\n"
            f"  pq_m: {pq_m}\n"
            f"  pq_k: {pq_k}\n"
            f"  top_k: {top_k}\n"
            f"  local_window: {local_window}\n"
            f"  global_window: {global_window}\n"
            f"  verbose: {verbose}"
        )

        patched_modules = 0
        for name, module in model.named_modules():
            if isinstance(module, Qwen2FlashAttention2):
                # Apply the PQCacheFlashAttention module
                module.flash_attention = PQCacheFlashAttention(
                    max_new_tokens=max_new_tokens,
                    pq_m=pq_m,
                    pq_k=pq_k,
                    top_k=top_k,
                    local_window=local_window,
                    global_window=global_window,
                    verbose=verbose,
                )
                patched_modules += 1

        if patched_modules > 0:
            logger.info(
                f"Successfully patched {patched_modules} flash attention modules with PQCache"
            )
        else:
            logger.warning("No Qwen2FlashAttention2 modules found to patch")

    except Exception as e:
        logger.error(f"Error applying PQCache patch: {e}")
