import json
import logging

from infer_utils.modeling_qwen2 import Qwen2FlashAttention2, Qwen2ForCausalLM
from methods.quest.attention import QuestFlashAttention

logger = logging.getLogger(__name__)


def load_quest_config(config_path: str) -> dict:
    """
    Load QUEST configuration parameters from a JSON file.

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

        # Get values with defaults
        quest_params = {
            "page_size": config.get("page_size", 64),
            "page_topk": config.get("page_topk", 4),
        }

        # Log the configuration being used
        logger.info(f"Loaded QUEST configuration: {quest_params}")
        return quest_params

    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using default values.")
        default_config = {"page_size": 64, "page_topk": 4}
        logger.info(f"Using default configuration: {default_config}")
        return default_config

    except json.JSONDecodeError:
        logger.warning(
            f"Invalid JSON in config file: {config_path}. Using default values."
        )
        default_config = {"page_size": 64, "page_topk": 4}
        logger.info(f"Using default configuration: {default_config}")
        return default_config


def patch_model(
    model: Qwen2ForCausalLM,
    max_new_tokens: int,
    sparse_decode_config: str,
    verbose: bool = False,
) -> None:
    """
    Apply QUEST patch to the model using configuration from a JSON file.

    Args:
        model (Qwen2ForCausalLM): The model to patch
        max_new_tokens (int): Maximum number of new tokens to generate
        sparse_decode_config (str): Path to the sparse decoding configuration file
        verbose (bool, optional): Enable verbose output. Defaults to False.
    """
    logger.info(f"Loading QUEST configuration from: {sparse_decode_config}")
    quest_config = load_quest_config(config_path=sparse_decode_config)

    page_size = quest_config["page_size"]
    page_topk = quest_config["page_topk"]

    logger.info(
        f"Applying QUEST patch with parameters: "
        f"max_new_tokens={max_new_tokens}, "
        f"page_size={page_size}, page_topk={page_topk}"
    )

    patched_modules = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen2FlashAttention2):
            module.flash_attention = QuestFlashAttention(
                max_new_tokens, page_size, page_topk, verbose
            )
            patched_modules += 1

    if patched_modules > 0:
        logger.info(
            f"Successfully patched {patched_modules} flash attention modules with QUEST"
        )
    else:
        logger.warning("No Qwen2FlashAttention2 modules found to patch")


# Example usage:
"""
# quest_config.json
{
    "page_size": 64,
    "page_topk": 4
}

# Application code
model = Qwen2ForCausalLM.from_pretrained(...)
patch_model(
    model=model,
    max_new_tokens=100,
    sparse_decode_config='path/to/quest_config.json',
    verbose=True
)
"""
