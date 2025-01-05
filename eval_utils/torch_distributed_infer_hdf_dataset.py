import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from data_utils.prompts import system_prompt
from eval_utils.metric import calculate_accuracy, extract_boxed_answer
from infer_utils.infer import batch_infer
from infer_utils.modeling_qwen2 import Qwen2ForCausalLM

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def apply_sparse_decode_method(
    llm: PreTrainedModel,
    sparse_decode_method: str,
    sparse_decode_config: str,
    max_tokens: int,
) -> Optional[PreTrainedModel]:
    """
    Apply the specified sparse decoding method to the model.

    Args:
        llm: The language model (PreTrainedModel)
        sparse_decode_method: String specifying the decoding method ('quest' or 'vanilla')
        max_tokens: Maximum number of tokens for generation

    Returns:
        Modified model with applied sparse decoding method, or None if error occurs

    Raises:
        AssertionError: If specified decode method is not implemented
    """
    logger.info(f"Applying sparse decode method: {sparse_decode_method}")

    if sparse_decode_method == "quest":
        try:
            logger.info("Importing QUEST patch module...")
            from methods.quest.patch import patch_model

            logger.info(
                f"Applying QUEST patch with max_tokens={max_tokens}, "
                f"page_size=64, page_topk=4"
            )
            patch_model(
                llm,
                max_new_tokens=max_tokens,
                sparse_decode_config=sparse_decode_config,
                verbose=False,
            )
            logger.info("QUEST patch applied successfully")

        except ImportError as e:
            logger.error(f"Failed to import QUEST patch module: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error occurred while applying QUEST patch: {str(e)}")
            return None

    elif sparse_decode_method == "pqcache":
        try:
            logger.info("Importing PQCache patch module...")
            from methods.pqcache.patch import patch_model

            patch_model(
                llm,
                max_new_tokens=max_tokens,
                sparse_decode_config=sparse_decode_config,
                verbose=False,
            )
            logger.info("PQCache patch applied successfully")
            return llm

        except ImportError as e:
            logger.error(f"Failed to import PQCache patch module: {str(e)}")
            return None

        except Exception as e:
            logger.error(f"Error occurred while applying PQCache patch: {str(e)}")
            return None

    elif sparse_decode_method == "vanilla":
        logger.info("Using vanilla decoding - no modifications applied")
        pass
    else:
        error_msg = f"Sparse decode method '{sparse_decode_method}' not implemented"
        logger.error(error_msg)
        raise AssertionError(error_msg)

    logger.info("Sparse decode method application completed")
    return llm


def check_all_processes_complete(base_dir: Path, num_processes: int) -> bool:
    """
    Check if all GPU processes have completed their work by counting done files
    Args:
        base_dir: Directory containing result files
        num_processes: Total number of GPU processes
    Returns:
        bool: True if all processes have completed
    """
    done_files = list(base_dir.glob("done_*.json"))
    # Also check the corresponding partial results exist
    partial_dirs = list(base_dir.glob("partial_*"))

    # Verify both done files and partial results exist for each process
    return len(done_files) == num_processes and len(partial_dirs) == num_processes


def save_partial_dataset(dataset: Dataset, output_path: Path, process_index: int):
    """Save partial dataset processed by each GPU"""
    os.makedirs(output_path, exist_ok=True)
    # Save directly in the base directory without gpu_ prefix
    dataset.save_to_disk(output_path / f"partial_{process_index}")

    # Write completion flag directly in base directory
    with open(output_path / f"done_{process_index}.json", "w") as f:
        json.dump(
            {
                "process_index": process_index,
                "timestamp": time.time(),
                "num_samples": len(dataset),
            },
            f,
        )


def load_and_concatenate_results(base_dir: Path) -> Dataset:
    """Load and concatenate results from all GPU processes"""
    partial_datasets = []

    # Look for partial_* directories directly in base_dir
    for partial_path in base_dir.glob("partial_*"):
        if partial_path.exists():
            partial_ds = Dataset.load_from_disk(str(partial_path))
            partial_datasets.append(partial_ds)

    return concatenate_datasets(partial_datasets)


def distributed_autoformalize_dataset(
    accelerator: Accelerator,
    llm: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sampling_params: dict,
    dataset: Dataset,
    output_base_dir: str,
    batch_size: int = 32,
    use_system_prompt: bool = True,
    user_prompt_function: Optional[callable] = None,
) -> Optional[Dataset]:
    """Process dataset with distributed computing using local storage"""

    process_index = accelerator.process_index

    # Split dataset across GPUs
    dataset = dataset.shard(
        num_shards=accelerator.num_processes,
        index=process_index,
    )

    def process_batch(examples):
        try:
            # Prepare prompts
            prompts = []
            batch_size = len(examples["problem"])
            for i in range(sampling_params["n"]):
                for idx in range(batch_size):
                    prompts.append(examples["problem"][idx])

            # Get base model
            model_for_inference = llm.module if hasattr(llm, "module") else llm

            # Run inference
            outputs = batch_infer(
                model_for_inference,
                tokenizer,
                prompts,
                system_prompt if use_system_prompt else None,
                user_prompt_function,
                sampling_params["max_length"],
                sampling_params["max_tokens"],
                sampling_params["temperature"] > 0,
                0,
                sampling_params["top_p"],
                sampling_params["temperature"],
            )

            # Format results
            result = {"prompt": prompts}
            for i in range(sampling_params["n"]):
                result[f"answer_{i + 1}"] = [
                    output["output"]
                    for output in outputs[batch_size * i : batch_size * (i + 1)]
                ]
            return result

        except Exception as e:
            print(f"Error in process_batch (GPU {process_index}): {str(e)}")
            return {
                "prompt": prompts,
                **{
                    f"answer_{i + 1}": [""] * len(prompts)
                    for i in range(sampling_params["n"])
                },
            }

    try:
        # Process dataset
        processed_dataset = dataset.map(
            process_batch,
            batched=True,
            batch_size=batch_size,
            desc=f"Processing batches (GPU {process_index})",
            load_from_cache_file=False,
        )

        # Save partial results
        save_partial_dataset(processed_dataset, Path(output_base_dir), process_index)

        # Check if this is the last process to complete
        if check_all_processes_complete(
            Path(output_base_dir), accelerator.num_processes
        ):
            print(f"GPU {process_index} is last to complete - concatenating results")
            return load_and_concatenate_results(Path(output_base_dir))

        return None

    except Exception as e:
        print(f"Error in dataset processing (GPU {process_index}): {str(e)}")
        return None


def distributed_infer_dataset(
    accelerator: Accelerator,
    model_path: str,
    dataset_id: str,
    output_dataset_id: str,
    sparse_decode_method: str,
    sparse_decode_config: str,
    output_base_dir: str,
    dataset_branch: str = "main",
    output_dataset_branch: str = "main",
    n_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    max_tokens: int = 8192,
    batch_size: int = 32,
    use_system_prompt: bool = True,
    user_prompt_function: Optional[callable] = None,
    max_length: int = 20480,
):
    """Main function to run distributed inference using local storage"""
    try:
        # Load dataset
        ds = load_dataset(dataset_id, split="train", revision=dataset_branch)

        # Initialize model and tokenizer
        try:
            llm = Qwen2ForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )
            llm = apply_sparse_decode_method(
                llm, sparse_decode_method, sparse_decode_config, max_tokens
            )
            llm.eval()
            for param in llm.parameters():
                param.requires_grad = False

            tokenizer = AutoTokenizer.from_pretrained(model_path)

        except Exception as e:
            print(
                f"Error loading model/tokenizer on GPU {accelerator.process_index}: {str(e)}"
            )
            return

        # Prepare model without synchronization
        llm = accelerator.prepare(llm)

        sampling_params = dict(
            n=n_samples,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            max_length=max_length,
        )

        # Process dataset
        final_dataset = distributed_autoformalize_dataset(
            accelerator,
            llm,
            tokenizer,
            sampling_params,
            ds,
            output_base_dir,
            batch_size=batch_size,
            use_system_prompt=use_system_prompt,
            user_prompt_function=user_prompt_function,
        )

        # Only the last finishing process handles the upload
        if final_dataset is not None:
            try:
                # Calculate metrics
                processed_dataset = final_dataset.map(
                    extract_boxed_answer,
                    batched=True,
                    batch_size=batch_size,
                )

                accuracy = calculate_accuracy(
                    ground_truth=processed_dataset["answer"],
                    predicted=processed_dataset["extracted_answers"],
                )

                # Format and push results
                accuracy_percentage = f"{accuracy * 100:.2f}%"
                commit_message = f"Updated dataset with extracted answers (Accuracy: {accuracy_percentage})"

                processed_dataset.push_to_hub(
                    output_dataset_id,
                    revision=output_dataset_branch,
                    private=True,
                    commit_message=commit_message,
                )

                print(
                    f"Successfully uploaded complete dataset with accuracy: {accuracy_percentage}"
                )

            except Exception as e:
                print(f"Error in final processing/upload: {str(e)}")

    except Exception as e:
        print(f"Global error on GPU {accelerator.process_index}: {str(e)}")
