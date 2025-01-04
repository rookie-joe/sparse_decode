from typing import Optional

import torch
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from data_utils.prompts import system_prompt
from infer_utils.infer import batch_infer
from infer_utils.modeling_qwen2 import Qwen2ForCausalLM


def distributed_autoformalize_dataset(
    accelerator: Accelerator,
    llm: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sampling_params: dict,
    dataset: Dataset,
    batch_size: int = 128,
    use_system_prompt: bool = True,
    user_prompt_function: Optional[callable] = None,
) -> Dataset:
    """
    Process dataset with distributed computing using Accelerator
    """
    if accelerator.is_main_process:
        print(f"Starting distributed processing with batch size {batch_size}")

    n_samples = sampling_params["n"]
    max_length = sampling_params["max_length"]
    max_new_tokens = sampling_params["max_tokens"]
    top_p = sampling_params["top_p"]
    temperature = sampling_params["temperature"]
    do_sample = temperature > 0
    sys_prompt = system_prompt if use_system_prompt else None

    # Split dataset across GPUs
    dataset = dataset.shard(
        num_shards=accelerator.num_processes,
        index=accelerator.process_index,
    )

    def process_batch(examples):
        # Prepare prompts for the batch
        prompts = []
        batch_size = len(examples["problem"])
        for i in range(n_samples):
            for idx in range(batch_size):
                problem = examples["problem"][idx]
                prompts.append(problem)

        # Get the base model if it's wrapped in DDP
        model_for_inference = llm.module if hasattr(llm, "module") else llm

        # Run inference on this GPU's portion
        outputs = batch_infer(
            model_for_inference,
            tokenizer,
            prompts,
            sys_prompt,
            None,
            max_length,
            max_new_tokens,
            do_sample,
            0,
            top_p,
            temperature,
        )

        # Format results
        result = {"prompt": prompts}
        for i in range(n_samples):
            result[f"answer_{i + 1}"] = [
                output["output"]
                for output in outputs[batch_size * i : batch_size * (i + 1)]
            ]
        return result

    # Process dataset in batches
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
    )

    # Gather datasets from all processes using torch.distributed
    all_datasets = [None] * accelerator.num_processes
    torch.distributed.all_gather_object(all_datasets, processed_dataset)

    # Wait for all processes to finish gathering
    accelerator.wait_for_everyone()

    # Concatenate datasets (on main process only)
    if accelerator.is_main_process:
        final_dataset = concatenate_datasets(all_datasets)
        return final_dataset
    else:
        return processed_dataset


def distributed_infer_dataset(
    accelerator: Accelerator,
    model_path: str,
    dataset_id: str,
    output_dataset_id: str,
    dataset_branch: str = "main",
    output_dataset_branch: str = "main",
    n_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    max_tokens: int = 8192,
    batch_size: int = 128,
    use_system_prompt: bool = True,
    user_prompt_function: Optional[callable] = None,
    max_length: int = 20480,
):
    """
    Main function to run distributed inference using Accelerator
    """

    # Load dataset
    ds = load_dataset(dataset_id, split="train", revision=dataset_branch)

    # Initialize model and tokenizer (on all processes)
    llm = Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    # Set model to evaluation mode and disable gradient computation
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    accelerator.wait_for_everyone()

    # Prepare model for distributed inference with no gradient computations
    with accelerator.no_sync(llm):
        llm = accelerator.prepare(llm)
    accelerator.wait_for_everyone()

    sampling_params = dict(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        max_length=max_length,
    )

    # Process dataset with distributed autoformalization
    processed_ds = distributed_autoformalize_dataset(
        accelerator,
        llm,
        tokenizer,
        sampling_params,
        ds,
        batch_size=batch_size,
        use_system_prompt=use_system_prompt,
        user_prompt_function=user_prompt_function,
    )

    # Only push to hub from main process
    if accelerator.is_main_process:
        new_columns = [col for col in processed_ds.column_names if col != "prompt"]
        for col in new_columns:
            ds = ds.add_column(col, processed_ds[col])

        ds.push_to_hub(
            output_dataset_id,
            revision=output_dataset_branch,
            private=True,
            commit_message=f"Inferred using {model_path} model with distributed processing.",
        )

    # Wait for all processes to complete
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    # Example usage
    distributed_infer_dataset(
        model_path="your_model_path",
        dataset_id="your_dataset_id",
        output_dataset_id="your_output_dataset_id",
        batch_size=128,  # Adjust based on GPU memory
        n_samples=1,
    )
