import os
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from data_utils.prompts import system_prompt
from infer_utils.modeling_qwen2 import Qwen2ForCausalLM
from infer_utils.infer import batch_infer


def autoformalize_dataset_batched(
    llm: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sampling_params: dict,
    dataset: Dataset,
    batch_size: int = 1024,
    use_system_prompt: bool = True,
    user_prompt_function=None,
) -> Dataset:

    batch_size = 1

    n_samples = sampling_params["n"]
    max_length = sampling_params["max_length"]
    max_new_tokens = sampling_params["max_tokens"]
    top_p = sampling_params["top_p"]
    temperature = sampling_params["temperature"]
    do_sample = temperature > 0
    sys_prompt = system_prompt if use_system_prompt else None

    def process_batch(examples):
        # Prepare prompts for the batch
        prompts = []
        batch_size = len(examples["problem"])
        for i in range(n_samples):
            for idx in range(batch_size):
                problem = examples["problem"][idx]
                prompts.append(problem)
        outputs = batch_infer(
            llm,
            tokenizer,
            prompts,
            sys_prompt,
            None,
            max_length,
            max_new_tokens,
            do_sample,
            top_p,
            temperature,
            False,
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

    return processed_dataset


def infer_hf_datasete(
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
    batch_size: int = 1024,
    use_system_prompt: bool = True,
    user_prompt_function=None,
    max_length: int = 20480,
):

    ds = load_dataset(dataset_id, split="train", revision=dataset_branch)

    llm = Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = dict(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        max_length=max_length,
    )

    # Process dataset with autoformalization
    processed_ds = autoformalize_dataset_batched(
        llm,
        tokenizer,
        sampling_params,
        ds,
        batch_size=batch_size,
        use_system_prompt=use_system_prompt,
        user_prompt_function=user_prompt_function,
    )

    new_columns = [col for col in processed_ds.column_names if col != "prompt"]
    for col in new_columns:
        ds = ds.add_column(col, processed_ds[col])

    # Push to HuggingFace Hub
    ds.push_to_hub(
        output_dataset_id,
        revision=output_dataset_branch,
        private=True,
        commit_message=f"Inferred using {model_path} model.",
    )
