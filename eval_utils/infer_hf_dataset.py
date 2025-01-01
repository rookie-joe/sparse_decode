import os

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from data_utils.prompts import system_prompt


def autoformalize_dataset_batched(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    dataset: Dataset,
    batch_size: int = 1024,
    use_system_prompt: bool = True,
    user_prompt_function=None,
) -> Dataset:

    n_samples = sampling_params.n

    def process_batch(examples):
        # Prepare prompts for the batch

        prompts = []
        for idx in range(len(examples["problem"])):
            problem = examples["problem"][idx]

            prompts.append(problem)

        # Prepare messages and generate
        messages = [
            (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                if use_system_prompt
                else [{"role": "user", "content": prompt}]
            )
            for prompt in prompts
        ]

        texts = [
            tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]

        outputs = llm.generate(texts, sampling_params)

        # Format results
        result = {"prompt": prompts}
        for i in range(n_samples):
            result[f"answer_{i + 1}"] = [output.outputs[i].text for output in outputs]

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
):

    ds = load_dataset(dataset_id, split="train", revision=dataset_branch)

    llm = LLM(
        model_path,
        download_dir=f"{os.getenv('HOME')}/.cache/vllm/",
        tensor_parallel_size=8,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
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
