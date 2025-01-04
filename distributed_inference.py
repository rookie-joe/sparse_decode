import random

import fire
from accelerate import Accelerator
from datasets import load_dataset

from eval_utils.metric import calculate_accuracy, extract_boxed_answer
from eval_utils.torch_distributed_infer_hdf_dataset import distributed_infer_dataset


def generate(
    model_path: str,
    bs_ds: str,
    save_ds: str,
    save_ds_branch: str = "main",
    n_samples: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    batch_size: int = 32,
):
    """
    Distributed generation and evaluation function
    """
    # Initialize accelerator
    accelerator = Accelerator()

    # Only main process loads and samples dataset
    if accelerator.is_main_process:
        bs_ds = load_dataset(bs_ds, split="train")
        n_samples = min(n_samples, len(bs_ds))
        random_rows = [random.randint(0, len(bs_ds) - 1) for _ in range(n_samples)]
        bs_ds_sample = bs_ds.select(random_rows)
        print(f"Sampled dataset size: {len(bs_ds_sample)}")

        # Push sampled dataset
        bs_ds_sample.push_to_hub(
            save_ds,
            revision=save_ds_branch,
            private=True,
            commit_message=f"sample {len(bs_ds_sample)} from {bs_ds}",
        )

    # Ensure all processes wait for the main process
    accelerator.wait_for_everyone()

    # Run distributed inference
    distributed_infer_dataset(
        accelerator=accelerator,
        model_path=model_path,
        dataset_id=save_ds,
        dataset_branch=save_ds_branch,
        output_dataset_id=save_ds,
        output_dataset_branch=save_ds_branch,
        n_samples=1,
        temperature=temperature,
        max_tokens=max_tokens,
        batch_size=batch_size,
    )

    # Only main process handles results and metrics
    if accelerator.is_main_process:
        # Load results and calculate metrics
        ds_with_results = load_dataset(save_ds, split="train", revision=save_ds_branch)

        # Extract answers and calculate accuracy
        processed_dataset = ds_with_results.map(
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
        commit_message = f"Updated dataset with extracted boxed answers (Accuracy: {accuracy_percentage})"

        processed_dataset.push_to_hub(
            save_ds,
            revision=save_ds_branch,
            private=True,
            commit_message=commit_message,
        )

    # Wait for all processes to complete
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(generate)
