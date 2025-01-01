import random

import fire
from datasets import load_dataset

from eval_utils.infer_hf_dataset import infer_hf_datasete
from eval_utils.metric import calculate_accuracy, extract_boxed_answer


def generate(
    model_path: str,
    bs_ds: str,
    save_ds: str,
    save_ds_branch: str,
    n_samples: int,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    batch_size: int = 32,
):

    bs_ds = load_dataset(bs_ds, split="train")

    # Sample n_samples from the reference dataset and push to hub
    n_samples = min(n_samples, len(bs_ds))
    random_rows = [random.randint(0, len(bs_ds) - 1) for _ in range(n_samples)]
    bs_ds_sample = bs_ds.select(random_rows)
    print(f"Sampled dataset size: {len(bs_ds_sample)}")

    bs_ds_sample.push_to_hub(
        save_ds,
        revision=save_ds_branch,
        private=True,
        commit_message=f"sample {len(bs_ds_sample)} from {bs_ds}",
    )

    # generate on the sampled dataset
    infer_hf_datasete(
        model_path=model_path,
        dataset_id=save_ds,
        dataset_branch=save_ds_branch,
        output_dataset_id=save_ds,
        output_dataset_branch=save_ds_branch,
        n_samples=1,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # do the metrics calculation
    ds_with_results = load_dataset(save_ds, split="train", revision=save_ds_branch)

    # Process dataset in batches
    # processed_dataset =  extract_boxed_answer(ds_with_results)
    processed_dataset = ds_with_results.map(
        extract_boxed_answer,
        batched=True,
        batch_size=batch_size,
    )

    # Calculate accuracy
    accuracy = calculate_accuracy(
        ground_truth=processed_dataset["answer"],
        predicted=processed_dataset["extracted_answers"],
    )

    # Format accuracy as percentage with 2 decimal places
    accuracy_percentage = f"{accuracy * 100:.2f}%"

    # Create informative commit message
    commit_message = f"Updated dataset with extracted boxed answers (Accuracy: {accuracy_percentage})"

    # Push to hub with the commit message
    processed_dataset.push_to_hub(
        save_ds,
        revision=save_ds_branch,
        private=True,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    fire.Fire(generate)
