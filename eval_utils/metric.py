import re
from typing import Dict, List, Union


def extract_boxed_answer(
    examples: Dict, answer_key: str = "answer_1"
) -> Dict[str, List[int]]:
    """
    Extracts LaTeX boxed answers from text and returns them as integers.

    Args:
        examples (dict): Dictionary containing answer texts
        answer_key (str): Key to access the answer text in the examples dict

    Returns:
        dict: Dictionary containing a list of:
            - extracted_answers: List of integer values from inside \boxed{} or None if not found/not convertible
    """
    # Initialize result dictionary with empty lists
    result = {
        "extracted_answers": [],
    }

    # Get number of examples in batch
    num_examples = len(examples[answer_key])

    # Process each example in the batch
    for i in range(num_examples):
        # Get the text from the current example
        text = examples[answer_key][i]
        if isinstance(text, list):
            text = text[0]

        # Split the text into lines
        lines = text.split("\n")

        # Regular expression pattern to match both \boxed{content} and \\boxed{content}
        pattern = r"\\?\\boxed{([^}]*)}"

        # Initialize default value
        extracted_answer = None

        # Iterate through lines from last to first
        for line in reversed(lines):
            match = re.search(pattern, line)
            if match:
                # Extract the content inside \boxed{}
                boxed_content = match.group(1).strip()
                try:
                    # Convert to float first to handle decimal points if present
                    float_val = float(boxed_content)
                    # Convert to int - this will round down for positive numbers
                    extracted_answer = int(float_val)
                    break
                except (ValueError, TypeError):
                    # If conversion fails, keep looking in previous lines
                    continue

        # Store result
        result["extracted_answers"].append(extracted_answer)

    return result


def calculate_accuracy(
    ground_truth: List[str], predicted: List[Union[int, None]]
) -> float:
    """
    Calculate accuracy by comparing extracted answers with ground truth answers.

    Args:
        ground_truth (List[str]): List of correct answers as strings
        predicted (List[Union[int, None]]): List of extracted answers (can contain None for failed extractions)

    Returns:
        float: Accuracy score between 0 and 1
    """
    if len(ground_truth) != len(predicted):
        raise ValueError("Ground truth and predicted lists must have the same length")

    if not ground_truth:  # Handle empty lists
        return 0.0

    correct = 0
    total = len(ground_truth)

    for true, pred in zip(ground_truth, predicted):
        try:
            # Convert ground truth to int for comparison
            true_int = int(float(true))
            # Check if prediction matches (accounting for None values)
            if pred is not None and true_int == pred:
                correct += 1
        except (ValueError, TypeError):
            continue

    return correct / total
