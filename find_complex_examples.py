"""
Find 100 complex examples from FACTUAL dataset for Experiment 1.

Uses the same complexity heuristic as the T5 evaluator.
"""

from datasets import load_dataset
from pathlib import Path


def classify_complexity(caption: str) -> str:
    """
    Classify caption complexity based on length and structure.

    Relaxed threshold for more complex examples:
    - Complex: >13 words OR >1 comma

    Args:
        caption: Input caption

    Returns:
        Complexity level: 'simple', 'medium', or 'complex'
    """
    word_count = len(caption.split())
    comma_count = caption.count(",")

    # Relaxed heuristic to get more complex examples
    if word_count > 13 or comma_count > 1:
        return "complex"
    elif word_count > 10 or comma_count > 0:
        return "medium"
    else:
        return "simple"


def find_complex_examples(num_samples: int = 100):
    """
    Find complex examples from the entire FACTUAL dataset.

    Args:
        num_samples: Number of complex examples to find
    """
    print("Loading FACTUAL dataset...")
    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True)

    # Load dataset
    dataset = load_dataset(
        "lizhuang144/FACTUAL_Scene_Graph",
        split="train",
        cache_dir=str(cache_dir)
    )

    total_len = len(dataset)
    print(f"Total dataset size: {total_len}")
    print(f"Searching entire dataset for complex examples...")

    # Find complex examples in entire dataset
    complex_indices = []
    complexity_counts = {"simple": 0, "medium": 0, "complex": 0}

    for i in range(total_len):
        sample = dataset[i]
        caption = sample.get("caption", "")
        complexity = classify_complexity(caption)
        complexity_counts[complexity] += 1

        if complexity == "complex":
            complex_indices.append(i)

        if len(complex_indices) >= num_samples:
            break

        # Progress update
        if (i + 1) % 5000 == 0:
            print(f"  Scanned {i + 1}/{total_len} samples, found {len(complex_indices)} complex so far...")

    print(f"\nComplexity distribution (scanned {i + 1} samples):")
    print(f"  Simple: {complexity_counts['simple']}")
    print(f"  Medium: {complexity_counts['medium']}")
    print(f"  Complex: {complexity_counts['complex']}")

    print(f"\nFound {len(complex_indices)} complex examples")
    print(f"\nIndices: {complex_indices}")

    # Show some example captions
    print("\n" + "="*80)
    print("Sample complex captions:")
    print("="*80)
    for idx in complex_indices[:5]:
        sample = dataset[idx]
        caption = sample.get("caption", "")
        word_count = len(caption.split())
        comma_count = caption.count(",")
        print(f"\nIndex {idx}:")
        print(f"  Caption: {caption}")
        print(f"  Words: {word_count}, Commas: {comma_count}")

    return complex_indices


if __name__ == "__main__":
    indices = find_complex_examples(num_samples=100)
