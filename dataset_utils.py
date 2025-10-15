"""
FACTUAL Dataset Loading and Test Data Utilities

This module provides centralized dataset loading and sampling functions
to ensure ALL experiments use identical test data for fair comparison.

Key Features:
- Single source of truth for dataset loading
- Consistent seed (42) for reproducibility
- Standard test split (10% of training data)
- Complex-only filtering (>20 words)
- Diverse sampling via linspace
"""

from typing import List, Dict
from datasets import load_dataset


# ============================================================================
# Dataset Configuration Constants
# ============================================================================

DATASET_NAME = "lizhuang144/FACTUAL_Scene_Graph"
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.1  # Use 10% of training data for test
COMPLEX_THRESHOLD = 20  # Words threshold for complex captions


# ============================================================================
# Dataset Loading Functions
# ============================================================================

def load_factual_dataset(
    split: str = "train",
    num_samples: int = 100,
    test_split: bool = True,
    use_complex_only: bool = False,
    cache_dir: str = "./cache"
) -> List[Dict]:
    """
    Load FACTUAL Scene Graph dataset with consistent configuration.

    This function ensures all experiments use the same dataset configuration
    for reproducible and fair comparisons.

    Args:
        split: Dataset split to load (default: "train")
        num_samples: Maximum number of samples to return (default: 100)
        test_split: Whether to use test split (10% of data, shuffled with seed 42)
        use_complex_only: Filter to only complex captions (>20 words)
        cache_dir: Directory to cache the dataset

    Returns:
        List of dataset samples (dictionaries)

    Example:
        >>> # Load same 100 complex samples as all experiments
        >>> samples = load_factual_dataset(
        ...     split="train",
        ...     num_samples=100,
        ...     test_split=True,
        ...     use_complex_only=True
        ... )
    """
    # Load dataset from HuggingFace
    dataset = load_dataset(DATASET_NAME, split=split, cache_dir=cache_dir)

    # Apply test split (10% of data, shuffled with consistent seed)
    if test_split:
        test_size = int(len(dataset) * TEST_SPLIT_RATIO)
        dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(test_size))

    # Filter to complex captions only (>20 words)
    if use_complex_only:
        dataset = dataset.filter(
            lambda x: len(x.get("caption", "").split()) > COMPLEX_THRESHOLD
        )

    # Select specified number of samples
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    return list(dataset)


def classify_complexity(caption: str) -> str:
    """
    Classify caption complexity based on word count.

    This function is used for breaking down results by complexity level.

    Args:
        caption: Image caption text

    Returns:
        Complexity level: "simple", "medium", or "complex"

    Classification:
        - simple: ≤10 words
        - medium: 11-20 words
        - complex: >20 words
    """
    word_count = len(caption.split())

    if word_count <= 10:
        return "simple"
    elif word_count <= 20:
        return "medium"
    else:
        return "complex"


# ============================================================================
# Standard Test Set Creators
# ============================================================================

def get_experiment_1_test_set() -> List[Dict]:
    """
    Get the standard test set used in Experiment 1 (T5 Baseline).

    Returns:
        100 complex samples from FACTUAL dataset
    """
    return load_factual_dataset(
        split="train",
        num_samples=100,
        test_split=True,
        use_complex_only=True
    )


def get_experiment_2_test_set() -> List[Dict]:
    """
    Get the standard test set used in Experiment 2 (LangExtract PoC).

    Selects 30 diverse samples from the same 100 complex samples
    used in Experiment 1 using evenly-spaced indices.

    Returns:
        30 diverse complex samples
    """
    import numpy as np

    all_samples = get_experiment_1_test_set()

    # Select 30 diverse samples using linspace
    indices = np.linspace(0, len(all_samples) - 1, 30, dtype=int)
    samples = [all_samples[int(i)] for i in indices]

    return samples


def get_experiment_3_test_set() -> List[Dict]:
    """
    Get the standard test set used in Experiment 3 (Format Optimization).

    Selects 50 diverse samples from the same 100 complex samples
    used in Experiment 1 using evenly-spaced indices.

    Returns:
        50 diverse complex samples
    """
    import numpy as np

    all_samples = get_experiment_1_test_set()

    # Select 50 diverse samples using linspace
    indices = np.linspace(0, len(all_samples) - 1, 50, dtype=int)
    samples = [all_samples[int(i)] for i in indices]

    return samples


def get_experiment_4_test_set() -> List[Dict]:
    """
    Get the standard test set used in Experiment 4 (Backend Comparison).

    Uses the SAME samples as Experiment 3 (50 diverse complex samples).

    Returns:
        50 diverse complex samples (identical to Experiment 3)
    """
    return get_experiment_3_test_set()


# ============================================================================
# Dataset Information
# ============================================================================

def print_dataset_info():
    """Print information about the FACTUAL dataset and test splits."""
    print("="*80)
    print("FACTUAL DATASET CONFIGURATION")
    print("="*80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Test Split: {TEST_SPLIT_RATIO*100:.0f}% of training data")
    print(f"Complex Threshold: >{COMPLEX_THRESHOLD} words")
    print()
    print("STANDARD TEST SETS:")
    print("-"*80)
    print("Experiment 1 (T5 Baseline):        100 complex samples")
    print("Experiment 2 (LangExtract PoC):     30 diverse samples (from Exp 1)")
    print("Experiment 3 (Format Optimization): 50 diverse samples (from Exp 1)")
    print("Experiment 4 (Backend Comparison):  50 diverse samples (same as Exp 3)")
    print("="*80)


if __name__ == "__main__":
    # Print dataset info when run as script
    print_dataset_info()

    # Test loading
    print("\nTesting dataset loading...")
    samples = load_factual_dataset(num_samples=5, test_split=True, use_complex_only=True)
    print(f"Loaded {len(samples)} samples")
    print(f"First caption: {samples[0].get('caption', '')[:80]}...")
    print("\nDataset utilities ready! ✓")
