"""Check available splits in FACTUAL dataset."""

from datasets import load_dataset

print("Loading FACTUAL dataset to check available splits...")

# Try to load and see what splits exist
try:
    dataset_info = load_dataset("lizhuang144/FACTUAL_Scene_Graph", cache_dir="./cache")
    print(f"\nAvailable splits: {list(dataset_info.keys())}")

    for split_name, split_data in dataset_info.items():
        print(f"\n{split_name}: {len(split_data)} samples")
        if len(split_data) > 0:
            print(f"  Example keys: {list(split_data[0].keys())}")
except Exception as e:
    print(f"Error: {e}")
