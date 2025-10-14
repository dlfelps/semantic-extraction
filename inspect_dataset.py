"""
Inspect FACTUAL dataset to understand the scene graph format.
"""

from datasets import load_dataset

# Load dataset
dataset = load_dataset("lizhuang144/FACTUAL_Scene_Graph", split="train", cache_dir="./cache")

# Show first 10 examples
print(f"Total samples: {len(dataset)}")
print(f"\nDataset fields: {dataset.column_names}")
print("\n" + "=" * 80)
print("FIRST 10 SAMPLES:")
print("=" * 80)

for i in range(min(10, len(dataset))):
    sample = dataset[i]
    print(f"\n--- Sample {i} ---")
    print(f"Caption: {sample.get('caption', 'N/A')}")
    print(f"Scene Graph: {sample.get('scene_graph', 'N/A')}")
    print(f"Image ID: {sample.get('image_id', 'N/A')}")
    print(f"Region ID: {sample.get('region_id', 'N/A')}")
