"""
Analyze predicate patterns where Native Gemini succeeded but LangExtract failed.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

def main():
    # Load relationship analysis
    with open("results/experiment_4_backend_comparison/relationship_analysis.json") as f:
        data = json.load(f)

    native_only = data["native_only_correct"]

    print("="*80)
    print("DETAILED ANALYSIS: Native Gemini Success Patterns")
    print("="*80)

    # Group by sample
    by_sample = defaultdict(list)
    for item in native_only:
        by_sample[item["sample_id"]].append(item)

    print(f"\nTotal relationships Native got right (LangExtract missed): {len(native_only)}")
    print(f"Across {len(by_sample)} unique samples\n")

    # Analyze by sample
    sample_counts = Counter({sid: len(items) for sid, items in by_sample.items()})

    print("="*80)
    print("TOP SAMPLES WITH MOST MISSED RELATIONSHIPS")
    print("="*80)

    for sample_id, count in sample_counts.most_common(10):
        items = by_sample[sample_id]
        caption = items[0]["caption"]

        print(f"\n--- Sample {sample_id}: {count} relationships missed ---")
        print(f"Caption: {caption}")
        print(f"\nRelationships LangExtract missed:")
        for item in items:
            print(f"  - ({item['subject']}, {item['predicate']}, {item['object']})")

    # Analyze predicate patterns
    print("\n" + "="*80)
    print("PREDICATE ANALYSIS")
    print("="*80)

    predicate_counts = Counter([item["predicate"] for item in native_only])

    print(f"\nPredicates where LangExtract failed most:")
    for predicate, count in predicate_counts.most_common():
        print(f"  '{predicate}': {count} times")

        # Show a few examples
        examples = [item for item in native_only if item["predicate"] == predicate][:3]
        print(f"  Examples:")
        for ex in examples:
            print(f"    - Sample {ex['sample_id']}: ({ex['subject']}, {ex['predicate']}, {ex['object']})")
            print(f"      Caption: {ex['caption'][:80]}...")

if __name__ == "__main__":
    main()
