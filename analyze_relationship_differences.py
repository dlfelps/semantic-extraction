"""
Analyze relationship extraction differences between LangExtract and Native Gemini.

Identifies specific relationships that Native Gemini extracted correctly but LangExtract missed.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

def load_results(approach: str) -> Dict:
    """Load results for a specific approach."""
    path = Path(f"results/experiment_4_backend_comparison/{approach}/results.json")
    with open(path, "r") as f:
        return json.load(f)

def parse_ground_truth_relationships(gt_str: str) -> Set[Tuple]:
    """Parse ground truth relationships from FACTUAL format string."""
    relationships = set()
    if not gt_str:
        return relationships

    # Split by commas and parse triplets
    parts = [p.strip() for p in gt_str.split(',')]

    i = 0
    while i < len(parts):
        part = parts[i].strip()

        # Look for opening parenthesis
        if part.startswith('('):
            # Extract subject
            subject = part[1:].strip()
            if i + 2 < len(parts):
                predicate = parts[i + 1].strip()
                obj_part = parts[i + 2].strip()

                # Remove closing parenthesis
                if obj_part.endswith(')'):
                    obj = obj_part[:-1].strip()
                    relationships.add((subject, predicate, obj))
                    i += 3
                    continue

        i += 1

    return relationships

def extract_relationships(detailed_result: Dict) -> Tuple[Set[Tuple], Set[Tuple]]:
    """Extract predicted and ground truth relationships from a detailed result."""
    # Predicted relationships
    pred_rels = set()
    if "predicted" in detailed_result and "relationships" in detailed_result["predicted"]:
        for rel in detailed_result["predicted"]["relationships"]:
            if isinstance(rel, list) and len(rel) == 3:
                pred_rels.add(tuple(rel))

    # Ground truth relationships - parse from string
    gt_rels = set()
    if "ground_truth" in detailed_result:
        gt_rels = parse_ground_truth_relationships(detailed_result["ground_truth"])

    return pred_rels, gt_rels

def main():
    """Main analysis function."""

    # Load both results
    langextract_results = load_results("langextract")
    native_results = load_results("native_gemini")

    # Track statistics
    native_only_correct = []  # Relationships Native got right but LangExtract missed
    both_correct = []  # Relationships both got right
    langextract_only_correct = []  # Relationships LangExtract got right but Native missed
    both_missed = []  # Relationships both missed

    # Predicate statistics
    native_success_predicates = Counter()
    langextract_miss_predicates = Counter()

    # Process each sample
    for i, (le_result, native_result) in enumerate(zip(
        langextract_results["detailed_results"],
        native_results["detailed_results"]
    )):
        # Get relationships
        le_pred, le_gt = extract_relationships(le_result)
        native_pred, native_gt = extract_relationships(native_result)

        # Sanity check: ground truth should be the same
        if le_gt != native_gt:
            print(f"WARNING: Sample {i} has different ground truth!")
            continue

        gt = le_gt

        # Find what each got correct
        le_correct = le_pred & gt
        native_correct = native_pred & gt

        # Categorize each ground truth relationship
        for rel in gt:
            subject, predicate, obj = rel

            native_got_it = rel in native_correct
            le_got_it = rel in le_correct

            caption = le_result.get("caption", "")

            if native_got_it and not le_got_it:
                # Native Gemini got it right, LangExtract missed
                native_only_correct.append({
                    "sample_id": i,
                    "caption": caption,
                    "relationship": rel,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })
                native_success_predicates[predicate] += 1
                langextract_miss_predicates[predicate] += 1

            elif le_got_it and not native_got_it:
                langextract_only_correct.append({
                    "sample_id": i,
                    "caption": caption,
                    "relationship": rel,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })

            elif native_got_it and le_got_it:
                both_correct.append({
                    "sample_id": i,
                    "caption": caption,
                    "relationship": rel,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })

            else:
                both_missed.append({
                    "sample_id": i,
                    "caption": caption,
                    "relationship": rel,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })

    # Print summary
    print("\n" + "="*80)
    print("RELATIONSHIP EXTRACTION ANALYSIS")
    print("="*80)

    total_rels = len(native_only_correct) + len(langextract_only_correct) + len(both_correct) + len(both_missed)

    print(f"\nTotal ground truth relationships: {total_rels}")
    if total_rels > 0:
        print(f"\nBoth correct: {len(both_correct)} ({len(both_correct)/total_rels*100:.1f}%)")
        print(f"Native only correct: {len(native_only_correct)} ({len(native_only_correct)/total_rels*100:.1f}%)")
        print(f"LangExtract only correct: {len(langextract_only_correct)} ({len(langextract_only_correct)/total_rels*100:.1f}%)")
        print(f"Both missed: {len(both_missed)} ({len(both_missed)/total_rels*100:.1f}%)")
    else:
        print("\nNo relationships found! Check data parsing.")

    # Analyze predicates where Native succeeded but LangExtract failed
    print("\n" + "="*80)
    print("PREDICATES: Native Gemini succeeded, LangExtract failed")
    print("="*80)

    for predicate, count in langextract_miss_predicates.most_common(20):
        print(f"  {predicate}: {count} times")

    # Show specific examples
    print("\n" + "="*80)
    print("EXAMPLES: Relationships Native Gemini got right, LangExtract missed")
    print("="*80)

    # Group by predicate
    by_predicate = defaultdict(list)
    for item in native_only_correct:
        by_predicate[item["predicate"]].append(item)

    # Show top examples for each common predicate
    for predicate in list(langextract_miss_predicates.keys())[:10]:
        examples = by_predicate.get(predicate, [])[:3]  # Show up to 3 examples
        if examples:
            print(f"\n--- Predicate: '{predicate}' ({langextract_miss_predicates[predicate]} total misses) ---")
            for ex in examples:
                print(f"\nSample {ex['sample_id']}:")
                print(f"  Caption: {ex['caption'][:100]}...")
                print(f"  Relationship: ({ex['subject']}, {ex['predicate']}, {ex['object']})")

    # Save detailed results
    output = {
        "summary": {
            "total_relationships": total_rels,
            "both_correct": len(both_correct),
            "native_only_correct": len(native_only_correct),
            "langextract_only_correct": len(langextract_only_correct),
            "both_missed": len(both_missed)
        },
        "predicate_analysis": {
            "native_success_predicates": dict(native_success_predicates.most_common()),
            "langextract_miss_predicates": dict(langextract_miss_predicates.most_common())
        },
        "native_only_correct": native_only_correct,
        "langextract_only_correct": langextract_only_correct,
        "both_correct": both_correct,
        "both_missed": both_missed
    }

    output_path = Path("results/experiment_4_backend_comparison/relationship_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nDetailed analysis saved to: {output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
