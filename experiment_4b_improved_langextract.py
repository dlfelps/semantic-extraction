"""
Experiment 4b: Test Improved LangExtract Examples

Tests whether adding targeted examples for "sitting on" -> "sit on" normalization
improves LangExtract's relationship extraction performance.

This experiment uses the SAME test set as Experiment 4, but with 5 additional
examples that explicitly demonstrate predicate normalization patterns where
LangExtract previously failed.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import asdict
from collections import defaultdict

# Fix Windows console UTF-8 encoding issue
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
import langextract as lx
from langextract.data import Document

# Import dataset and metrics utilities
from dataset_utils import get_experiment_4_test_set, classify_complexity
from factual_metrics import (
    convert_json_structured_to_factual,
    parse_ground_truth_factual,
    compute_factual_metrics,
    aggregate_factual_metrics
)

# Import improved examples
from improved_langextract_examples import create_targeted_sitting_examples

load_dotenv()


class ImprovedLangExtractEvaluator:
    """LangExtract evaluator with improved examples for predicate normalization."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

        # Check for API key
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or LANGEXTRACT_API_KEY not found")

        self.api_key = api_key

        # Create improved examples (original 2 + 5 new targeted examples)
        self.examples = self._create_all_examples()

        print(f"Initialized Improved LangExtract Evaluator")
        print(f"  Model: {model_name}")
        print(f"  Examples: {len(self.examples)} (including 5 targeted 'sit on' examples)")

    def _create_all_examples(self):
        """Create all examples: original + targeted improvements."""
        from experiment_4_backend_comparison import create_langextract_examples

        # Get original examples (2 examples from experiment 4)
        original_examples = create_langextract_examples()

        # Get new targeted examples (5 examples for "sitting on" normalization)
        targeted_examples = create_targeted_sitting_examples()

        # Combine them
        all_examples = original_examples + targeted_examples

        print(f"  - Original examples: {len(original_examples)}")
        print(f"  - Targeted examples: {len(targeted_examples)}")
        print(f"  - Total: {len(all_examples)}")

        return all_examples

    def extract_batch(self, captions: List[str]) -> tuple[List[Dict], float]:
        """Extract scene graphs from batch of captions."""
        try:
            documents = [Document(text=caption) for caption in captions]

            start_time = time.time()

            results = lx.extract(
                text_or_documents=documents,
                prompt_description=(
                    "Extract entities, attributes, and relationships from the image caption. "
                    "Return structured JSON with three arrays: entities, attributes, and relationships. "
                    "CRITICAL RULES: "
                    "Use BASE VERB FORMS in predicates (sitting on becomes sit on, standing on becomes stand on, wearing becomes wear, holding becomes hold). "
                    "Extract sitting or standing as an ATTRIBUTE of the entity, but use sit on or stand on in the RELATIONSHIP predicate. "
                    "For spatial relationships, use standard forms like at the left of and on the right side of. "
                    "Extract core entity names without descriptive prefixes."
                ),
                examples=self.examples,
                model_id=self.model_name,
                api_key=self.api_key,
                temperature=0.3,
                use_schema_constraints=True,
                fence_output=False,
                batch_length=50,
                max_workers=10,
            )

            total_time = time.time() - start_time

            # Convert to standard format
            all_extractions = []
            for result in results:
                extraction = {"entities": [], "attributes": [], "relationships": []}

                if result.extractions:
                    for ext in result.extractions:
                        if ext.extraction_class == "entity" and ext.attributes and "name" in ext.attributes:
                            extraction["entities"].append({"name": ext.attributes["name"]})
                        elif ext.extraction_class == "attribute" and ext.attributes:
                            if "entity" in ext.attributes and "attribute" in ext.attributes:
                                extraction["attributes"].append({
                                    "entity": ext.attributes["entity"],
                                    "attribute": ext.attributes["attribute"]
                                })
                        elif ext.extraction_class == "relationship" and ext.attributes:
                            if all(k in ext.attributes for k in ["subject", "predicate", "object"]):
                                extraction["relationships"].append({
                                    "subject": ext.attributes["subject"],
                                    "predicate": ext.attributes["predicate"],
                                    "object": ext.attributes["object"]
                                })

                all_extractions.append(extraction)

            return all_extractions, total_time

        except Exception as e:
            import traceback
            print(f"Batch extraction error: {e}")
            print(f"Error type: {type(e).__name__}")
            traceback.print_exc()
            return [{"entity": [], "attribute": [], "relationship": []} for _ in captions], 0.0

    def evaluate(self, samples: List[Dict], save_dir: str):
        """Evaluate improved LangExtract on test samples."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\nEvaluating Improved LangExtract on {len(samples)} samples...")
        print("="*80)

        # Extract
        captions = [s.get("caption", "") for s in samples]
        ground_truth_strs = [s.get("scene_graph", "") for s in samples]

        print(f"Processing {len(captions)} samples in batch...")
        all_extracted, total_time = self.extract_batch(captions)

        avg_time = total_time / len(captions) if captions else 0.0
        print(f"Batch completed in {total_time:.2f}s ({avg_time:.3f}s per sample)")

        # Compute metrics
        all_metrics = []
        complexity_groups = defaultdict(list)
        detailed_results = []
        success_count = 0

        for i, (caption, gt_str, extracted) in enumerate(zip(captions, ground_truth_strs, all_extracted)):
            # Convert to FACTUAL format
            pred_entities, pred_attrs, pred_rels = convert_json_structured_to_factual(extracted)
            gt_entities, gt_attrs, gt_rels = parse_ground_truth_factual(gt_str)

            # Check success
            has_results = len(pred_entities) > 0 or len(pred_attrs) > 0 or len(pred_rels) > 0
            if has_results:
                success_count += 1

            # Compute metrics
            metrics = compute_factual_metrics(pred_entities, pred_attrs, pred_rels, gt_entities, gt_attrs, gt_rels)
            all_metrics.append(metrics)

            # Classify complexity
            complexity = classify_complexity(caption)
            complexity_groups[complexity].append(metrics)

            # Store result
            detailed_results.append({
                "sample_id": i,
                "caption": caption,
                "complexity": complexity,
                "ground_truth": gt_str,
                "predicted": extracted,
                "factual_metrics": {k: asdict(v) for k, v in metrics.items()},
                "extraction_success": has_results
            })

        # Aggregate metrics
        overall_metrics = aggregate_factual_metrics(all_metrics)
        complexity_metrics = {
            comp: aggregate_factual_metrics(metrics)
            for comp, metrics in complexity_groups.items()
        }

        success_rate = success_count / len(samples) if samples else 0.0

        # Compile results
        results = {
            "experiment": "Experiment 4b: Improved LangExtract",
            "model": self.model_name,
            "num_examples": len(self.examples),
            "num_samples": len(samples),
            "extraction_success_rate": success_rate,
            "overall_metrics": overall_metrics,
            "complexity_metrics": complexity_metrics,
            "total_inference_time": total_time,
            "avg_inference_time_per_sample": avg_time,
            "detailed_results": detailed_results
        }

        # Save
        with open(save_path / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EXPERIMENT 4B: IMPROVED LANGEXTRACT - EVALUATION SUMMARY")
        print("="*80)
        print(f"\nModel: {results['model']}")
        print(f"Examples: {results['num_examples']}")
        print(f"Samples: {results['num_samples']}")
        print(f"Success Rate: {results['extraction_success_rate']:.1%}")
        print(f"Total Time: {results['total_inference_time']:.2f}s")
        print(f"Avg Time: {results['avg_inference_time_per_sample']:.3f}s/sample")

        overall = results["overall_metrics"]
        print(f"\n--- Overall Performance ---")
        print(f"Macro F1: {overall['macro_f1']:.3f}")

        for component in ["entities", "attributes", "relationships"]:
            comp = overall[component]
            print(f"\n{component.capitalize()}:")
            print(f"  Precision: {comp['precision']:.3f}")
            print(f"  Recall:    {comp['recall']:.3f}")
            print(f"  F1:        {comp['f1']:.3f}")

        print("\n" + "="*80)


def main():
    """Run Experiment 4b."""

    print("\n" + "="*80)
    print("EXPERIMENT 4B: IMPROVED LANGEXTRACT WITH TARGETED EXAMPLES")
    print("="*80 + "\n")

    # Load same test set as experiment 4
    samples = get_experiment_4_test_set()
    print(f"Loaded {len(samples)} samples (same as Experiment 4)\n")

    # Initialize evaluator with improved examples
    evaluator = ImprovedLangExtractEvaluator(model_name="gemini-2.5-flash")

    # Evaluate
    results = evaluator.evaluate(
        samples,
        save_dir="./results/experiment_4b_improved_langextract"
    )

    # Compare with original LangExtract (Experiment 4)
    exp4_path = Path("results/experiment_4_backend_comparison/langextract/results.json")
    if exp4_path.exists():
        with open(exp4_path) as f:
            exp4_results = json.load(f)

        print("\n" + "="*80)
        print("COMPARISON: ORIGINAL VS IMPROVED LANGEXTRACT")
        print("="*80)

        orig_f1 = exp4_results["overall_metrics"]["macro_f1"]
        improved_f1 = results["overall_metrics"]["macro_f1"]

        print(f"\nOriginal LangExtract Macro F1:  {orig_f1:.3f}")
        print(f"Improved LangExtract Macro F1:  {improved_f1:.3f}")
        print(f"Improvement:                     {improved_f1 - orig_f1:+.3f}")

        # Component breakdown
        print(f"\nComponent-wise comparison:")
        for comp in ["entities", "attributes", "relationships"]:
            orig = exp4_results["overall_metrics"][comp]["f1"]
            improved = results["overall_metrics"][comp]["f1"]
            print(f"  {comp.capitalize():15} {orig:.3f} -> {improved:.3f} ({improved - orig:+.3f})")

        print("\n" + "="*80)

    print("\nExperiment 4b complete!")


if __name__ == "__main__":
    main()
