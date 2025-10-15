"""
Experiment 2: LangExtract Proof of Concept (BATCHED VERSION)

Uses Google's LangExtract with Gemini to extract scene graphs in Flat Entities format.
Tests on 30 diverse samples from the same test set as Experiment 1.

This batched version processes all samples in a single API call for better performance.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

# Import langextract correctly
import langextract as lx
from langextract.data import ExampleData, Extraction, Document

# Import shared components from experiment 1 (for dataset loading and complexity classification)
from experiment_1_t5_baseline import T5BaselineEvaluator

# Import FACTUAL metrics utilities
from factual_metrics import (
    EvaluationMetrics,
    convert_flat_entities_to_factual,
    parse_ground_truth_factual,
    compute_factual_metrics,
    aggregate_factual_metrics
)

load_dotenv()


def create_example_data() -> List[ExampleData]:
    """
    Create few-shot examples for LangExtract in Flat Entities format.

    IMPORTANT INSTRUCTIONS DEMONSTRATED IN EXAMPLES:
    1. Use base verb forms in predicates, NOT present participles (wear, not wearing)
    2. Extract entity names as they appear, without adding possessive forms (head, not man's head)
    3. Extract core entities, not descriptive phrases (area, not sandy area)
    4. Use standard spatial predicates (at the left of, on the right side of)
    5. Use standard action predicates (under, not beneath)
    """
    examples = []

    # Example 1: Complex scene with multiple entities and relationships
    ex1 = ExampleData(
        text="A large brown dog with a red collar sitting on a wooden bench in a park next to a metal trash can while a small bird perches on the bench arm.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="dog"),
            Extraction(extraction_class="entity", extraction_text="collar"),
            Extraction(extraction_class="entity", extraction_text="bench"),
            Extraction(extraction_class="entity", extraction_text="park"),
            Extraction(extraction_class="entity", extraction_text="trash can"),
            Extraction(extraction_class="entity", extraction_text="bird"),
            Extraction(extraction_class="entity", extraction_text="bench arm"),
            Extraction(extraction_class="attribute", extraction_text="large", attributes={"entity": "dog"}),
            Extraction(extraction_class="attribute", extraction_text="brown", attributes={"entity": "dog"}),
            Extraction(extraction_class="attribute", extraction_text="sitting", attributes={"entity": "dog"}),
            Extraction(extraction_class="attribute", extraction_text="red", attributes={"entity": "collar"}),
            Extraction(extraction_class="attribute", extraction_text="wooden", attributes={"entity": "bench"}),
            Extraction(extraction_class="attribute", extraction_text="metal", attributes={"entity": "trash can"}),
            Extraction(extraction_class="attribute", extraction_text="small", attributes={"entity": "bird"}),
            Extraction(extraction_class="attribute", extraction_text="perches", attributes={"entity": "bird"}),
            Extraction(extraction_class="relationship", extraction_text="dog with collar",
                      attributes={"subject": "dog", "predicate": "with", "object": "collar"}),
            Extraction(extraction_class="relationship", extraction_text="dog sit on bench",
                      attributes={"subject": "dog", "predicate": "sit on", "object": "bench"}),
            Extraction(extraction_class="relationship", extraction_text="bench in park",
                      attributes={"subject": "bench", "predicate": "in", "object": "park"}),
            Extraction(extraction_class="relationship", extraction_text="bench next to trash can",
                      attributes={"subject": "bench", "predicate": "next to", "object": "trash can"}),
            Extraction(extraction_class="relationship", extraction_text="bird perch on bench arm",
                      attributes={"subject": "bird", "predicate": "perch on", "object": "bench arm"}),
        ]
    )
    examples.append(ex1)

    # Example 2: Demonstrating predicate normalization
    ex2 = ExampleData(
        text="A young woman wearing a blue dress holds a cup of coffee while talking on her phone standing beside a black car.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="woman"),
            Extraction(extraction_class="entity", extraction_text="dress"),
            Extraction(extraction_class="entity", extraction_text="cup of coffee"),
            Extraction(extraction_class="entity", extraction_text="phone"),
            Extraction(extraction_class="entity", extraction_text="car"),
            Extraction(extraction_class="attribute", extraction_text="young", attributes={"entity": "woman"}),
            Extraction(extraction_class="attribute", extraction_text="blue", attributes={"entity": "dress"}),
            Extraction(extraction_class="attribute", extraction_text="standing", attributes={"entity": "woman"}),
            Extraction(extraction_class="attribute", extraction_text="black", attributes={"entity": "car"}),
            # IMPORTANT: Use base verb forms
            # "wearing" in text -> "wear" in predicate
            # "holds" in text -> "hold" in predicate
            # "talking" in text -> "talk on" in predicate
            Extraction(extraction_class="relationship", extraction_text="woman wear dress",
                      attributes={"subject": "woman", "predicate": "wear", "object": "dress"}),
            Extraction(extraction_class="relationship", extraction_text="woman hold cup of coffee",
                      attributes={"subject": "woman", "predicate": "hold", "object": "cup of coffee"}),
            Extraction(extraction_class="relationship", extraction_text="woman talk on phone",
                      attributes={"subject": "woman", "predicate": "talk on", "object": "phone"}),
            Extraction(extraction_class="relationship", extraction_text="woman beside car",
                      attributes={"subject": "woman", "predicate": "beside", "object": "car"}),
        ]
    )
    examples.append(ex2)

    # Example 3: Entity extraction guidance - use simple entity names
    ex3 = ExampleData(
        text="Small stairs from the lower area to the upper area of the sandy beach.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="stairs"),
            # Extract core entity "area" and "beach", not "lower area" or "sandy beach"
            Extraction(extraction_class="entity", extraction_text="area"),
            Extraction(extraction_class="entity", extraction_text="beach"),
            Extraction(extraction_class="attribute", extraction_text="small", attributes={"entity": "stairs"}),
            Extraction(extraction_class="attribute", extraction_text="lower", attributes={"entity": "area"}),
            Extraction(extraction_class="attribute", extraction_text="upper", attributes={"entity": "area"}),
            Extraction(extraction_class="attribute", extraction_text="sandy", attributes={"entity": "beach"}),
            Extraction(extraction_class="relationship", extraction_text="stairs on area",
                      attributes={"subject": "stairs", "predicate": "on", "object": "area"}),
            Extraction(extraction_class="relationship", extraction_text="area on beach",
                      attributes={"subject": "area", "predicate": "on", "object": "beach"}),
        ]
    )
    examples.append(ex3)

    # Example 4: Spatial relationships - use standard forms
    ex4 = ExampleData(
        text="The building to the left of another building with clocks on it.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="building"),
            Extraction(extraction_class="entity", extraction_text="clocks"),
            # Use "at the left of" not "to the left of"
            Extraction(extraction_class="relationship", extraction_text="building at the left of building",
                      attributes={"subject": "building", "predicate": "at the left of", "object": "building"}),
            Extraction(extraction_class="relationship", extraction_text="clocks on building",
                      attributes={"subject": "clocks", "predicate": "on", "object": "building"}),
        ]
    )
    examples.append(ex4)

    # Example 5: More predicate normalization patterns
    ex5 = ExampleData(
        text="A dog watching birds lying beneath a tree holding a stick.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="dog"),
            Extraction(extraction_class="entity", extraction_text="birds"),
            Extraction(extraction_class="entity", extraction_text="tree"),
            Extraction(extraction_class="entity", extraction_text="stick"),
            Extraction(extraction_class="attribute", extraction_text="lying", attributes={"entity": "dog"}),
            # Use base forms: "watch" not "watching", "under" not "beneath", "hold" not "holding"
            Extraction(extraction_class="relationship", extraction_text="dog watch birds",
                      attributes={"subject": "dog", "predicate": "watch", "object": "birds"}),
            Extraction(extraction_class="relationship", extraction_text="dog under tree",
                      attributes={"subject": "dog", "predicate": "under", "object": "tree"}),
            Extraction(extraction_class="relationship", extraction_text="dog hold stick",
                      attributes={"subject": "dog", "predicate": "hold", "object": "stick"}),
        ]
    )
    examples.append(ex5)

    return examples


class BatchedLangExtractEvaluator:
    """Evaluates LangExtract performance on FACTUAL dataset using batch processing."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        cache_dir: str = "./cache",
        batch_length: int = 30,
        max_workers: int = 10
    ):
        """
        Initialize the evaluator.

        Args:
            model_name: Gemini model identifier
            cache_dir: Directory to cache datasets
            batch_length: Number of samples per batch
            max_workers: Maximum parallel workers
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.batch_length = batch_length
        self.max_workers = max_workers

        # Check for API key
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or LANGEXTRACT_API_KEY not found in environment variables")

        self.api_key = api_key

        # Create examples
        self.examples = create_example_data()

        print(f"Initialized LangExtract with model: {model_name}")
        print(f"Using {len(self.examples)} few-shot examples")
        print(f"Batch settings: batch_length={batch_length}, max_workers={max_workers}")

    def extract_batch(self, captions: List[str]) -> Tuple[List[Dict], float]:
        """
        Extract scene graphs from a batch of captions using LangExtract.

        Args:
            captions: List of input image captions

        Returns:
            Tuple of (list of extracted dictionaries, total time)
        """
        try:
            # Create Document objects for batch processing
            documents = [Document(text=caption) for caption in captions]

            # Use langextract.extract() with batch processing
            start_time = time.time()

            results = lx.extract(
                text_or_documents=documents,
                prompt_description=(
                    "Extract entities (objects, people, things), attributes (properties like color, size, state), "
                    "and relationships (interactions between entities) from the text. "
                    "For attributes, specify which entity they describe. "
                    "For relationships, specify subject, predicate, and object. "
                    "\n\nIMPORTANT RULES:\n"
                    "1. Use base verb forms in predicates (wear, hold, watch), NOT present participles (wearing, holding, watching)\n"
                    "2. Extract entity names as they appear, without possessive forms (head, not man's head)\n"
                    "3. Extract core entities only, not descriptive phrases (area, not sandy area or lower part)\n"
                    "4. Use standard spatial predicates: 'at the left of' (not 'to the left of'), 'on the right side of' (not 'to the right of')\n"
                    "5. Use standard action predicates: under (not beneath), across (not going across)"
                ),
                examples=self.examples,
                model_id=self.model_name,
                api_key=self.api_key,
                temperature=0.3,
                use_schema_constraints=True,
                fence_output=False,
                batch_length=self.batch_length,
                max_workers=self.max_workers,
            )

            total_time = time.time() - start_time

            # Convert AnnotatedDocuments to dictionary format
            all_extractions = []

            for result in results:
                extractions = {"entity": [], "attribute": [], "relationship": []}

                if result.extractions:
                    for ext in result.extractions:
                        if ext.extraction_class == "entity":
                            extractions["entity"].append({"name": ext.extraction_text})
                        elif ext.extraction_class == "attribute":
                            if ext.attributes and "entity" in ext.attributes:
                                extractions["attribute"].append({
                                    "entity": ext.attributes["entity"],
                                    "value": ext.extraction_text
                                })
                        elif ext.extraction_class == "relationship":
                            if ext.attributes and all(k in ext.attributes for k in ["subject", "predicate", "object"]):
                                extractions["relationship"].append({
                                    "subject": ext.attributes["subject"],
                                    "predicate": ext.attributes["predicate"],
                                    "object": ext.attributes["object"]
                                })

                all_extractions.append(extractions)

            return all_extractions, total_time

        except Exception as e:
            print(f"Batch extraction error: {e}")
            # Return empty results for all samples
            return [{"entity": [], "attribute": [], "relationship": []} for _ in captions], 0.0


    def evaluate(
        self,
        samples: List[Dict],
        save_dir: str = "./results/experiment_2_langextract_poc_batched"
    ) -> Dict:
        """
        Evaluate LangExtract on samples using batch processing.

        Args:
            samples: List of dataset samples
            save_dir: Directory to save results

        Returns:
            Dictionary containing all evaluation results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\nEvaluating LangExtract on {len(samples)} samples (BATCHED)...")
        print("=" * 80)

        # Reuse T5 evaluator for parsing ground truth and computing metrics
        t5_eval = T5BaselineEvaluator()

        # Extract all captions
        captions = [sample.get("caption", "") for sample in samples]
        ground_truth_strs = [sample.get("scene_graph", "") for sample in samples]

        # Batch extract
        print(f"Processing {len(captions)} samples in batch...")
        all_extracted, total_time = self.extract_batch(captions)

        avg_time_per_sample = total_time / len(captions) if captions else 0.0
        print(f"Batch completed in {total_time:.2f}s ({avg_time_per_sample:.3f}s per sample)")

        # Process results - compute FACTUAL metrics only
        all_factual_metrics = []
        complexity_groups_factual = defaultdict(list)
        detailed_results = []
        extraction_success_count = 0

        for i, (caption, gt_scene_graph_str, extracted) in enumerate(zip(captions, ground_truth_strs, all_extracted)):
            # Convert to FACTUAL format
            pred_entities, pred_attrs, pred_rels = convert_flat_entities_to_factual(extracted)
            gt_entities, gt_attrs, gt_rels = parse_ground_truth_factual(gt_scene_graph_str)

            # Check if extraction was successful (got at least some results)
            has_results = len(pred_entities) > 0 or len(pred_attrs) > 0 or len(pred_rels) > 0
            if has_results:
                extraction_success_count += 1

            # Compute FACTUAL-based metrics
            factual_metrics = compute_factual_metrics(pred_entities, pred_attrs, pred_rels, gt_entities, gt_attrs, gt_rels)
            all_factual_metrics.append(factual_metrics)

            # Classify complexity
            complexity = t5_eval.classify_complexity(caption)
            complexity_groups_factual[complexity].append(factual_metrics)

            # Store detailed result
            detailed_results.append({
                "sample_id": i,
                "caption": caption,
                "complexity": complexity,
                "ground_truth": gt_scene_graph_str,
                "extracted_raw": extracted,
                "predicted_factual": {
                    "entities": list(pred_entities),
                    "attributes": [list(attr) for attr in pred_attrs],
                    "relationships": [list(rel) for rel in pred_rels]
                },
                "factual_metrics": {k: asdict(v) for k, v in factual_metrics.items()},
                "extraction_success": has_results
            })

        print("=" * 80)
        print("\nComputing aggregate metrics...")

        # Aggregate overall metrics
        overall_factual_metrics = aggregate_factual_metrics(all_factual_metrics)

        # Aggregate by complexity
        complexity_factual_metrics = {
            complexity: aggregate_factual_metrics(metrics_list)
            for complexity, metrics_list in complexity_groups_factual.items()
        }

        # Calculate extraction success rate
        extraction_success_rate = extraction_success_count / len(samples) if samples else 0.0

        # Compile results
        results = {
            "model": self.model_name,
            "representation_format": "Flat Entities",
            "num_samples": len(samples),
            "extraction_success_rate": extraction_success_rate,
            "extraction_success_count": extraction_success_count,
            "overall_metrics": overall_factual_metrics,
            "complexity_metrics": complexity_factual_metrics,
            "total_inference_time": total_time,
            "avg_inference_time_per_sample": avg_time_per_sample,
            "detailed_results": detailed_results,
            "batch_settings": {
                "batch_length": self.batch_length,
                "max_workers": self.max_workers
            }
        }

        # Save results
        self._save_results(results, save_path)

        # Print summary
        self._print_summary(results)

        return results

    def _save_results(self, results: Dict, save_path: Path):
        """Save results to disk."""
        # Save JSON
        json_path = save_path / "results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {json_path}")

    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: LANGEXTRACT PROOF OF CONCEPT (BATCHED) - EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nModel: {results['model']}")
        print(f"Representation Format: {results['representation_format']}")
        print(f"Samples: {results['num_samples']}")
        print(f"Extraction Success Rate: {results['extraction_success_rate']:.1%} ({results['extraction_success_count']}/{results['num_samples']})")
        print(f"Total Inference Time: {results['total_inference_time']:.2f}s")
        print(f"Avg Time per Sample: {results['avg_inference_time_per_sample']:.3f}s")
        print(f"Batch Settings: batch_length={results['batch_settings']['batch_length']}, max_workers={results['batch_settings']['max_workers']}")

        # Show FACTUAL metrics
        overall = results["overall_metrics"]

        print("\n--- Overall Performance (FACTUAL Metrics) ---")
        print(f"Macro F1: {overall['macro_f1']:.3f}")

        for component in ["entities", "attributes", "relationships"]:
            comp_metrics = overall[component]
            print(f"\n{component.capitalize()}:")
            print(f"  Precision: {comp_metrics['precision']:.3f}")
            print(f"  Recall:    {comp_metrics['recall']:.3f}")
            print(f"  F1:        {comp_metrics['f1']:.3f}")
            print(f"  Support:   {comp_metrics['support']}")

        if results["complexity_metrics"]:
            print("\n--- Performance by Complexity ---")
            for complexity in sorted(results["complexity_metrics"].keys()):
                print(f"\n{complexity.capitalize()}:")
                comp_metrics = results["complexity_metrics"][complexity]
                print(f"  Macro F1: {comp_metrics['macro_f1']:.3f}")
                for component in ["entities", "attributes", "relationships"]:
                    print(f"  {component}: F1 = {comp_metrics[component]['f1']:.3f}")

        print("\n" + "=" * 80)


def main():
    """Main evaluation function for Experiment 2 (Batched)."""

    # Load the same test set as Experiment 1 (100 complex examples)
    # We'll select 30 diverse samples from this set
    t5_eval = T5BaselineEvaluator()
    all_samples = t5_eval.load_factual_dataset(
        split="train",
        num_samples=100,
        test_split=True,
        use_complex_only=True
    )

    # Select 30 diverse samples (every ~3rd sample to ensure diversity)
    indices = np.linspace(0, len(all_samples) - 1, 30, dtype=int)
    samples = [all_samples[int(i)] for i in indices]

    print(f"Selected {len(samples)} diverse samples from Experiment 1 test set")

    # Initialize LangExtract evaluator with batch settings
    evaluator = BatchedLangExtractEvaluator(
        model_name="gemini-2.5-flash",
        batch_length=30,  # Process all 30 samples at once
        max_workers=10    # Use 10 parallel workers
    )

    # Run evaluation
    results = evaluator.evaluate(samples, save_dir="./results/experiment_2_langextract_poc_batched")

    # Load Experiment 1 baseline for comparison
    exp1_results_path = Path("./results/experiment_1_complex/results.json")
    if exp1_results_path.exists():
        with open(exp1_results_path, "r") as f:
            exp1_results = json.load(f)

        print("\n" + "=" * 80)
        print("COMPARISON WITH EXPERIMENT 1 BASELINE (T5)")
        print("=" * 80)

        t5_macro_f1 = exp1_results["overall_metrics"]["macro_f1"]
        langextract_macro_f1 = results["overall_metrics"]["macro_f1"]

        print(f"\nT5 Baseline Macro F1:     {t5_macro_f1:.3f}")
        print(f"LangExtract Macro F1:     {langextract_macro_f1:.3f}")
        print(f"Difference:               {langextract_macro_f1 - t5_macro_f1:+.3f}")
        print(f"Within 20% threshold:     {'✓ Yes' if langextract_macro_f1 >= t5_macro_f1 * 0.8 else '✗ No'}")
        print(f"Extraction success rate:  {results['extraction_success_rate']:.1%}")
        print(f"Success criteria (>70%):  {'✓ Yes' if results['extraction_success_rate'] > 0.7 else '✗ No'}")

        print("\n" + "=" * 80)

    print("\nExperiment 2 (LangExtract Proof of Concept - BATCHED) complete!")


if __name__ == "__main__":
    main()
