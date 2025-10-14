"""
Experiment 2: LangExtract Proof of Concept

Uses LangExtract with Gemini to extract scene graphs in Flat Entities format.
Tests on 30 diverse samples from the same test set as Experiment 1.
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
from langextract import LangExtract

# Import shared components from experiment 1
from experiment_1_t5_baseline import (
    SceneGraph,
    EvaluationMetrics,
    T5BaselineEvaluator
)

load_dotenv()


# LangExtract Schema Definitions for Flat Entities Format
ENTITY_SCHEMA = {
    "name": "entity",
    "description": "A physical object, person, or thing mentioned in the caption",
    "attributes": {
        "name": {
            "type": "string",
            "description": "The name of the entity (e.g., 'man', 'dog', 'tree')"
        }
    },
    "examples": [
        {"name": "man"},
        {"name": "dog"},
        {"name": "bicycle"}
    ]
}

ATTRIBUTE_SCHEMA = {
    "name": "attribute",
    "description": "A property or characteristic of an entity (color, size, state, etc.)",
    "attributes": {
        "entity": {
            "type": "string",
            "description": "The entity this attribute describes"
        },
        "value": {
            "type": "string",
            "description": "The attribute value (e.g., 'red', 'large', 'sitting')"
        }
    },
    "examples": [
        {"entity": "shirt", "value": "red"},
        {"entity": "dog", "value": "large"},
        {"entity": "man", "value": "standing"}
    ]
}

RELATIONSHIP_SCHEMA = {
    "name": "relationship",
    "description": "A relationship or interaction between two entities",
    "attributes": {
        "subject": {
            "type": "string",
            "description": "The entity performing the action or in the relationship"
        },
        "predicate": {
            "type": "string",
            "description": "The action or relationship type (e.g., 'riding', 'next to', 'holding')"
        },
        "object": {
            "type": "string",
            "description": "The entity being acted upon or related to"
        }
    },
    "examples": [
        {"subject": "man", "predicate": "riding", "object": "bicycle"},
        {"subject": "dog", "predicate": "next to", "object": "tree"},
        {"subject": "woman", "predicate": "holding", "object": "umbrella"}
    ]
}


class LangExtractEvaluator:
    """Evaluates LangExtract performance on FACTUAL dataset."""

    def __init__(
        self,
        model_name: str = "gemini/gemini-1.5-flash",
        cache_dir: str = "./cache"
    ):
        """
        Initialize the evaluator.

        Args:
            model_name: LangExtract model identifier
            cache_dir: Directory to cache datasets
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize LangExtract with Gemini
        # Note: Requires GEMINI_API_KEY in environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.extractor = LangExtract(
            model=model_name,
            api_key=api_key
        )

        # Register schemas
        self.extractor.add_type(ENTITY_SCHEMA)
        self.extractor.add_type(ATTRIBUTE_SCHEMA)
        self.extractor.add_type(RELATIONSHIP_SCHEMA)

        print(f"Initialized LangExtract with model: {model_name}")

    def extract_scene_graph(self, caption: str) -> Dict:
        """
        Extract scene graph from caption using LangExtract.

        Args:
            caption: Input image caption

        Returns:
            Dictionary with extracted entities, attributes, and relationships
        """
        try:
            result = self.extractor.extract(caption)
            return result
        except Exception as e:
            print(f"Extraction error: {e}")
            return {"entity": [], "attribute": [], "relationship": []}

    def convert_to_scene_graph(self, extracted: Dict) -> SceneGraph:
        """
        Convert LangExtract output to SceneGraph format.

        Args:
            extracted: Dictionary from LangExtract

        Returns:
            SceneGraph object
        """
        entities = set()
        attributes = set()
        relationships = set()

        # Extract entities
        for entity_obj in extracted.get("entity", []):
            if isinstance(entity_obj, dict) and "name" in entity_obj:
                entities.add(entity_obj["name"])

        # Extract attributes as (entity, value) tuples
        for attr_obj in extracted.get("attribute", []):
            if isinstance(attr_obj, dict) and "entity" in attr_obj and "value" in attr_obj:
                entity = attr_obj["entity"]
                value = attr_obj["value"]
                attributes.add((entity, value))
                # Ensure entity is in entities set
                entities.add(entity)

        # Extract relationships as (subject, predicate, object) tuples
        for rel_obj in extracted.get("relationship", []):
            if isinstance(rel_obj, dict) and all(k in rel_obj for k in ["subject", "predicate", "object"]):
                subject = rel_obj["subject"]
                predicate = rel_obj["predicate"]
                obj = rel_obj["object"]
                relationships.add((subject, predicate, obj))
                # Ensure both entities are in entities set
                entities.add(subject)
                entities.add(obj)

        return SceneGraph(entities, attributes, relationships)

    def evaluate(
        self,
        samples: List[Dict],
        save_dir: str = "./results/experiment_2_langextract_poc"
    ) -> Dict:
        """
        Evaluate LangExtract on samples.

        Args:
            samples: List of dataset samples
            save_dir: Directory to save results

        Returns:
            Dictionary containing all evaluation results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\nEvaluating LangExtract on {len(samples)} samples...")
        print("=" * 80)

        # Reuse T5 evaluator for parsing ground truth and computing metrics
        t5_eval = T5BaselineEvaluator()

        all_metrics = []
        complexity_groups = defaultdict(list)
        inference_times = []
        detailed_results = []
        extraction_success_count = 0

        for i, sample in enumerate(samples):
            caption = sample.get("caption", "")
            gt_scene_graph_str = sample.get("scene_graph", "")

            # Extract using LangExtract
            start_time = time.time()
            extracted = self.extract_scene_graph(caption)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Convert to SceneGraph format
            predicted = self.convert_to_scene_graph(extracted)

            # Parse ground truth using T5 evaluator's parser
            ground_truth = t5_eval.parse_factual_format(gt_scene_graph_str)

            # Check if extraction was successful (got at least some results)
            has_results = len(predicted.entities) > 0 or len(predicted.attributes) > 0 or len(predicted.relationships) > 0
            if has_results:
                extraction_success_count += 1

            # Compute metrics
            metrics = t5_eval.compute_metrics(predicted, ground_truth)
            all_metrics.append(metrics)

            # Classify complexity
            complexity = t5_eval.classify_complexity(caption)
            complexity_groups[complexity].append(metrics)

            # Store detailed result
            detailed_results.append({
                "sample_id": i,
                "caption": caption,
                "complexity": complexity,
                "ground_truth": gt_scene_graph_str,
                "extracted_raw": extracted,
                "predicted": {
                    "entities": list(predicted.entities),
                    "attributes": [list(attr) for attr in predicted.attributes],
                    "relationships": [list(rel) for rel in predicted.relationships]
                },
                "metrics": {k: asdict(v) for k, v in metrics.items()},
                "inference_time": inference_time,
                "extraction_success": has_results
            })

            # Progress update
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples...")

        print("=" * 80)
        print("\nComputing aggregate metrics...")

        # Aggregate overall metrics
        overall_metrics = t5_eval._aggregate_metrics(all_metrics)

        # Aggregate by complexity
        complexity_metrics = {
            complexity: t5_eval._aggregate_metrics(metrics_list)
            for complexity, metrics_list in complexity_groups.items()
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
            "overall_metrics": overall_metrics,
            "complexity_metrics": complexity_metrics,
            "avg_inference_time": np.mean(inference_times),
            "std_inference_time": np.std(inference_times),
            "detailed_results": detailed_results
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
        print("EXPERIMENT 2: LANGEXTRACT PROOF OF CONCEPT - EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nModel: {results['model']}")
        print(f"Representation Format: {results['representation_format']}")
        print(f"Samples: {results['num_samples']}")
        print(f"Extraction Success Rate: {results['extraction_success_rate']:.1%} ({results['extraction_success_count']}/{results['num_samples']})")
        print(f"Avg Inference Time: {results['avg_inference_time']:.3f}s ± {results['std_inference_time']:.3f}s")

        print("\n--- Overall Performance ---")
        overall = results["overall_metrics"]
        print(f"Macro F1: {overall['macro_f1']:.3f}")

        for component in ["entities", "attributes", "relationships"]:
            comp_metrics = overall[component]
            print(f"\n{component.capitalize()}:")
            print(f"  Precision: {comp_metrics['precision']:.3f}")
            print(f"  Recall:    {comp_metrics['recall']:.3f}")
            print(f"  F1:        {comp_metrics['f1']:.3f}")
            print(f"  Support:   {comp_metrics['total_support']}")

        if results["complexity_metrics"]:
            print("\n--- Performance by Complexity ---")
            for complexity in sorted(results["complexity_metrics"].keys()):
                print(f"\n{complexity.capitalize()}:")
                comp_metrics = results["complexity_metrics"][complexity]
                for component in ["entities", "attributes", "relationships"]:
                    print(f"  {component}: F1 = {comp_metrics[component]['f1']:.3f}")

        print("\n" + "=" * 80)


def main():
    """Main evaluation function for Experiment 2."""

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

    # Initialize LangExtract evaluator
    evaluator = LangExtractEvaluator(model_name="gemini/gemini-1.5-flash")

    # Run evaluation
    results = evaluator.evaluate(samples, save_dir="./results/experiment_2_langextract_poc")

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

    print("\nExperiment 2 (LangExtract Proof of Concept) complete!")


if __name__ == "__main__":
    main()
