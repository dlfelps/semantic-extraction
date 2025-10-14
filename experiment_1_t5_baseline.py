"""
Experiment 1: T5 Baseline Performance Evaluation

Evaluates the flan-t5-base-VG-factual-sg model on FACTUAL dataset.
Measures precision, recall, F1 for entities, attributes, and relationships.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import numpy as np
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SceneGraph:
    """Represents a parsed scene graph."""
    entities: Set[str]
    attributes: Set[Tuple[str, str]]  # (entity, attribute)
    relationships: Set[Tuple[str, str, str]]  # (subject, predicate, object)


@dataclass
class EvaluationMetrics:
    """Stores evaluation metrics for a component."""
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ComplexityMetrics:
    """Metrics grouped by caption complexity."""
    simple: Dict[str, EvaluationMetrics]
    medium: Dict[str, EvaluationMetrics]
    complex: Dict[str, EvaluationMetrics]


class T5BaselineEvaluator:
    """Evaluates T5 model performance on FACTUAL dataset."""

    def __init__(
        self,
        model_name: str = "lizhuang144/flan-t5-base-VG-factual-sg",
        device: str = None,
        cache_dir: str = "./cache"
    ):
        """
        Initialize the evaluator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache models and datasets
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        ).to(self.device)
        self.model.eval()

        print("Model loaded successfully")

    def load_factual_dataset(self, split: str = "train", num_samples: int = 100, test_split: bool = True, use_complex_only: bool = False) -> List[Dict]:
        """
        Load FACTUAL dataset samples.

        Args:
            split: Dataset split (default 'train')
            num_samples: Number of samples to load
            test_split: If True, use last 20% as test set, otherwise use random samples
            use_complex_only: If True, use hardcoded complex example indices

        Returns:
            List of dataset samples
        """
        print(f"Loading FACTUAL dataset ({split} split, {num_samples} samples)")

        dataset = load_dataset(
            "lizhuang144/FACTUAL_Scene_Graph",
            split=split,
            cache_dir=str(self.cache_dir)
        )

        # Hardcoded indices for 100 complex examples (>13 words OR >1 comma)
        # These are from the entire train split
        COMPLEX_INDICES = [322, 337, 371, 413, 598, 690, 704, 763, 1018, 1088,
                          1114, 1249, 1261, 1301, 1305, 1311, 1331, 1448, 1660, 1694,
                          1755, 1799, 2045, 2136, 2214, 2233, 2239, 2321, 2324, 2428,
                          2448, 2450, 2474, 2548, 2671, 2738, 2784, 2978, 3012, 3084,
                          3149, 3161, 3298, 3319, 3328, 3419, 3427, 3498, 3600, 3629,
                          3699, 3735, 3934, 4089, 4412, 4419, 4456, 4535, 4762, 5101,
                          5158, 5180, 5206, 5215, 5230, 5305, 5574, 5600, 5737, 5762,
                          5839, 5858, 5922, 5968, 6031, 6192, 6325, 6486, 6568, 6618,
                          6706, 6739, 6858, 6893, 6949, 7061, 7166, 7311, 7363, 7493,
                          7681, 7689, 7760, 7902, 7927, 7947, 7955, 8243, 8270, 8441]

        if use_complex_only:
            print(f"Using hardcoded complex example indices (100 samples)")
            samples = [dataset[idx] for idx in COMPLEX_INDICES]
        else:
            # If test_split is True, use last 20% of data as test set
            if test_split and split == "train":
                total_len = len(dataset)
                test_start_idx = int(total_len * 0.8)
                dataset = dataset.select(range(test_start_idx, total_len))
                print(f"Using last 20% of train split as test set ({len(dataset)} samples available)")

            # Sample diverse examples
            if len(dataset) > num_samples:
                indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
                samples = [dataset[int(i)] for i in indices]
            else:
                samples = list(dataset)

        print(f"Loaded {len(samples)} samples")
        return samples

    def parse_factual_format(self, scene_graph_str: str) -> SceneGraph:
        """
        Parse FACTUAL format scene graph string.

        Format: "( subject , predicate , object ) , ( entity , is , attribute )"
        - Attributes: predicates like "is", "are", "has" followed by descriptive terms
        - Relationships: other predicates connecting different entities

        Args:
            scene_graph_str: Scene graph in FACTUAL format

        Returns:
            Parsed SceneGraph object
        """
        entities = set()
        attributes = set()
        relationships = set()

        if not scene_graph_str or scene_graph_str.strip() == "":
            return SceneGraph(entities, attributes, relationships)

        # Attribute predicates (linking verbs that indicate attributes)
        attribute_predicates = {"is", "are", "has", "have"}

        # Split by commas outside parentheses to get individual tuples
        tuples = []
        current_tuple = ""
        paren_depth = 0

        for char in scene_graph_str:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
                current_tuple += char
                if paren_depth == 0:
                    tuples.append(current_tuple.strip())
                    current_tuple = ""
                    continue
            elif char == "," and paren_depth == 0:
                continue

            if paren_depth > 0:
                current_tuple += char

        for tuple_str in tuples:
            tuple_str = tuple_str.strip()
            if not tuple_str:
                continue

            # Extract tuple content between parentheses
            if tuple_str.startswith("(") and tuple_str.endswith(")"):
                tuple_content = tuple_str[1:-1]
                elements = [e.strip() for e in tuple_content.split(",")]

                if len(elements) == 3:
                    subj, pred, obj = elements

                    # Add entities (always add subject, conditionally add object)
                    entities.add(subj)

                    # Check if this is an attribute or relationship
                    if pred.lower() in attribute_predicates:
                        # This is an attribute: (entity, is, attribute_value)
                        attributes.add((subj, obj))
                    else:
                        # This is a relationship
                        entities.add(obj)
                        relationships.add((subj, pred, obj))

        return SceneGraph(entities, attributes, relationships)

    def generate_scene_graph(self, caption: str) -> str:
        """
        Generate scene graph from caption using T5 model.

        Args:
            caption: Input image caption

        Returns:
            Generated scene graph string
        """
        # Prepare input
        input_text = f"generate scene graph: {caption}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def classify_complexity(self, caption: str) -> str:
        """
        Classify caption complexity based on length and structure.

        Args:
            caption: Input caption

        Returns:
            Complexity level: 'simple', 'medium', or 'complex'
        """
        word_count = len(caption.split())
        comma_count = caption.count(",")
        and_count = caption.lower().count(" and ")

        # Simple heuristic
        if word_count <= 10 and comma_count == 0:
            return "simple"
        elif word_count <= 20 and comma_count <= 2:
            return "medium"
        else:
            return "complex"

    def compute_metrics(
        self,
        predicted: SceneGraph,
        ground_truth: SceneGraph
    ) -> Dict[str, EvaluationMetrics]:
        """
        Compute precision, recall, F1 for each component.

        Args:
            predicted: Predicted scene graph
            ground_truth: Ground truth scene graph

        Returns:
            Dictionary of metrics for each component
        """
        metrics = {}

        # Compute for each component
        for component_name, pred_set, gt_set in [
            ("entities", predicted.entities, ground_truth.entities),
            ("attributes", predicted.attributes, ground_truth.attributes),
            ("relationships", predicted.relationships, ground_truth.relationships)
        ]:
            if len(gt_set) == 0:
                # No ground truth for this component
                metrics[component_name] = EvaluationMetrics(0.0, 0.0, 0.0, 0)
                continue

            true_positives = len(pred_set & gt_set)
            false_positives = len(pred_set - gt_set)
            false_negatives = len(gt_set - pred_set)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[component_name] = EvaluationMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
                support=len(gt_set)
            )

        return metrics

    def evaluate(
        self,
        samples: List[Dict],
        save_dir: str = "./results/experiment_1"
    ) -> Dict:
        """
        Evaluate model on samples.

        Args:
            samples: List of dataset samples
            save_dir: Directory to save results

        Returns:
            Dictionary containing all evaluation results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\nEvaluating on {len(samples)} samples...")
        print("=" * 80)

        all_metrics = []
        complexity_groups = defaultdict(list)
        inference_times = []
        detailed_results = []

        for i, sample in enumerate(samples):
            caption = sample.get("caption", "")
            gt_scene_graph_str = sample.get("scene_graph", "")

            # Generate prediction
            start_time = time.time()
            pred_scene_graph_str = self.generate_scene_graph(caption)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Parse both
            predicted = self.parse_factual_format(pred_scene_graph_str)
            ground_truth = self.parse_factual_format(gt_scene_graph_str)

            # Compute metrics
            metrics = self.compute_metrics(predicted, ground_truth)
            all_metrics.append(metrics)

            # Classify complexity
            complexity = self.classify_complexity(caption)
            complexity_groups[complexity].append(metrics)

            # Store detailed result
            detailed_results.append({
                "sample_id": i,
                "caption": caption,
                "complexity": complexity,
                "ground_truth": gt_scene_graph_str,
                "predicted": pred_scene_graph_str,
                "metrics": {k: asdict(v) for k, v in metrics.items()},
                "inference_time": inference_time
            })

            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples...")

        print("=" * 80)
        print("\nComputing aggregate metrics...")

        # Aggregate overall metrics
        overall_metrics = self._aggregate_metrics(all_metrics)

        # Aggregate by complexity
        complexity_metrics = {
            complexity: self._aggregate_metrics(metrics_list)
            for complexity, metrics_list in complexity_groups.items()
        }

        # Compile results
        results = {
            "model": self.model_name,
            "num_samples": len(samples),
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

    def _aggregate_metrics(self, metrics_list: List[Dict[str, EvaluationMetrics]]) -> Dict:
        """Aggregate metrics across multiple samples."""
        if not metrics_list:
            return {}

        aggregated = {}
        component_names = ["entities", "attributes", "relationships"]

        for component in component_names:
            precisions = [m[component].precision for m in metrics_list if m[component].support > 0]
            recalls = [m[component].recall for m in metrics_list if m[component].support > 0]
            f1s = [m[component].f1 for m in metrics_list if m[component].support > 0]
            supports = [m[component].support for m in metrics_list]

            aggregated[component] = {
                "precision": np.mean(precisions) if precisions else 0.0,
                "recall": np.mean(recalls) if recalls else 0.0,
                "f1": np.mean(f1s) if f1s else 0.0,
                "total_support": sum(supports)
            }

        # Macro average F1
        aggregated["macro_f1"] = np.mean([
            aggregated[comp]["f1"] for comp in component_names
        ])

        return aggregated

    def _save_results(self, results: Dict, save_path: Path):
        """Save results to disk."""
        # Save JSON
        json_path = save_path / "results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {json_path}")

        # Create visualizations
        self._create_visualizations(results, save_path)

    def _create_visualizations(self, results: Dict, save_path: Path):
        """Create visualization plots."""
        sns.set_style("whitegrid")

        # Plot 1: Overall component performance
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        components = ["entities", "attributes", "relationships"]
        metrics_types = ["precision", "recall", "f1"]

        data = []
        for comp in components:
            comp_metrics = results["overall_metrics"][comp]
            for metric_type in metrics_types:
                data.append({
                    "Component": comp.capitalize(),
                    "Metric": metric_type.capitalize(),
                    "Score": comp_metrics[metric_type]
                })

        import pandas as pd
        df = pd.DataFrame(data)

        pivot_df = df.pivot(index="Component", columns="Metric", values="Score")
        pivot_df.plot(kind="bar", ax=ax, rot=0)
        ax.set_ylabel("Score")
        ax.set_title("T5 Baseline Performance by Component")
        ax.set_ylim([0, 1])
        ax.legend(title="Metric")

        plt.tight_layout()
        plt.savefig(save_path / "component_performance.png", dpi=300)
        plt.close()

        # Plot 2: Performance by complexity
        if len(results["complexity_metrics"]) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for idx, comp in enumerate(components):
                ax = axes[idx]

                complexity_levels = sorted(results["complexity_metrics"].keys())
                f1_scores = [
                    results["complexity_metrics"][level][comp]["f1"]
                    for level in complexity_levels
                ]

                ax.bar(complexity_levels, f1_scores, color=["green", "orange", "red"])
                ax.set_ylabel("F1 Score")
                ax.set_title(f"{comp.capitalize()} by Complexity")
                ax.set_ylim([0, 1])

            plt.tight_layout()
            plt.savefig(save_path / "complexity_analysis.png", dpi=300)
            plt.close()

        print(f"Visualizations saved to: {save_path}")

    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nModel: {results['model']}")
        print(f"Samples: {results['num_samples']}")
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
    """Main evaluation function."""
    # Initialize evaluator
    evaluator = T5BaselineEvaluator()

    # Load dataset (using hardcoded complex examples)
    samples = evaluator.load_factual_dataset(split="train", num_samples=100, test_split=True, use_complex_only=True)

    # Run evaluation
    results = evaluator.evaluate(samples, save_dir="./results/experiment_1_complex")

    print("\nExperiment 1 (Complex Examples) complete!")


if __name__ == "__main__":
    main()
