"""
Relationship Error Analysis for Experiment 3

Deep dive into why relationship extraction F1 is so low (0.086-0.114).
Analyzes predicted vs ground truth relationships to identify error patterns.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


class RelationshipErrorAnalyzer:
    """Analyzes relationship extraction errors across formats."""

    def __init__(self, results_dir: str = "./results/experiment_3_format_optimization_batched"):
        """Initialize analyzer."""
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "relationship_analysis"
        self.plots_dir.mkdir(exist_ok=True)

        # Load all format results
        self.format_results = {}
        self.load_all_results()

        print(f"Loaded results from: {self.results_dir}")
        print(f"Output directory: {self.plots_dir}")

    def load_all_results(self):
        """Load detailed results from all formats."""
        format_dirs = {
            "flat_entities": "Flat Entities",
            "tuple_format": "Tuple Format",
            "hierarchical": "Hierarchical",
            "json_structured": "JSON Structured"
        }

        for format_key, format_name in format_dirs.items():
            results_file = self.results_dir / format_key / "results.json"
            if results_file.exists():
                with open(results_file, "r") as f:
                    data = json.load(f)
                    self.format_results[format_name] = data
                    print(f"  Loaded: {format_name}")

    def extract_relationships_from_ground_truth(self, gt_str: str) -> Set[Tuple[str, str, str]]:
        """
        Parse ground truth string to extract relationships.

        FACTUAL format: "( subject , predicate , object ) , ( entity , is , attribute )"

        Relationships are tuples where the predicate is NOT an attribute predicate.
        Attribute predicates: "is", "are", "has", "have", "be"
        """
        relationships = set()

        if not gt_str or gt_str.strip() == "":
            return relationships

        # Attribute predicates (linking verbs that indicate attributes, not relationships)
        # These match what's used in experiment_1_t5_baseline.py
        attribute_predicates = {"is", "are", "has", "have", "be"}

        # Split by commas outside parentheses to get individual tuples
        tuples = []
        current_tuple = ""
        paren_depth = 0

        for char in gt_str:
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

        # Parse each tuple
        for tuple_str in tuples:
            tuple_str = tuple_str.strip()
            if not tuple_str:
                continue

            # Remove parentheses
            if tuple_str.startswith("(") and tuple_str.endswith(")"):
                tuple_str = tuple_str[1:-1]

            # Split by commas
            parts = [p.strip() for p in tuple_str.split(",")]

            if len(parts) != 3:
                continue

            subject, predicate, obj = parts

            # Classify as relationship if predicate is NOT an attribute linking verb
            if predicate.lower() not in attribute_predicates:
                relationships.add((subject, predicate, obj))

        return relationships

    def analyze_format_errors(self, format_name: str) -> Dict:
        """Analyze relationship errors for a specific format."""
        if format_name not in self.format_results:
            return None

        results = self.format_results[format_name]
        detailed = results["detailed_results"]

        # Collect all errors
        total_gt = 0
        total_pred = 0
        correct = 0

        missing_relationships = []  # In GT but not predicted
        hallucinated_relationships = []  # Predicted but not in GT

        predicate_errors = defaultdict(int)
        entity_mismatches = defaultdict(int)

        all_gt_predicates = []
        all_pred_predicates = []

        for sample in detailed:
            caption = sample["caption"]
            gt_str = sample.get("ground_truth", "")

            # Parse ground truth
            gt_rels = self.extract_relationships_from_ground_truth(gt_str)

            # Get predicted relationships
            pred_rels = set()
            for rel in sample["predicted"]["relationships"]:
                if len(rel) == 3:
                    pred_rels.add(tuple(rel))

            total_gt += len(gt_rels)
            total_pred += len(pred_rels)

            # Find matches (exact matches only)
            matches = gt_rels & pred_rels
            correct += len(matches)

            # Find errors
            missing = gt_rels - pred_rels
            hallucinated = pred_rels - gt_rels

            for rel in missing:
                missing_relationships.append({
                    "caption": caption,
                    "relationship": rel,
                    "predicted_rels": list(pred_rels)
                })

            for rel in hallucinated:
                hallucinated_relationships.append({
                    "caption": caption,
                    "relationship": rel,
                    "ground_truth_rels": list(gt_rels)
                })

            # Collect predicates
            for _, pred, _ in gt_rels:
                all_gt_predicates.append(pred)

            for _, pred, _ in pred_rels:
                all_pred_predicates.append(pred)

            # Analyze predicate errors (relaxed matching)
            for gt_rel in gt_rels:
                gt_subj, gt_pred, gt_obj = gt_rel

                # Check if subject and object match but predicate differs
                for pred_rel in pred_rels:
                    pred_subj, pred_pred, pred_obj = pred_rel

                    if gt_subj == pred_subj and gt_obj == pred_obj and gt_pred != pred_pred:
                        predicate_errors[(gt_pred, pred_pred)] += 1

                    # Check if predicate matches but entities differ
                    if gt_pred == pred_pred and (gt_subj != pred_subj or gt_obj != pred_obj):
                        entity_mismatches[gt_pred] += 1

        # Calculate metrics
        precision = correct / total_pred if total_pred > 0 else 0
        recall = correct / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "format": format_name,
            "total_ground_truth": total_gt,
            "total_predicted": total_pred,
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "missing_relationships": missing_relationships,
            "hallucinated_relationships": hallucinated_relationships,
            "predicate_errors": dict(predicate_errors),
            "entity_mismatches": dict(entity_mismatches),
            "gt_predicates": Counter(all_gt_predicates),
            "pred_predicates": Counter(all_pred_predicates)
        }

    def plot_predicate_distribution(self, analysis_results: Dict):
        """Plot distribution of predicates in GT vs Predicted."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (format_name, analysis) in enumerate(analysis_results.items()):
            ax = axes[idx]

            # Get top predicates from both GT and predicted
            gt_preds = analysis["gt_predicates"]
            pred_preds = analysis["pred_predicates"]

            # Combine and get top 15
            all_preds = set(list(gt_preds.keys()) + list(pred_preds.keys()))
            top_preds = sorted(all_preds, key=lambda x: gt_preds.get(x, 0) + pred_preds.get(x, 0), reverse=True)[:15]

            # Prepare data
            gt_counts = [gt_preds.get(p, 0) for p in top_preds]
            pred_counts = [pred_preds.get(p, 0) for p in top_preds]

            x = range(len(top_preds))
            width = 0.35

            bars1 = ax.bar([i - width/2 for i in x], gt_counts, width, label='Ground Truth', alpha=0.8)
            bars2 = ax.bar([i + width/2 for i in x], pred_counts, width, label='Predicted', alpha=0.8)

            ax.set_xlabel("Predicate", fontsize=10, fontweight='bold')
            ax.set_ylabel("Count", fontsize=10, fontweight='bold')
            ax.set_title(f"{format_name}", fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(top_preds, rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "predicate_distribution.png", dpi=300, bbox_inches='tight')
        print(f"Saved: predicate_distribution.png")
        plt.close()

    def plot_error_breakdown(self, analysis_results: Dict):
        """Plot breakdown of relationship errors."""
        fig, ax = plt.subplots(figsize=(12, 7))

        formats = list(analysis_results.keys())

        correct = [analysis_results[f]["correct"] for f in formats]
        missing = [len(analysis_results[f]["missing_relationships"]) for f in formats]
        hallucinated = [len(analysis_results[f]["hallucinated_relationships"]) for f in formats]

        x = range(len(formats))
        width = 0.25

        bars1 = ax.bar([i - width for i in x], correct, width, label='Correct', color='green', alpha=0.8)
        bars2 = ax.bar(x, missing, width, label='Missing (False Negatives)', color='red', alpha=0.8)
        bars3 = ax.bar([i + width for i in x], hallucinated, width, label='Hallucinated (False Positives)', color='orange', alpha=0.8)

        ax.set_xlabel("Format", fontsize=12, fontweight='bold')
        ax.set_ylabel("Count", fontsize=12, fontweight='bold')
        ax.set_title("Relationship Error Breakdown", fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(formats, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "error_breakdown.png", dpi=300, bbox_inches='tight')
        print(f"Saved: error_breakdown.png")
        plt.close()

    def create_error_examples_report(self, analysis_results: Dict):
        """Create detailed report of error examples."""
        report_path = self.plots_dir / "error_examples.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("RELATIONSHIP ERROR ANALYSIS - DETAILED EXAMPLES\n")
            f.write("="*80 + "\n\n")

            for format_name, analysis in analysis_results.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"FORMAT: {format_name}\n")
                f.write(f"{'='*80}\n\n")

                f.write(f"Summary:\n")
                f.write(f"  Ground Truth Relationships: {analysis['total_ground_truth']}\n")
                f.write(f"  Predicted Relationships:    {analysis['total_predicted']}\n")
                f.write(f"  Correct:                    {analysis['correct']}\n")
                f.write(f"  Precision:                  {analysis['precision']:.3f}\n")
                f.write(f"  Recall:                     {analysis['recall']:.3f}\n")
                f.write(f"  F1:                         {analysis['f1']:.3f}\n\n")

                # Missing relationships (first 10 examples)
                f.write(f"\n--- MISSING RELATIONSHIPS (False Negatives) ---\n")
                f.write(f"Total: {len(analysis['missing_relationships'])}\n\n")

                for i, error in enumerate(analysis['missing_relationships'][:10], 1):
                    rel = error['relationship']
                    f.write(f"{i}. Caption: {error['caption'][:80]}...\n")
                    f.write(f"   Missing: ({rel[0]} | {rel[1]} | {rel[2]})\n")
                    f.write(f"   Predicted: {error['predicted_rels'][:3]}\n\n")

                # Hallucinated relationships (first 10 examples)
                f.write(f"\n--- HALLUCINATED RELATIONSHIPS (False Positives) ---\n")
                f.write(f"Total: {len(analysis['hallucinated_relationships'])}\n\n")

                for i, error in enumerate(analysis['hallucinated_relationships'][:10], 1):
                    rel = error['relationship']
                    f.write(f"{i}. Caption: {error['caption'][:80]}...\n")
                    f.write(f"   Hallucinated: ({rel[0]} | {rel[1]} | {rel[2]})\n")
                    f.write(f"   Ground Truth: {error['ground_truth_rels'][:3]}\n\n")

                # Predicate confusion matrix
                if analysis['predicate_errors']:
                    f.write(f"\n--- PREDICATE ERRORS (Same entities, wrong predicate) ---\n")
                    f.write(f"Total unique errors: {len(analysis['predicate_errors'])}\n\n")

                    sorted_errors = sorted(analysis['predicate_errors'].items(),
                                          key=lambda x: x[1], reverse=True)[:10]

                    for (gt_pred, pred_pred), count in sorted_errors:
                        f.write(f"  GT: '{gt_pred}' -> Predicted: '{pred_pred}' ({count} times)\n")

                f.write("\n")

        print(f"Saved: error_examples.txt")
        return report_path

    def create_summary_csv(self, analysis_results: Dict):
        """Create CSV with analysis summary."""
        rows = []

        for format_name, analysis in analysis_results.items():
            row = {
                "Format": format_name,
                "GT Relationships": analysis["total_ground_truth"],
                "Predicted Relationships": analysis["total_predicted"],
                "Correct": analysis["correct"],
                "Missing (FN)": len(analysis["missing_relationships"]),
                "Hallucinated (FP)": len(analysis["hallucinated_relationships"]),
                "Precision": f"{analysis['precision']:.3f}",
                "Recall": f"{analysis['recall']:.3f}",
                "F1": f"{analysis['f1']:.3f}",
                "Unique GT Predicates": len(analysis["gt_predicates"]),
                "Unique Pred Predicates": len(analysis["pred_predicates"])
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        csv_path = self.plots_dir / "relationship_analysis_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: relationship_analysis_summary.csv")

        return df

    def analyze_all_formats(self):
        """Run complete analysis across all formats."""
        print("\n" + "="*80)
        print("ANALYZING RELATIONSHIP ERRORS")
        print("="*80 + "\n")

        # Analyze each format
        analysis_results = {}

        for format_name in self.format_results.keys():
            print(f"Analyzing {format_name}...")
            analysis = self.analyze_format_errors(format_name)
            if analysis:
                analysis_results[format_name] = analysis
                print(f"  F1: {analysis['f1']:.3f}, "
                      f"Missing: {len(analysis['missing_relationships'])}, "
                      f"Hallucinated: {len(analysis['hallucinated_relationships'])}")

        print("\nGenerating visualizations...")
        self.plot_predicate_distribution(analysis_results)
        self.plot_error_breakdown(analysis_results)

        print("\nCreating detailed reports...")
        self.create_error_examples_report(analysis_results)
        df = self.create_summary_csv(analysis_results)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll outputs saved to: {self.plots_dir}\n")

        print("Summary:")
        print(df.to_string(index=False))
        print("\n" + "="*80)

        return analysis_results


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze relationship extraction errors")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results/experiment_3_format_optimization_batched",
        help="Directory containing experiment results"
    )

    args = parser.parse_args()

    try:
        analyzer = RelationshipErrorAnalyzer(results_dir=args.results_dir)
        analyzer.analyze_all_formats()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you've run experiment 3 first:")
        print("  uv run experiment_3_format_optimization_batched.py")


if __name__ == "__main__":
    main()
