"""
Analysis Script for Experiment 3 Format Optimization Results

Creates visualizations comparing performance across different representation formats:
- Bar charts for F1 scores by format
- Component-wise performance comparison
- Success rates and processing times
- Complexity-based performance breakdown
"""

import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")


class FormatComparisonAnalyzer:
    """Analyzes and visualizes format comparison results."""

    def __init__(self, results_dir: str = "./results/experiment_3_format_optimization_batched"):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory containing experiment 3 results
        """
        self.results_dir = Path(results_dir)
        self.comparison_file = self.results_dir / "format_comparison.json"

        if not self.comparison_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.comparison_file}")

        # Load results
        with open(self.comparison_file, "r") as f:
            self.comparison = json.load(f)

        self.formats = self.comparison["formats"]
        self.summary = self.comparison["comparison_summary"]

        # Create output directory for plots
        self.plots_dir = self.results_dir / "analysis_plots"
        self.plots_dir.mkdir(exist_ok=True)

        print(f"Loaded results from: {self.results_dir}")
        print(f"Number of formats: {len(self.formats)}")
        print(f"Output directory: {self.plots_dir}")

    def plot_overall_performance(self):
        """Create bar chart comparing overall Macro F1 scores."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data
        format_names = []
        f1_scores = []

        for item in self.summary["by_macro_f1"]:
            format_names.append(item["format"])
            f1_scores.append(item["macro_f1"])

        # Create bar chart
        bars = ax.bar(range(len(format_names)), f1_scores, alpha=0.8, edgecolor='black')

        # Color bars by performance
        colors = sns.color_palette("RdYlGn", len(bars))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Formatting
        ax.set_xlabel("Representation Format", fontsize=12, fontweight='bold')
        ax.set_ylabel("Macro F1 Score", fontsize=12, fontweight='bold')
        ax.set_title("Overall Performance Comparison Across Formats", fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(format_names)))
        ax.set_xticklabels(format_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, f1_scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.plots_dir / "overall_f1_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Saved: overall_f1_comparison.png")
        plt.close()

    def plot_component_performance(self):
        """Create grouped bar chart for component-wise performance."""
        fig, ax = plt.subplots(figsize=(12, 7))

        # Prepare data
        components = ["entities", "attributes", "relationships"]
        format_names = [item["format"] for item in self.summary["by_macro_f1"]]

        # Get F1 scores for each component and format
        data = {component: [] for component in components}

        for format_name in format_names:
            # Find this format's results
            format_key = None
            for key, val in self.formats.items():
                if val["format"] == format_name:
                    format_key = key
                    break

            if format_key:
                overall = self.formats[format_key]["overall_metrics"]
                for component in components:
                    data[component].append(overall[component]["f1"])

        # Set up bar positions
        x = np.arange(len(format_names))
        width = 0.25

        # Create bars
        bars1 = ax.bar(x - width, data["entities"], width, label='Entities', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, data["attributes"], width, label='Attributes', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, data["relationships"], width, label='Relationships', alpha=0.8, edgecolor='black')

        # Formatting
        ax.set_xlabel("Representation Format", fontsize=12, fontweight='bold')
        ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
        ax.set_title("Component-wise Performance Comparison", fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(format_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right', fontsize=10)

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.plots_dir / "component_performance.png", dpi=300, bbox_inches='tight')
        print(f"Saved: component_performance.png")
        plt.close()

    def plot_precision_recall_f1(self):
        """Create detailed precision/recall/F1 comparison for each component."""
        components = ["entities", "attributes", "relationships"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, component in enumerate(components):
            ax = axes[idx]

            # Prepare data
            format_names = []
            precision = []
            recall = []
            f1 = []

            for format_key, format_data in self.formats.items():
                format_names.append(format_data["format"])
                metrics = format_data["overall_metrics"][component]
                precision.append(metrics["precision"])
                recall.append(metrics["recall"])
                f1.append(metrics["f1"])

            # Set up bars
            x = np.arange(len(format_names))
            width = 0.25

            bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, edgecolor='black')
            bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, edgecolor='black')
            bars3 = ax.bar(x + width, f1, width, label='F1', alpha=0.8, edgecolor='black')

            # Formatting
            ax.set_xlabel("Format", fontsize=11, fontweight='bold')
            ax.set_ylabel("Score", fontsize=11, fontweight='bold')
            ax.set_title(f"{component.capitalize()}", fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(format_names, rotation=45, ha='right', fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

        fig.suptitle("Precision, Recall, and F1 by Component", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "precision_recall_f1.png", dpi=300, bbox_inches='tight')
        print(f"Saved: precision_recall_f1.png")
        plt.close()

    def plot_success_rate(self):
        """Create bar chart for extraction success rates."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data
        format_names = []
        success_rates = []

        for item in self.summary["by_success_rate"]:
            format_names.append(item["format"])
            success_rates.append(item["success_rate"] * 100)  # Convert to percentage

        # Create bar chart
        bars = ax.bar(range(len(format_names)), success_rates, alpha=0.8, edgecolor='black')

        # Color bars by success rate
        colors = sns.color_palette("RdYlGn", len(bars))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Formatting
        ax.set_xlabel("Representation Format", fontsize=12, fontweight='bold')
        ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight='bold')
        ax.set_title("Extraction Success Rate by Format", fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(format_names)))
        ax.set_xticklabels(format_names, rotation=45, ha='right')
        ax.set_ylim(0, 100)

        # Add value labels
        for bar, val in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Add reference line at 70% (success threshold)
        ax.axhline(y=70, color='red', linestyle='--', linewidth=2, alpha=0.5, label='70% Threshold')
        ax.legend(loc='lower right')

        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.plots_dir / "success_rate.png", dpi=300, bbox_inches='tight')
        print(f"Saved: success_rate.png")
        plt.close()

    def plot_processing_time(self):
        """Create bar chart for processing times."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data
        format_names = []
        avg_times = []

        for item in self.summary["by_total_time"]:
            format_names.append(item["format"])
            avg_times.append(item["avg_time_per_sample"])

        # Create bar chart
        bars = ax.bar(range(len(format_names)), avg_times, alpha=0.8, edgecolor='black', color='skyblue')

        # Formatting
        ax.set_xlabel("Representation Format", fontsize=12, fontweight='bold')
        ax.set_ylabel("Avg Time per Sample (seconds)", fontsize=12, fontweight='bold')
        ax.set_title("Processing Time Comparison", fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(format_names)))
        ax.set_xticklabels(format_names, rotation=45, ha='right')

        # Add value labels
        for bar, val in zip(bars, avg_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}s',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.plots_dir / "processing_time.png", dpi=300, bbox_inches='tight')
        print(f"Saved: processing_time.png")
        plt.close()

    def plot_complexity_breakdown(self):
        """Create stacked bar chart showing performance by caption complexity."""
        # Check if we have complexity data
        has_complexity_data = False
        for format_data in self.formats.values():
            if format_data.get("complexity_metrics"):
                has_complexity_data = True
                break

        if not has_complexity_data:
            print("No complexity metrics found, skipping complexity breakdown plot")
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        # Prepare data
        format_names = [data["format"] for data in self.formats.values()]
        complexities = set()

        # Collect all complexity levels
        for format_data in self.formats.values():
            if format_data.get("complexity_metrics"):
                complexities.update(format_data["complexity_metrics"].keys())

        complexities = sorted(list(complexities))

        # Build data matrix
        data = {complexity: [] for complexity in complexities}

        for format_data in self.formats.values():
            complexity_metrics = format_data.get("complexity_metrics", {})
            for complexity in complexities:
                if complexity in complexity_metrics:
                    f1 = complexity_metrics[complexity]["macro_f1"]
                    data[complexity].append(f1)
                else:
                    data[complexity].append(0)

        # Create grouped bars
        x = np.arange(len(format_names))
        width = 0.25
        offsets = np.linspace(-width, width, len(complexities))

        for i, complexity in enumerate(complexities):
            ax.bar(x + offsets[i], data[complexity], width * 0.9,
                  label=complexity.capitalize(), alpha=0.8, edgecolor='black')

        # Formatting
        ax.set_xlabel("Representation Format", fontsize=12, fontweight='bold')
        ax.set_ylabel("Macro F1 Score", fontsize=12, fontweight='bold')
        ax.set_title("Performance by Caption Complexity", fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(format_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.legend(title='Complexity', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.plots_dir / "complexity_breakdown.png", dpi=300, bbox_inches='tight')
        print(f"Saved: complexity_breakdown.png")
        plt.close()

    def create_summary_table(self):
        """Create a summary table with key metrics."""
        # Prepare data
        rows = []

        for format_key, format_data in self.formats.items():
            overall = format_data["overall_metrics"]
            row = {
                "Format": format_data["format"],
                "Macro F1": f"{overall['macro_f1']:.3f}",
                "Entities F1": f"{overall['entities']['f1']:.3f}",
                "Attributes F1": f"{overall['attributes']['f1']:.3f}",
                "Relationships F1": f"{overall['relationships']['f1']:.3f}",
                "Success Rate": f"{format_data['success_rate']:.1%}",
                "Avg Time (s)": f"{format_data['avg_inference_time_per_sample']:.3f}"
            }
            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Sort by Macro F1
        df['_sort_key'] = df['Macro F1'].astype(float)
        df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)

        # Save as CSV
        csv_path = self.plots_dir / "summary_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: summary_table.csv")

        # Create styled table image
        fig, ax = plt.subplots(figsize=(14, len(df) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colWidths=[0.18] * len(df.columns))

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')

        plt.title("Format Comparison Summary Table", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "summary_table.png", dpi=300, bbox_inches='tight')
        print(f"Saved: summary_table.png")
        plt.close()

        return df

    def generate_all_plots(self):
        """Generate all analysis plots."""
        print("\n" + "="*80)
        print("GENERATING ANALYSIS PLOTS")
        print("="*80 + "\n")

        print("Creating plots...")
        self.plot_overall_performance()
        self.plot_component_performance()
        self.plot_precision_recall_f1()
        self.plot_success_rate()
        self.plot_processing_time()
        self.plot_complexity_breakdown()

        print("\nCreating summary table...")
        df = self.create_summary_table()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll plots saved to: {self.plots_dir}")
        print("\nGenerated files:")
        print("  - overall_f1_comparison.png")
        print("  - component_performance.png")
        print("  - precision_recall_f1.png")
        print("  - success_rate.png")
        print("  - processing_time.png")
        print("  - complexity_breakdown.png")
        print("  - summary_table.png")
        print("  - summary_table.csv")

        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze format comparison results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results/experiment_3_format_optimization_batched",
        help="Directory containing experiment results"
    )

    args = parser.parse_args()

    try:
        analyzer = FormatComparisonAnalyzer(results_dir=args.results_dir)
        analyzer.generate_all_plots()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you've run experiment 3 first:")
        print("  uv run experiment_3_format_optimization_batched.py")
        print("\nOr specify the correct results directory:")
        print("  uv run analyze_format_comparison.py --results-dir ./results/experiment_3_format_optimization")


if __name__ == "__main__":
    main()
