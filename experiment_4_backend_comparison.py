"""
Experiment 4: Backend Comparison - LangExtract vs Native Gemini Structured Output

Compares two approaches for extracting scene graphs from image captions:
1. LangExtract (Gemini backend) - Few-shot learning framework with examples
2. Native Gemini Structured Output - Direct API with response_schema

Both use the JSON Structured format (winner from Experiment 3).
Tests on 50-100 diverse samples from FACTUAL test set.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv

# Import langextract for comparison
import langextract as lx
from langextract.data import ExampleData, Extraction, Document

# Import Google Generative AI for native structured output
import google.generativeai as genai

# Import shared components
from factual_metrics import (
    EvaluationMetrics,
    parse_ground_truth_factual,
    compute_factual_metrics,
    aggregate_factual_metrics
)

load_dotenv()


# ============================================================================
# Helper Functions (from experiment_1_t5_baseline.py - without loading T5 model)
# ============================================================================

def load_factual_dataset(split: str = "train", num_samples: int = 100, test_split: bool = True, use_complex_only: bool = False):
    """Load FACTUAL dataset without requiring T5 model."""
    dataset = load_dataset("lizhuang144/FACTUAL_Scene_Graph", split=split, cache_dir="./cache")

    if test_split:
        dataset = dataset.shuffle(seed=42).select(range(len(dataset) // 10))

    if use_complex_only:
        # Filter to complex captions (>20 words)
        dataset = dataset.filter(lambda x: len(x.get("caption", "").split()) > 20)

    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    return list(dataset)


def classify_complexity(caption: str) -> str:
    """Classify caption complexity without requiring T5 model."""
    word_count = len(caption.split())

    if word_count <= 10:
        return "simple"
    elif word_count <= 20:
        return "medium"
    else:
        return "complex"


# ============================================================================
# Schema Definitions for Gemini Structured Output
# ============================================================================

# Define the schema using type hints for Gemini's response_schema
@dataclass
class Entity:
    """An entity (object, person, thing) in the scene."""
    name: str


@dataclass
class Attribute:
    """An attribute describing an entity's properties."""
    entity: str
    attribute: str


@dataclass
class Relationship:
    """A relationship between two entities."""
    subject: str
    predicate: str
    object: str


@dataclass
class SceneGraphExtraction:
    """Complete scene graph extraction with entities, attributes, and relationships."""
    entities: List[Entity]
    attributes: List[Attribute]
    relationships: List[Relationship]


# ============================================================================
# LangExtract Examples (from Experiment 3 winner - JSON Structured)
# ============================================================================

def create_langextract_examples() -> List[ExampleData]:
    """Create few-shot examples for LangExtract JSON Structured format."""
    examples = []

    # Example 1: Complex scene
    ex1 = ExampleData(
        text="A large brown dog with a red collar sitting on a wooden bench in a park next to a metal trash can while a small bird perches on the bench arm.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="dog",
                      attributes={"name": "dog"}),
            Extraction(extraction_class="entity", extraction_text="collar",
                      attributes={"name": "collar"}),
            Extraction(extraction_class="entity", extraction_text="bench",
                      attributes={"name": "bench"}),
            Extraction(extraction_class="entity", extraction_text="park",
                      attributes={"name": "park"}),
            Extraction(extraction_class="entity", extraction_text="trash can",
                      attributes={"name": "trash can"}),
            Extraction(extraction_class="entity", extraction_text="bird",
                      attributes={"name": "bird"}),
            Extraction(extraction_class="entity", extraction_text="bench arm",
                      attributes={"name": "bench arm"}),
            Extraction(extraction_class="attribute", extraction_text="large",
                      attributes={"entity": "dog", "attribute": "large"}),
            Extraction(extraction_class="attribute", extraction_text="brown",
                      attributes={"entity": "dog", "attribute": "brown"}),
            Extraction(extraction_class="attribute", extraction_text="sitting",
                      attributes={"entity": "dog", "attribute": "sitting"}),
            Extraction(extraction_class="attribute", extraction_text="red",
                      attributes={"entity": "collar", "attribute": "red"}),
            Extraction(extraction_class="attribute", extraction_text="wooden",
                      attributes={"entity": "bench", "attribute": "wooden"}),
            Extraction(extraction_class="attribute", extraction_text="metal",
                      attributes={"entity": "trash can", "attribute": "metal"}),
            Extraction(extraction_class="attribute", extraction_text="small",
                      attributes={"entity": "bird", "attribute": "small"}),
            Extraction(extraction_class="attribute", extraction_text="perches",
                      attributes={"entity": "bird", "attribute": "perches"}),
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

    # Example 2: Predicate normalization
    ex2 = ExampleData(
        text="A young woman wearing a blue dress holds a cup of coffee while talking on her phone standing beside a black car.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="woman",
                      attributes={"name": "woman"}),
            Extraction(extraction_class="entity", extraction_text="dress",
                      attributes={"name": "dress"}),
            Extraction(extraction_class="entity", extraction_text="cup of coffee",
                      attributes={"name": "cup of coffee"}),
            Extraction(extraction_class="entity", extraction_text="phone",
                      attributes={"name": "phone"}),
            Extraction(extraction_class="entity", extraction_text="car",
                      attributes={"name": "car"}),
            Extraction(extraction_class="attribute", extraction_text="young",
                      attributes={"entity": "woman", "attribute": "young"}),
            Extraction(extraction_class="attribute", extraction_text="blue",
                      attributes={"entity": "dress", "attribute": "blue"}),
            Extraction(extraction_class="attribute", extraction_text="standing",
                      attributes={"entity": "woman", "attribute": "standing"}),
            Extraction(extraction_class="attribute", extraction_text="black",
                      attributes={"entity": "car", "attribute": "black"}),
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

    return examples


# ============================================================================
# Evaluator Classes
# ============================================================================

class LangExtractEvaluator:
    """Evaluator using LangExtract with Gemini backend."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        cache_dir: str = "./cache",
        batch_length: int = 50,
        max_workers: int = 10
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.batch_length = batch_length
        self.max_workers = max_workers

        # Check for API key
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or LANGEXTRACT_API_KEY not found")
        self.api_key = api_key

        # Create examples
        self.examples = create_langextract_examples()

        print(f"Initialized LangExtract with model: {model_name}")
        print(f"Using {len(self.examples)} few-shot examples")

    def extract_batch(self, captions: List[str]) -> Tuple[List[Dict], float]:
        """Extract scene graphs using LangExtract."""
        try:
            documents = [Document(text=caption) for caption in captions]
            start_time = time.time()

            results = lx.extract(
                text_or_documents=documents,
                prompt_description=(
                    "Extract entities (objects, people, things), attributes (properties like color, size, state), "
                    "and relationships (interactions between entities) from the text. "
                    "For attributes, specify which entity they describe. "
                    "For relationships, specify subject, predicate, and object."
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

            # Convert to dictionary format
            all_extractions = []
            for result in results:
                extraction = {"entities": [], "attributes": [], "relationships": []}
                if result.extractions:
                    for ext in result.extractions:
                        if ext.extraction_class == "entity":
                            if ext.attributes and "name" in ext.attributes:
                                extraction["entities"].append({"name": ext.attributes["name"]})
                        elif ext.extraction_class == "attribute":
                            if ext.attributes and "entity" in ext.attributes and "attribute" in ext.attributes:
                                extraction["attributes"].append({
                                    "entity": ext.attributes["entity"],
                                    "attribute": ext.attributes["attribute"]
                                })
                        elif ext.extraction_class == "relationship":
                            if ext.attributes and all(k in ext.attributes for k in ["subject", "predicate", "object"]):
                                extraction["relationships"].append({
                                    "subject": ext.attributes["subject"],
                                    "predicate": ext.attributes["predicate"],
                                    "object": ext.attributes["object"]
                                })
                all_extractions.append(extraction)

            return all_extractions, total_time

        except Exception as e:
            print(f"LangExtract batch error: {e}")
            return [{"entities": [], "attributes": [], "relationships": []} for _ in captions], 0.0


class NativeGeminiEvaluator:
    """Evaluator using native Gemini API with structured output."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        cache_dir: str = "./cache"
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Check for API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        # Store API key for thread-safe model creation
        self.api_key = api_key

        # Configure Gemini globally
        genai.configure(api_key=api_key)

        # Store model config for batch processing - array of scene graphs
        self.model_config = {
            "temperature": 0.3,
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entities": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"}
                                        },
                                        "required": ["name"]
                                    }
                                },
                                "attributes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "entity": {"type": "string"},
                                            "attribute": {"type": "string"}
                                        },
                                        "required": ["entity", "attribute"]
                                    }
                                },
                                "relationships": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "subject": {"type": "string"},
                                            "predicate": {"type": "string"},
                                            "object": {"type": "string"}
                                        },
                                        "required": ["subject", "predicate", "object"]
                                    }
                                }
                            },
                            "required": ["entities", "attributes", "relationships"]
                        }
                    }
                },
                "required": ["results"]
            }
        }

        # Create model instance for batch processing
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.model_config
        )

        print(f"Initialized Native Gemini with model: {model_name}")
        print(f"Using structured output schema with true batch processing")

    def extract_batch(self, captions: List[str]) -> Tuple[List[Dict], float]:
        """Extract scene graphs using native Gemini structured output with true batch processing."""
        start_time = time.time()

        # Create prompt with instructions and examples
        system_prompt = """Extract a structured scene graph from the image caption.

INSTRUCTIONS:
1. Entities: Identify all objects, people, and things mentioned (e.g., "dog", "bench", "woman")
2. Attributes: Extract properties describing entities (color, size, state, action)
   - Link each attribute to its entity
   - Use single-word attributes when possible
3. Relationships: Extract spatial and action relationships between entities
   - Use base verb forms (e.g., "wear" not "wearing", "hold" not "holds")
   - Use standard spatial predicates ("on", "in", "next to", "beside")

Extract core entity names without descriptive prefixes (e.g., "area" not "sandy area").

EXAMPLES:

Example 1:
Caption: "A large brown dog with a red collar sitting on a wooden bench in a park next to a metal trash can while a small bird perches on the bench arm."

Output:
{
  "entities": [
    {"name": "dog"},
    {"name": "collar"},
    {"name": "bench"},
    {"name": "park"},
    {"name": "trash can"},
    {"name": "bird"},
    {"name": "bench arm"}
  ],
  "attributes": [
    {"entity": "dog", "attribute": "large"},
    {"entity": "dog", "attribute": "brown"},
    {"entity": "dog", "attribute": "sitting"},
    {"entity": "collar", "attribute": "red"},
    {"entity": "bench", "attribute": "wooden"},
    {"entity": "trash can", "attribute": "metal"},
    {"entity": "bird", "attribute": "small"},
    {"entity": "bird", "attribute": "perches"}
  ],
  "relationships": [
    {"subject": "dog", "predicate": "with", "object": "collar"},
    {"subject": "dog", "predicate": "sit on", "object": "bench"},
    {"subject": "bench", "predicate": "in", "object": "park"},
    {"subject": "bench", "predicate": "next to", "object": "trash can"},
    {"subject": "bird", "predicate": "perch on", "object": "bench arm"}
  ]
}

Example 2:
Caption: "A young woman wearing a blue dress holds a cup of coffee while talking on her phone standing beside a black car."

Output:
{
  "entities": [
    {"name": "woman"},
    {"name": "dress"},
    {"name": "cup of coffee"},
    {"name": "phone"},
    {"name": "car"}
  ],
  "attributes": [
    {"entity": "woman", "attribute": "young"},
    {"entity": "dress", "attribute": "blue"},
    {"entity": "woman", "attribute": "standing"},
    {"entity": "car", "attribute": "black"}
  ],
  "relationships": [
    {"subject": "woman", "predicate": "wear", "object": "dress"},
    {"subject": "woman", "predicate": "hold", "object": "cup of coffee"},
    {"subject": "woman", "predicate": "talk on", "object": "phone"},
    {"subject": "woman", "predicate": "beside", "object": "car"}
  ]
}

Now extract scene graphs from ALL of the following captions. Return a JSON array with one result per caption, in the same order:"""

        # Build batch prompt with all captions numbered
        batch_prompt = system_prompt + "\n\n"
        for i, caption in enumerate(captions, 1):
            batch_prompt += f"\nCaption {i}: {caption}"

        batch_prompt += "\n\nReturn a JSON object with a 'results' array containing one scene graph extraction for each caption, in order."

        try:
            # Single API call for all captions (true batch processing)
            response = self.model.generate_content(batch_prompt)

            # Parse JSON response
            result = json.loads(response.text)
            all_extractions = result.get("results", [])

            # Ensure we have the right number of results
            if len(all_extractions) != len(captions):
                print(f"Warning: Expected {len(captions)} results, got {len(all_extractions)}")
                # Pad with empty results if needed
                while len(all_extractions) < len(captions):
                    all_extractions.append({"entities": [], "attributes": [], "relationships": []})

            total_time = time.time() - start_time
            return all_extractions, total_time

        except Exception as e:
            print(f"Batch extraction error: {e}")
            # Return empty results for all samples
            total_time = time.time() - start_time
            return [{"entities": [], "attributes": [], "relationships": []} for _ in captions], total_time


# ============================================================================
# Shared Evaluation Logic
# ============================================================================

def convert_to_factual(extraction: Dict) -> Tuple[Set[str], Set[Tuple[str, str, str]], Set[Tuple[str, str, str]]]:
    """Convert extraction dictionary to FACTUAL format."""
    entities = set()
    attribute_triplets = set()
    relationship_triplets = set()

    # Extract entities
    for entity_obj in extraction.get("entities", []):
        if isinstance(entity_obj, dict) and "name" in entity_obj:
            entities.add(entity_obj["name"])

    # Extract attributes
    for attr_obj in extraction.get("attributes", []):
        if isinstance(attr_obj, dict) and "entity" in attr_obj and "attribute" in attr_obj:
            entity = attr_obj["entity"]
            attribute = attr_obj["attribute"]
            attribute_triplets.add((entity, "has_attribute", attribute))
            entities.add(entity)

    # Extract relationships
    for rel_obj in extraction.get("relationships", []):
        if isinstance(rel_obj, dict) and all(k in rel_obj for k in ["subject", "predicate", "object"]):
            subject = rel_obj["subject"]
            predicate = rel_obj["predicate"]
            obj = rel_obj["object"]
            relationship_triplets.add((subject, predicate, obj))
            entities.add(subject)
            entities.add(obj)

    return entities, attribute_triplets, relationship_triplets


def evaluate_approach(
    evaluator,
    approach_name: str,
    samples: List[Dict],
    save_dir: str
) -> Dict:
    """Evaluate an approach on samples."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Evaluating {approach_name}")
    print(f"{'='*80}\n")

    # Extract captions and ground truth
    captions = [sample.get("caption", "") for sample in samples]
    ground_truth_strs = [sample.get("scene_graph", "") for sample in samples]

    # Extract
    print(f"Processing {len(captions)} samples...")
    all_extracted, total_time = evaluator.extract_batch(captions)
    avg_time = total_time / len(captions) if captions else 0.0
    print(f"Completed in {total_time:.2f}s ({avg_time:.3f}s per sample)")

    # Evaluate
    all_metrics = []
    complexity_groups = defaultdict(list)
    detailed_results = []
    extraction_success_count = 0

    for i, (caption, gt_str, extracted) in enumerate(zip(captions, ground_truth_strs, all_extracted)):
        # Convert to FACTUAL
        pred_entities, pred_attrs, pred_rels = convert_to_factual(extracted)
        gt_entities, gt_attrs, gt_rels = parse_ground_truth_factual(gt_str)

        # Check success
        has_results = len(pred_entities) > 0 or len(pred_attrs) > 0 or len(pred_rels) > 0
        if has_results:
            extraction_success_count += 1

        # Compute metrics
        metrics = compute_factual_metrics(pred_entities, pred_attrs, pred_rels, gt_entities, gt_attrs, gt_rels)
        all_metrics.append(metrics)

        # Complexity
        complexity = classify_complexity(caption)
        complexity_groups[complexity].append(metrics)

        # Store details
        detailed_results.append({
            "sample_id": i,
            "caption": caption,
            "complexity": complexity,
            "ground_truth": gt_str,
            "predicted": {
                "entities": list(pred_entities),
                "attributes": [list(attr) for attr in pred_attrs],
                "relationships": [list(rel) for rel in pred_rels]
            },
            "metrics": {k: asdict(v) for k, v in metrics.items()},
            "extraction_success": has_results
        })

    # Aggregate
    overall_metrics = aggregate_factual_metrics(all_metrics)
    complexity_metrics = {
        complexity: aggregate_factual_metrics(metrics_list)
        for complexity, metrics_list in complexity_groups.items()
    }

    extraction_success_rate = extraction_success_count / len(samples) if samples else 0.0

    # Compile results
    results = {
        "approach": approach_name,
        "model": evaluator.model_name,
        "num_samples": len(samples),
        "extraction_success_rate": extraction_success_rate,
        "extraction_success_count": extraction_success_count,
        "overall_metrics": overall_metrics,
        "complexity_metrics": complexity_metrics,
        "total_inference_time": total_time,
        "avg_inference_time_per_sample": avg_time,
        "detailed_results": detailed_results
    }

    # Save
    with open(save_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {save_path / 'results.json'}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"{approach_name} - EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Samples: {len(samples)}")
    print(f"Success Rate: {extraction_success_rate:.1%}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time/Sample: {avg_time:.3f}s")
    print(f"\nMacro F1: {overall_metrics['macro_f1']:.4f}")
    for comp in ["entities", "attributes", "relationships"]:
        m = overall_metrics[comp]
        print(f"  {comp}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")

    return results


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    """Run Experiment 4: LangExtract vs Native Gemini comparison."""

    print("\n" + "="*80)
    print("EXPERIMENT 4: LANGEXTRACT VS NATIVE GEMINI STRUCTURED OUTPUT")
    print("="*80 + "\n")

    # Load samples (same as experiments 2 & 3 - complex only)
    all_samples = load_factual_dataset(
        split="train",
        num_samples=100,
        test_split=True,
        use_complex_only=True
    )

    # Select 50 diverse samples
    indices = np.linspace(0, len(all_samples) - 1, 50, dtype=int)
    samples = [all_samples[int(i)] for i in indices]
    print(f"Selected {len(samples)} diverse samples\n")

    # Initialize evaluators
    langextract_eval = LangExtractEvaluator(
        model_name="gemini-2.5-flash",
        batch_length=50,
        max_workers=10
    )

    native_gemini_eval = NativeGeminiEvaluator(
        model_name="gemini-2.5-flash"
    )

    # Run evaluations
    langextract_results = evaluate_approach(
        langextract_eval,
        "LangExtract (Few-Shot Learning)",
        samples,
        "./results/experiment_4_backend_comparison/langextract"
    )

    native_results = evaluate_approach(
        native_gemini_eval,
        "Native Gemini (Structured Output)",
        samples,
        "./results/experiment_4_backend_comparison/native_gemini"
    )

    # Create comparison
    comparison = {
        "experiment": "Experiment 4: LangExtract vs Native Gemini",
        "num_samples": len(samples),
        "approaches": {
            "langextract": {
                "macro_f1": langextract_results["overall_metrics"]["macro_f1"],
                "entities_f1": langextract_results["overall_metrics"]["entities"]["f1"],
                "attributes_f1": langextract_results["overall_metrics"]["attributes"]["f1"],
                "relationships_f1": langextract_results["overall_metrics"]["relationships"]["f1"],
                "avg_time_per_sample": langextract_results["avg_inference_time_per_sample"],
                "success_rate": langextract_results["extraction_success_rate"]
            },
            "native_gemini": {
                "macro_f1": native_results["overall_metrics"]["macro_f1"],
                "entities_f1": native_results["overall_metrics"]["entities"]["f1"],
                "attributes_f1": native_results["overall_metrics"]["attributes"]["f1"],
                "relationships_f1": native_results["overall_metrics"]["relationships"]["f1"],
                "avg_time_per_sample": native_results["avg_inference_time_per_sample"],
                "success_rate": native_results["extraction_success_rate"]
            }
        },
        "winner": "langextract" if langextract_results["overall_metrics"]["macro_f1"] > native_results["overall_metrics"]["macro_f1"] else "native_gemini"
    }

    # Save comparison
    comparison_path = Path("./results/experiment_4_backend_comparison/comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # Print comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<25} {'LangExtract':<20} {'Native Gemini':<20} {'Difference':<15}")
    print("-"*80)

    le_f1 = comparison["approaches"]["langextract"]["macro_f1"]
    ng_f1 = comparison["approaches"]["native_gemini"]["macro_f1"]
    print(f"{'Macro F1':<25} {le_f1:<20.4f} {ng_f1:<20.4f} {le_f1-ng_f1:<+15.4f}")

    for comp in ["entities", "attributes", "relationships"]:
        le_val = comparison["approaches"]["langextract"][f"{comp}_f1"]
        ng_val = comparison["approaches"]["native_gemini"][f"{comp}_f1"]
        print(f"{f'{comp.capitalize()} F1':<25} {le_val:<20.4f} {ng_val:<20.4f} {le_val-ng_val:<+15.4f}")

    le_time = comparison["approaches"]["langextract"]["avg_time_per_sample"]
    ng_time = comparison["approaches"]["native_gemini"]["avg_time_per_sample"]
    print(f"{'Avg Time (s)':<25} {le_time:<20.3f} {ng_time:<20.3f} {le_time-ng_time:<+15.3f}")

    print(f"\n{'Winner':<25} {comparison['winner'].upper()}")
    print("="*80)


if __name__ == "__main__":
    main()
