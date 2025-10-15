"""
Experiment 3: Format Optimization (BATCHED VERSION)

Tests 4 different representation formats for scene graph extraction:
1. Flat Entities - Separate classes: entity, attribute, relationship
2. Tuple Format - Direct FACTUAL format: (subject, predicate, object)
3. Hierarchical - Objects with nested properties + relationships
4. JSON Structured - Clean nested JSON with entities/attributes/relationships

Uses Google's LangExtract with Gemini to extract scene graphs.
Tests on 40-50 diverse samples from the same test set as Experiment 1.

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

# Import FACTUAL metrics utilities
from factual_metrics import (
    EvaluationMetrics,
    convert_flat_entities_to_factual,
    convert_tuple_format_to_factual,
    convert_hierarchical_to_factual,
    convert_json_structured_to_factual,
    parse_ground_truth_factual,
    compute_factual_metrics,
    aggregate_factual_metrics
)

load_dotenv()


# ============================================================================
# Helper Functions (without loading T5 model)
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
# FORMAT 1: FLAT ENTITIES
# ============================================================================

def create_flat_entities_examples() -> List[ExampleData]:
    """Create few-shot examples for Flat Entities format."""
    examples = []

    # Example 1: Complex scene
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

    # Example 2: Predicate normalization
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


def convert_flat_entities_extractions(result) -> Dict:
    """Convert AnnotatedDocument extractions to Flat Entities dictionary format."""
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

    return extractions


# ============================================================================
# FORMAT 2: TUPLE FORMAT
# ============================================================================

def create_tuple_format_examples() -> List[ExampleData]:
    """Create few-shot examples for Tuple Format."""
    examples = []

    # Example 1: Complex scene
    ex1 = ExampleData(
        text="A large brown dog with a red collar sitting on a wooden bench in a park next to a metal trash can while a small bird perches on the bench arm.",
        extractions=[
            Extraction(extraction_class="triplet", extraction_text="dog has large",
                      attributes={"subject": "dog", "predicate": "has", "object": "large"}),
            Extraction(extraction_class="triplet", extraction_text="dog has brown",
                      attributes={"subject": "dog", "predicate": "has", "object": "brown"}),
            Extraction(extraction_class="triplet", extraction_text="dog has sitting",
                      attributes={"subject": "dog", "predicate": "has", "object": "sitting"}),
            Extraction(extraction_class="triplet", extraction_text="collar has red",
                      attributes={"subject": "collar", "predicate": "has", "object": "red"}),
            Extraction(extraction_class="triplet", extraction_text="bench has wooden",
                      attributes={"subject": "bench", "predicate": "has", "object": "wooden"}),
            Extraction(extraction_class="triplet", extraction_text="trash can has metal",
                      attributes={"subject": "trash can", "predicate": "has", "object": "metal"}),
            Extraction(extraction_class="triplet", extraction_text="bird has small",
                      attributes={"subject": "bird", "predicate": "has", "object": "small"}),
            Extraction(extraction_class="triplet", extraction_text="bird has perches",
                      attributes={"subject": "bird", "predicate": "has", "object": "perches"}),
            Extraction(extraction_class="triplet", extraction_text="dog with collar",
                      attributes={"subject": "dog", "predicate": "with", "object": "collar"}),
            Extraction(extraction_class="triplet", extraction_text="dog sit on bench",
                      attributes={"subject": "dog", "predicate": "sit on", "object": "bench"}),
            Extraction(extraction_class="triplet", extraction_text="bench in park",
                      attributes={"subject": "bench", "predicate": "in", "object": "park"}),
            Extraction(extraction_class="triplet", extraction_text="bench next to trash can",
                      attributes={"subject": "bench", "predicate": "next to", "object": "trash can"}),
            Extraction(extraction_class="triplet", extraction_text="bird perch on bench arm",
                      attributes={"subject": "bird", "predicate": "perch on", "object": "bench arm"}),
        ]
    )
    examples.append(ex1)

    # Example 2: Predicate normalization
    ex2 = ExampleData(
        text="A young woman wearing a blue dress holds a cup of coffee while talking on her phone standing beside a black car.",
        extractions=[
            Extraction(extraction_class="triplet", extraction_text="woman has young",
                      attributes={"subject": "woman", "predicate": "has", "object": "young"}),
            Extraction(extraction_class="triplet", extraction_text="dress has blue",
                      attributes={"subject": "dress", "predicate": "has", "object": "blue"}),
            Extraction(extraction_class="triplet", extraction_text="woman has standing",
                      attributes={"subject": "woman", "predicate": "has", "object": "standing"}),
            Extraction(extraction_class="triplet", extraction_text="car has black",
                      attributes={"subject": "car", "predicate": "has", "object": "black"}),
            Extraction(extraction_class="triplet", extraction_text="woman wear dress",
                      attributes={"subject": "woman", "predicate": "wear", "object": "dress"}),
            Extraction(extraction_class="triplet", extraction_text="woman hold cup of coffee",
                      attributes={"subject": "woman", "predicate": "hold", "object": "cup of coffee"}),
            Extraction(extraction_class="triplet", extraction_text="woman talk on phone",
                      attributes={"subject": "woman", "predicate": "talk on", "object": "phone"}),
            Extraction(extraction_class="triplet", extraction_text="woman beside car",
                      attributes={"subject": "woman", "predicate": "beside", "object": "car"}),
        ]
    )
    examples.append(ex2)

    return examples


def convert_tuple_format_extractions(result) -> List[Tuple[str, str, str]]:
    """Convert AnnotatedDocument extractions to Tuple Format list."""
    triplets = []

    if result.extractions:
        for ext in result.extractions:
            if ext.extraction_class == "triplet":
                if ext.attributes and all(k in ext.attributes for k in ["subject", "predicate", "object"]):
                    triplets.append((
                        ext.attributes["subject"],
                        ext.attributes["predicate"],
                        ext.attributes["object"]
                    ))

    return triplets


# ============================================================================
# FORMAT 3: HIERARCHICAL
# ============================================================================

def create_hierarchical_examples() -> List[ExampleData]:
    """Create few-shot examples for Hierarchical format."""
    examples = []

    # Example 1: Complex scene
    ex1 = ExampleData(
        text="A large brown dog with a red collar sitting on a wooden bench in a park next to a metal trash can while a small bird perches on the bench arm.",
        extractions=[
            Extraction(extraction_class="object", extraction_text="dog",
                      attributes={"name": "dog", "attributes": ["large", "brown", "sitting"]}),
            Extraction(extraction_class="object", extraction_text="collar",
                      attributes={"name": "collar", "attributes": ["red"]}),
            Extraction(extraction_class="object", extraction_text="bench",
                      attributes={"name": "bench", "attributes": ["wooden"]}),
            Extraction(extraction_class="object", extraction_text="park",
                      attributes={"name": "park", "attributes": []}),
            Extraction(extraction_class="object", extraction_text="trash can",
                      attributes={"name": "trash can", "attributes": ["metal"]}),
            Extraction(extraction_class="object", extraction_text="bird",
                      attributes={"name": "bird", "attributes": ["small", "perches"]}),
            Extraction(extraction_class="object", extraction_text="bench arm",
                      attributes={"name": "bench arm", "attributes": []}),
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
            Extraction(extraction_class="object", extraction_text="woman",
                      attributes={"name": "woman", "attributes": ["young", "standing"]}),
            Extraction(extraction_class="object", extraction_text="dress",
                      attributes={"name": "dress", "attributes": ["blue"]}),
            Extraction(extraction_class="object", extraction_text="cup of coffee",
                      attributes={"name": "cup of coffee", "attributes": []}),
            Extraction(extraction_class="object", extraction_text="phone",
                      attributes={"name": "phone", "attributes": []}),
            Extraction(extraction_class="object", extraction_text="car",
                      attributes={"name": "car", "attributes": ["black"]}),
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


def convert_hierarchical_extractions(result) -> Dict:
    """Convert AnnotatedDocument extractions to Hierarchical format."""
    objects = []
    relationships = []

    if result.extractions:
        for ext in result.extractions:
            if ext.extraction_class == "object":
                if ext.attributes and "name" in ext.attributes:
                    objects.append({
                        "name": ext.attributes["name"],
                        "attributes": ext.attributes.get("attributes", [])
                    })
            elif ext.extraction_class == "relationship":
                if ext.attributes and all(k in ext.attributes for k in ["subject", "predicate", "object"]):
                    relationships.append({
                        "subject": ext.attributes["subject"],
                        "predicate": ext.attributes["predicate"],
                        "object": ext.attributes["object"]
                    })

    return {"objects": objects, "relationships": relationships}


# ============================================================================
# FORMAT 4: JSON STRUCTURED
# ============================================================================

def create_json_structured_examples() -> List[ExampleData]:
    """Create few-shot examples for JSON Structured format."""
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


def convert_json_structured_extractions(result) -> Dict:
    """Convert AnnotatedDocument extractions to JSON Structured format."""
    entities = []
    attributes = []
    relationships = []

    if result.extractions:
        for ext in result.extractions:
            if ext.extraction_class == "entity":
                if ext.attributes and "name" in ext.attributes:
                    entities.append({"name": ext.attributes["name"]})
            elif ext.extraction_class == "attribute":
                if ext.attributes and "entity" in ext.attributes and "attribute" in ext.attributes:
                    attributes.append({
                        "entity": ext.attributes["entity"],
                        "attribute": ext.attributes["attribute"]
                    })
            elif ext.extraction_class == "relationship":
                if ext.attributes and all(k in ext.attributes for k in ["subject", "predicate", "object"]):
                    relationships.append({
                        "subject": ext.attributes["subject"],
                        "predicate": ext.attributes["predicate"],
                        "object": ext.attributes["object"]
                    })

    return {"entities": entities, "attributes": attributes, "relationships": relationships}


# ============================================================================
# FORMAT CONFIGURATIONS
# ============================================================================

FORMAT_CONFIGS = {
    "flat_entities": {
        "name": "Flat Entities",
        "description": "Separate classes: entity, attribute, relationship",
        "examples_fn": create_flat_entities_examples,
        "converter_fn": convert_flat_entities_extractions,
        "to_factual_fn": convert_flat_entities_to_factual,
        "prompt": (
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
        )
    },
    "tuple_format": {
        "name": "Tuple Format",
        "description": "Direct FACTUAL format: (subject, predicate, object)",
        "examples_fn": create_tuple_format_examples,
        "converter_fn": convert_tuple_format_extractions,
        "to_factual_fn": convert_tuple_format_to_factual,
        "prompt": (
            "Extract scene graph triplets in the format (subject, predicate, object). "
            "For attributes, use predicates 'has', 'is', or 'are'. "
            "For relationships, use appropriate action or spatial predicates. "
            "\n\nIMPORTANT RULES:\n"
            "1. Use base verb forms in predicates (wear, hold, watch), NOT present participles (wearing, holding, watching)\n"
            "2. Extract entity names as they appear, without possessive forms (head, not man's head)\n"
            "3. Extract core entities only, not descriptive phrases (area, not sandy area)\n"
            "4. Use standard spatial predicates: 'at the left of', 'on the right side of'\n"
            "5. Use 'has' for attributes: (dog, has, large), (collar, has, red)"
        )
    },
    "hierarchical": {
        "name": "Hierarchical",
        "description": "Objects with nested properties + relationships",
        "examples_fn": create_hierarchical_examples,
        "converter_fn": convert_hierarchical_extractions,
        "to_factual_fn": convert_hierarchical_to_factual,
        "prompt": (
            "Extract objects with their nested attributes, and separate relationships between objects. "
            "Each object should have a name and a list of attributes. "
            "Relationships should specify subject, predicate, and object. "
            "\n\nIMPORTANT RULES:\n"
            "1. Use base verb forms in relationship predicates (wear, hold, watch), NOT present participles\n"
            "2. Extract entity names as they appear, without possessive forms\n"
            "3. Extract core entity names only, not descriptive phrases\n"
            "4. Group all attributes of an object in its attributes list\n"
            "5. Use standard spatial and action predicates in relationships"
        )
    },
    "json_structured": {
        "name": "JSON Structured",
        "description": "Clean nested JSON with entities/attributes/relationships",
        "examples_fn": create_json_structured_examples,
        "converter_fn": convert_json_structured_extractions,
        "to_factual_fn": convert_json_structured_to_factual,
        "prompt": (
            "Extract structured scene graph with separate entities, attributes, and relationships. "
            "Entities have a 'name' field. "
            "Attributes have 'entity' and 'attribute' fields. "
            "Relationships have 'subject', 'predicate', and 'object' fields. "
            "\n\nIMPORTANT RULES:\n"
            "1. Use base verb forms in predicates (wear, hold, watch), NOT present participles\n"
            "2. Extract entity names as they appear, without possessive forms\n"
            "3. Extract core entities only, not descriptive phrases\n"
            "4. Link attributes to their entities using the entity name\n"
            "5. Use standard spatial and action predicates"
        )
    }
}


# ============================================================================
# BATCHED FORMAT EVALUATOR
# ============================================================================

class BatchedFormatEvaluator:
    """Evaluates different representation formats on FACTUAL dataset using batch processing."""

    def __init__(
        self,
        format_key: str,
        model_name: str = "gemini-2.5-flash",
        cache_dir: str = "./cache",
        batch_length: int = 50,
        max_workers: int = 10
    ):
        """
        Initialize the evaluator for a specific format.

        Args:
            format_key: Key identifying the format configuration
            model_name: Gemini model identifier
            cache_dir: Directory to cache datasets
            batch_length: Number of samples per batch
            max_workers: Maximum parallel workers
        """
        if format_key not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format: {format_key}. Must be one of {list(FORMAT_CONFIGS.keys())}")

        self.format_key = format_key
        self.format_config = FORMAT_CONFIGS[format_key]
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
        self.examples = self.format_config["examples_fn"]()

        print(f"Initialized {self.format_config['name']} format evaluator")
        print(f"Using {len(self.examples)} few-shot examples")
        print(f"Batch settings: batch_length={batch_length}, max_workers={max_workers}")

    def extract_batch(self, captions: List[str]) -> Tuple[List, float]:
        """
        Extract scene graphs from a batch of captions using LangExtract.

        Args:
            captions: List of input image captions

        Returns:
            Tuple of (list of extracted data in format-specific structure, total time)
        """
        try:
            # Create Document objects for batch processing
            documents = [Document(text=caption) for caption in captions]

            # Use langextract.extract() with batch processing
            start_time = time.time()

            results = lx.extract(
                text_or_documents=documents,
                prompt_description=self.format_config["prompt"],
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

            # Convert AnnotatedDocuments to format-specific structure
            converter_fn = self.format_config["converter_fn"]
            all_extractions = [converter_fn(result) for result in results]

            return all_extractions, total_time

        except Exception as e:
            print(f"Batch extraction error: {e}")
            # Return empty results for all samples
            if self.format_key == "tuple_format":
                return [[] for _ in captions], 0.0
            else:
                return [{} for _ in captions], 0.0

    def evaluate(
        self,
        samples: List[Dict],
        save_dir: str
    ) -> Dict:
        """
        Evaluate format on samples using batch processing.

        Args:
            samples: List of dataset samples
            save_dir: Directory to save results

        Returns:
            Dictionary containing all evaluation results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\nEvaluating {self.format_config['name']} format on {len(samples)} samples (BATCHED)...")
        print("=" * 80)

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

        to_factual_fn = self.format_config["to_factual_fn"]

        for i, (caption, gt_scene_graph_str, extracted) in enumerate(zip(captions, ground_truth_strs, all_extracted)):
            # Convert to FACTUAL format
            pred_entities, pred_attrs, pred_rels = to_factual_fn(extracted)
            gt_entities, gt_attrs, gt_rels = parse_ground_truth_factual(gt_scene_graph_str)

            # Check if extraction was successful (got at least some results)
            has_results = len(pred_entities) > 0 or len(pred_attrs) > 0 or len(pred_rels) > 0
            if has_results:
                extraction_success_count += 1

            # Compute FACTUAL-based metrics
            factual_metrics = compute_factual_metrics(pred_entities, pred_attrs, pred_rels, gt_entities, gt_attrs, gt_rels)
            all_factual_metrics.append(factual_metrics)

            # Classify complexity
            complexity = classify_complexity(caption)
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
            "format": self.format_config["name"],
            "format_key": self.format_key,
            "model": self.model_name,
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
        print(f"FORMAT: {results['format']} - EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nModel: {results['model']}")
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


# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

def main():
    """Main evaluation function for Experiment 3 (Format Optimization - Batched)."""

    # Load the same test set as Experiment 1
    all_samples = load_factual_dataset(
        split="train",
        num_samples=100,
        test_split=True,
        use_complex_only=True
    )

    # Select 50 diverse samples
    indices = np.linspace(0, len(all_samples) - 1, 50, dtype=int)
    samples = [all_samples[int(i)] for i in indices]

    print(f"Selected {len(samples)} diverse samples from Experiment 1 test set")
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: FORMAT OPTIMIZATION (BATCHED)")
    print("=" * 80)

    # Evaluate each format
    all_format_results = {}

    for format_key in FORMAT_CONFIGS.keys():
        print(f"\n\n{'='*80}")
        print(f"TESTING FORMAT: {FORMAT_CONFIGS[format_key]['name']}")
        print(f"Description: {FORMAT_CONFIGS[format_key]['description']}")
        print(f"{'='*80}\n")

        # Initialize evaluator for this format
        evaluator = BatchedFormatEvaluator(
            format_key=format_key,
            model_name="gemini-2.5-flash",
            batch_length=50,  # Process all 50 samples at once
            max_workers=10    # Use 10 parallel workers
        )

        # Run evaluation
        save_dir = f"./results/experiment_3_format_optimization_batched/{format_key}"
        results = evaluator.evaluate(samples, save_dir=save_dir)

        all_format_results[format_key] = results

    # Create comparison summary
    print("\n\n" + "=" * 80)
    print("FORMAT COMPARISON SUMMARY")
    print("=" * 80)

    comparison_data = []
    for format_key, results in all_format_results.items():
        comparison_data.append({
            "format": results["format"],
            "format_key": format_key,
            "macro_f1": results["overall_metrics"]["macro_f1"],
            "entities_f1": results["overall_metrics"]["entities"]["f1"],
            "attributes_f1": results["overall_metrics"]["attributes"]["f1"],
            "relationships_f1": results["overall_metrics"]["relationships"]["f1"],
            "extraction_success_rate": results["extraction_success_rate"],
            "avg_inference_time": results["avg_inference_time_per_sample"]
        })

    # Sort by macro F1
    comparison_data.sort(key=lambda x: x["macro_f1"], reverse=True)

    # Print comparison table
    print("\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10}".format(
        "Format", "Macro F1", "Ent F1", "Attr F1", "Rel F1", "Success %", "Time (s)"
    ))
    print("-" * 90)
    for data in comparison_data:
        print("{:<20} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>11.1f}% {:>10.3f}".format(
            data["format"],
            data["macro_f1"],
            data["entities_f1"],
            data["attributes_f1"],
            data["relationships_f1"],
            data["extraction_success_rate"] * 100,
            data["avg_inference_time"]
        ))

    # Save comparison summary
    comparison_path = Path("./results/experiment_3_format_optimization_batched/format_comparison.json")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w") as f:
        json.dump({
            "comparison_data": comparison_data,
            "best_format": comparison_data[0]["format"],
            "best_format_key": comparison_data[0]["format_key"],
            "best_macro_f1": comparison_data[0]["macro_f1"]
        }, f, indent=2)

    print(f"\nComparison summary saved to: {comparison_path}")
    print(f"\nBest format: {comparison_data[0]['format']} (Macro F1: {comparison_data[0]['macro_f1']:.3f})")
    print("\n" + "=" * 80)
    print("\nExperiment 3 (Format Optimization - BATCHED) complete!")


if __name__ == "__main__":
    main()
