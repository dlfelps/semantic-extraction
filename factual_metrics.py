"""
FACTUAL Format Metrics and Utilities

This module provides functions for converting various extraction formats to FACTUAL format
and computing evaluation metrics directly on FACTUAL triplets.

FACTUAL format represents scene graphs as triplets: (subject, predicate, object)
- Entities are extracted from all triplets
- Attributes have predicates: is, are, has, have (normalized to "has_attribute")
- Relationships have other predicates (e.g., on, in, wearing, holding)
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a single component."""
    precision: float
    recall: float
    f1: float
    support: int


# ============================================================================
# FACTUAL Format Converters
# ============================================================================

def convert_flat_entities_to_factual(extracted: Dict) -> Tuple[Set[str], Set[Tuple[str, str, str]], Set[Tuple[str, str, str]]]:
    """
    Convert Flat Entities dictionary format to FACTUAL format.

    Args:
        extracted: Dictionary with 'entity', 'attribute', 'relationship' keys

    Returns:
        Tuple of (entities, attribute_triplets, relationship_triplets)
    """
    entities = set()
    attribute_triplets = set()
    relationship_triplets = set()

    # Extract entities
    for entity_obj in extracted.get("entity", []):
        if isinstance(entity_obj, dict) and "name" in entity_obj:
            entities.add(entity_obj["name"])

    # Extract attributes as triplets
    for attr_obj in extracted.get("attribute", []):
        if isinstance(attr_obj, dict) and "entity" in attr_obj and "value" in attr_obj:
            entity = attr_obj["entity"]
            value = attr_obj["value"]
            attribute_triplets.add((entity, "has_attribute", value))
            # Add entity to entities set
            entities.add(entity)

    # Extract relationships as triplets
    for rel_obj in extracted.get("relationship", []):
        if isinstance(rel_obj, dict) and all(k in rel_obj for k in ["subject", "predicate", "object"]):
            subject = rel_obj["subject"]
            predicate = rel_obj["predicate"]
            obj = rel_obj["object"]
            relationship_triplets.add((subject, predicate, obj))
            # Add both entities to entities set
            entities.add(subject)
            entities.add(obj)

    return entities, attribute_triplets, relationship_triplets


def convert_tuple_format_to_factual(extracted: List) -> Tuple[Set[str], Set[Tuple[str, str, str]], Set[Tuple[str, str, str]]]:
    """
    Convert Tuple Format (list of triplets) to FACTUAL format.

    Args:
        extracted: List of tuples [(subject, predicate, object), ...]

    Returns:
        Tuple of (entities, attribute_triplets, relationship_triplets)
    """
    entities = set()
    attribute_triplets = set()
    relationship_triplets = set()
    attribute_predicates = {"is", "are", "has", "have"}

    for item in extracted:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            subject, predicate, obj = item

            # Add entities
            entities.add(subject)

            # Classify as attribute or relationship
            if predicate.lower() in attribute_predicates:
                attribute_triplets.add((subject, "has_attribute", obj))
                # For attributes, object is the attribute value, not an entity
            else:
                relationship_triplets.add((subject, predicate, obj))
                # For relationships, object is also an entity
                entities.add(obj)

    return entities, attribute_triplets, relationship_triplets


def convert_hierarchical_to_factual(extracted: Dict) -> Tuple[Set[str], Set[Tuple[str, str, str]], Set[Tuple[str, str, str]]]:
    """
    Convert Hierarchical format (nested objects) to FACTUAL format.

    Args:
        extracted: Dictionary with 'objects' and 'relationships' keys

    Returns:
        Tuple of (entities, attribute_triplets, relationship_triplets)
    """
    entities = set()
    attribute_triplets = set()
    relationship_triplets = set()

    # Extract entities and attributes from objects
    for obj in extracted.get("objects", []):
        if isinstance(obj, dict):
            entity_name = obj.get("name")
            if entity_name:
                entities.add(entity_name)

                # Extract attributes
                for attr in obj.get("attributes", []):
                    attribute_triplets.add((entity_name, "has_attribute", attr))

    # Extract relationships
    for rel in extracted.get("relationships", []):
        if isinstance(rel, dict) and all(k in rel for k in ["subject", "predicate", "object"]):
            subject = rel["subject"]
            predicate = rel["predicate"]
            obj = rel["object"]
            relationship_triplets.add((subject, predicate, obj))
            # Add both entities to entities set
            entities.add(subject)
            entities.add(obj)

    return entities, attribute_triplets, relationship_triplets


def convert_json_structured_to_factual(extracted: Dict) -> Tuple[Set[str], Set[Tuple[str, str, str]], Set[Tuple[str, str, str]]]:
    """
    Convert JSON Structured format to FACTUAL format.

    Args:
        extracted: Dictionary with 'entities', 'attributes', 'relationships' keys

    Returns:
        Tuple of (entities, attribute_triplets, relationship_triplets)
    """
    entities = set()
    attribute_triplets = set()
    relationship_triplets = set()

    # Extract entities
    for entity_obj in extracted.get("entities", []):
        if isinstance(entity_obj, dict) and "name" in entity_obj:
            entities.add(entity_obj["name"])

    # Extract attributes
    for attr_obj in extracted.get("attributes", []):
        if isinstance(attr_obj, dict) and "entity" in attr_obj and "attribute" in attr_obj:
            entity = attr_obj["entity"]
            attribute = attr_obj["attribute"]
            attribute_triplets.add((entity, "has_attribute", attribute))
            entities.add(entity)

    # Extract relationships
    for rel_obj in extracted.get("relationships", []):
        if isinstance(rel_obj, dict) and all(k in rel_obj for k in ["subject", "predicate", "object"]):
            subject = rel_obj["subject"]
            predicate = rel_obj["predicate"]
            obj = rel_obj["object"]
            relationship_triplets.add((subject, predicate, obj))
            entities.add(subject)
            entities.add(obj)

    return entities, attribute_triplets, relationship_triplets


def parse_ground_truth_factual(gt_str: str) -> Tuple[Set[str], Set[Tuple[str, str, str]], Set[Tuple[str, str, str]]]:
    """
    Parse ground truth FACTUAL format string into entities, attribute triplets, and relationship triplets.

    FACTUAL format: "( subject , predicate , object ) , ( subject , predicate , object )"
    Attributes have predicates: is, are, has, have
    Relationships have other predicates

    Args:
        gt_str: Ground truth string in FACTUAL format

    Returns:
        Tuple of (entities, attribute_triplets, relationship_triplets)
    """
    entities = set()
    attribute_triplets = set()
    relationship_triplets = set()
    attribute_predicates = {"is", "are", "has", "have"}

    if not gt_str or gt_str.strip() == "":
        return entities, attribute_triplets, relationship_triplets

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
        if tuple_str.startswith("(") and tuple_str.endswith(")"):
            tuple_str = tuple_str[1:-1]
        parts = [p.strip() for p in tuple_str.split(",")]
        if len(parts) == 3:
            subject, predicate, obj = parts
            # Add entities from all triplets
            entities.add(subject)
            # For attributes, the object is the attribute value, not an entity
            # For relationships, both subject and object are entities
            if predicate.lower() not in attribute_predicates:
                entities.add(obj)

            # Normalize to has_attribute for consistency
            if predicate.lower() in attribute_predicates:
                attribute_triplets.add((subject, "has_attribute", obj))
            else:
                relationship_triplets.add((subject, predicate, obj))

    return entities, attribute_triplets, relationship_triplets


# ============================================================================
# FACTUAL Metrics Computation
# ============================================================================

def compute_factual_metrics(
    pred_entities: Set[str],
    pred_attrs: Set[Tuple[str, str, str]],
    pred_rels: Set[Tuple[str, str, str]],
    gt_entities: Set[str],
    gt_attrs: Set[Tuple[str, str, str]],
    gt_rels: Set[Tuple[str, str, str]]
) -> Dict[str, EvaluationMetrics]:
    """
    Compute metrics directly on FACTUAL format (entities, attribute triplets, relationship triplets).

    Args:
        pred_entities: Predicted entities
        pred_attrs: Predicted attribute triplets
        pred_rels: Predicted relationship triplets
        gt_entities: Ground truth entities
        gt_attrs: Ground truth attribute triplets
        gt_rels: Ground truth relationship triplets

    Returns:
        Dictionary with 'entities', 'attributes', and 'relationships' metrics
    """
    metrics = {}

    for component_name, pred_set, gt_set in [
        ("entities", pred_entities, gt_entities),
        ("attributes", pred_attrs, gt_attrs),
        ("relationships", pred_rels, gt_rels)
    ]:
        if len(gt_set) == 0:
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


def aggregate_factual_metrics(metrics_list: List[Dict[str, EvaluationMetrics]]) -> Dict:
    """
    Aggregate FACTUAL metrics across multiple samples.

    Args:
        metrics_list: List of metric dictionaries (entities, attributes, and relationships)

    Returns:
        Dictionary with aggregated metrics including macro F1
    """
    if not metrics_list:
        return {"entities": {}, "attributes": {}, "relationships": {}, "macro_f1": 0.0}

    # Aggregate entities, attributes, and relationships
    aggregated = {}

    for component in ["entities", "attributes", "relationships"]:
        component_metrics = [m[component] for m in metrics_list]

        # Calculate averages
        precisions = [m.precision for m in component_metrics]
        recalls = [m.recall for m in component_metrics]
        f1s = [m.f1 for m in component_metrics]
        supports = [m.support for m in component_metrics]

        aggregated[component] = {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s),
            "support": sum(supports)
        }

    # Calculate macro F1 (average of entities, attributes, and relationships F1)
    aggregated["macro_f1"] = (
        aggregated["entities"]["f1"] +
        aggregated["attributes"]["f1"] +
        aggregated["relationships"]["f1"]
    ) / 3.0

    return aggregated
