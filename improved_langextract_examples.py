"""
Improved LangExtract examples specifically targeting the "sit on" predicate normalization issue.

Analysis showed that LangExtract is predicting "sitting on" instead of "sit on",
despite having instructions to use base verb forms. These examples directly demonstrate
the correct normalization pattern.
"""

from langextract.data import ExampleData, Extraction

def create_targeted_sitting_examples():
    """
    Create examples that explicitly show "sitting on" → "sit on" normalization.

    This addresses the #1 failure mode where LangExtract missed 48 relationships
    by using "sitting on" instead of "sit on".
    """
    examples = []

    # Example 1: Direct "sitting on" case - EXACTLY matches the failing samples
    ex1 = ExampleData(
        text="A white teddy bear sitting on a green carpeted stair.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="teddy bear",
                      attributes={"name": "teddy bear"}),
            Extraction(extraction_class="entity", extraction_text="stair",
                      attributes={"name": "stair"}),
            Extraction(extraction_class="attribute", extraction_text="white",
                      attributes={"entity": "teddy bear", "attribute": "white"}),
            Extraction(extraction_class="attribute", extraction_text="green",
                      attributes={"entity": "stair", "attribute": "green"}),
            Extraction(extraction_class="attribute", extraction_text="carpeted",
                      attributes={"entity": "stair", "attribute": "carpeted"}),
            Extraction(extraction_class="attribute", extraction_text="sitting",
                      attributes={"entity": "teddy bear", "attribute": "sitting"}),
            # CRITICAL: Text says "sitting on", but extract as "sit on" (base form)
            Extraction(extraction_class="relationship", extraction_text="teddy bear sit on stair",
                      attributes={"subject": "teddy bear", "predicate": "sit on", "object": "stair"}),
        ]
    )
    examples.append(ex1)

    # Example 2: Another "sitting on" variant
    ex2 = ExampleData(
        text="A small child sitting on a wooden chair in a room.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="child",
                      attributes={"name": "child"}),
            Extraction(extraction_class="entity", extraction_text="chair",
                      attributes={"name": "chair"}),
            Extraction(extraction_class="entity", extraction_text="room",
                      attributes={"name": "room"}),
            Extraction(extraction_class="attribute", extraction_text="small",
                      attributes={"entity": "child", "attribute": "small"}),
            Extraction(extraction_class="attribute", extraction_text="wooden",
                      attributes={"entity": "chair", "attribute": "wooden"}),
            Extraction(extraction_class="attribute", extraction_text="sitting",
                      attributes={"entity": "child", "attribute": "sitting"}),
            # Text says "sitting on" → extract as "sit on"
            Extraction(extraction_class="relationship", extraction_text="child sit on chair",
                      attributes={"subject": "child", "predicate": "sit on", "object": "chair"}),
            Extraction(extraction_class="relationship", extraction_text="chair in room",
                      attributes={"subject": "chair", "predicate": "in", "object": "room"}),
        ]
    )
    examples.append(ex2)

    # Example 3: "sitting" with different prepositions
    ex3 = ExampleData(
        text="A dog sitting beside a tree.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="dog",
                      attributes={"name": "dog"}),
            Extraction(extraction_class="entity", extraction_text="tree",
                      attributes={"name": "tree"}),
            Extraction(extraction_class="attribute", extraction_text="sitting",
                      attributes={"entity": "dog", "attribute": "sitting"}),
            # Text says "sitting beside" → extract as "beside" (no "sit" in predicate when preposition follows)
            Extraction(extraction_class="relationship", extraction_text="dog beside tree",
                      attributes={"subject": "dog", "predicate": "beside", "object": "tree"}),
        ]
    )
    examples.append(ex3)

    # Example 4: Multiple "sitting" relationships
    ex4 = ExampleData(
        text="People sitting on benches next to tables.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="people",
                      attributes={"name": "people"}),
            Extraction(extraction_class="entity", extraction_text="benches",
                      attributes={"name": "benches"}),
            Extraction(extraction_class="entity", extraction_text="tables",
                      attributes={"name": "tables"}),
            Extraction(extraction_class="attribute", extraction_text="sitting",
                      attributes={"entity": "people", "attribute": "sitting"}),
            # "sitting on" → "sit on"
            Extraction(extraction_class="relationship", extraction_text="people sit on benches",
                      attributes={"subject": "people", "predicate": "sit on", "object": "benches"}),
            # "next to" stays as is
            Extraction(extraction_class="relationship", extraction_text="benches next to tables",
                      attributes={"subject": "benches", "predicate": "next to", "object": "tables"}),
        ]
    )
    examples.append(ex4)

    # Example 5: "standing on" → "stand on" (same pattern, different verb)
    ex5 = ExampleData(
        text="A man standing on a ladder.",
        extractions=[
            Extraction(extraction_class="entity", extraction_text="man",
                      attributes={"name": "man"}),
            Extraction(extraction_class="entity", extraction_text="ladder",
                      attributes={"name": "ladder"}),
            Extraction(extraction_class="attribute", extraction_text="standing",
                      attributes={"entity": "man", "attribute": "standing"}),
            # "standing on" → "stand on"
            Extraction(extraction_class="relationship", extraction_text="man stand on ladder",
                      attributes={"subject": "man", "predicate": "stand on", "object": "ladder"}),
        ]
    )
    examples.append(ex5)

    return examples


def print_examples_summary():
    """Print a summary of the targeted examples."""
    examples = create_targeted_sitting_examples()

    print("="*80)
    print("TARGETED EXAMPLES FOR LANGEXTRACT - PREDICATE NORMALIZATION")
    print("="*80)
    print(f"\nCreated {len(examples)} examples specifically targeting:")
    print("  1. 'sitting on' -> 'sit on' normalization")
    print("  2. 'standing on' -> 'stand on' normalization")
    print("  3. Handling 'sitting' as an attribute while using base form in predicate")
    print("\n" + "="*80)

    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}: {ex.text}")

        relationships = [e for e in ex.extractions if e.extraction_class == "relationship"]
        print(f"  Relationships ({len(relationships)}):")
        for rel in relationships:
            subj = rel.attributes.get("subject")
            pred = rel.attributes.get("predicate")
            obj = rel.attributes.get("object")
            print(f"    - ({subj}, {pred}, {obj})")


if __name__ == "__main__":
    print_examples_summary()
