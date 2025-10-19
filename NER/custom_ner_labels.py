"""
Step 1: Define Custom NER Labels
Define your domain-specific entity types using IOB tagging scheme
"""

# Define your custom entity types
# Modify these based on your domain!
CUSTOM_ENTITY_TYPES = [
    'PERSON',      # Person names
    'COMPANY',     # Company/Organization names
    'PRODUCT',     # Product names
    'LOCATION',    # Geographic locations
    'DATE',        # Dates and time expressions
    'MONEY',       # Monetary amounts
]

# Generate IOB labels automatically
def generate_iob_labels(entity_types):
    """
    Generate IOB tag scheme from entity types
    
    Args:
        entity_types (list): List of entity type names
    
    Returns:
        dict: Label mappings
    """
    labels = ['O']  # Outside tag always first
    
    for entity in entity_types:
        labels.append(f'B-{entity}')  # Beginning tag
        labels.append(f'I-{entity}')  # Inside tag
    
    # Create mappings
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    
    return labels, label2id, id2label

# Generate the label scheme
LABELS, LABEL2ID, ID2LABEL = generate_iob_labels(CUSTOM_ENTITY_TYPES)

def display_label_scheme():
    """Display the generated label scheme"""
    print("="*80)
    print("CUSTOM NER LABEL SCHEME")
    print("="*80)
    print(f"\nTotal Labels: {len(LABELS)}")
    print(f"\nEntity Types: {', '.join(CUSTOM_ENTITY_TYPES)}")
    print(f"\n{'ID':<5} {'Label':<15} {'Description':<50}")
    print("-"*70)
    
    for label_id, label in ID2LABEL.items():
        if label == 'O':
            desc = "Outside any named entity"
        elif label.startswith('B-'):
            entity = label.split('-')[1]
            desc = f"Beginning of {entity} entity"
        elif label.startswith('I-'):
            entity = label.split('-')[1]
            desc = f"Inside/Continuation of {entity} entity"
        else:
            desc = "Unknown"
        
        print(f"{label_id:<5} {label:<15} {desc:<50}")

if __name__ == "__main__":
    display_label_scheme()
    
    print("\n" + "="*80)
    print("EXAMPLE USAGE")
    print("="*80)
    
    # Example sentence with annotations
    example_tokens = ['Apple', 'Inc.', 'released', 'iPhone', '15', 'on', 'September', '15', 'for', '$799']
    example_labels = ['B-COMPANY', 'I-COMPANY', 'O', 'B-PRODUCT', 'I-PRODUCT', 'O', 'B-DATE', 'I-DATE', 'O', 'B-MONEY']
    
    print(f"\n{'Token':<15} {'Label':<15} {'Label ID':<10}")
    print("-"*40)
    for token, label in zip(example_tokens, example_labels):
        label_id = LABEL2ID[label]
        print(f"{token:<15} {label:<15} {label_id:<10}")
