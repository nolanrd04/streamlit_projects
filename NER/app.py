"""
Streamlit App for Named Entity Recognition (NER)
Interactive interface to test the trained NER model
"""

import streamlit as st
import sys
from pathlib import Path
import json
import pickle

# Add parent directory to path
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir))

from custom_ner_labels import LABELS, CUSTOM_ENTITY_TYPES
from model_training import NERModelTrainer

# Page configuration
st.set_page_config(
    page_title="NER Model Tester",
    page_icon="🏷️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .entity-company { background-color: #ffcccc; padding: 2px 6px; border-radius: 3px; margin: 2px; }
    .entity-person { background-color: #ccffcc; padding: 2px 6px; border-radius: 3px; margin: 2px; }
    .entity-product { background-color: #ccccff; padding: 2px 6px; border-radius: 3px; margin: 2px; }
    .entity-location { background-color: #ffffcc; padding: 2px 6px; border-radius: 3px; margin: 2px; }
    .entity-date { background-color: #ffccff; padding: 2px 6px; border-radius: 3px; margin: 2px; }
    .entity-money { background-color: #ccffff; padding: 2px 6px; border-radius: 3px; margin: 2px; }
    .stAlert { margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained NER model"""
    try:
        model_path = script_dir / 'annotated_data' / 'ner_model.pkl'
        
        if not model_path.exists():
            return None, "Model not found. Please run ner_pipeline.py first to train a model."
        
        trainer = NERModelTrainer(LABELS)
        trainer.load_model(str(model_path))
        
        return trainer, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

@st.cache_data
def load_training_examples():
    """Load example sentences from training data"""
    try:
        annotations_path = script_dir / 'annotated_data' / 'training_annotations.json'
        
        if not annotations_path.exists():
            return None
        
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        return annotations
    except:
        return None

def extract_entities(tokens, labels):
    """Extract entities from IOB-tagged sequence"""
    entities = []
    current_entity = None
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B-'):
            # Save previous entity if exists
            if current_entity:
                entities.append(current_entity)
            
            # Start new entity
            entity_type = label.split('-')[1]
            current_entity = {
                'start': i,
                'end': i + 1,
                'type': entity_type,
                'text': [token]
            }
        
        elif label.startswith('I-'):
            # Continue entity
            if current_entity:
                entity_type = label.split('-')[1]
                if current_entity['type'] == entity_type:
                    current_entity['end'] = i + 1
                    current_entity['text'].append(token)
        
        elif label == 'O':
            # End current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Don't forget last entity
    if current_entity:
        entities.append(current_entity)
    
    # Convert text lists to strings
    for entity in entities:
        entity['text'] = ' '.join(entity['text'])
    
    return entities

def highlight_entities(tokens, labels):
    """Create HTML with highlighted entities"""
    html = '<div style="line-height: 2.5; font-size: 18px;">'
    
    entity_colors = {
        'COMPANY': 'entity-company',
        'PERSON': 'entity-person',
        'PRODUCT': 'entity-product',
        'LOCATION': 'entity-location',
        'DATE': 'entity-date',
        'MONEY': 'entity-money'
    }
    
    current_entity = None
    entity_tokens = []
    
    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            # Save previous entity
            if current_entity and entity_tokens:
                entity_type = current_entity.split('-')[1]
                css_class = entity_colors.get(entity_type, 'entity-company')
                html += f'<span class="{css_class}">{" ".join(entity_tokens)} <sup>{entity_type}</sup></span> '
                entity_tokens = []
            
            # Start new entity
            current_entity = label
            entity_tokens = [token]
        
        elif label.startswith('I-'):
            # Continue entity
            if current_entity:
                entity_tokens.append(token)
        
        elif label == 'O':
            # End current entity
            if current_entity and entity_tokens:
                entity_type = current_entity.split('-')[1]
                css_class = entity_colors.get(entity_type, 'entity-company')
                html += f'<span class="{css_class}">{" ".join(entity_tokens)} <sup>{entity_type}</sup></span> '
                entity_tokens = []
                current_entity = None
            
            # Add non-entity token
            html += f'{token} '
    
    # Handle last entity
    if current_entity and entity_tokens:
        entity_type = current_entity.split('-')[1]
        css_class = entity_colors.get(entity_type, 'entity-company')
        html += f'<span class="{css_class}">{" ".join(entity_tokens)} <sup>{entity_type}</sup></span> '
    
    html += '</div>'
    return html

def main():
    st.title("🏷️ Named Entity Recognition Model Tester")
    st.markdown("Test your trained NER model on custom text input")
    
    # Load model
    trainer, error = load_model()
    
    if error:
        st.error(error)
        st.info("💡 Run `python ner_pipeline.py` in the NER directory to train a model first.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar - Information
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown(f"**Entity Types:** {len(CUSTOM_ENTITY_TYPES)}")
        
        # Display entity types with color legend
        st.markdown("### 🎨 Entity Legend")
        st.markdown('<span class="entity-company">COMPANY</span>', unsafe_allow_html=True)
        st.markdown('<span class="entity-person">PERSON</span>', unsafe_allow_html=True)
        st.markdown('<span class="entity-product">PRODUCT</span>', unsafe_allow_html=True)
        st.markdown('<span class="entity-location">LOCATION</span>', unsafe_allow_html=True)
        st.markdown('<span class="entity-date">DATE</span>', unsafe_allow_html=True)
        st.markdown('<span class="entity-money">MONEY</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Load training examples
        st.header("📚 Training Data Info")
        training_examples = load_training_examples()
        
        if training_examples:
            st.markdown(f"**Training samples:** {len(training_examples)}")
            
            # Count entity types in training data
            entity_counts = {et: 0 for et in CUSTOM_ENTITY_TYPES}
            for example in training_examples:
                for label in example['ner_tags']:
                    if label.startswith('B-'):
                        entity_type = label.split('-')[1]
                        if entity_type in entity_counts:
                            entity_counts[entity_type] += 1
            
            st.markdown("**Entity distribution:**")
            for entity_type, count in entity_counts.items():
                if count > 0:
                    st.markdown(f"- {entity_type}: {count}")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use entities similar to training data
        - Include company names, products, people, locations
        - Try tech companies and products
        - Include monetary amounts and dates
        """)
    
    # Main content
    tabs = st.tabs(["🔍 Test Model", "📝 Example Sentences", "📊 Training Data"])
    
    # Tab 1: Test Model
    with tabs[0]:
        st.header("Enter Text to Analyze")
        
        # Text input
        user_input = st.text_area(
            "Input Text:",
            height=100,
            placeholder="Example: Apple Inc. released iPhone 15 for $799",
            help="Enter a sentence to identify named entities"
        )
        
        # Predict button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            predict_button = st.button("🎯 Predict", type="primary")
        with col2:
            clear_button = st.button("🗑️ Clear")
        
        if clear_button:
            st.rerun()
        
        if predict_button and user_input.strip():
            # Tokenize and predict
            tokens = user_input.split()
            
            with st.spinner("Analyzing text..."):
                try:
                    predictions = trainer.predict(tokens)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    # Highlighted text
                    st.markdown("### Highlighted Entities")
                    highlighted_html = highlight_entities(tokens, predictions)
                    st.markdown(highlighted_html, unsafe_allow_html=True)
                    
                    # Extract entities
                    entities = extract_entities(tokens, predictions)
                    
                    if entities:
                        st.markdown("### 🎯 Detected Entities")
                        
                        # Display in columns
                        cols = st.columns(3)
                        for idx, entity in enumerate(entities):
                            with cols[idx % 3]:
                                st.info(f"**{entity['text']}**\n\n`{entity['type']}`")
                        
                        # Entity summary
                        st.markdown("### 📈 Entity Summary")
                        entity_type_counts = {}
                        for entity in entities:
                            entity_type = entity['type']
                            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
                        
                        summary_cols = st.columns(len(entity_type_counts))
                        for idx, (entity_type, count) in enumerate(entity_type_counts.items()):
                            with summary_cols[idx]:
                                st.metric(entity_type, count)
                    else:
                        st.warning("⚠️ No entities detected in the text")
                    
                    # Detailed token table
                    with st.expander("🔍 View Detailed Token Labels"):
                        st.markdown("### Token-Level Predictions")
                        
                        # Create DataFrame for display
                        import pandas as pd
                        df = pd.DataFrame({
                            'Token': tokens,
                            'Label': predictions,
                            'Description': [
                                'Beginning of ' + pred.split('-')[1] if pred.startswith('B-')
                                else 'Inside ' + pred.split('-')[1] if pred.startswith('I-')
                                else 'Outside entity'
                                for pred in predictions
                            ]
                        })
                        
                        st.dataframe(df, use_container_width=True, hide_index=True)
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        
        elif predict_button:
            st.warning("⚠️ Please enter some text to analyze")
    
    # Tab 2: Example Sentences
    with tabs[1]:
        st.header("📝 Example Sentences to Try")
        st.markdown("Click on any example to test it instantly!")
        
        example_categories = {
            "🏢 Companies & Products": [
                "Apple Inc. released iPhone 15 on September 15 for $799",
                "Microsoft CEO Satya Nadella announced Windows 12",
                "Tesla stock rose to $250 in New York",
                "Google acquired DeepMind for $500 million in London",
                "Amazon launched Prime Video in California",
                "Samsung released Galaxy S24 on January 15",
                "Meta CEO Mark Zuckerberg introduced Threads",
                "IBM announced Watson AI at CES in Las Vegas"
            ],
            "👤 People & Organizations": [
                "Tim Cook became Apple CEO in 2011",
                "Elon Musk founded SpaceX in California",
                "Bill Gates donated $10 billion to charity",
                "Jeff Bezos stepped down as Amazon CEO"
            ],
            "💰 Financial News": [
                "Netflix stock dropped to $450 yesterday",
                "Oracle acquired Sun Microsystems for $7 billion",
                "Nvidia reached $500 market cap in New York",
                "PayPal processed $1 trillion in payments"
            ],
            "🌍 Locations & Events": [
                "The conference was held in San Francisco on March 15",
                "Apple opened a new store in Tokyo last week",
                "Microsoft announced Azure at Build in Seattle"
            ]
        }
        
        for category, examples in example_categories.items():
            with st.expander(category, expanded=True):
                for example in examples:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"• {example}")
                    with col2:
                        if st.button("Try", key=example):
                            # Set the example as input and trigger prediction
                            st.session_state['example_text'] = example
                            st.switch_page
    
    # Tab 3: Training Data
    with tabs[2]:
        st.header("📊 Training Data Overview")
        
        training_examples = load_training_examples()
        
        if training_examples:
            st.success(f"✅ Loaded {len(training_examples)} training examples")
            
            # Show samples
            st.markdown("### Sample Annotations")
            
            for idx, example in enumerate(training_examples[:5]):
                with st.expander(f"Sample {idx + 1}: {example['text'][:60]}..."):
                    tokens = example['tokens']
                    labels = example['ner_tags']
                    
                    # Highlighted view
                    highlighted_html = highlight_entities(tokens, labels)
                    st.markdown(highlighted_html, unsafe_allow_html=True)
                    
                    # Entities
                    entities = extract_entities(tokens, labels)
                    if entities:
                        st.markdown("**Entities:**")
                        for entity in entities:
                            st.markdown(f"- `{entity['text']}` → **{entity['type']}**")
            
            # Full data view
            if st.checkbox("Show all training data"):
                import pandas as pd
                
                data = []
                for example in training_examples:
                    entities = extract_entities(example['tokens'], example['ner_tags'])
                    entity_text = ", ".join([f"{e['text']} ({e['type']})" for e in entities])
                    
                    data.append({
                        'Text': example['text'],
                        'Entities': entity_text if entities else 'None'
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ No training data found")
            st.info("Run `python ner_pipeline.py` to generate training data")

if __name__ == "__main__":
    main()
