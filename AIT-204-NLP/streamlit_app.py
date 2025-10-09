"""
Streamlit App: Multi-Scale Sentiment Analyzer
Deploy-ready sentiment analysis web application

Students: Complete the TODOs to enhance the UI and functionality
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_base_model():
    """Load the base model (not fine-tuned)"""
    model_name = "distilbert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=7,  # -3 to +3 scale
            ignore_mismatched_sizes=True
        )
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading base model: {e}")
        st.stop()


@st.cache_resource
def load_trained_model():
    """Load the fine-tuned model from Hugging Face Hub or local"""
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try local model first (for development)
    model_path = './sentiment_model_7point'
    
    # Replace with your Hugging Face model ID after uploading
    # Format: "your-username/model-name"
    huggingface_model_id = "nolanrd04/AIT-204-NLP_sentiment-model"  # CHANGE THIS to your username/model-name
    
    # Try loading from local first
    if os.path.exists(model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(device)
            return model, tokenizer, device
        except Exception as e:
            st.warning(f"Local model failed to load: {e}. Trying Hugging Face...")
    
    # If local fails or doesn't exist, try Hugging Face
    try:
        st.info(f"📥 Downloading model from Hugging Face Hub: {huggingface_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model_id)
        model = AutoModelForSequenceClassification.from_pretrained(huggingface_model_id)
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"❌ Failed to load trained model from Hugging Face: {e}")
        st.info("💡 Options:")
        st.info("1. Train a model locally using sentiment_scale_analyzer.py")
        st.info(f"2. Upload your model to Hugging Face Hub as: {huggingface_model_id}")
        st.info("3. Use the base model instead (select in sidebar)")
        st.stop()


def class_to_sentiment_score(class_id):
    """Convert class ID (0-6) to sentiment score (-3 to +3)"""
    return class_id - 3


def get_sentiment_label(score):
    """Get descriptive label for sentiment score"""
    labels = {
        -3: "Very Negative",
        -2: "Negative",
        -1: "Slightly Negative",
        0: "Neutral",
        1: "Slightly Positive",
        2: "Positive",
        3: "Very Positive"
    }
    return labels.get(score, "Unknown")


def get_sentiment_emoji(score):
    """Get emoji for sentiment score"""
    emojis = {
        -3: "😢",
        -2: "😞",
        -1: "😐",
        0: "😶",
        1: "🙂",
        2: "😊",
        3: "🤩"
    }
    return emojis.get(score, "❓")


def get_sentiment_color(score):
    """Get color for sentiment score"""
    if score <= -2:
        return "#ff4444"  # Red
    elif score == -1:
        return "#ff9944"  # Orange
    elif score == 0:
        return "#ffdd44"  # Yellow
    elif score == 1:
        return "#99dd44"  # Light green
    else:
        return "#44dd44"  # Green


def create_sentiment_gauge(score, confidence):
    """Create a gauge chart for sentiment visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score", 'font': {'size': 24}},
        delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [-3, 3], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': get_sentiment_color(score)},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-3, -2], 'color': '#ffcccc'},
                {'range': [-2, -1], 'color': '#ffe6cc'},
                {'range': [-1, 0], 'color': '#ffffcc'},
                {'range': [0, 1], 'color': '#e6ffcc'},
                {'range': [1, 2], 'color': '#ccffcc'},
                {'range': [2, 3], 'color': '#ccffee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"},
        height=300
    )

    return fig


def analyze_sentiment(text, model, tokenizer, device):
    """Analyze sentiment of text"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = probabilities.argmax(dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    # Convert to sentiment score
    sentiment_score = class_to_sentiment_score(predicted_class)

    return sentiment_score, confidence, probabilities[0].cpu().numpy()


def main():
    # TODO 1: Add an attractive title and description
    st.title("🎬 Movie Review Sentiment Analyzer")  # Customize this!
    st.markdown("""
    **Analyze movie reviews on a scale from -3 (Very Negative) to +3 (Very Positive)**

    This app uses a DistilBERT transformer model to analyze sentiment with nuance,
    going beyond simple positive/negative classification.
    """)  # Customize this description!

    # Sidebar with examples and information
    with st.sidebar:
        st.header("🤖 Model Selection")
        
        model_option = st.radio(
            "Choose a model:",
            options=[
                "🎯 Fine-tuned Model (Best Accuracy)",
                "📦 Base Model (Baseline)"
            ],
            index=0,
            disabled=False,
            help="Fine-tuned model loads from Hugging Face Hub or local directory"
        )
        
        st.caption("💡 Fine-tuned model loads from Hugging Face Hub (cloud) or local directory (development)")
        
        st.divider()
        
        st.header("ℹ️ About")
        st.markdown("""
        This sentiment analyzer uses a **7-point scale**:

        - **+3**: Very Positive 🤩
        - **+2**: Positive 😊
        - **+1**: Slightly Positive 🙂
        - **0**: Neutral 😶
        - **-1**: Slightly Negative 😐
        - **-2**: Negative 😞
        - **-3**: Very Negative 😢
        """)

        st.divider()

        # TODO 4: Add example reviews that users can click
        st.header("📝 Try These Examples")

        if st.button("🤩 Very Positive Example"):
            st.session_state.example_text = "This movie was absolutely phenomenal! The best film I've seen this year. Every aspect from acting to cinematography was perfect!"

        if st.button("😊 Positive Example"):
            st.session_state.example_text = "Really enjoyed this film. Great performances and a compelling story."

        if st.button("😶 Neutral Example"):
            st.session_state.example_text = "It was an okay movie. Nothing particularly special, but not bad either."

        if st.button("😞 Negative Example"):
            st.session_state.example_text = "Quite disappointing. The plot was weak and the pacing was off."

        if st.button("😢 Very Negative Example"):
            st.session_state.example_text = "Absolutely terrible! Complete waste of time. One of the worst films I've ever seen."

        # TODO: Add more example buttons for other sentiment levels

    # Load the selected model
    if "Fine-tuned" in model_option:
        with st.spinner("🤖 Loading fine-tuned model..."):
            model, tokenizer, device = load_trained_model()
        st.success("✅ Fine-tuned model loaded successfully! (Best accuracy)")
    else:
        with st.spinner("🤖 Loading base model..."):
            model, tokenizer, device = load_base_model()
        st.info("ℹ️ Using base model (may be less accurate for 7-point scale)")

    # Main input area
    st.divider()

    # TODO 2: Create a text area for user input
    review_text = st.text_area(
        "📝 Enter a movie review to analyze:",
        value=st.session_state.get('example_text', ''),
        placeholder="Type or paste a movie review here...\n\nExample: 'This movie was incredible! The acting was superb and the plot kept me engaged throughout.'",
        height=150,
        key="review_input"
    )

    # Clear the example text after it's been used
    if 'example_text' in st.session_state:
        del st.session_state.example_text

    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("⚠️ Please enter a review to analyze.")
        else:
            with st.spinner("🧠 Analyzing sentiment..."):
                # Perform analysis
                sentiment_score, confidence, all_probabilities = analyze_sentiment(
                    review_text, model, tokenizer, device
                )

                # Get label and emoji
                sentiment_label = get_sentiment_label(sentiment_score)
                sentiment_emoji = get_sentiment_emoji(sentiment_score)

            # Display results
            st.divider()
            st.subheader("📊 Analysis Results")

            # Create columns for metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Sentiment Score",
                    value=f"{sentiment_score:+d}/3",
                    delta=None
                )

            with col2:
                st.metric(
                    label="Classification",
                    value=f"{sentiment_label} {sentiment_emoji}",
                    delta=None
                )

            with col3:
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.1%}",
                    delta=None
                )

            # TODO 3: Display results with sentiment visualization
            # Visual sentiment indicator
            if sentiment_score >= 2:
                st.success(f"✅ **{sentiment_label}** {sentiment_emoji}")
            elif sentiment_score >= 1:
                st.info(f"👍 **{sentiment_label}** {sentiment_emoji}")
            elif sentiment_score == 0:
                st.warning(f"➖ **{sentiment_label}** {sentiment_emoji}")
            elif sentiment_score >= -1:
                st.warning(f"👎 **{sentiment_label}** {sentiment_emoji}")
            else:
                st.error(f"❌ **{sentiment_label}** {sentiment_emoji}")

            # Gauge chart
            st.plotly_chart(
                create_sentiment_gauge(sentiment_score, confidence),
                use_container_width=True
            )

            # Probability distribution
            st.subheader("📈 Probability Distribution")

            # Create bar chart of all class probabilities
            import pandas as pd

            prob_df = pd.DataFrame({
                'Sentiment': [f"{i-3:+d}: {get_sentiment_label(i-3)}" for i in range(7)],
                'Probability': all_probabilities * 100
            })

            st.bar_chart(prob_df.set_index('Sentiment'))

            # Detailed breakdown
            with st.expander("🔬 Detailed Probability Breakdown"):
                for i, prob in enumerate(all_probabilities):
                    score = i - 3
                    label = get_sentiment_label(score)
                    emoji = get_sentiment_emoji(score)
                    st.write(f"{emoji} **{score:+d} ({label})**: {prob:.2%}")

    # Footer
    st.divider()
    model_type = "Fine-tuned" if "Fine-tuned" in model_option else "Base"
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit and Hugging Face Transformers 🤗</p>
        <p>Model: DistilBERT ({model_type}) | 7-Point Sentiment Scale</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
