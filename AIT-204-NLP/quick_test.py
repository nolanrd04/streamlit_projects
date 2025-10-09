"""
Quick Test Script for Binary Sentiment Classifier
Part 1 of Class Activity - Test the existing implementation
"""

import torch
print("Imported torch")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
print("Imported transformers")
import warnings
print("Imported warnings")
warnings.filterwarnings('ignore')


def test_binary_classifier():
    """Test the original binary sentiment classifier"""

    print("=" * 80)
    print("Part 1: Testing Binary Sentiment Classifier")
    print("=" * 80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}")

    # Load model
    print("\n🤖 Loading DistilBERT model...")
    model_name = "distilbert-base-uncased"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Binary: Positive/Negative
        )
        model.to(device)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Tip: Make sure you have internet connection for first-time model download.")
        print("    The model will be cached locally after the first download.")
        return

    # Test reviews
    print("\n" + "=" * 80)
    print("🧪 Testing with Sample Reviews")
    print("=" * 80)

    test_reviews = [
        "This movie was absolutely fantastic! Best film I've seen this year!",
        "Terrible waste of time. I want my money back.",
        "It was okay, nothing special.",  # This will expose binary classifier limitation
        "Amazing performance! Loved every minute of it.",
        "Boring and predictable. Very disappointing.",
        "The film was fine, neither good nor bad.",  # Another neutral example
    ]

    labels_map = {0: "Negative 😞", 1: "Positive 😊"}

    for i, review in enumerate(test_reviews, 1):
        print(f"\n{i}. Review:")
        print(f"   \"{review}\"")

        # Tokenize
        inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = probabilities.argmax(dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        # Display results
        print(f"   Prediction: {labels_map[predicted_class]}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Probabilities: Negative={probabilities[0][0].item():.1%}, Positive={probabilities[0][1].item():.1%}")

        # Highlight limitation for neutral reviews
        if "okay" in review.lower() or "fine" in review.lower() or "neither" in review.lower():
            print(f"   ⚠️  Note: This review seems neutral, but binary classifier must choose positive or negative!")

    # Discussion prompts
    print("\n" + "=" * 80)
    print("🤔 Discussion Questions:")
    print("=" * 80)
    print("""
1. What happened with the neutral reviews (e.g., "It was okay")?
   → The model had to classify them as either positive or negative!

2. Can this model distinguish between "good" and "amazing"?
   → No, both would just be classified as "positive"

3. How would you represent different intensities of sentiment?
   → We need a multi-class classifier with a sentiment scale!

This is why we're building a 7-point scale classifier (-3 to +3) in Part 2!
    """)

    # Summary statistics
    print("=" * 80)
    print("📊 Summary")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Classification: Binary (Positive/Negative)")
    print(f"Number of classes: 2")
    print(f"Limitation: Cannot handle neutral sentiment or intensity variations")
    print(f"\nNext Step: Build a multi-scale classifier (-3 to +3) to solve these limitations!")
    print("=" * 80)


if __name__ == "__main__":
    test_binary_classifier()
