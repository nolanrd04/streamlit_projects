"""
IMDB Sentiment Analysis Demo with Transformers
This demo version works without downloading external datasets
"""

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')


def compute_metrics(pred):
    """Compute accuracy, precision, recall, and F1 score"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def create_sample_dataset():
    """Create a sample IMDB-like dataset for demonstration"""

    # Positive reviews
    positive_reviews = [
        "This movie was absolutely fantastic! Best film I've seen this year.",
        "Amazing performance by the cast. Highly recommended!",
        "Loved every minute of it. A masterpiece of cinema.",
        "Brilliant storytelling and excellent direction. Five stars!",
        "One of the best movies ever made. Truly inspiring.",
        "Exceptional film with great acting and beautiful cinematography.",
        "A must-watch! Incredible plot and stunning visuals.",
        "Outstanding movie. The director did an amazing job.",
        "Perfect movie for the whole family. Absolutely loved it.",
        "Superb acting and a gripping storyline. Highly entertaining.",
        "This film exceeded all my expectations. Wonderful experience.",
        "Great movie with an emotional and powerful message.",
        "Fantastic performances all around. A real treat to watch.",
        "Excellent movie that keeps you engaged from start to finish.",
        "Absolutely brilliant! One of my favorite films of all time.",
        "Wonderfully crafted story with memorable characters.",
        "Impressive cinematography and a captivating narrative.",
        "This movie is a triumph. Truly remarkable filmmaking.",
        "Phenomenal acting and an unforgettable story.",
        "A cinematic gem. Highly recommend to everyone.",
        "Beautifully made film with heart and soul.",
        "Terrific movie with great dialogue and performances.",
        "Stunning visuals and an emotionally resonant story.",
        "Incredible film that will stay with you long after viewing.",
        "A perfect blend of entertainment and artistry.",
        "Magnificent movie with outstanding production values.",
        "Truly inspiring film with powerful performances.",
        "A delightful movie that brings joy and laughter.",
        "Exceptionally well-made with brilliant direction.",
        "A masterclass in filmmaking. Absolutely spectacular."
    ] * 100  # Repeat to create more samples

    # Negative reviews
    negative_reviews = [
        "Terrible movie, waste of time. I want my money back.",
        "Awful film with poor acting and a boring plot.",
        "Completely disappointing. One of the worst movies I've seen.",
        "Horrible movie. Bad acting, bad story, bad everything.",
        "Don't waste your time on this film. It's terrible.",
        "Painfully bad. I couldn't even finish watching it.",
        "Absolutely dreadful. Poor script and worse execution.",
        "This movie was a complete disaster. Avoid at all costs.",
        "Boring and predictable. Not worth watching.",
        "Terrible acting and a ridiculous storyline.",
        "One of the worst films ever made. Completely unwatchable.",
        "Disappointing on every level. Skip this one.",
        "Awful movie with no redeeming qualities.",
        "Poorly directed and badly acted. A total mess.",
        "Horrible film that makes no sense whatsoever.",
        "Terrible waste of time and money. Very disappointing.",
        "Bad movie with a weak plot and poor performances.",
        "Completely boring and uninspired. Avoid this film.",
        "Dreadful movie that fails in every aspect.",
        "Awful film with terrible writing and acting.",
        "Painfully bad movie. Couldn't wait for it to end.",
        "Horrible experience. One of the worst films I've seen.",
        "Terrible movie that I regret watching.",
        "Disappointing film with nothing good about it.",
        "Awful and boring. Don't bother watching.",
        "Horrible movie with a nonsensical plot.",
        "Terrible film that's not worth your time.",
        "Completely bad movie. Poor in every way.",
        "Dreadful film with awful performances.",
        "Terrible waste of time. Highly disappointing."
    ] * 100  # Repeat to create more samples

    # Create train and test splits
    train_positive = positive_reviews[:2400]
    train_negative = negative_reviews[:2400]
    test_positive = positive_reviews[2400:3000]
    test_negative = negative_reviews[2400:3000]

    # Create datasets
    train_data = {
        'text': train_positive + train_negative,
        'label': [1] * len(train_positive) + [0] * len(train_negative)
    }

    test_data = {
        'text': test_positive + test_negative,
        'label': [1] * len(test_positive) + [0] * len(test_negative)
    }

    # Shuffle the data
    train_indices = np.random.permutation(len(train_data['text']))
    test_indices = np.random.permutation(len(test_data['text']))

    train_data['text'] = [train_data['text'][i] for i in train_indices]
    train_data['label'] = [train_data['label'][i] for i in train_indices]
    test_data['text'] = [test_data['text'][i] for i in test_indices]
    test_data['label'] = [test_data['label'][i] for i in test_indices]

    return Dataset.from_dict(train_data), Dataset.from_dict(test_data)


def preprocess_function(examples, tokenizer, max_length=128):
    """Tokenize the text data"""
    return tokenizer(examples['text'], truncation=True, max_length=max_length, padding=True)


def main():
    print("=" * 80)
    print("IMDB Sentiment Analysis Demo with Transformer Architecture")
    print("=" * 80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}")

    # Create sample dataset
    print("\n📚 Creating sample IMDB-like dataset...")
    train_dataset, test_dataset = create_sample_dataset()
    print(f"✓ Dataset created successfully!")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")

    # Load pre-trained model and tokenizer
    model_name = "distilbert-base-uncased"
    print(f"\n🤖 Loading pre-trained model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # Binary classification: positive/negative
    )
    print("✓ Model and tokenizer loaded successfully!")

    # Tokenize the dataset
    print("\n🔄 Tokenizing dataset...")
    train_tokenized = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    test_tokenized = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    print("✓ Tokenization complete!")

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments (reduced for demo)
    training_args = TrainingArguments(
        output_dir="./imdb_demo_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_dir='./logs',
        logging_steps=100,
        warmup_steps=200,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("\n🚀 Starting training...")
    print("=" * 80)
    trainer.train()

    # Evaluate the model
    print("\n📊 Evaluating model on test set...")
    print("=" * 80)
    results = trainer.evaluate()

    print("\n✅ Final Results:")
    print(f"  - Accuracy: {results['eval_accuracy']:.4f}")
    print(f"  - Precision: {results['eval_precision']:.4f}")
    print(f"  - Recall: {results['eval_recall']:.4f}")
    print(f"  - F1 Score: {results['eval_f1']:.4f}")

    # Save the final model
    print("\n💾 Saving model...")
    trainer.save_model("./imdb_demo_model_final")
    tokenizer.save_pretrained("./imdb_demo_model_final")
    print("✓ Model saved to './imdb_demo_model_final'")

    # Test with sample predictions
    print("\n🧪 Testing with sample predictions...")
    print("=" * 80)

    test_reviews = [
        "This movie was absolutely fantastic! Best film I've seen this year.",
        "Terrible movie, waste of time. I want my money back.",
        "The acting was decent but the plot was confusing.",
        "Amazing performance by the cast. Highly recommended!",
        "Boring and predictable. Not worth watching.",
    ]

    # Tokenize test reviews
    inputs = tokenizer(test_reviews, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_classes = predictions.argmax(dim=-1)

    labels_map = {0: "Negative 😞", 1: "Positive 😊"}

    for i, review in enumerate(test_reviews):
        print(f"\nReview: \"{review}\"")
        print(f"Prediction: {labels_map[predicted_classes[i].item()]}")
        print(f"Confidence: Negative={predictions[i][0].item():.2%}, Positive={predictions[i][1].item():.2%}")

    print("\n" + "=" * 80)
    print("✅ Training and evaluation complete!")
    print("=" * 80)

    # Demonstrate attention mechanism concept
    print("\n🔍 Understanding the Transformer:")
    print("=" * 80)
    print("\n1. TOKENIZATION:")
    sample = "This movie was fantastic!"
    tokens = tokenizer.tokenize(sample)
    print(f"   Input: '{sample}'")
    print(f"   Tokens: {tokens}")
    print(f"   Token IDs: {tokenizer.convert_tokens_to_ids(tokens)}")

    print("\n2. EMBEDDING & POSITIONAL ENCODING:")
    print("   - Each token is converted to a 768-dimensional vector")
    print("   - Position information is added to preserve word order")

    print("\n3. MULTI-HEAD ATTENTION:")
    print("   - Model learns to focus on relevant words")
    print("   - Example: 'fantastic' strongly influences positive sentiment")
    print("   - 12 attention heads capture different relationships")

    print("\n4. FEED FORWARD & NORMALIZATION:")
    print("   - Non-linear transformations refine representations")
    print("   - Layer normalization stabilizes training")

    print("\n5. CLASSIFICATION:")
    print("   - [CLS] token representation → Linear layer → Softmax")
    print("   - Output: Probability distribution over classes")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
