"""
IMDB Sentiment Analysis with Transformers
Using Hugging Face's transformers library to fine-tune a pre-trained model
"""

import numpy as np
import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Set environment variables for better timeout handling
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['CURL_CA_BUNDLE'] = ''


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


def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenize the text data"""
    return tokenizer(examples['text'], truncation=True, max_length=max_length, padding=True)


def main():
    print("=" * 80)
    print("IMDB Sentiment Analysis with Transformer Architecture")
    print("=" * 80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}")

    # Load IMDB dataset with retry logic
    print("\n📚 Loading IMDB dataset...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            dataset = load_dataset("imdb", cache_dir="./cache", download_mode="reuse_cache_if_exists")
            print(f"✓ Dataset loaded successfully!")
            print(f"  - Train samples: {len(dataset['train'])}")
            print(f"  - Test samples: {len(dataset['test'])}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Attempt {attempt + 1} failed. Retrying...")
                import time
                time.sleep(5)
            else:
                print(f"❌ Failed to load dataset after {max_retries} attempts: {e}")
                print("Please check your internet connection and try again.")
                return

    # Use a smaller subset for faster training (optional)
    # Uncomment the following lines to use a smaller dataset for quick testing
    # dataset['train'] = dataset['train'].select(range(1000))
    # dataset['test'] = dataset['test'].select(range(500))

    # Load pre-trained model and tokenizer
    # Using DistilBERT for efficiency (smaller and faster than BERT)
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
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    print("✓ Tokenization complete!")

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./imdb_sentiment_model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_dir='./logs',
        logging_steps=500,
        warmup_steps=500,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
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
    trainer.save_model("./imdb_sentiment_model_final")
    tokenizer.save_pretrained("./imdb_sentiment_model_final")
    print("✓ Model saved to './imdb_sentiment_model_final'")

    # Test with sample predictions
    print("\n🧪 Testing with sample predictions...")
    print("=" * 80)

    test_reviews = [
        "This movie was absolutely fantastic! Best film I've seen this year.",
        "Terrible movie, waste of time. I want my money back.",
        "The acting was decent but the plot was confusing.",
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


if __name__ == "__main__":
    main()
