"""
Multi-Scale Sentiment Analyzer: -3 to +3
Students: Complete the TODOs to implement 7-point sentiment scale
"""
print(">    Warning: running on a Linux subsystem takes a while...")

import torch
print("Imported torch")
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
print("Imported transformers")
from torch.utils.data import Dataset
import warnings
print("Imported warnings")
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import os


def class_to_sentiment_score(class_id):
    """
    TODO 3: Convert class ID (0-6) to sentiment score (-3 to +3)

    Hint: If class_id ranges from 0 to 6, and we want -3 to +3,
    what mathematical operation would you use?

    Args:
        class_id: Integer from 0 to 6
    Returns:
        sentiment_score: Integer from -3 to +3
    """
    # TODO: Write the conversion formula here
    sentiment_score = class_id - 3  # SOLUTION: Remove this line and let students figure it out

    return sentiment_score


def get_sentiment_label(score):
    """Convert numeric score to descriptive label"""
    labels = {
        -3: "Very Negative 😢",
        -2: "Negative 😞",
        -1: "Slightly Negative 😐",
        0: "Neutral 😶",
        1: "Slightly Positive 🙂",
        2: "Positive 😊",
        3: "Very Positive 🤩"
    }
    return labels.get(score, "Unknown")


def create_training_data():
    """
    TODO 2: Create labeled training data for 7-point scale

    Add more examples for each sentiment level.
    Format: (review_text, sentiment_class)
    where sentiment_class: 0=Very Negative, 3=Neutral, 6=Very Positive
    """

    training_data = [
        # Very Negative (-3) → Class 0
        ("This is the worst movie I have ever seen in my entire life!", 0),
        ("Absolutely terrible! Complete waste of time and money.", 0),
        ("Horrible film. I want my money back. Awful in every way.", 0),
        ("Utter garbage. I couldn’t even finish watching it.", 0),
        ("Painfully bad acting and a laughable script.", 0),
        ("Disgusting. I can’t believe this even got released.", 0),
        ("This movie is an insult to cinema.", 0),
        ("The plot made absolutely no sense at all.", 0),
        ("I regret pressing play on this disaster.", 0),
        ("One of the most poorly made films in history.", 0),
        ("Everything about this was wrong — acting, music, story.", 0),
        ("Just awful. I felt embarrassed watching it.", 0),
        ("Terrible pacing and cringe-worthy dialogue.", 0),
        ("I hated every second of this mess.", 0),
        ("The editing was so bad it gave me a headache.", 0),
        ("A complete disaster from start to finish.", 0),
        ("So bad I couldn’t stop laughing for the wrong reasons.", 0),
        ("One of the worst experiences I’ve had watching a movie.", 0),
        ("Truly unwatchable. Zero redeeming qualities.", 0),
        ("I’d rather watch paint dry than sit through this again.", 0),

        # Negative (-2) → Class 1
        ("This movie was quite disappointing and boring.", 1),
        ("Not good. Poor acting and weak plot.", 1),
        ("Mediocre at best, with many dull moments.", 1),
        ("It failed to keep my attention for more than ten minutes.", 1),
        ("The storyline was predictable and unoriginal.", 1),
        ("I didn’t hate it, but it definitely wasn’t enjoyable.", 1),
        ("The direction felt lazy and uninspired.", 1),
        ("Bad acting overshadowed any good parts.", 1),
        ("Some scenes were okay, but overall it was subpar.", 1),
        ("Soundtrack didn’t match the mood at all.", 1),
        ("The pacing dragged too much to stay interesting.", 1),
        ("Felt more like a chore than entertainment.", 1),
        ("The script was weak and full of clichés.", 1),
        ("There were moments of promise, but they fell flat.", 1),
        ("The lead performance couldn’t save the bad writing.", 1),
        ("It looked cheap and unfinished.", 1),
        ("Not terrible, but I wouldn’t recommend it.", 1),
        ("Dialogue was stiff and unnatural.", 1),
        ("Forgettable film that added nothing new.", 1),
        ("Below expectations in every way.", 1),

        # Slightly Negative (-1) → Class 2
        ("The movie had potential but didn't deliver.", 2),
        ("It was below average, not terrible but not good either.", 2),
        ("Some parts were interesting, but it didn’t hold up overall.", 2),
        ("Mildly disappointing, though it had its moments.", 2),
        ("It felt rushed, like they didn’t polish it enough.", 2),
        ("Could’ve been great with a better script.", 2),
        ("Average at best; not something I’d watch again.", 2),
        ("It started strong but lost momentum halfway.", 2),
        ("Acting was decent but the plot was weak.", 2),
        ("Not the worst, but far from memorable.", 2),
        ("There was effort, but the execution wasn’t there.", 2),
        ("I didn’t enjoy it much, but I can see others might.", 2),
        ("A bit disappointing considering the hype.", 2),
        ("Okay visuals, but emotionally flat.", 2),
        ("Felt incomplete, like something was missing.", 2),
        ("It had potential but poor pacing ruined it.", 2),
        ("A slightly underwhelming experience.", 2),
        ("Fine for background noise, not much else.", 2),
        ("It wasn’t terrible, just kind of dull.", 2),
        ("Could’ve been better with a few small changes.", 2),

        # Neutral (0) → Class 3
        ("It was okay, nothing special. Just average.", 3),
        ("The film was fine. Neither good nor bad.", 3),
        ("Completely average experience, nothing stood out.", 3),
        ("It was serviceable, but I don’t feel strongly either way.", 3),
        ("An okay movie to pass the time.", 3),
        ("Nothing memorable, but nothing awful either.", 3),
        ("It was a middle-of-the-road movie.", 3),
        ("Could take it or leave it.", 3),
        ("A decent way to spend an evening, I guess.", 3),
        ("It’s fine, but I won’t be rewatching it.", 3),
        ("An average film, pretty standard stuff.", 3),
        ("Didn’t love it, didn’t hate it.", 3),
        ("Neutral feelings — it just exists.", 3),
        ("A movie that’s hard to have an opinion about.", 3),
        ("It filled time, that’s about it.", 3),
        ("Watchable, but not exciting.", 3),
        ("Mediocre, but not offensively bad.", 3),
        ("An okay effort that neither impresses nor disappoints.", 3),
        ("A passable but unremarkable film.", 3),
        ("Pretty neutral — nothing to complain about.", 3),

        # Slightly Positive (+1) → Class 4
        ("Pretty decent movie. I enjoyed it overall.", 4),
        ("Good film with some nice moments.", 4),
        ("Solid effort, even if not groundbreaking.", 4),
        ("Pleasantly surprised by how much I liked it.", 4),
        ("Had a few flaws, but overall enjoyable.", 4),
        ("A charming film with heart.", 4),
        ("Entertaining enough for a one-time watch.", 4),
        ("A light, feel-good experience.", 4),
        ("Not perfect, but worth a watch.", 4),
        ("Good pacing and likable characters.", 4),
        ("Some parts dragged, but overall fun.", 4),
        ("Had a few strong performances.", 4),
        ("Enjoyable, though not a masterpiece.", 4),
        ("Decent direction and solid acting.", 4),
        ("I’d recommend it to casual viewers.", 4),
        ("Left me with a good impression.", 4),
        ("Nice visuals and good energy.", 4),
        ("Had heart and effort behind it.", 4),
        ("A pleasant surprise overall.", 4),
        ("A bit generic, but still fun.", 4),

        # Positive (+2) → Class 5
        ("Really great movie! I thoroughly enjoyed it.", 5),
        ("Excellent film with strong performances.", 5),
        ("One of the better movies I’ve seen this year.", 5),
        ("Well-made and genuinely engaging.", 5),
        ("Loved the story and the acting.", 5),
        ("A high-quality production with great direction.", 5),
        ("The cast delivered fantastic performances.", 5),
        ("Beautiful cinematography and great pacing.", 5),
        ("A wonderful mix of emotion and entertainment.", 5),
        ("It exceeded my expectations in a good way.", 5),
        ("Smart, entertaining, and heartfelt.", 5),
        ("I’d gladly watch it again.", 5),
        ("Fantastic visuals and memorable music.", 5),
        ("A well-rounded film with charm and depth.", 5),
        ("I was impressed by how well it was made.", 5),
        ("Feels polished and professional.", 5),
        ("Delivers exactly what it promises.", 5),
        ("Captivating from start to finish.", 5),
        ("A great example of modern filmmaking done right.", 5),
        ("Very enjoyable with a solid story arc.", 5),

        # Very Positive (+3) → Class 6
        ("Absolutely amazing! Best movie I've seen this year!", 6),
        ("Masterpiece! Incredible storytelling and perfect execution.", 6),
        ("This movie was phenomenal! A true work of art!", 6),
        ("One of the greatest films ever made!", 6),
        ("Blew me away — everything was flawless.", 6),
        ("Pure perfection. Every scene was breathtaking.", 6),
        ("Emotionally powerful and beautifully acted.", 6),
        ("A cinematic masterpiece that moved me deeply.", 6),
        ("This should win every award possible.", 6),
        ("An instant classic that I’ll remember forever.", 6),
        ("The writing, acting, and direction were all top-notch.", 6),
        ("Completely blew my expectations out of the water.", 6),
        ("Absolutely loved every second of it.", 6),
        ("It touched my heart in ways I didn’t expect.", 6),
        ("Flawless execution and emotional impact.", 6),
        ("Unforgettable. I can’t stop thinking about it.", 6),
        ("A film that defines excellence.", 6),
        ("A perfect balance of story and emotion.", 6),
        ("The best cinematic experience I’ve had in years.", 6),
        ("Beyond extraordinary — truly special.", 6),
    ]


    return training_data


class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def train_sentiment_model(training_data, model, tokenizer, device):
    """
    Fine-tune model on 7-point sentiment data
    
    Args:
        training_data: List of (text, label) tuples
        model: Pre-trained model to fine-tune
        tokenizer: Tokenizer
        device: CPU or GPU
    Returns:
        trained_model: Fine-tuned model
    """
    print("\n" + "="*80)
    print("🎓 TRAINING MODE: Fine-tuning model on custom 7-point scale")
    print("="*80)
    
    # Separate texts and labels
    texts = [text for text, _ in training_data]
    labels = [label for _, label in training_data]
    
    # Split into train and validation sets (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training examples: {len(X_train)}")
    print(f"   Validation examples: {len(X_val)}")
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,              # More epochs for small dataset
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",           # Changed from evaluation_strategy
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    print("\n🚀 Starting training... (this may take a few minutes)")
    print("-" * 80)
    trainer.train()
    
    # Save model
    print("\n💾 Saving trained model...")
    model.save_pretrained('./sentiment_model_7point')
    tokenizer.save_pretrained('./sentiment_model_7point')
    print("✅ Model saved to ./sentiment_model_7point")
    
    return model


def analyze_sentiment(text, model, tokenizer, device):
    """
    Analyze sentiment of a single text using the trained model

    Args:
        text: Review text to analyze
        model: Trained model
        tokenizer: Tokenizer
        device: CPU or GPU
    Returns:
        sentiment_score: -3 to +3
        confidence: 0 to 1
        sentiment_label: Descriptive label
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = probabilities.argmax(dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    # Convert to sentiment score
    sentiment_score = class_to_sentiment_score(predicted_class)
    sentiment_label = get_sentiment_label(sentiment_score)

    return sentiment_score, confidence, sentiment_label


def main():
    print("=" * 80)
    print("Multi-Scale Sentiment Analyzer: -3 to +3")
    print("=" * 80)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}")

    # Check if a previously trained model exists
    model_loaded_from_saved = False
    user_wants_to_retrain = False
    
    if os.path.exists('./sentiment_model_7point'):
        print("\n✅ Found previously trained model!")
        print("\nWhat would you like to do?")
        print("1. Load existing trained model (fastest)")
        print("2. Retrain model on data (overwrites existing model)")
        print("3. Load base model without fine-tuning (for testing)")
        
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            print("\n🤖 Loading saved model...")
            try:
                model = AutoModelForSequenceClassification.from_pretrained('./sentiment_model_7point')
                tokenizer = AutoTokenizer.from_pretrained('./sentiment_model_7point')
                model.to(device)
                model_loaded_from_saved = True
                print("✓ Saved model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading saved model: {e}")
                print("Will load base model instead...")
        elif choice == '2':
            print("\n⚠️  Retraining will overwrite the existing model.")
            confirm = input("Are you sure? (y/n): ").strip().lower()
            if confirm == 'y':
                user_wants_to_retrain = True
                print("\n🤖 Loading base model for retraining...")
            else:
                print("Retraining cancelled. Loading saved model instead...")
                try:
                    model = AutoModelForSequenceClassification.from_pretrained('./sentiment_model_7point')
                    tokenizer = AutoTokenizer.from_pretrained('./sentiment_model_7point')
                    model.to(device)
                    model_loaded_from_saved = True
                    print("✓ Saved model loaded successfully!")
                except Exception as e:
                    print(f"❌ Error loading saved model: {e}")
                    print("Will load base model instead...")
        else:  # choice == '3' or any other input
            print("\n🤖 Loading base model without using saved model...")

    # Load base model if we didn't load a saved one
    if not model_loaded_from_saved:
        # TODO 1: Load model with correct number of labels
        print("\n🤖 Loading base model...")
        model_name = "distilbert-base-uncased"

        # TODO: Modify num_labels for 7-point scale (-3 to +3)
        # How many classes do we need? Fill in the blank below:
        num_labels = 7  # SOLUTION: Students should figure out this is 7

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True  # Allow different output size
            )
            model.to(device)
            print("✓ Base model loaded successfully!")
            print(f"  - Model: {model_name}")
            print(f"  - Number of sentiment classes: {num_labels}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\nNote: If you see a network error, the model needs to be downloaded first.")
            print("Make sure you have internet access or use a pre-downloaded model.")
            return

    # Show training data structure
    print("\n📊 Training Data Structure:")
    training_data = create_training_data()
    print(f"  - Total examples: {len(training_data)}")

    # Count examples per class
    from collections import Counter
    class_counts = Counter([label for _, label in training_data])
    print("\n  Examples per sentiment level:")
    for class_id in sorted(class_counts.keys()):
        score = class_to_sentiment_score(class_id)
        label = get_sentiment_label(score)
        count = class_counts[class_id]
        print(f"    Class {class_id} (Score {score:+d}): {label} - {count} examples")

    # Ask user if they want to train the model (skip if already loaded from saved)
    if user_wants_to_retrain:
        # User chose to retrain, so train the model
        print("\n" + "=" * 80)
        print("🎯 RETRAINING MODEL")
        print("=" * 80)
        model = train_sentiment_model(training_data, model, tokenizer, device)
        print("\n✅ Retraining complete! Now using newly fine-tuned model.")
    elif not model_loaded_from_saved:
        # No saved model was loaded, so ask about training
        print("\n" + "=" * 80)
        print("🎯 TRAINING OPTIONS")
        print("=" * 80)
        print("1. Train model on custom data (recommended for better accuracy)")
        print("2. Use pre-trained model without fine-tuning (faster, less accurate)")
        print("\nNote: Training takes 5-10 minutes depending on your hardware.")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            model = train_sentiment_model(training_data, model, tokenizer, device)
            print("\n✅ Training complete! Now using fine-tuned model.")
        else:
            print("\n⚠️  Using pre-trained model without fine-tuning")
            print("    Results may be less accurate for the 7-point scale.")
    else:
        print("\n✅ Using previously trained model. Skipping training options.")

    # Test with example reviews
    print("\n🧪 Testing Multi-Scale Sentiment Analysis")
    print("=" * 80)

    test_reviews = [
        "This movie was absolutely phenomenal! Best film of the decade!",
        "Really enjoyed this film. Great performances all around.",
        "Pretty good movie, had some nice moments.",
        "It was okay, nothing particularly memorable.",
        "The film had potential but fell short.",
        "Quite disappointing. Expected much better.",
        "Absolutely terrible! Worst movie I've ever seen!",
    ]

    for i, review in enumerate(test_reviews, 1):
        print(f"\n{i}. Review: \"{review}\"")

        # Analyze sentiment
        score, confidence, label = analyze_sentiment(review, model, tokenizer, device)

        # Display results
        print(f"   Sentiment Score: {score:+d}/3")
        print(f"   Label: {label}")
        print(f"   Confidence: {confidence:.1%}")

        # Visual representation
        visual_bar = "█" * (score + 3) + "░" * (3 - score)
        print(f"   Scale: [{visual_bar}] ({score:+d})")

    # Interactive testing
    print("\n" + "=" * 80)
    print("🎮 Interactive Mode: Test Your Own Reviews")
    print("=" * 80)
    print("Enter movie reviews to analyze (or 'quit' to exit)\n")

    while True:
        user_input = input("Enter review: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using the Multi-Scale Sentiment Analyzer!")
            break

        if not user_input:
            continue

        # Analyze
        score, confidence, label = analyze_sentiment(user_input, model, tokenizer, device)

        # Display
        print(f"\n  → Sentiment Score: {score:+d}/3")
        print(f"  → Label: {label}")
        print(f"  → Confidence: {confidence:.1%}\n")


if __name__ == "__main__":
    main()
