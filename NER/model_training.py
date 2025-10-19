"""
Step 4: Model Training
Train a CRF (Conditional Random Field) model for NER
"""

import sklearn_crfsuite
from sklearn_crfsuite import metrics
import json
import pickle
from feature_extraction import NERFeatureExtractor
from custom_ner_labels import LABELS, LABEL2ID, ID2LABEL

class NERModelTrainer:
    """
    Train and manage NER models using CRF
    """
    
    def __init__(self, labels):
        """
        Initialize trainer
        
        Args:
            labels (list): List of NER labels
        """
        self.labels = labels
        self.feature_extractor = NERFeatureExtractor()
        self.model = None
    
    def prepare_training_data(self, annotations_file):
        """
        Load and prepare training data
        
        Args:
            annotations_file (str): Path to annotations JSON file
        
        Returns:
            tuple: (X_train, y_train) - features and labels
        """
        print(f"Loading annotations from: {annotations_file}")
        
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"✓ Loaded {len(annotations)} annotated samples")
        
        # Build vocabulary
        self.feature_extractor.build_vocabulary(annotations)
        
        # Extract features for each sample
        X_train = []
        y_train = []
        
        for annotation in annotations:
            tokens = annotation['tokens']
            ner_tags = annotation['ner_tags']
            
            # Extract features for all tokens in this sentence
            sentence_features = self.feature_extractor.extract_all_features(tokens)
            
            X_train.append(sentence_features)
            y_train.append(ner_tags)
        
        print(f"✓ Prepared {len(X_train)} training samples")
        
        return X_train, y_train
    
    def train(self, X_train, y_train, algorithm='lbfgs', max_iterations=100):
        """
        Train CRF model
        
        Args:
            X_train (list): Training features
            y_train (list): Training labels
            algorithm (str): Training algorithm
            max_iterations (int): Maximum training iterations
        """
        print("\n" + "="*80)
        print("TRAINING CRF MODEL")
        print("="*80)
        
        # Initialize CRF model
        self.model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            max_iterations=max_iterations,
            all_possible_transitions=True,
            verbose=True
        )
        
        # Train
        print(f"\nTraining with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        print("\n✓ Training complete!")
    
    def predict(self, tokens, pos_tags=None):
        """
        Predict NER tags for tokens
        
        Args:
            tokens (list): List of tokens
            pos_tags (list): List of POS tags (optional)
        
        Returns:
            list: Predicted NER tags
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(tokens, pos_tags)
        
        # Predict
        predictions = self.model.predict([features])[0]
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test (list): Test features
            y_test (list): Test labels
        
        Returns:
            dict: Evaluation metrics
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        # Predict
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        print("\n📊 Classification Report:")
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=self.labels, digits=3
        ))
        
        # Per-entity metrics
        print("\n📊 Per-Entity Metrics:")
        entity_labels = [label for label in self.labels if label != 'O']
        
        for label in entity_labels:
            precision = metrics.flat_precision_score(y_test, y_pred, 
                                                     average=None, labels=[label])
            recall = metrics.flat_recall_score(y_test, y_pred, 
                                               average=None, labels=[label])
            f1 = metrics.flat_f1_score(y_test, y_pred, 
                                       average=None, labels=[label])
            
            if len(precision) > 0:
                print(f"  {label:<15} Precision: {precision[0]:.3f}  "
                      f"Recall: {recall[0]:.3f}  F1: {f1[0]:.3f}")
        
        # Overall metrics
        accuracy = metrics.flat_accuracy_score(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': metrics.flat_classification_report(
                y_test, y_pred, labels=self.labels, digits=3, output_dict=True
            )
        }
        
        print(f"\n✓ Overall Accuracy: {accuracy:.3f}")
        
        return results
    
    def save_model(self, model_path='ner_model.pkl'):
        """
        Save trained model
        
        Args:
            model_path (str): Path to save model
        """
        model_data = {
            'model': self.model,
            'labels': self.labels,
            'word_vocab': self.feature_extractor.word_vocab,
            'char_vocab': self.feature_extractor.char_vocab
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to: {model_path}")
    
    def load_model(self, model_path='ner_model.pkl'):
        """
        Load trained model
        
        Args:
            model_path (str): Path to model file
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.labels = model_data['labels']
        self.feature_extractor.word_vocab = model_data['word_vocab']
        self.feature_extractor.char_vocab = model_data['char_vocab']
        
        print(f"✓ Model loaded from: {model_path}")
    
    def analyze_predictions(self, tokens, true_labels, pred_labels):
        """
        Analyze and display predictions
        
        Args:
            tokens (list): List of tokens
            true_labels (list): True NER labels
            pred_labels (list): Predicted NER labels
        """
        print("\n" + "="*80)
        print("PREDICTION ANALYSIS")
        print("="*80)
        
        print(f"\n{'Token':<15} {'True Label':<15} {'Predicted':<15} {'Match':<10}")
        print("-"*55)
        
        correct = 0
        for token, true_label, pred_label in zip(tokens, true_labels, pred_labels):
            match = "✓" if true_label == pred_label else "✗"
            if true_label == pred_label:
                correct += 1
            
            print(f"{token:<15} {true_label:<15} {pred_label:<15} {match:<10}")
        
        accuracy = correct / len(tokens) if tokens else 0
        print(f"\n✓ Token-level accuracy: {accuracy:.3f} ({correct}/{len(tokens)})")


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("NER MODEL TRAINING")
    print("="*80)
    
    # Check if annotations exist
    import os
    from pathlib import Path
    
    script_dir = Path(__file__).parent.resolve()
    annotation_file = script_dir / 'annotated_data' / 'example_annotations.json'
    
    if not os.path.exists(annotation_file):
        print(f"\n⚠ No annotations found at: {annotation_file}")
        print("Please run annotation_tool.py first to create training data!")
    else:
        # Initialize trainer
        trainer = NERModelTrainer(LABELS)
        
        # Prepare data
        X_train, y_train = trainer.prepare_training_data(str(annotation_file))
        
        # Split into train/test (80/20)
        split_idx = int(0.8 * len(X_train))
        X_train_split = X_train[:split_idx]
        y_train_split = y_train[:split_idx]
        X_test = X_train[split_idx:]
        y_test = y_train[split_idx:]
        
        print(f"\nDataset split:")
        print(f"  - Training samples: {len(X_train_split)}")
        print(f"  - Test samples: {len(X_test)}")
        
        # Train model
        trainer.train(X_train_split, y_train_split, max_iterations=100)
        
        # Evaluate
        if len(X_test) > 0:
            results = trainer.evaluate(X_test, y_test)
        
        # Save model
        model_path = script_dir / 'annotated_data' / 'ner_model.pkl'
        trainer.save_model(str(model_path))
        
        # Example prediction
        print("\n" + "="*80)
        print("EXAMPLE PREDICTION")
        print("="*80)
        
        test_tokens = ['Google', 'released', 'Pixel', '8', 'for', '$699']
        predictions = trainer.predict(test_tokens)
        
        print(f"\nInput: {' '.join(test_tokens)}")
        print(f"\n{'Token':<15} {'Predicted Label':<15}")
        print("-"*30)
        for token, label in zip(test_tokens, predictions):
            print(f"{token:<15} {label:<15}")
