"""
Step 3: Feature Extraction
Extract features from annotated data for model training
"""

import numpy as np
from collections import Counter
import json

class NERFeatureExtractor:
    """
    Extract features from tokens for NER model training
    """
    
    def __init__(self):
        self.word_vocab = {}
        self.char_vocab = {}
        self.pos_vocab = {}
        
    def extract_word_features(self, token):
        """
        Extract word-level features
        
        Args:
            token (str): Input token
        
        Returns:
            dict: Feature dictionary
        """
        features = {
            # Basic features
            'token_lower': token.lower(),
            'token_isupper': token.isupper(),
            'token_istitle': token.istitle(),
            'token_isdigit': token.isdigit(),
            
            # Shape features
            'token_length': len(token),
            'has_hyphen': '-' in token,
            'has_digit': any(c.isdigit() for c in token),
            'has_upper': any(c.isupper() for c in token),
            
            # Prefix/Suffix features
            'prefix_1': token[0] if len(token) > 0 else '',
            'prefix_2': token[:2] if len(token) > 1 else '',
            'prefix_3': token[:3] if len(token) > 2 else '',
            'suffix_1': token[-1] if len(token) > 0 else '',
            'suffix_2': token[-2:] if len(token) > 1 else '',
            'suffix_3': token[-3:] if len(token) > 2 else '',
            
            # Pattern features
            'is_punctuation': token in '.,!?;:',
            'is_bracket': token in '()[]{}',
            'is_quote': token in '"\'"',
        }
        
        return features
    
    def extract_context_features(self, tokens, index, window=2):
        """
        Extract context features (surrounding tokens)
        
        Args:
            tokens (list): List of tokens
            index (int): Current token index
            window (int): Context window size
        
        Returns:
            dict: Context features
        """
        features = {}
        
        # Previous tokens
        for i in range(1, window + 1):
            if index - i >= 0:
                features[f'prev_{i}'] = tokens[index - i].lower()
            else:
                features[f'prev_{i}'] = '<BOS>'  # Beginning of sentence
        
        # Next tokens
        for i in range(1, window + 1):
            if index + i < len(tokens):
                features[f'next_{i}'] = tokens[index + i].lower()
            else:
                features[f'next_{i}'] = '<EOS>'  # End of sentence
        
        return features
    
    def extract_pos_features(self, pos_tags, index, window=1):
        """
        Extract POS tag features
        
        Args:
            pos_tags (list): List of POS tags
            index (int): Current token index
            window (int): Context window size
        
        Returns:
            dict: POS features
        """
        features = {}
        
        # Current POS
        features['pos'] = pos_tags[index] if pos_tags else 'UNKNOWN'
        
        # Previous POS
        for i in range(1, window + 1):
            if index - i >= 0:
                features[f'pos_prev_{i}'] = pos_tags[index - i]
            else:
                features[f'pos_prev_{i}'] = '<BOS>'
        
        # Next POS
        for i in range(1, window + 1):
            if index + i < len(pos_tags):
                features[f'pos_next_{i}'] = pos_tags[index + i]
            else:
                features[f'pos_next_{i}'] = '<EOS>'
        
        return features
    
    def extract_all_features(self, tokens, pos_tags=None, index=None):
        """
        Extract all features for tokens
        
        Args:
            tokens (list): List of tokens
            pos_tags (list): List of POS tags (optional)
            index (int): Specific token index (optional)
        
        Returns:
            list or dict: Features for all tokens or specific token
        """
        if index is not None:
            # Extract features for single token
            features = {}
            features.update(self.extract_word_features(tokens[index]))
            features.update(self.extract_context_features(tokens, index))
            if pos_tags:
                features.update(self.extract_pos_features(pos_tags, index))
            return features
        else:
            # Extract features for all tokens
            all_features = []
            for i in range(len(tokens)):
                features = {}
                features.update(self.extract_word_features(tokens[i]))
                features.update(self.extract_context_features(tokens, i))
                if pos_tags:
                    features.update(self.extract_pos_features(pos_tags, i))
                all_features.append(features)
            return all_features
    
    def build_vocabulary(self, annotations):
        """
        Build vocabulary from annotations
        
        Args:
            annotations (list): List of annotated samples
        """
        word_counts = Counter()
        char_counts = Counter()
        
        for annotation in annotations:
            for token in annotation['tokens']:
                word_counts[token.lower()] += 1
                for char in token:
                    char_counts[char] += 1
        
        # Create vocabularies (with special tokens)
        self.word_vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, (word, _) in enumerate(word_counts.most_common(), start=2):
            self.word_vocab[word] = i
        
        self.char_vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, (char, _) in enumerate(char_counts.most_common(), start=2):
            self.char_vocab[char] = i
        
        print(f"✓ Vocabulary built:")
        print(f"  - Words: {len(self.word_vocab)}")
        print(f"  - Characters: {len(self.char_vocab)}")
    
    def tokens_to_ids(self, tokens):
        """
        Convert tokens to IDs using vocabulary
        
        Args:
            tokens (list): List of tokens
        
        Returns:
            list: List of token IDs
        """
        return [self.word_vocab.get(token.lower(), self.word_vocab['<UNK>']) 
                for token in tokens]
    
    def save_vocab(self, filename):
        """Save vocabulary to file"""
        vocab_data = {
            'word_vocab': self.word_vocab,
            'char_vocab': self.char_vocab
        }
        with open(filename, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        print(f"✓ Vocabulary saved to: {filename}")
    
    def load_vocab(self, filename):
        """Load vocabulary from file"""
        with open(filename, 'r') as f:
            vocab_data = json.load(f)
        self.word_vocab = vocab_data['word_vocab']
        self.char_vocab = vocab_data['char_vocab']
        print(f"✓ Vocabulary loaded from: {filename}")


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("NER FEATURE EXTRACTION")
    print("="*80)
    
    # Initialize feature extractor
    extractor = NERFeatureExtractor()
    
    # Example sentence
    tokens = ['Apple', 'Inc.', 'released', 'iPhone', '15', 'in', 'California']
    pos_tags = ['NNP', 'NNP', 'VBD', 'NNP', 'CD', 'IN', 'NNP']
    
    print("\nExample sentence:")
    print(f"Tokens: {tokens}")
    print(f"POS Tags: {pos_tags}")
    
    # Extract features for each token
    print("\n" + "="*80)
    print("EXTRACTED FEATURES")
    print("="*80)
    
    for i, token in enumerate(tokens):
        print(f"\n📝 Token: '{token}' (Position {i})")
        print("-"*80)
        
        features = extractor.extract_all_features(tokens, pos_tags, index=i)
        
        # Display important features
        print(f"  Word features:")
        print(f"    - Lowercase: {features['token_lower']}")
        print(f"    - Is title case: {features['token_istitle']}")
        print(f"    - Is digit: {features['token_isdigit']}")
        print(f"    - Prefix-3: {features['prefix_3']}")
        print(f"    - Suffix-3: {features['suffix_3']}")
        
        print(f"  Context features:")
        print(f"    - Previous token: {features['prev_1']}")
        print(f"    - Next token: {features['next_1']}")
        
        print(f"  POS features:")
        print(f"    - Current POS: {features['pos']}")
        print(f"    - Previous POS: {features['pos_prev_1']}")
        print(f"    - Next POS: {features['pos_next_1']}")
    
    # Load example annotations
    print("\n" + "="*80)
    print("BUILDING VOCABULARY")
    print("="*80)
    
    try:
        # Get script directory
        from pathlib import Path
        script_dir = Path(__file__).parent.resolve()
        annotations_path = script_dir / 'annotated_data' / 'example_annotations.json'
        
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        extractor.build_vocabulary(annotations)
        
        # Convert tokens to IDs
        print(f"\nToken to ID conversion:")
        token_ids = extractor.tokens_to_ids(tokens)
        for token, tid in zip(tokens, token_ids):
            print(f"  {token:<15} → {tid}")
        
        # Save vocabulary
        vocab_path = script_dir / 'annotated_data' / 'vocabulary.json'
        extractor.save_vocab(str(vocab_path))
        
    except FileNotFoundError:
        print("⚠ No annotations found. Run annotation_tool.py first!")
