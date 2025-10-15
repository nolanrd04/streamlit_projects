# Named Entity Recognition (NER) System with IOB Tagging

A complete, from-scratch implementation of a Named Entity Recognition system using IOB tagging, feature extraction, and CRF (Conditional Random Field) model training.

## 📁 Project Structure

```
NER/
├── custom_ner_labels.py      # Step 1: Define custom entity types & IOB labels
├── annotation_tool.py         # Step 2: Annotate training data
├── feature_extraction.py      # Step 3: Extract features from tokens
├── model_training.py          # Step 4: Train CRF model
├── model_evaluation.py        # Step 5: Evaluate model performance
├── ner_pipeline.py            # Complete end-to-end pipeline
├── preprocessor.py            # NLTK tokenization & POS tagging
├── requirements.txt           # Dependencies
└── annotated_data/            # Output directory (auto-created)
    ├── training_annotations.json
    ├── training_annotations.conll
    ├── ner_model.pkl
    └── evaluation_report.json
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python ner_pipeline.py
```

This will:
- ✅ Create annotated training data
- ✅ Train a CRF model
- ✅ Evaluate the model
- ✅ Test predictions on new sentences

## 📚 Detailed Workflow

### Step 1: Define Custom Labels

**File:** `custom_ner_labels.py`

Define your domain-specific entity types:

```python
CUSTOM_ENTITY_TYPES = [
    'PERSON',      # Person names
    'COMPANY',     # Company/Organization names
    'PRODUCT',     # Product names
    'LOCATION',    # Geographic locations
    'DATE',        # Dates
    'MONEY',       # Monetary amounts
]
```

Run to see the generated IOB scheme:

```bash
python custom_ner_labels.py
```

**Output:**
- O (Outside)
- B-PERSON, I-PERSON
- B-COMPANY, I-COMPANY
- B-PRODUCT, I-PRODUCT
- etc.

### Step 2: Annotate Data

**File:** `annotation_tool.py`

Two annotation methods:

#### A. Quick Annotation (Programmatic)

```python
from annotation_tool import NERAnnotator
from custom_ner_labels import LABELS

annotator = NERAnnotator(LABELS)

# Annotate with entity spans
text = "Apple Inc. released iPhone 15 for $799"
entities = [
    (0, 10, 'COMPANY'),    # Apple Inc.
    (20, 29, 'PRODUCT'),   # iPhone 15
    (34, 38, 'MONEY')      # $799
]

annotator.quick_annotate(text, entities)
annotator.save_annotations('my_annotations.json')
```

#### B. Interactive Annotation

```python
sentences = [
    "Apple Inc. launched the new MacBook Pro.",
    "John Smith works at Google in California."
]
annotator.annotate_batch(sentences)
```

Run standalone:
```bash
python annotation_tool.py
```

### Step 3: Feature Extraction

**File:** `feature_extraction.py`

Extracts rich features from tokens:

**Features Extracted:**
- **Word features**: lowercase, case patterns, length, digits
- **Shape features**: capitalization, hyphens, special characters
- **Prefix/Suffix**: 1-3 character prefixes and suffixes
- **Context features**: surrounding words (window=2)
- **POS features**: Part-of-speech tags from NLTK

Run to see feature examples:
```bash
python feature_extraction.py
```

### Step 4: Train Model

**File:** `model_training.py`

Trains a CRF (Conditional Random Field) model:

```python
from model_training import NERModelTrainer

trainer = NERModelTrainer(LABELS)
X_train, y_train = trainer.prepare_training_data('annotations.json')
trainer.train(X_train, y_train, max_iterations=100)
trainer.save_model('ner_model.pkl')
```

Run standalone:
```bash
python model_training.py
```

### Step 5: Evaluate Model

**File:** `model_evaluation.py`

Comprehensive evaluation metrics:

**Metrics Calculated:**
- ✅ Token-level accuracy
- ✅ Entity-level precision, recall, F1
- ✅ Per-entity-type metrics
- ✅ Confusion matrix
- ✅ True/False positives/negatives

Run standalone:
```bash
python model_evaluation.py
```

## 🎯 Example Usage

### Train and Predict

```python
from ner_pipeline import NERPipeline

# Initialize
pipeline = NERPipeline()

# Train
training_data = [
    ("Apple Inc. released iPhone 15", 
     [(0, 10, 'COMPANY'), (20, 29, 'PRODUCT')]),
    # ... more examples
]

pipeline.step1_annotate(training_data)
pipeline.step2_train('annotated_data/training_annotations.json')

# Predict
pipeline.display_prediction("Microsoft launched Surface Pro")
```

### Load Existing Model

```python
from model_training import NERModelTrainer

trainer = NERModelTrainer(LABELS)
trainer.load_model('annotated_data/ner_model.pkl')

# Predict
tokens = ['Google', 'released', 'Pixel', '8']
predictions = trainer.predict(tokens)
```

## 📊 Evaluation Metrics Explained

### Token-Level Metrics
- **Accuracy**: Percentage of correctly labeled tokens

### Entity-Level Metrics
- **Precision**: Of predicted entities, how many are correct?
- **Recall**: Of true entities, how many did we find?
- **F1-Score**: Harmonic mean of precision and recall

### Example Output

```
TOKEN-LEVEL METRICS
  Accuracy: 0.9200
  Correct:  92 / 100

ENTITY-LEVEL METRICS (Overall)
  Precision: 0.8750
  Recall:    0.8235
  F1-Score:  0.8485

PER-ENTITY METRICS
Entity Type     Precision    Recall       F1-Score     Support
COMPANY         0.9000       0.8571       0.8780       14
PERSON          0.8571       0.7500       0.8000       8
PRODUCT         0.9231       0.9231       0.9231       13
```

## 🎨 Customization

### Add New Entity Types

Edit `custom_ner_labels.py`:

```python
CUSTOM_ENTITY_TYPES = [
    'PERSON',
    'COMPANY',
    'YOUR_NEW_TYPE',  # Add here!
]
```

### Adjust Features

Edit `feature_extraction.py`:

```python
def extract_word_features(self, token):
    features = {
        'your_custom_feature': your_function(token),
        # Add more features...
    }
```

### Tune Model

Edit `model_training.py`:

```python
self.model = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,  # Increase for better training
    c1=0.1,              # L1 regularization
    c2=0.1,              # L2 regularization
)
```

## 📈 Performance Tips

### 1. More Training Data
- Aim for 1000+ annotated sentences
- Diverse examples covering different contexts

### 2. Better Features
- Add domain-specific features
- Include word embeddings (Word2Vec, GloVe)
- Add gazetteer features (known entity lists)

### 3. Model Tuning
- Experiment with hyperparameters
- Try different algorithms (lbfgs, l2sgd, ap)
- Adjust regularization (c1, c2)

## 🔧 Advanced Usage

### Use with NLTK POS Tags

```python
from preprocessor import pos_tag_text
from model_training import NERModelTrainer

tokens = ['Apple', 'Inc.', 'released', 'iPhone']
pos_tags = pos_tag_text(tokens)

trainer.predict(tokens, pos_tags)
```

### Export to CoNLL Format

```python
annotator.export_to_conll('output.conll')
```

### Batch Processing

```python
texts = ["sentence 1", "sentence 2", "..."]
for text in texts:
    predictions = pipeline.predict(text)
    print(predictions)
```

## 📝 IOB Tagging Reference

### Tag Format
- **B-TYPE**: **B**eginning of entity of TYPE
- **I-TYPE**: **I**nside/continuation of entity of TYPE
- **O**: **O**utside any entity

### Example

```
Text:   Apple Inc. released iPhone 15
Tokens: Apple  Inc.  released  iPhone  15
Labels: B-COMP I-COMP O        B-PROD  I-PROD
```

## 🐛 Troubleshooting

### NLTK Resources Not Found
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger_eng')"
```

### sklearn-crfsuite Install Issues
```bash
pip install --upgrade pip setuptools wheel
pip install sklearn-crfsuite
```

### Low Performance
- ✅ Add more training data
- ✅ Check annotation quality
- ✅ Add more features
- ✅ Increase training iterations

## 📚 References

- IOB Tagging: https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)
- CRF Model: https://sklearn-crfsuite.readthedocs.io/
- NLTK: https://www.nltk.org/
- CoNLL-2003: https://www.aclweb.org/anthology/W03-0419/

## 🎓 For CST-435 Students

This complete implementation covers:
1. ✅ **Annotation** - IOB tagging scheme, custom entity types
2. ✅ **Model Development** - Feature extraction, CRF training
3. ✅ **Evaluation** - Comprehensive metrics, confusion matrix

Use this as a foundation for your NER project. Customize entity types, add features, and train on your own domain-specific data!

## 📧 Questions?

Check the code comments or run individual modules to see examples:
- `python custom_ner_labels.py`
- `python annotation_tool.py`
- `python feature_extraction.py`
- `python model_training.py`
- `python model_evaluation.py`
- `python ner_pipeline.py`

---
**Happy NER Building! 🚀**
