
# Nepali Name Generator

A modularized version of the [makemore project](https://github.com/karpathy/makemore) (by Andrej Karpathy) adapted for generating Nepali names. This project takes Nepali name datasets and generates new name suggestions using various neural network architectures.

## Project Structure

```
nepali-name-generator/
├── app.py                 # Streamlit web interface
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration classes and hyperparameters
│   ├── models.py          # All neural network model implementations
│   ├── data.py            # Data loading and preprocessing utilities
│   ├── utils.py           # Helper functions for generation, evaluation, and sampling
│   └── train.py           # Main training script
├── data/
│   ├── male.txt           # Male Nepali names (cleaned, 6,212 names)
│   ├── female.txt         # Female Nepali names (cleaned, 2,575 names)
│   ├── names_cleaned.txt  # Combined cleaned dataset
│   ├── names.txt          # Original combined dataset
│   └── backup/            # Original datasets before cleaning
│       ├── male_original.txt
│       └── female_original.txt
├── models/                # Saved model checkpoints
│   ├── base/              # Initial trained model
│   ├── best_results/      # Optimized combined model
│   ├── male_finetuned/    # Male-specific fine-tuned model
│   └── female_finetuned/  # Female-specific fine-tuned model
├── scripts/               # Utility scripts
│   ├── analyze_data.py    # Data analysis tools
│   ├── clean_data.py      # Data cleaning pipeline
│   ├── config_examples.py # Configuration examples
│   ├── fine_tune_model.py # Model fine-tuning script
│   └── generate_names.py  # Command-line name generation
├── tests/                 # Unit tests
├── README.md
├── DATA_CLEANING_REPORT.md # Detailed data cleaning documentation
├── NAME_GENERATION_README.md # Generation-specific documentation
└── requirements.txt       # Python dependencies
```

## Features

- **Streamlit Web Interface**: Beautiful, interactive web application for easy name generation
- **Modular Design**: Clean separation of concerns for easy maintenance and extension
- **Multiple Models**: Support for various architectures from simple bigrams to Transformers
- **Nepali Name Support**: Handles Devanagari script and multiple name files
- **Data Cleaning Pipeline**: Automated data preprocessing and backup management
- **Flexible Data Loading**: Can combine male.txt and female.txt or use single files
- **Gender-Specific Models**: Fine-tuned models for male, female, or combined name generation
- **Easy Fine-Tuning**: Scripts to create specialized models from base models
- **Advanced Configuration**: Centralized config system with presets and command-line overrides

## Technical Summary

This project trains and samples from character-level neural language models implemented in PyTorch. The primary model is a GPT-2-like Transformer (causal self-attention) with options for simpler architectures (RNN/GRU, BoW, MLP). Names are generated autoregressively with temperature / top-k sampling.

## Quick Start

### Option 1: Web Interface (Recommended)
Run the interactive Streamlit web application:

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web interface
streamlit run app.py
```

Then open your browser to `http://localhost:8501` for an intuitive web interface with:
- Real-time name generation with customizable parameters
- Gender-specific model selection (male/female/both)
- Advanced settings (temperature, top-k sampling, name length)
- Bulk generation and download capabilities
- Beautiful, responsive design

### Option 2: Command Line Training and Generation

## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration System

The project uses a centralized configuration system for all hyperparameters and settings. All configurable parameters are defined in `src/config.py` with sensible defaults.

### Configuration Classes

- **DataConfig**: Data loading settings (input files, train/test split, encoding)
- **TrainingConfig**: Training parameters (batch size, learning rate, epochs, intervals)
- **SystemConfig**: System settings (device, seed, work directory, model type)
- **SamplingConfig**: Generation parameters (temperature, max length, number of samples)

### Using the Configuration System

#### Default Configuration
```bash
python src/train.py
```

#### Custom Configuration via Command Line
```bash
python src/train.py --input-files data/male.txt data/female.txt --batch-size 64 --learning-rate 0.001 --max-steps 5000
```

#### Configuration Presets
The system provides several preset configurations:

- `get_default_config()` - Standard settings for general use
- `get_quick_train_config()` - Fast training for testing (1000 steps, smaller batch)
- `get_production_config()` - Optimized for production (larger batch, lower learning rate)

#### Example: Quick Training
```bash
python src/train.py --input-files data/male.txt data/female.txt --work-dir models/quick --max-steps 1000 --batch-size 16
```

### Configuration Parameters

#### Data Configuration
- `input_file`: Single input file path
- `input_files`: List of input files (for combining datasets)
- `test_split_ratio`: Fraction of data for testing (default: 0.1)
- `max_test_size`: Maximum test set size (default: 1000)
- `encoding`: File encoding (default: 'utf-8')

#### Training Configuration
- `max_steps`: Maximum training steps (-1 for infinite)
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay for regularization
- `eval_interval`: Evaluate every N steps
- `sample_interval`: Generate samples every N steps

#### System Configuration
- `device`: Computing device ('cpu', 'cuda', 'mps')
- `seed`: Random seed for reproducibility
- `work_dir`: Output directory for models and logs
- `model_type`: Model architecture ('transformer', 'bigram', etc.)
- `resume`: Resume training from checkpoint
- `sample_only`: Only generate samples, don't train

### Configuration Examples

See `scripts/config_examples.py` for detailed examples of how to use the configuration system programmatically.

```python
from config import get_quick_train_config, Config, DataConfig

# Use preset
config = get_quick_train_config()

# Custom configuration
config = Config(
    data=DataConfig(
        input_files=['data/male.txt', 'data/female.txt'],
        test_split_ratio=0.15
    ),
    training=TrainingConfig(
        max_steps=2000,
        batch_size=32
    )
)
```

## Gender-Specific Name Generation

The project now supports gender-specific name generation through fine-tuned models. You can generate names that are more likely to be male, female, or use the combined model.

### Available Models

- **Combined Model** (`models/best_results/`): Trained on both male and female names
- **Male Model** (`models/male_finetuned/`): Fine-tuned specifically for male names
- **Female Model** (`models/female_finetuned/`): Fine-tuned specifically for female names

### Quick Name Generation

#### Generate Male Names
```bash
# Edit scripts/generate_names.py and set:
GENDER_PREFERENCE = "male"
# Then run:
python scripts/generate_names.py
```

#### Generate Female Names
```bash
# Edit scripts/generate_names.py and set:
GENDER_PREFERENCE = "female"
# Then run:
python scripts/generate_names.py
```

#### Generate Mixed Names
```bash
# Edit scripts/generate_names.py and set:
GENDER_PREFERENCE = "both"
# Then run:
python scripts/generate_names.py
```

### Fine-Tuning Your Own Models

Create specialized models by fine-tuning the base model on gender-specific data:

#### Fine-Tune Male Model
```bash
# Edit scripts/fine_tune_model.py and set:
TARGET_DATA = ["data/male.txt"]
OUTPUT_DIR = "models/male_finetuned"
# Then run:
python scripts/fine_tune_model.py
```

#### Fine-Tune Female Model
```bash
# Edit scripts/fine_tune_model.py and set:
TARGET_DATA = ["data/female.txt"]
OUTPUT_DIR = "models/female_finetuned"
# Then run:
python scripts/fine_tune_model.py
```

### Fine-Tuning Parameters

- `MAX_STEPS`: Number of fine-tuning steps (default: 500)
- `LEARNING_RATE`: Fine-tuning learning rate (default: 0.0001)
- `BATCH_SIZE`: Batch size for fine-tuning (default: 16)
- `TARGET_DATA`: Data files to fine-tune on

## Data Files

The project includes cleaned and organized datasets:

- `data/male.txt` - Male Nepali names (6,212 cleaned names)
- `data/female.txt` - Female Nepali names (2,575 cleaned names)
- `data/names_cleaned.txt` - Combined cleaned dataset
- `data/backup/` - Original datasets before cleaning

**Data Cleaning Process**: All names have been preprocessed to use only lowercase a-z characters, removing special characters, numbers, and mixed case for consistent and stable training. See `DATA_CLEANING_REPORT.md` for detailed information.

## Model Architectures

The project implements several neural network architectures:

- **Bigram**: Simple character-to-character prediction using a lookup table
- **MLP**: Multi-layer perceptron following Bengio et al. 2003
- **RNN/GRU**: Recurrent neural networks for sequence modeling
- **BoW**: Bag of Words model with averaging
- **Transformer**: Full transformer architecture as in GPT (primary model)

### Available Pre-trained Models

- **Base Model** (`models/base/`): Initial trained model
- **Best Results** (`models/best_results/`): Optimized combined model
- **Male Fine-tuned** (`models/male_finetuned/`): Specialized for male names
- **Female Fine-tuned** (`models/female_finetuned/`): Specialized for female names

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Adding New Models

1. Add model class to `src/models.py`
2. Update the model selection logic in `src/train.py`
3. Add model type to argument parser

### Adding New Data Sources

1. Add data loading logic to `src/data.py`
2. Update `create_datasets()` function if needed
3. Add new command-line arguments to `src/train.py`

## How the Transformer Network Works Under the Hood

### The Magic Behind Nepali Name Generation

This project leverages a **GPT-2 style transformer architecture** to understand and generate Nepali names. Here's how the neural network learns to create authentic-sounding names from scratch:

### 1. Character-Level Language Modeling

Unlike word-based models, our transformer operates at the **character level**:

```
Input:  "र ा म"  →  Tokens: [र, ा, म]
Output: Probability distribution over next character
```

The model learns patterns like:
- After "र", "ा" is very likely (forming "रा")
- After "क", "म" often follows (forming common endings)
- Certain character combinations are distinctly Nepali

### 2. Self-Attention: Understanding Name Patterns

The **self-attention mechanism** is the core innovation that makes the transformer so effective:

#### What Self-Attention Does:
- Each character position "looks at" all previous characters in the name
- Learns which characters are most relevant for predicting the next one
- Captures both short-range (adjacent letters) and long-range dependencies

#### Example in Action:
For generating the name "सुर्या":
```
Position 1: "स" → looks at: [start]
Position 2: "ु" → looks at: [start, स] → learns "स" + "ु" pattern
Position 3: "र" → looks at: [start, स, ु] → learns common "सुर" combination
Position 4: "्" → looks at: [start, स, ु, र] → learns conjunct formation
Position 5: "य" → looks at all previous → completes the name pattern
```

### 3. Multi-Head Attention: Multiple Perspectives

The model uses **multiple attention heads** (default: 4) simultaneously:
- **Head 1**: Might focus on vowel-consonant patterns
- **Head 2**: Might learn name length and ending patterns  
- **Head 3**: Might capture gender-specific characteristics
- **Head 4**: Might understand syllable structures

### 4. Architecture Components

#### **Embedding Layers**:
```python
# Character embeddings: Convert characters to vectors
wte = nn.Embedding(vocab_size, n_embd)  # Character → Vector
wpe = nn.Embedding(block_size, n_embd)  # Position → Vector
```

#### **Transformer Blocks** (4 layers):
Each block contains:
1. **Layer Normalization**: Stabilizes training
2. **Multi-Head Self-Attention**: Learns character relationships
3. **Feed-Forward Network**: Processes attention outputs
4. **Residual Connections**: Allows deep learning

#### **Language Model Head**:
```python
# Final prediction layer
lm_head = nn.Linear(n_embd, vocab_size)  # Vector → Character probabilities
```

### 5. Training Process: How It Learns Names

#### **Autoregressive Training**:
The model learns by predicting the next character given all previous characters:

```
Training Example: "राम"
Input:  [START] → Target: "र"
Input:  [START, र] → Target: "ा" 
Input:  [START, र, ा] → Target: "म"
Input:  [START, र, ा, म] → Target: [END]
```

#### **Loss Function**:
Cross-entropy loss measures how well the model predicts the correct next character:
```python
loss = CrossEntropyLoss(predicted_probabilities, actual_next_character)
```

### 6. Generation Process: Creating New Names

#### **Autoregressive Sampling**:
1. Start with `[START]` token
2. Model predicts probability distribution over all characters
3. Sample next character using temperature and top-k sampling
4. Add sampled character to sequence
5. Repeat until `[END]` token or max length

#### **Temperature Control**:
- **Low temperature (0.5)**: More predictable, common names
- **High temperature (1.5)**: More creative, unusual combinations

#### **Top-K Sampling**:
- Restricts sampling to the K most likely characters
- Prevents completely nonsensical combinations
- Balances creativity with linguistic validity

### 7. Gender-Specific Learning

The fine-tuned models learn gender-specific patterns:

#### **Male Names** tend to have:
- Certain ending patterns (-देव, -प्रसाद, -बहादुर)
- Specific consonant clusters
- Traditional masculine morphology

#### **Female Names** often feature:
- Different vowel patterns
- Characteristic endings (-माया, -कुमारी, -देवी)
- Distinct phonetic structures

### 8. Why This Works So Well

#### **Contextual Understanding**:
Unlike simple n-gram models, transformers understand:
- **Long-range dependencies**: Beginning of name influences the end
- **Hierarchical patterns**: Syllables, morphemes, and full name structure
- **Cultural constraints**: Only generates linguistically valid combinations

#### **Attention Visualization** (conceptual):
```
Name: "श्याम"
श् → High attention to: [start] (name beginning patterns)
या → High attention to: [श्] (consonant-vowel harmony)  
म → High attention to: [श्, या] (common name endings after "श्या")
```

### 9. Technical Specifications

- **Model Size**: ~100K parameters (compact but effective)
- **Context Window**: 32 characters (sufficient for Nepali names)
- **Vocabulary**: 27 tokens (a-z + special tokens)
- **Architecture**: 4 layers, 64 dimensions, 4 attention heads
- **Training Data**: 8,787 cleaned Nepali names

This transformer architecture successfully captures the intricate patterns of Nepali name formation, from basic character sequences to complex cultural and linguistic rules, enabling it to generate authentic and diverse new names that sound naturally Nepali.

