
# Nepali Name Generator

A modularized version of the makemore project adapted for generating Nepali names. This project takes Nepali name datasets and generates new name suggestions using various neural network architectures.

## Project Structure

```
nepali-name-generator/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration classes and hyperparameters
│   ├── models.py          # All neural network model implementations
│   ├── data.py            # Data loading and preprocessing utilities
│   ├── utils.py           # Helper functions for generation, evaluation, and sampling
│   └── train.py           # Main training script
├── data/
│   ├── male.txt           # Male Nepali names
│   └── female.txt         # Female Nepali names
├── models/                # Saved model checkpoints
├── tests/                 # Unit tests
├── scripts/               # Utility scripts
├── README.md
└── requirements.txt       # Python dependencies
```

## Features

- **Modular Design**: Clean separation of concerns for easy maintenance and extension
- **Multiple Models**: Support for various architectures from simple bigrams to Transformers
- **Nepali Name Support**: Handles Devanagari script and multiple name files
- **Flexible Data Loading**: Can combine male.txt and female.txt or use single files
- **Gender-Specific Models**: Fine-tuned models for male, female, or combined name generation
- **Easy Fine-Tuning**: Scripts to create specialized models from base models

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

- `data/male.txt` - Male Nepali names (one per line)
- `data/female.txt` - Female Nepali names (one per line)

## Model Architectures

The project implements several neural network architectures:

- **Bigram**: Simple character-to-character prediction using a lookup table
- **MLP**: Multi-layer perceptron following Bengio et al. 2003
- **RNN/GRU**: Recurrent neural networks for sequence modeling
- **BoW**: Bag of Words model with averaging
- **Transformer**: Full transformer architecture as in GPT

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
