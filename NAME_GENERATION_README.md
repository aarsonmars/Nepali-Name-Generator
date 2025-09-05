# 🇳🇵 Nepali Name Generator - Custom Name Generation

A powerful script to generate authentic Nepali names using a trained Transformer model. Customize parameters to create names with your preferred style and characteristics.

## 🚀 Scripts Available

### 1. Name Generation Script
```bash
python scripts/generate_names.py
```
**Purpose**: Generate names using trained models with customizable parameters.

### 2. Fine-Tuning Script
```bash
python scripts/fine_tune_model.py
```
**Purpose**: Fine-tune existing models on new datasets for specialization.

## 📋 Features

- ✅ **Authentic Nepali Names**: Trained on 8,789+ real Nepali names
- ✅ **Fully Customizable**: Adjust all generation parameters
- ✅ **Multiple Styles**: From traditional to modern names
- ✅ **Letter-Specific**: Generate names starting with specific letters
- ✅ **Quality Control**: Top-k sampling and temperature control
- ✅ **Easy to Use**: Simple script with clear documentation

## 🎛️ Customization Parameters

Edit the variables at the top of `scripts/generate_names.py`:

### Basic Settings
```python
NUM_NAMES = 20              # How many names to generate (1-100)
TOP_K = 5                  # Sampling focus (1-20)
TEMPERATURE = 1.0          # Creativity level (0.5-2.0)
MAX_LENGTH = 15            # Maximum name length (5-20)
```

### Advanced Settings
```python
START_WITH = ""            # Names starting with specific letters
GENDER_PREFERENCE = "both" # 'male', 'female', or 'both'
```

## 👥 Gender Preference Feature

The `GENDER_PREFERENCE` variable controls which dataset is used for generation:

### How It Works
- **`"both"`** (default): Uses both `data/male.txt` and `data/female.txt`
- **`"male"`**: Uses only `data/male.txt`
- **`"female"`**: Uses only `data/female.txt`

### Important Notes
⚠️ **Current Limitation**: The model was trained on the combined dataset, so using `"male"` or `"female"` alone may cause compatibility issues. For best results, use `"both"`.

### Future Enhancement
To fully support gender-specific generation:
1. Train separate models on male-only and female-only datasets
2. Or retrain the current model with gender-specific data loading

### Example Usage
```python
# Use both genders (recommended)
GENDER_PREFERENCE = "both"

# For future use with gender-specific models
GENDER_PREFERENCE = "male"   # Male names only
GENDER_PREFERENCE = "female" # Female names only
```

## 📊 Parameter Guide

### TOP_K (Sampling Focus)
| Value | Effect | Best For |
|-------|--------|----------|
| 1-3 | Very focused, consistent | Traditional names |
| 4-7 | Balanced quality & variety | General use |
| 8-15 | Creative and diverse | Modern names |
| 16-20 | Very experimental | Unique names |

### TEMPERATURE (Creativity)
| Value | Effect | Best For |
|-------|--------|----------|
| 0.5-0.7 | Very predictable | Safe, traditional names |
| 0.8-1.2 | Balanced | General use |
| 1.3-1.8 | Creative | Modern, unique names |
| 1.9-2.0 | Very experimental | Artistic names |

## 🎯 Usage Examples

### 1. Traditional Nepali Names
```python
NUM_NAMES = 10
TOP_K = 3
TEMPERATURE = 0.7
START_WITH = ""
```
**Output:** Rajesh, Sushil, Ramesh, Bishnu, Krishna...

### 2. Modern Creative Names
```python
NUM_NAMES = 15
TOP_K = 10
TEMPERATURE = 1.3
START_WITH = ""
```
**Output:** Prabin, Sandesh, Roshan, Bibek, Sujan...

### 3. Names Starting with 'S'
```python
NUM_NAMES = 12
TOP_K = 5
TEMPERATURE = 0.9
START_WITH = "S"
```
**Output:** Sushil, Sandesh, Sujan, Sabina, Suman...

### 4. Short Names Only
```python
NUM_NAMES = 8
TOP_K = 3
TEMPERATURE = 0.8
MAX_LENGTH = 8
```
**Output:** Ram, Sita, Hari, Gita, Mohan...

## 🛠️ How to Customize

### Method 1: Direct Edit
1. Open `scripts/generate_names.py`
2. Modify the parameters at the top
3. Run: `python scripts/generate_names.py`

### Method 2: Create Multiple Versions
```bash
# Create your own version
cp scripts/generate_names.py scripts/my_names.py
# Edit parameters in my_names.py
python scripts/my_names.py
```

## 📈 Model Information

- **Architecture**: Transformer (4 layers, 64 embeddings)
- **Training Data**: 8,789 Nepali names (male + female)
- **Vocabulary**: 55 characters (Devanagari + English)
- **Model Size**: 208K parameters
- **Training**: 2,000 steps, final loss: ~1.78

## 🎨 Sample Generated Names

### Traditional Style (TOP_K=3)
- Rajesh, Sushil, Ramesh, Bishnu, Krishna
- Sunita, Sabita, Manju, Shanti, Radha

### Modern Style (TOP_K=10)
- Prabin, Sandesh, Roshan, Bibek, Sujan
- Anisha, Priya, Sapana, Rashmi, Kabita

### Creative Style (TOP_K=15, TEMP=1.5)
- Prashant, Sudip, Roshan, Bibek, Sujan
- Anushka, Priyanka, Sapana, Rashmika, Kabita

## 🔧 Advanced Usage

### Generate Names for Specific Use Cases

```python
# Baby names (short, traditional)
NUM_NAMES = 20
TOP_K = 2
TEMPERATURE = 0.6
MAX_LENGTH = 10

# Business names (modern, professional)
NUM_NAMES = 15
TOP_K = 8
TEMPERATURE = 1.1
MAX_LENGTH = 12

# Character names (creative, unique)
NUM_NAMES = 10
TOP_K = 20
TEMPERATURE = 1.8
MAX_LENGTH = 15
```

### Batch Generation
```bash
# Generate multiple batches with different settings
for i in 1..5; do
    echo "Batch $i:"
    python scripts/generate_names.py
    echo "---"
done
```

## 📋 Requirements

- Python 3.7+
- PyTorch
- Trained model in `models/best_results/model.pt`
- Data files in `data/` directory

## 🐛 Troubleshooting

### "Model not found" Error
```bash
# Train a model first
python src/train.py --input-files data/male.txt data/female.txt --work-dir models/best_results
```

### "No such file or directory: 'data/names.txt'" Error
```bash
# Create combined names file
Get-Content data/male.txt, data/female.txt | Set-Content data/names.txt
```

### Names look strange
- Try lower TOP_K (1-3) for more consistent names
- Try lower TEMPERATURE (0.5-0.8) for more predictable names
- Check MAX_LENGTH isn't too short

## 📚 Technical Details

### How It Works
1. **Load Model**: Trained Transformer from `models/best_results/`
2. **Load Data**: Vocabulary and token mappings from training data
3. **Generate**: Use sampling parameters to create new names
4. **Filter**: Remove invalid names and duplicates
5. **Display**: Show results with statistics

### Sampling Process
- **Top-K**: Limits choices to top K most likely characters
- **Temperature**: Controls randomness in selection
- **Max Length**: Prevents overly long names
- **Start With**: Forces names to begin with specific letters

## 🔧 Advanced: Fine-Tuning for Custom Models

### Create Gender-Specific or Custom Models

Use the fine-tuning script to adapt your existing model to specific datasets:

```bash
python scripts/fine_tune_model.py
```

### Fine-Tuning Parameters

Edit `scripts/fine_tune_model.py` to customize:

```python
# Model Configuration
BASE_MODEL = "models/best_results/model.pt"  # Existing model to fine-tune
OUTPUT_DIR = "models/finetuned_model"        # Where to save fine-tuned model

# Data Configuration
TARGET_DATA = ["data/male.txt"]  # Data files for fine-tuning

# Training Configuration
MAX_STEPS = 500              # Fine-tuning steps (500-1000 recommended)
LEARNING_RATE = 0.0001       # Learning rate (10x smaller than initial)
BATCH_SIZE = 16              # Batch size (smaller for fine-tuning)
```

### Fine-Tuning Script Parameters

Edit `scripts/fine_tune_model.py` to customize:

```python
# Model Configuration
BASE_MODEL = "models/best_results/model.pt"  # Existing model to fine-tune
OUTPUT_DIR = "models/finetuned_model"        # Where to save fine-tuned model

# Data Configuration
TARGET_DATA = ["data/male.txt"]  # Data files for fine-tuning

# Training Configuration
MAX_STEPS = 500              # Fine-tuning steps (500-1000 recommended)
LEARNING_RATE = 0.0001       # Learning rate (10x smaller than initial)
BATCH_SIZE = 16              # Batch size (smaller for fine-tuning)
EVAL_INTERVAL = 50           # Evaluate every N steps
SAVE_INTERVAL = 100          # Save model every N steps
```

### Fine-Tuning Workflow

1. **Prepare Data**: Ensure target data files exist
2. **Configure Script**: Edit parameters in `fine_tune_model.py`
3. **Run Fine-Tuning**: `python scripts/fine_tune_model.py`
4. **Use Fine-Tuned Model**: Update `MODEL_DIR` in generation script

### Example: Create Male-Specific Model

```python
# In fine_tune_model.py
BASE_MODEL = "models/best_results/model.pt"
OUTPUT_DIR = "models/male_finetuned"
TARGET_DATA = ["data/male.txt"]
MAX_STEPS = 500
LEARNING_RATE = 0.0001
```

After fine-tuning, update generation script:
```python
# In generate_names.py
MODEL_DIR = "models/male_finetuned"
GENDER_PREFERENCE = "male"
```

### Expected Results

Fine-tuned models will generate more specialized names:
- **Male Model**: Rajesh, Suresh, Bishnu, Krishna, Ramesh
- **Female Model**: Sunita, Sabita, Radha, Gita, Manju
- **Custom Model**: Names specific to your custom dataset

## 📄 License

This project is part of the Nepali Name Generator system. See main README for license information.

---

**Happy Name Generating! 🎉**

*Generate thousands of beautiful Nepali names with just a few parameter tweaks!* 🇳🇵✨
