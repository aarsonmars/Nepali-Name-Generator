#!/usr/bin/env python3
"""
Nepali Name Generator - Custom Name Generation Script

This script uses the best trained model to generate Nepali names with
configurable parameters. Users can customize various settings to get
names of their choice.

USAGE:
    python scripts/generate_names.py

CUSTOMIZATION:
    Edit the variables at the top of this script to change:
    - NUM_NAMES: How many names to generate
    - TOP_K: Sampling focus (1-3 = focused, 10-20 = diverse)
    - TEMPERATURE: Creativity (0.5-0.8 = consistent, 1.2-2.0 = unique)
    - MAX_LENGTH: Maximum name length
    - START_WITH: Names starting with specific letters
    - GENDER_PREFERENCE: 'male', 'female', or 'both'

EXAMPLES:
    # Generate 10 focused names starting with 'A'
    NUM_NAMES = 10
    TOP_K = 3
    START_WITH = "A"

    # Generate 25 diverse, creative names
    NUM_NAMES = 25
    TOP_K = 15
    TEMPERATURE = 1.5

    # Generate 5 very consistent names
    NUM_NAMES = 5
    TOP_K = 1
    TEMPERATURE = 0.7
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    Config, DataConfig, TrainingConfig, SamplingConfig, SystemConfig
)
from models import Transformer
from data import create_datasets
from utils import generate
import torch

# ===== CONFIGURABLE PARAMETERS =====
# Change these values to customize name generation

NUM_NAMES = 20              # Number of names to generate
TOP_K = 5                  # Top-k sampling (1-20, lower = more focused)
TEMPERATURE = 1.0          # Sampling temperature (0.5-2.0, lower = more consistent)
MAX_LENGTH = 15            # Maximum name length
START_WITH = "K"            # Start names with these letters (empty = any)
GENDER_PREFERENCE = "female" # 'male', 'female', or 'both'
                           # ⚠️ Note: Currently 'both' is recommended as the model was trained on combined data

# Model configuration
MODEL_DIR = "models/best_results"  # Default to combined model

# ===== END CONFIGURABLE PARAMETERS =====

def get_model_dir():
    """Get model directory based on gender preference."""
    if GENDER_PREFERENCE.lower() == "male":
        return "models/male_finetuned"
    elif GENDER_PREFERENCE.lower() == "female":
        return "models/female_finetuned"
    else:  # "both" or any other value
        return "models/best_results"

def get_data_files():
    """Get data files based on gender preference."""
    if GENDER_PREFERENCE.lower() == "male":
        return ["data/male.txt"]
    elif GENDER_PREFERENCE.lower() == "female":
        return ["data/female.txt"]
    else:  # "both" or any other value
        return ["data/male.txt", "data/female.txt"]

def load_model_and_data():
    """Load the trained model and prepare data."""
    print("🔄 Loading model and data...")

    # Get data files based on gender preference
    data_files = get_data_files()
    print(f"📂 Using data files: {data_files}")
    print(f"👥 Gender preference: {GENDER_PREFERENCE}")

    # Get model directory based on gender preference
    model_dir = get_model_dir()
    print(f"🤖 Using model: {model_dir}")

    # For loading the model, we need to use the ORIGINAL training data configuration
    # The model was trained on combined male/female data, so we need that vocab
    original_data = ["data/male.txt", "data/female.txt"]
    temp_config = Config(data=DataConfig(input_files=original_data))
    train_dataset, _ = create_datasets(original_data, temp_config.data)

    # Use the original model's vocabulary and block size
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()

    # Initialize model with original architecture
    model_config = Config().model
    model_config.vocab_size = vocab_size
    model_config.block_size = block_size

    model = Transformer(model_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load trained weights
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        # Try model_final.pt if model.pt doesn't exist (for fine-tuned models)
        model_path = os.path.join(model_dir, "model_final.pt")
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {os.path.join(model_dir, 'model.pt')} or {model_path}")
            print("💡 Train a model first using: python src/train.py --input-files data/male.txt data/female.txt")
            return None, None, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"✅ Model loaded from {model_path}")
    print(f"📊 Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🔤 Vocabulary size: {vocab_size}")
    print(f"📏 Max name length: {block_size}")

    # Now create dataset for generation using the gender preference
    gen_dataset, _ = create_datasets(data_files, Config(data=DataConfig(input_files=data_files)).data)

    return model, gen_dataset, Config(
        data=DataConfig(input_files=data_files),
        system=SystemConfig(work_dir=model_dir, sample_only=True),
        sampling=SamplingConfig(
            temperature=TEMPERATURE,
            top_k=TOP_K,
            max_new_tokens=MAX_LENGTH,
            num_samples=NUM_NAMES
        )
    )

def generate_names(model, dataset, config):
    """Generate names using the trained model."""
    print(f"\n🎯 Generating {NUM_NAMES} names...")
    print(f"⚙️  Settings: top_k={TOP_K}, temperature={TEMPERATURE}, max_length={MAX_LENGTH}")
    if START_WITH:
        print(f"🎭 Starting with: '{START_WITH}'")
    print("-" * 50)

    device = next(model.parameters()).device
    generated_names = []

    for i in range(NUM_NAMES):
        # Start generation
        if START_WITH:
            # Convert starting letters to indices
            start_indices = [dataset.stoi.get(c, 0) for c in START_WITH]
            context = torch.tensor([[0] + start_indices], dtype=torch.long, device=device)  # Include start token
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)

        # Generate name
        with torch.no_grad():
            # Use the generate function from utils.py
            generated = generate(
                model,
                context,
                max_new_tokens=MAX_LENGTH,
                temperature=TEMPERATURE,
                do_sample=True,
                top_k=TOP_K if TOP_K > 0 else None
            )

            name_indices = generated[0].tolist()

        # Convert back to text using dataset's decode method
        name_indices = generated[0].tolist()
        # Skip the first token (start token) and find the stop token
        row = name_indices[1:]  # Skip start token
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        name = dataset.decode(row)

        # Clean up the name
        name = name.strip()
        if name and len(name) > 1:  # Only add names longer than 1 character
            generated_names.append(name)

    return generated_names[:NUM_NAMES]  # Return exactly NUM_NAMES

def display_results(names):
    """Display the generated names in a nice format."""
    print(f"\n🎉 Generated {len(names)} Nepali Names:")
    print("=" * 50)

    for i, name in enumerate(names, 1):
        print(f"{i:2d}. {name}")

    print("=" * 50)
    print("💡 Tip: Adjust TOP_K and TEMPERATURE in the script for different styles!")
    print("   - Lower TOP_K (1-3): More focused, consistent names")
    print("   - Higher TOP_K (10-20): More diverse, creative names")
    print("   - Lower TEMPERATURE (0.5-0.8): More predictable names")
    print("   - Higher TEMPERATURE (1.2-2.0): More unique names")

def main():
    """Main function."""
    print("🇳🇵 Nepali Name Generator")
    print("=" * 50)

    # Load model and data
    model, dataset, config = load_model_and_data()
    if model is None:
        return

    # Generate names
    names = generate_names(model, dataset, config)

    # Display results
    display_results(names)

if __name__ == "__main__":
    main()
