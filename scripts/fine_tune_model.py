#!/usr/bin/env python3
"""
Nepali Name Generator - Fine-Tuning Script

This# Training Configuration
MAX_STEPS = 500              # Fine-tuning steps for male-specific model
LEARNING_RATE = 0.0001       # Learning rate (10x smaller than initial)
BATCH_SIZE = 16              # Batch size (smaller for fine-tuning)
EVAL_INTERVAL = 50           # Evaluate every 50 steps
SAVE_INTERVAL = 100          # Save model every 100 stepst fine-tunes an existing trained model on new data.
Perfect for creating gender-specific models or adapting to new datasets.

USAGE:
    python scripts/finetune_model.py

CUSTOMIZATION:
    Edit the variables below to customize your fine-tuning:

    BASE_MODEL: Path to existing model to fine-tune
    TARGET_DATA: Data files for fine-tuning
    OUTPUT_DIR: Where to save the fine-tuned model
    TRAINING_PARAMS: Learning rate, steps, batch size, etc.
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
from utils import evaluate

import torch

# Copy InfiniteDataLoader from train.py
class InfiniteDataLoader:
    """Data loader that cycles through the dataset indefinitely."""
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        if self.shuffle:
            self.data_iter = iter(torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0
            ))
        else:
            self.data_iter = iter(torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            ))

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.reset()
            batch = next(self.data_iter)
        return batch

# ===== CONFIGURABLE PARAMETERS =====
# Change these values to customize your fine-tuning

# Model Configuration
BASE_MODEL = "models/best_results/model.pt"  # Existing model to fine-tune
OUTPUT_DIR = "models/female_finetuned"        # Where to save female-specific model

# Data Configuration
TARGET_DATA = ["data/female.txt"]  # Female data for fine-tuning
# Options: ["data/male.txt"], ["data/female.txt"], ["data/male.txt", "data/female.txt"]

# Training Configuration
MAX_STEPS = 500              # Fine-tuning steps (500-1000 recommended)
LEARNING_RATE = 0.0001       # Learning rate (0.0001 = 10x smaller than initial)
BATCH_SIZE = 16              # Batch size (smaller for fine-tuning)
EVAL_INTERVAL = 50           # Evaluate every N steps
SAVE_INTERVAL = 100          # Save model every N steps

# Advanced Training Parameters
WEIGHT_DECAY = 0.01          # Weight decay for regularization
GRADIENT_CLIP = 1.0          # Gradient clipping value
WARMUP_STEPS = 10            # Learning rate warmup steps

# ===== END CONFIGURABLE PARAMETERS =====

def validate_setup():
    """Validate that all required files exist."""
    print("🔍 Validating setup...")

    # Check base model
    if not os.path.exists(BASE_MODEL):
        print(f"❌ Base model not found: {BASE_MODEL}")
        print("💡 Train a base model first or update BASE_MODEL path")
        return False

    # Check target data files
    for data_file in TARGET_DATA:
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return False

    print("✅ Setup validation passed!")
    return True

def load_base_model():
    """Load the base model for fine-tuning."""
    print(f"🔄 Loading base model from {BASE_MODEL}")

    # First, create a temporary config to load the ORIGINAL data and determine vocab
    # We need to use the same vocab as the base model was trained on
    original_data = ["data/male.txt", "data/female.txt"]  # Base model was trained on combined data
    temp_config = Config(data=DataConfig(input_files=original_data))
    train_dataset, _ = create_datasets(original_data, temp_config.data)

    # Model configuration - use SAME as base model
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()

    model_config = Config().model
    model_config.vocab_size = vocab_size
    model_config.block_size = block_size

    # Initialize model with SAME architecture as base model
    model = Transformer(model_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Load base model weights
    try:
        model.load_state_dict(torch.load(BASE_MODEL, map_location=device))
        print("✅ Base model loaded successfully!")
        print(f"📊 Base model: {vocab_size} chars, max length {block_size}")
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        return None, None

    return model, device

def setup_fine_tuning(model, device):
    """Setup optimizer and data for fine-tuning."""
    print("⚙️ Setting up fine-tuning...")

    # Create configuration - use the model's existing vocab/block_size
    # The model already has the correct architecture from the base model
    config = Config(
        data=DataConfig(input_files=TARGET_DATA),
        training=TrainingConfig(
            max_steps=MAX_STEPS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            eval_interval=EVAL_INTERVAL,
            save_interval=SAVE_INTERVAL
        ),
        system=SystemConfig(
            device=device,
            work_dir=OUTPUT_DIR
        )
    )

    # Load datasets with target data
    train_dataset, test_dataset = create_datasets(TARGET_DATA, config.data)

    print(f"📊 Target data: {len(train_dataset)} training, {len(test_dataset)} test samples")
    print(f"🔤 Target vocab: {train_dataset.get_vocab_size()} chars, max length {train_dataset.get_output_length()}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.99)
    )

    # Learning rate scheduler (optional)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.max_steps
    )

    print("✅ Fine-tuning setup complete!")
    return config, train_dataset, test_dataset, optimizer, scheduler

def fine_tune_model(model, config, train_dataset, test_dataset, optimizer, scheduler, device):
    """Perform fine-tuning."""
    print("🎯 Starting fine-tuning...")
    print(f"📊 Target data: {TARGET_DATA}")
    print(f"🔄 Steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("-" * 60)

    model.train()

    # Create data loader
    batch_loader = InfiniteDataLoader(
        train_dataset,
        batch_size=config.training.batch_size
    )

    best_loss = float('inf')

    for step in range(config.training.max_steps):
        # Get batch
        batch = batch_loader.next()
        batch = [t.to(device) for t in batch]
        X, Y = batch

        # Forward pass
        logits, loss = model(X, Y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

        optimizer.step()
        scheduler.step()

        # Logging
        if step % 10 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.6f}")

        # Evaluation
        if step > 0 and step % config.training.eval_interval == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10, device=device)
            test_loss = evaluate(model, test_dataset, batch_size=100, max_batches=10, device=device)
            print(f"step {step:4d} | train loss {train_loss:.4f} | test loss {test_loss:.4f}")

            # Save best model
            if test_loss < best_loss:
                best_loss = test_loss
                save_model(model, os.path.join(OUTPUT_DIR, "model.pt"))
                print(f"💾 Saved best model with test loss: {test_loss:.4f}")

        # Regular saving
        if step > 0 and step % config.training.save_interval == 0:
            save_model(model, os.path.join(OUTPUT_DIR, f"model_step_{step}.pt"))

    print("🎉 Fine-tuning complete!")
    print(f"🏆 Best test loss achieved: {best_loss:.4f}")
def evaluate_model(model, dataset, device, num_batches=10):
    """Evaluate model on dataset."""
    model.eval()
    losses = []

    for _ in range(num_batches):
        x, y = dataset.get_batch(32, device=device)
        with torch.no_grad():
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return torch.tensor(losses).mean().item()

def save_model(model, path):
    """Save model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def main():
    """Main fine-tuning function."""
    print("🔥 Nepali Name Generator - Fine-Tuning Script")
    print("=" * 60)

    # Validate setup
    if not validate_setup():
        return

    # Load base model
    model, device = load_base_model()
    if model is None:
        return

    # Setup fine-tuning
    config, train_dataset, test_dataset, optimizer, scheduler = setup_fine_tuning(model, device)

    # Fine-tune
    fine_tune_model(model, config, train_dataset, test_dataset, optimizer, scheduler, device)

    # Final save
    final_path = os.path.join(OUTPUT_DIR, "model_final.pt")
    save_model(model, final_path)
    print(f"💾 Final model saved to: {final_path}")

    print("\n🎯 Fine-tuning Summary:")
    print(f"   📁 Base model: {BASE_MODEL}")
    print(f"   📊 Target data: {TARGET_DATA}")
    print(f"   💾 Output: {OUTPUT_DIR}")
    print(f"   🔄 Steps: {MAX_STEPS}")
    print("\n✅ Ready to generate names with your fine-tuned model!")
    print(f"   python scripts/generate_names.py  # Update MODEL_DIR to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
