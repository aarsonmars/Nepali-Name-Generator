#!/usr/bin/env python3
"""
Example script showing how to use the new configuration system.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    Config, DataConfig, TrainingConfig, SamplingConfig, SystemConfig,
    get_default_config, get_quick_train_config, get_production_config
)

def main():
    print("=== Nepali Name Generator Configuration Examples ===\n")

    # Example 1: Default configuration
    print("1. Default Configuration:")
    config = get_default_config()
    print(f"   Data: {config.data.input_file}")
    print(f"   Model: {config.system.model_type}")
    print(f"   Training: {config.training.max_steps} steps, batch_size={config.training.batch_size}")
    print()

    # Example 2: Quick training configuration
    print("2. Quick Training Configuration:")
    config = get_quick_train_config()
    print(f"   Max steps: {config.training.max_steps}")
    print(f"   Batch size: {config.training.batch_size}")
    print(f"   Work dir: {config.system.work_dir}")
    print()

    # Example 3: Custom configuration
    print("3. Custom Configuration for Nepali Names:")
    config = Config(
        data=DataConfig(
            input_files=['data/male.txt', 'data/female.txt'],
            test_split_ratio=0.15  # 15% for testing
        ),
        training=TrainingConfig(
            max_steps=2000,
            batch_size=32,
            learning_rate=3e-4,
            eval_interval=250
        ),
        system=SystemConfig(
            model_type='transformer',
            work_dir='models/nepali_custom'
        )
    )
    print(f"   Input files: {config.data.input_files}")
    print(f"   Test split: {config.data.test_split_ratio}")
    print(f"   Model: {config.system.model_type}")
    print(f"   Training steps: {config.training.max_steps}")
    print()

    # Example 4: Production configuration
    print("4. Production Configuration:")
    config = get_production_config()
    print(f"   Device: {config.system.device}")
    print(f"   Learning rate: {config.training.learning_rate}")
    print(f"   Batch size: {config.training.batch_size}")
    print(f"   Work dir: {config.system.work_dir}")
    print()

    print("=== Configuration Parameters ===")
    print("\nDataConfig:")
    print("  - input_file: Default input file path")
    print("  - input_files: List of input files (for multiple datasets)")
    print("  - test_split_ratio: Fraction of data for testing (default: 0.1)")
    print("  - max_test_size: Maximum test set size (default: 1000)")
    print("  - encoding: File encoding (default: 'utf-8')")

    print("\nTrainingConfig:")
    print("  - max_steps: Maximum training steps (-1 for infinite)")
    print("  - batch_size: Batch size for training")
    print("  - learning_rate: Learning rate")
    print("  - weight_decay: Weight decay for regularization")
    print("  - eval_interval: Evaluate every N steps")
    print("  - sample_interval: Generate samples every N steps")

    print("\nSystemConfig:")
    print("  - device: Computing device ('cpu', 'cuda', etc.)")
    print("  - seed: Random seed")
    print("  - work_dir: Output directory")
    print("  - model_type: Model architecture")
    print("  - resume: Resume training from checkpoint")
    print("  - sample_only: Only generate samples, don't train")

if __name__ == "__main__":
    main()
