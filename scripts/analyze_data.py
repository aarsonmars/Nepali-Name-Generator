#!/usr/bin/env python3
"""
Utility script for common tasks with the Nepali Name Generator.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import create_datasets


def analyze_data(input_files):
    """Analyze the data files and print statistics."""
    print("Analyzing data files...")
    train_dataset, test_dataset = create_datasets(input_files)

    print(f"Total names: {len(train_dataset) + len(test_dataset)}")
    print(f"Training names: {len(train_dataset)}")
    print(f"Test names: {len(test_dataset)}")
    print(f"Vocabulary size: {train_dataset.get_vocab_size()}")
    print(f"Max word length: {train_dataset.max_word_length}")

    # Show some sample names
    print("\nSample training names:")
    for i in range(min(5, len(train_dataset.words))):
        print(f"  {train_dataset.words[i]}")


def main():
    parser = argparse.ArgumentParser(description="Nepali Name Generator Utilities")
    parser.add_argument('command', choices=['analyze'], help="Command to run")
    parser.add_argument('--input-files', type=str, nargs='+',
                       default=['data/male.txt', 'data/female.txt'],
                       help="Input data files")

    args = parser.parse_args()

    if args.command == 'analyze':
        analyze_data(args.input_files)


if __name__ == "__main__":
    main()
