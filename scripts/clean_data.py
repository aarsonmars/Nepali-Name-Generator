#!/usr/bin/env python3
"""
Data Cleaning Script for Nepali Name Generator

This script cleans the training data to ensure only valid characters (a-z) are used
for optimal model performance.
"""

import os
import re
from pathlib import Path

def clean_names(names):
    """
    Clean names to contain only a-z characters.
    - Convert to lowercase
    - Remove special characters, numbers, and punctuation
    - Filter out names that become too short or empty
    """
    cleaned_names = []
    
    for name in names:
        # Convert to lowercase
        cleaned = name.lower()
        
        # Remove everything except a-z letters
        cleaned = re.sub(r'[^a-z]', '', cleaned)
        
        # Skip names that are too short after cleaning
        if len(cleaned) >= 2:
            cleaned_names.append(cleaned)
    
    return cleaned_names

def analyze_data():
    """Analyze the current data and show cleaning results."""
    print("🔍 Analyzing Nepali Name Data")
    print("=" * 50)
    
    # Read original data
    with open('data/male.txt', 'r', encoding='utf-8') as f:
        male_names = [name.strip() for name in f.readlines() if name.strip()]
    
    with open('data/female.txt', 'r', encoding='utf-8') as f:
        female_names = [name.strip() for name in f.readlines() if name.strip()]
    
    print(f"📊 Original Data:")
    print(f"   Male names: {len(male_names)}")
    print(f"   Female names: {len(female_names)}")
    print(f"   Total names: {len(male_names + female_names)}")
    
    # Show problematic characters
    all_chars = set(''.join(male_names + female_names))
    problematic_chars = [c for c in all_chars if not c.islower() or not c.isalpha()]
    
    print(f"\n🚨 Problematic Characters Found: {len(problematic_chars)}")
    print(f"   Characters: {''.join(sorted(problematic_chars))}")
    
    # Clean the data
    cleaned_male = clean_names(male_names)
    cleaned_female = clean_names(female_names)
    
    print(f"\n✅ After Cleaning:")
    print(f"   Male names: {len(cleaned_male)} (removed {len(male_names) - len(cleaned_male)})")
    print(f"   Female names: {len(cleaned_female)} (removed {len(female_names) - len(cleaned_female)})")
    print(f"   Total names: {len(cleaned_male + cleaned_female)}")
    
    # Show character distribution
    all_cleaned_chars = set(''.join(cleaned_male + cleaned_female))
    print(f"\n🔤 Final Character Set: {len(all_cleaned_chars)} characters")
    print(f"   Characters: {''.join(sorted(all_cleaned_chars))}")
    
    # Show examples of changes
    print(f"\n📝 Examples of Cleaning:")
    examples = [
        ("Geeta(lamsal)", "geetalamsal"),
        ("M.d.", "md"),
        ("Md.alsahud", "mdalsahud"),
        ("Krishna", "krishna"),
        ("AADARSH", "aadarsh")
    ]
    
    for original, expected in examples:
        if original in male_names + female_names:
            cleaned = clean_names([original])[0] if clean_names([original]) else "REMOVED"
            print(f"   {original} → {cleaned}")
    
    return cleaned_male, cleaned_female

def create_cleaned_files(cleaned_male, cleaned_female):
    """Create cleaned data files."""
    print(f"\n💾 Creating Cleaned Data Files:")
    
    # Backup original files
    os.makedirs('data/backup', exist_ok=True)
    
    if not os.path.exists('data/backup/male_original.txt'):
        import shutil
        shutil.copy('data/male.txt', 'data/backup/male_original.txt')
        shutil.copy('data/female.txt', 'data/backup/female_original.txt')
        print("   ✅ Original files backed up to data/backup/")
    
    # Write cleaned files
    with open('data/male.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(set(cleaned_male))))
    
    with open('data/female.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(set(cleaned_female))))
    
    # Create combined file for convenience
    with open('data/names_cleaned.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(set(cleaned_male + cleaned_female))))
    
    print("   ✅ Cleaned files created:")
    print("      - data/male.txt (cleaned)")
    print("      - data/female.txt (cleaned)")
    print("      - data/names_cleaned.txt (combined)")

def main():
    """Main cleaning process."""
    # Change to project directory (parent of scripts directory)
    os.chdir(Path(__file__).parent.parent)
    
    # Analyze and clean data
    cleaned_male, cleaned_female = analyze_data()
    
    # Create cleaned files
    create_cleaned_files(cleaned_male, cleaned_female)
    
    print(f"\n🎉 Data cleaning completed!")
    print(f"   The model will now train on clean a-z characters only.")
    print(f"   This should improve training stability and generation quality.")

if __name__ == "__main__":
    main()
