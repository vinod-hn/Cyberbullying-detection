"""
Data Splitter Script
Splits raw datasets from 00_data/raw into train (70%), validation (15%), and test (15%) sets.
Outputs: train_data.csv, val_data.csv, test_data.csv in 00_data/processed/
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_raw_datasets():
    """Load all raw datasets and combine them with source tracking."""
    
    all_data = []
    
    # 1. Load english.csv
    english_path = os.path.join(RAW_DIR, 'english.csv')
    if os.path.exists(english_path):
        df_english = pd.read_csv(english_path)
        df_english['source'] = 'english'
        df_english['label'] = df_english['label'].str.lower()
        all_data.append(df_english)
        print(f"Loaded english.csv: {len(df_english)} records")
    
    # 2. Load kannada.csv
    kannada_path = os.path.join(RAW_DIR, 'kannada.csv')
    if os.path.exists(kannada_path):
        df_kannada = pd.read_csv(kannada_path)
        df_kannada['source'] = 'kannada'
        df_kannada['label'] = df_kannada['label'].str.lower()
        # Normalize label names
        df_kannada['label'] = df_kannada['label'].replace({
            'sexual harassment': 'sexual_harassment'
        })
        all_data.append(df_kannada)
        print(f"Loaded kannada.csv: {len(df_kannada)} records")
    
    # 3. Load kannad english.csv (code-mixed)
    codemix_path = os.path.join(RAW_DIR, 'kannad english.csv')
    if os.path.exists(codemix_path):
        df_codemix = pd.read_csv(codemix_path)
        df_codemix['source'] = 'codemix'
        df_codemix['label'] = df_codemix['label'].str.lower()
        all_data.append(df_codemix)
        print(f"Loaded kannad english.csv: {len(df_codemix)} records")
    
    # 4. Load emoji_cyberbullying_dataset.csv
    emoji_path = os.path.join(RAW_DIR, 'emoji_cyberbullying_dataset.csv')
    if os.path.exists(emoji_path):
        df_emoji = pd.read_csv(emoji_path)
        df_emoji['source'] = 'emoji'
        df_emoji['label'] = df_emoji['label'].str.lower()
        # Drop extra columns not in other datasets
        if 'explanation' in df_emoji.columns:
            df_emoji = df_emoji.drop(columns=['explanation'])
        all_data.append(df_emoji)
        print(f"Loaded emoji_cyberbullying_dataset.csv: {len(df_emoji)} records")
    
    # 5. Load bad_words.csv
    badwords_path = os.path.join(RAW_DIR, 'bad_words.csv')
    if os.path.exists(badwords_path):
        df_badwords = pd.read_csv(badwords_path)
        df_badwords['source'] = 'bad_words'
        df_badwords['label'] = df_badwords['label'].str.lower()
        # Normalize label names
        df_badwords['label'] = df_badwords['label'].replace({
            'sexual harassment': 'sexual_harassment'
        })
        # Rename severity_level to severity for consistency
        if 'severity_level' in df_badwords.columns:
            df_badwords = df_badwords.rename(columns={'severity_level': 'severity'})
        # Drop context column if exists
        if 'context' in df_badwords.columns:
            df_badwords = df_badwords.drop(columns=['context'])
        all_data.append(df_badwords)
        print(f"Loaded bad_words.csv: {len(df_badwords)} records")
    
    return all_data


def standardize_columns(dataframes):
    """Standardize columns across all dataframes."""
    
    # Define standard columns
    standard_cols = ['message', 'label', 'target_type', 'severity', 'severity_score', 'source']
    
    standardized_dfs = []
    
    for df in dataframes:
        # Ensure all standard columns exist
        for col in standard_cols:
            if col not in df.columns:
                df[col] = None
        
        # Keep only standard columns
        df = df[standard_cols]
        standardized_dfs.append(df)
    
    return standardized_dfs


def split_data(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data into train, validation, and test sets.
    Uses stratified splitting based on label to maintain class distribution.
    """
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df['label'],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def print_statistics(train_df, val_df, test_df):
    """Print statistics about the splits."""
    
    total = len(train_df) + len(val_df) + len(test_df)
    
    print("\n" + "="*60)
    print("DATASET SPLIT STATISTICS")
    print("="*60)
    
    print(f"\nTotal Records: {total}")
    print(f"Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df)/total*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/total*100:.1f}%)")
    
    print("\n--- Label Distribution ---")
    print("\nTrain Set:")
    print(train_df['label'].value_counts())
    
    print("\nValidation Set:")
    print(val_df['label'].value_counts())
    
    print("\nTest Set:")
    print(test_df['label'].value_counts())
    
    print("\n--- Source Distribution ---")
    print("\nTrain Set:")
    print(train_df['source'].value_counts())
    
    print("\nValidation Set:")
    print(val_df['source'].value_counts())
    
    print("\nTest Set:")
    print(test_df['source'].value_counts())


def main():
    """Main function to execute the data splitting pipeline."""
    
    print("="*60)
    print("CYBERBULLYING DATASET SPLITTER")
    print("="*60)
    print(f"\nRaw Data Directory: {RAW_DIR}")
    print(f"Output Directory: {PROCESSED_DIR}")
    
    # Step 1: Load all raw datasets
    print("\n[Step 1] Loading raw datasets...")
    dataframes = load_raw_datasets()
    
    if not dataframes:
        print("ERROR: No datasets found in raw directory!")
        return
    
    # Step 2: Standardize columns
    print("\n[Step 2] Standardizing columns...")
    standardized_dfs = standardize_columns(dataframes)
    
    # Step 3: Combine all datasets
    print("\n[Step 3] Combining datasets...")
    combined_df = pd.concat(standardized_dfs, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} total records")
    
    # Step 4: Clean data
    print("\n[Step 4] Cleaning data...")
    # Remove duplicates based on message
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['message'], keep='first')
    print(f"Removed {before_dedup - len(combined_df)} duplicate messages")
    
    # Remove rows with missing messages or labels
    combined_df = combined_df.dropna(subset=['message', 'label'])
    print(f"Final dataset size: {len(combined_df)} records")
    
    # Step 5: Split data
    print("\n[Step 5] Splitting data (70/15/15)...")
    train_df, val_df, test_df = split_data(combined_df)
    
    # Step 6: Save splits
    print("\n[Step 6] Saving split files...")
    
    train_path = os.path.join(PROCESSED_DIR, 'train_data.csv')
    val_path = os.path.join(PROCESSED_DIR, 'val_data.csv')
    test_path = os.path.join(PROCESSED_DIR, 'test_data.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")
    
    # Step 7: Print statistics
    print_statistics(train_df, val_df, test_df)
    
    print("\n" + "="*60)
    print("DATA SPLITTING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
