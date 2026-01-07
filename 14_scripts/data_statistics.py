"""
Data Statistics and Analysis Script
Analyzes the cyberbullying detection dataset structure and provides comprehensive statistics.
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def analyze_dataset(df: pd.DataFrame, dataset_name: str) -> dict:
    """Analyze a single dataset and return statistics."""
    stats = {
        'name': dataset_name,
        'total_samples': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    # Label distribution
    if 'label' in df.columns:
        label_counts = Counter(df['label'].dropna())
        stats['label_distribution'] = dict(label_counts)
        stats['num_classes'] = len(label_counts)
    
    # Severity distribution
    if 'severity' in df.columns:
        severity_counts = Counter(df['severity'].dropna())
        stats['severity_distribution'] = dict(severity_counts)
    
    # Message length statistics
    if 'message' in df.columns:
        msg_lengths = df['message'].dropna().str.len()
        stats['message_length'] = {
            'mean': float(msg_lengths.mean()),
            'median': float(msg_lengths.median()),
            'min': int(msg_lengths.min()),
            'max': int(msg_lengths.max()),
            'std': float(msg_lengths.std())
        }
        
        # Word count statistics
        word_counts = df['message'].dropna().str.split().str.len()
        stats['word_count'] = {
            'mean': float(word_counts.mean()),
            'median': float(word_counts.median()),
            'min': int(word_counts.min()),
            'max': int(word_counts.max()),
            'std': float(word_counts.std())
        }
    
    return stats


def print_statistics(stats: dict):
    """Print statistics in a formatted manner."""
    print(f"\n{'='*80}")
    print(f"Dataset: {stats['name']}")
    print(f"{'='*80}")
    print(f"Total Samples: {stats['total_samples']:,}")
    print(f"Columns: {', '.join(stats['columns'])}")
    
    # Missing values
    if any(stats['missing_values'].values()):
        print(f"\nMissing Values:")
        for col, count in stats['missing_values'].items():
            if count > 0:
                percentage = 100 * count / stats['total_samples']
                print(f"  {col}: {count:,} ({percentage:.2f}%)")
    else:
        print("\nNo missing values")
    
    # Label distribution
    if 'label_distribution' in stats:
        print(f"\nLabel Distribution ({stats['num_classes']} classes):")
        sorted_labels = sorted(stats['label_distribution'].items(), 
                             key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels:
            percentage = 100 * count / stats['total_samples']
            print(f"  {label:20s}: {count:6,} ({percentage:5.2f}%)")
    
    # Severity distribution
    if 'severity_distribution' in stats:
        print(f"\nSeverity Distribution:")
        sorted_severity = sorted(stats['severity_distribution'].items(), 
                                key=lambda x: x[1], reverse=True)
        for severity, count in sorted_severity:
            percentage = 100 * count / stats['total_samples']
            print(f"  {severity:15s}: {count:6,} ({percentage:5.2f}%)")
    
    # Message length statistics
    if 'message_length' in stats:
        ml = stats['message_length']
        print(f"\nMessage Length (characters):")
        print(f"  Mean:   {ml['mean']:.1f} ± {ml['std']:.1f}")
        print(f"  Median: {ml['median']:.1f}")
        print(f"  Range:  [{ml['min']}, {ml['max']}]")
    
    # Word count statistics
    if 'word_count' in stats:
        wc = stats['word_count']
        print(f"\nWord Count:")
        print(f"  Mean:   {wc['mean']:.1f} ± {wc['std']:.1f}")
        print(f"  Median: {wc['median']:.1f}")
        print(f"  Range:  [{wc['min']}, {wc['max']}]")


def main():
    """Main function to analyze all datasets."""
    # Define paths
    data_root = project_root / '00_data'
    raw_path = data_root / 'raw'
    processed_path = data_root / 'processed'
    
    print("="*80)
    print("CYBERBULLYING DETECTION DATASET ANALYSIS")
    print("="*80)
    
    # Analyze raw datasets
    print("\n" + "="*80)
    print("RAW DATASETS")
    print("="*80)
    
    raw_files = [
        ('english.csv', 'English Dataset'),
        ('kannada.csv', 'Kannada Dataset'),
        ('kannad english.csv', 'Code-Mixed (Kannada-English) Dataset'),
        ('emoji_cyberbullying_dataset.csv', 'Emoji Cyberbullying Dataset'),
        ('bad_words.csv', 'Bad Words Dataset')
    ]
    
    raw_stats = []
    total_raw_samples = 0
    
    for filename, name in raw_files:
        filepath = raw_path / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            stats = analyze_dataset(df, name)
            raw_stats.append(stats)
            total_raw_samples += stats['total_samples']
            print_statistics(stats)
        else:
            print(f"\nWarning: {filename} not found at {filepath}")
    
    # Print raw data summary
    print(f"\n{'='*80}")
    print("RAW DATA SUMMARY")
    print(f"{'='*80}")
    print(f"Total Raw Samples: {total_raw_samples:,}")
    print(f"Number of Raw Datasets: {len(raw_stats)}")
    
    # Analyze processed datasets
    print("\n" + "="*80)
    print("PROCESSED DATASETS")
    print("="*80)
    
    processed_files = [
        ('train_data.csv', 'Training Set'),
        ('val_data.csv', 'Validation Set'),
        ('test_data.csv', 'Test Set')
    ]
    
    processed_stats = []
    total_processed_samples = 0
    
    for filename, name in processed_files:
        filepath = processed_path / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            stats = analyze_dataset(df, name)
            processed_stats.append(stats)
            total_processed_samples += stats['total_samples']
            print_statistics(stats)
        else:
            print(f"\nWarning: {filename} not found at {filepath}")
    
    # Print processed data summary
    print(f"\n{'='*80}")
    print("PROCESSED DATA SUMMARY")
    print(f"{'='*80}")
    print(f"Total Processed Samples: {total_processed_samples:,}")
    print(f"Number of Processed Splits: {len(processed_stats)}")
    
    if processed_stats:
        train_size = processed_stats[0]['total_samples']
        val_size = processed_stats[1]['total_samples'] if len(processed_stats) > 1 else 0
        test_size = processed_stats[2]['total_samples'] if len(processed_stats) > 2 else 0
        
        print(f"\nData Split Ratios:")
        print(f"  Train: {100*train_size/total_processed_samples:.1f}%")
        print(f"  Val:   {100*val_size/total_processed_samples:.1f}%")
        print(f"  Test:  {100*test_size/total_processed_samples:.1f}%")
    
    # Compare raw vs processed
    print(f"\n{'='*80}")
    print("DATA PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Raw Samples:       {total_raw_samples:,}")
    print(f"Processed Samples: {total_processed_samples:,}")
    
    if total_raw_samples > 0:
        retention_rate = 100 * total_processed_samples / total_raw_samples
        print(f"Retention Rate:    {retention_rate:.2f}%")
        
        if retention_rate < 100:
            removed = total_raw_samples - total_processed_samples
            print(f"Removed Samples:   {removed:,} (duplicates/invalid)")
    
    # Check for class imbalance
    if processed_stats and 'label_distribution' in processed_stats[0]:
        print(f"\n{'='*80}")
        print("CLASS BALANCE ANALYSIS")
        print(f"{'='*80}")
        
        label_dist = processed_stats[0]['label_distribution']
        total = processed_stats[0]['total_samples']
        
        most_common = max(label_dist.values())
        least_common = min(label_dist.values())
        imbalance_ratio = most_common / least_common
        
        print(f"Number of Classes: {len(label_dist)}")
        print(f"Most Common Class: {most_common:,} samples")
        print(f"Least Common Class: {least_common:,} samples")
        print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print("\n⚠️  HIGH CLASS IMBALANCE DETECTED")
            print("   Consider using:")
            print("   - Class weights in loss function")
            print("   - Oversampling minority classes")
            print("   - Focal loss")
        elif imbalance_ratio > 5:
            print("\n⚠️  MODERATE CLASS IMBALANCE DETECTED")
            print("   Consider using class weights")
        else:
            print("\n✓ Classes are reasonably balanced")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
