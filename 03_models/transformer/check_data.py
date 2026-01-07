"""Quick check of dataset structure."""
import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent.parent.parent / '00_data' / 'processed'

train = pd.read_csv(data_path / 'train_data.csv')
val = pd.read_csv(data_path / 'val_data.csv')
test = pd.read_csv(data_path / 'test_data.csv')

print(f"Train: {len(train)} samples")
print(f"Val: {len(val)} samples")
print(f"Test: {len(test)} samples")
print(f"\nColumns: {list(train.columns)}")
print(f"\nLabel distribution in train:")
print(train['label'].value_counts())
