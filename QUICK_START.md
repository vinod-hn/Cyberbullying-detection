# Quick Start Guide - Cyberbullying Detection

## Dataset Overview

### Dataset Statistics
- **Total Raw Samples**: 11,120
- **Processed Samples**: 10,968 (98.63% retention)
- **Number of Classes**: 11 cyberbullying categories
- **Data Split**: 70% train (7,677), 15% val (1,645), 15% test (1,646)

### Class Distribution (Training Set)
| Label | Count | Percentage |
|-------|-------|------------|
| insult | 1,556 | 20.27% |
| neutral | 1,479 | 19.27% |
| harassment | 1,196 | 15.58% |
| threat | 706 | 9.20% |
| exclusion | 629 | 8.19% |
| aggression | 459 | 5.98% |
| toxicity | 412 | 5.37% |
| stalking | 378 | 4.92% |
| sexual_harassment | 363 | 4.73% |
| hate | 356 | 4.64% |
| cyberstalking | 143 | 1.86% |

⚠️ **Class Imbalance Ratio**: 10.88:1 (High)

### Dataset Composition
1. **English Dataset** (2,000 samples)
   - 9 classes, Medium/High severity
   - Mean message length: 40 characters, 7 words

2. **Kannada Dataset** (4,000 samples)
   - 7 classes, 50% neutral, Low/Medium severity
   - Mean message length: 56 characters, 9 words

3. **Code-Mixed (Kannada-English)** (2,000 samples)
   - 9 classes, 84% medium severity
   - Mean message length: 51 characters, 9 words

4. **Emoji Cyberbullying** (1,120 samples)
   - 9 classes, 47% high severity
   - Mean message length: 26 characters, 6 words

5. **Bad Words** (2,000 samples)
   - 4 classes focusing on profanity
   - Mean message length: 56 characters, 10 words

## Installation

### Required Dependencies

```bash
# Core dependencies
pip install pandas numpy scikit-learn

# For visualization (optional)
pip install matplotlib seaborn

# For deep learning models (optional)
pip install torch torchvision
```

### Quick Install All
```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch
```

## Running the Models

### 1. View Dataset Statistics

```bash
cd cyberbullying-detection/14_scripts
python data_statistics.py
```

This will display comprehensive statistics about raw and processed datasets.

### 2. Train Baseline Models (Recommended First)

#### Train All Baseline Models
```bash
cd cyberbullying-detection/14_scripts
python train_models.py --type baseline --model all
```

#### Train Specific Baseline Model
```bash
# Naive Bayes
python train_models.py --type baseline --model nb

# SVM
python train_models.py --type baseline --model svm

# TF-IDF + Logistic Regression
python train_models.py --type baseline --model tfidf
```

#### Train with Hyperparameter Tuning
```bash
python train_models.py --type baseline --model all --tune
```

**Expected Runtime**: 
- Without tuning: ~5-10 minutes
- With tuning: ~20-30 minutes

### 3. Train Context-Aware Models (Requires PyTorch)

#### Train All Context-Aware Models
```bash
python train_models.py --type context --model all --epochs 10
```

#### Train Specific Context-Aware Model
```bash
# LSTM
python train_models.py --type context --model lstm --epochs 10 --batch-size 32

# BiLSTM
python train_models.py --type context --model bilstm --epochs 10

# Transformer
python train_models.py --type context --model transformer --epochs 15
```

#### Advanced Configuration
```bash
# Use GloVe embeddings (if available)
python train_models.py --type context --model all --embedding glove --epochs 20

# Larger batch size for faster training (if GPU available)
python train_models.py --type context --model all --batch-size 64 --epochs 15
```

**Expected Runtime**: 
- CPU: ~30-60 minutes per model
- GPU: ~10-20 minutes per model

### 4. Train Both Model Types
```bash
python train_models.py --type both --model all --epochs 10
```

## Model Features

### Baseline Models

All baseline models handle class imbalance using:
- **Class Weights**: Automatically balanced (`class_weight='balanced'`)
- **TF-IDF Vectorization**: Max 15,000 features, 1-3 ngrams
- **Optimized Hyperparameters**: Pre-tuned for the dataset

#### Naive Bayes
- Type: ComplementNB (better for imbalanced data)
- Alpha: 0.1 (smoothing)
- Best for: Fast training, interpretable results

#### SVM
- Type: LinearSVC with calibration
- C: 1.0 (regularization)
- Best for: High accuracy, robust to noise

#### TF-IDF + Logistic Regression
- C: 1.0 (regularization)
- Max iterations: 1000
- Best for: Good balance of speed and accuracy

### Context-Aware Models

All context-aware models include:
- **Class Weights**: Computed from training data
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Patience of 5 epochs
- **Multi-Task Learning**: Classification + severity scoring

#### LSTM
- Hidden dim: 256, Layers: 2
- Dropout: 0.3
- Best for: Sequential pattern detection

#### BiLSTM
- Bidirectional encoding
- Attention mechanism
- Best for: Context understanding from both directions

#### Transformer
- Multi-head attention (8 heads)
- Positional encoding
- Best for: Long-range dependencies, complex patterns

## Output and Results

### Results Directory Structure
```
04_evaluation/results/
├── baseline/
│   ├── naive_bayes/
│   │   ├── confusion_matrix.png
│   │   ├── classification_report.txt
│   │   └── metrics.json
│   ├── svm/
│   └── tfidf/
└── context_aware/
    ├── lstm/
    ├── bilstm/
    └── transformer/
```

### Saved Models
```
03_models/saved_models/
├── baseline/
│   ├── naive_bayes/
│   ├── svm/
│   └── tfidf/
└── context_aware/
    ├── lstm/
    ├── bilstm/
    └── transformer/
```

### Logs
```
17_logs/
├── training_YYYYMMDD_HHMMSS.log
└── baseline_training_YYYYMMDD_HHMMSS.log
```

## Performance Tips

### For CPU Training
- Use smaller batch sizes (16-32) for context models
- Start with baseline models first
- Consider training one model at a time

### For GPU Training
- Increase batch size to 64 or 128
- Train all models simultaneously
- Enable mixed precision training (modify code)

### For Memory Issues
- Reduce max_features in TF-IDF (from 15000 to 10000)
- Reduce vocab_size in context models (from 10000 to 5000)
- Use smaller batch sizes

## Troubleshooting

### Issue: ModuleNotFoundError for sklearn
```bash
pip install scikit-learn
```

### Issue: PyTorch not available
```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# With CUDA (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Matplotlib warnings
- These are harmless and can be ignored
- Or install: `pip install matplotlib seaborn`

### Issue: Out of memory
- Reduce batch size: `--batch-size 16`
- Train one model at a time
- Close other applications

### Issue: Training too slow
- Use GPU if available
- Reduce epochs: `--epochs 5`
- Train specific models instead of all

## Next Steps

1. **Analyze Results**: Compare model performance in `04_evaluation/results/`
2. **Tune Hyperparameters**: Use `--tune` flag for baseline models
3. **Ensemble Methods**: Combine multiple models (see `03_models/ensemble/`)
4. **Deploy Best Model**: Use API in `06_api/` folder
5. **Create Dashboard**: Use dashboard in `08_dashboard/` folder

## Command Cheat Sheet

```bash
# Quick start - Train all baseline models
python train_models.py --type baseline --model all

# Full training pipeline
python train_models.py --type both --model all --tune --epochs 15

# Check dataset statistics
python data_statistics.py

# Train best performing baseline (usually SVM)
python train_models.py --type baseline --model svm --tune

# Train best performing deep model (usually BiLSTM)
python train_models.py --type context --model bilstm --epochs 20 --batch-size 32
```

## Expected Performance

Based on the class imbalance, expected metrics:

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|
| Naive Bayes | 65-70% | 45-50% | 70-75% |
| SVM | 70-75% | 50-55% | 75-80% |
| TF-IDF+LogReg | 70-75% | 50-55% | 75-80% |
| LSTM | 72-77% | 52-57% | 77-82% |
| BiLSTM | 74-79% | 54-59% | 79-84% |
| Transformer | 75-80% | 55-60% | 80-85% |

*Note: Macro F1 is lower due to class imbalance. Weighted F1 is more representative of overall performance.*

## Support

For issues or questions:
1. Check logs in `17_logs/`
2. Review error messages in terminal
3. Ensure all dependencies are installed
4. Check that data files exist in `00_data/processed/`
