# Ensemble Models for Cyberbullying Detection

This folder contains ensemble learning implementations that combine multiple models for improved cyberbullying detection performance.

## Overview

Ensemble methods combine predictions from multiple models to achieve better accuracy, robustness, and generalization than individual models. This implementation provides four main ensemble strategies plus multi-task learning.

## Ensemble Strategies

### 1. Voting Ensemble (`VotingEnsemble`)
Combines model predictions through voting:
- **Hard Voting**: Majority vote from model predictions
- **Soft Voting**: Average of predicted probabilities (recommended)

**Best for**: Quick ensemble without additional training

### 2. Stacking Ensemble (`StackingEnsemble`)
Uses base model predictions as features for a meta-learner:
- Base models make predictions
- Meta-classifier (Logistic Regression/Random Forest) learns from these predictions
- Can include original features for better performance

**Best for**: Maximum accuracy when you have validation data

### 3. Weighted Ensemble (`WeightedEnsemble`)
Learns optimal weights for each model:
- Automatically finds best weight combination
- Uses validation performance to optimize
- More flexible than equal-weight voting

**Best for**: When models have varying performance

### 4. Feature Ensemble (`FeatureEnsemble`)
Combines multiple feature extractors:
- Linguistic features (syntax, sentiment, readability)
- Emoji features (emoji patterns, sentiment)
- Contextual features (conversation history, escalation)
- Behavioral features (user patterns, targeting)
- Single classifier trained on combined features

**Best for**: Rich feature-based modeling without multiple model training

## Files

### Core Modules

- **`ensemble_model.py`**: Main ensemble implementations
  - `CyberbullyingEnsemble`: Main interface for all ensemble types
  - `VotingEnsemble`: Voting-based combination
  - `StackingEnsemble`: Stacking with meta-learner
  - `WeightedEnsemble`: Optimized weight combination
  - `FeatureEnsemble`: Multi-feature classifier

- **`weight_calibrator.py`**: Weight optimization utilities
  - `WeightCalibrator`: Optimize model weights using various methods
    - Grid search (exhaustive)
    - Random search (faster)
    - Differential evolution (global optimization)
    - Gradient descent (local optimization)
  - `TemperatureScaling`: Probability calibration

- **`multi_task_learning.py`**: Multi-task learning models
  - `MultiTaskLearner`: Train single model for multiple tasks
  - Simultaneous prediction of:
    - Cyberbullying classification (main)
    - Severity score (auxiliary)
    - Target type (auxiliary)

- **`train_ensemble.py`**: Training script for all ensemble types

- **`__init__.py`**: Package initialization

## Usage

### Basic Voting Ensemble

```python
from ensemble_model import CyberbullyingEnsemble

# Create ensemble
ensemble = CyberbullyingEnsemble(ensemble_type='voting', config={'voting': 'soft'})

# Load trained models
model_paths = [
    'saved_models/baseline/naive_bayes',
    'saved_models/baseline/svm',
    'saved_models/baseline/tfidf'
]
ensemble.load_models(model_paths)

# Build ensemble
ensemble.build(X_val, y_val)

# Make predictions
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)

# Evaluate
results = ensemble.evaluate(X_test, y_test)
```

### Stacking Ensemble

```python
ensemble = CyberbullyingEnsemble(
    ensemble_type='stacking',
    config={'use_features': True}  # Include original features
)

ensemble.load_models(model_paths)
ensemble.build(X_val, y_val)  # Trains meta-classifier

predictions = ensemble.predict(X_test)
```

### Weighted Ensemble with Optimization

```python
ensemble = CyberbullyingEnsemble(ensemble_type='weighted')

ensemble.load_models(model_paths)
ensemble.build(X_val, y_val)  # Optimizes weights

# Optimal weights are learned automatically
predictions = ensemble.predict(X_test)
```

### Feature Ensemble

```python
ensemble = CyberbullyingEnsemble(
    ensemble_type='feature',
    config={
        'feature_types': ['linguistic', 'emoji', 'contextual']
    }
)

# No need to load models - builds its own classifier
ensemble.build(X_train, y_train)

predictions = ensemble.predict(X_test)
```

### Weight Calibration

```python
from weight_calibrator import WeightCalibrator

# Get predictions from all models
predictions = [model1.predict_proba(X_val), model2.predict_proba(X_val), ...]

# Calibrate weights
calibrator = WeightCalibrator(metric='f1_weighted', method='evolution')
best_weights, best_score = calibrator.calibrate(predictions, y_val)

print(f"Optimal weights: {best_weights}")
print(f"Validation score: {best_score:.4f}")
```

### Multi-Task Learning

```python
from multi_task_learning import MultiTaskLearner

learner = MultiTaskLearner(config={
    'vocab_size': 10000,
    'hidden_dim': 256,
    'epochs': 10
})

# Train on multiple tasks
learner.fit(X_train, {
    'class': y_class,
    'severity': y_severity,
    'target': y_target_type
})

# Get predictions for all tasks
predictions = learner.predict(X_test)
# Returns: {'class': [...], 'severity': [...], 'target': [...]}

# Or just main classification
class_preds = learner.predict_class(X_test)
```

## Training Scripts

### Train Single Ensemble Type

```bash
# Voting ensemble
python train_ensemble.py --type voting --models baseline

# Stacking ensemble
python train_ensemble.py --type stacking --models baseline

# Weighted ensemble
python train_ensemble.py --type weighted --models baseline

# Feature ensemble
python train_ensemble.py --type feature --models baseline
```

### Train and Compare All Ensembles

```bash
python train_ensemble.py --type all --models baseline --compare
```

This will:
1. Train all four ensemble types
2. Evaluate each on test data
3. Print comparison table
4. Identify best ensemble strategy

## Requirements

### Core Requirements
- `scikit-learn`: Base ensemble functionality
- `numpy`: Numerical operations
- `pandas`: Data handling

### Optional Requirements
- `scipy`: Advanced optimization (differential evolution)
- `torch`: Multi-task learning with neural networks

### Feature Extraction (for Feature Ensemble)
- `nltk`: Natural language processing
- `textblob`: Sentiment analysis

Install all:
```bash
pip install scikit-learn numpy pandas scipy torch nltk textblob
```

## Configuration

### Ensemble Configuration

```python
config = {
    # Voting
    'voting': 'soft',  # 'hard' or 'soft'
    
    # Stacking
    'use_features': True,  # Include original features
    
    # Feature Ensemble
    'feature_types': [
        'linguistic',  # Lexical, syntactic, semantic
        'behavioral',  # User patterns, targeting
        'emoji',       # Emoji sentiment, patterns
        'contextual'   # Conversation context
    ]
}
```

### Multi-Task Configuration

```python
config = {
    'vocab_size': 10000,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10,
    'task_weights': {
        'classification': 1.0,  # Main task
        'severity': 0.5,        # Auxiliary
        'target': 0.3           # Auxiliary
    }
}
```

## Expected Performance

Based on baseline models achieving ~99% accuracy:

| Ensemble Type | Expected Accuracy | Training Time | Inference Speed |
|--------------|------------------|---------------|-----------------|
| Voting       | 99.0-99.5%       | None (instant)| Fast            |
| Stacking     | 99.2-99.7%       | ~1 min        | Medium          |
| Weighted     | 99.1-99.6%       | ~2-5 min      | Fast            |
| Feature      | 98.5-99.3%       | ~2-10 min     | Medium          |
| Multi-Task   | 98.8-99.5%       | ~10-30 min    | Medium          |

## Best Practices

### 1. Model Selection
- Use diverse base models (NB, SVM, TF-IDF) for better ensemble
- Avoid highly correlated models
- Include at least 3 base models

### 2. Weight Optimization
- Use validation set for weight optimization
- Try different optimization methods:
  - Grid search for 2-3 models
  - Evolution for 4+ models
  - Random search for quick approximation

### 3. Stacking Strategy
- Include original features (`use_features=True`) for best results
- Use simple meta-learners (Logistic Regression) to avoid overfitting
- Ensure base models are well-calibrated

### 4. Feature Ensemble
- Use all feature types for maximum coverage
- Feature extraction can be slow - cache features if possible
- Best when you need interpretable features

### 5. Multi-Task Learning
- Main task should have most weight (1.0)
- Auxiliary tasks help regularization (0.3-0.5)
- Requires sufficient training data (5000+ samples)

## Troubleshooting

### Issue: Ensemble performs worse than best single model
**Solution**: Check model diversity. If models are too similar, ensemble won't help. Try different model types.

### Issue: Stacking overfits
**Solution**: 
- Use simpler meta-learner
- Add regularization (higher C in LogisticRegression)
- Use cross-validation for meta-features

### Issue: Weight optimization takes too long
**Solution**:
- Use random search instead of grid search
- Reduce resolution: `resolution=5` instead of `resolution=10`
- Use gradient descent for local optimization

### Issue: Feature ensemble is slow
**Solution**:
- Reduce feature types
- Cache extracted features
- Use parallel processing for batch prediction

### Issue: Multi-task learning requires PyTorch
**Solution**: Install PyTorch:
```bash
pip install torch
```

## Examples

See [QUICK_START.md](../../QUICK_START.md) for complete examples and tutorials.

## Citation

If you use this ensemble implementation in your research, please cite:
```
Cyberbullying Detection Ensemble Models
Kannada-English Code-Mixed Text
2026
```

## License

Part of the Cyberbullying Detection Project.

## Support

For issues or questions:
1. Check logs in `17_logs/`
2. Review error messages
3. Ensure all dependencies are installed
4. Verify base models are trained and saved correctly
