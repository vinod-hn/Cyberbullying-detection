# Cyberbullying Detection System

A comprehensive Kannada-English code-mixed cyberbullying detection system using state-of-the-art machine learning and deep learning techniques.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-hn/Cyberbullying-detection/blob/main/Cyberbullying_Detection_Colab_Training.ipynb)

## 🚀 Quick Start

### For Google Colab (Recommended for Training)

1. **Click the "Open In Colab" badge above** or open [`Cyberbullying_Detection_Colab_Training.ipynb`](Cyberbullying_Detection_Colab_Training.ipynb)
2. **Enable GPU runtime**: Runtime → Change runtime type → GPU
3. **Follow the comprehensive guide**: See [COLAB_TRAINING.md](COLAB_TRAINING.md)

### For Local Development

```bash
# Clone the repository
git clone https://github.com/vinod-hn/Cyberbullying-detection.git
cd Cyberbullying-detection

# Install Git LFS (if not already installed)
git lfs install
git lfs pull

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .

# Validate setup
python validate_setup.py
```

## 📋 Features

- **Multi-lingual Support**: Optimized for Kannada-English code-mixed text
- **Advanced Preprocessing**: Text normalization, emoji handling, transliteration
- **Multiple Models**:
  - Baseline: Naive Bayes, SVM, TF-IDF + Logistic Regression
  - Context-Aware: LSTM with Attention, Transformers
  - Transformer-based: IndicBERT, XLM-R variants
- **Ensemble Learning**: Combined models with optimized weights
- **GPU Acceleration**: Full support for training on Google Colab GPUs
- **Reproducible**: Pinned dependencies and configuration management

## 📁 Project Structure

```
.
├── 00_data/              # Dataset and annotations
├── 01_preprocessing/     # Text preprocessing modules
├── 02_feature_extraction/# Feature extraction
├── 03_models/            # ML/DL models
│   ├── baseline/         # Traditional ML models
│   ├── context_aware/    # LSTM, attention mechanisms
│   ├── transformer/      # BERT, IndicBERT models
│   └── ensemble/         # Ensemble methods
├── 04_evaluation/        # Evaluation metrics
├── 05_severity_scoring/  # Severity assessment
├── 06_api/               # REST API
├── 07_database/          # Database schemas
├── 08_dashboard/         # Visualization dashboard
├── 09_privacy_security/  # Security measures
├── 10_deployment/        # Deployment configs
├── 11_notebooks/         # Analysis notebooks
└── requirements.txt      # Python dependencies
```

## 🔧 Installation

### Requirements

- Python 3.9+
- Git LFS (for large model files)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or with Conda:

```bash
conda env create -f environment.yml
conda activate cyberbullying-detection
```

### Verify Installation

```bash
python validate_setup.py
```

## 🎓 Training Models

### On Google Colab (GPU)

See the comprehensive guide: [**COLAB_TRAINING.md**](COLAB_TRAINING.md)

Quick reference commands are available in: [`colab_commands.sh`](colab_commands.sh)

### Locally

```bash
# Train baseline models
python 03_models/baseline/train_baseline.py --model all --tune

# Train LSTM with attention
python 03_models/context_aware/train_context_model.py --model lstm --epochs 50

# Train ensemble
python 03_models/ensemble/train_ensemble.py
```

## 📊 Data

The dataset consists of Kannada-English code-mixed social media text with cyberbullying annotations:

- **11 Categories**: Harassment, hate speech, threats, etc.
- **Severity Scores**: Multi-level severity assessment
- **Context**: Conversation threading for context-aware detection

Data is located in `00_data/` with preprocessing in `01_preprocessing/`.

## 🧪 Testing

```bash
# Run preprocessing tests
python -m pytest 01_preprocessing/tests/ -v

# Validate setup
python validate_setup.py
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Documentation

- [**COLAB_TRAINING.md**](COLAB_TRAINING.md) - Comprehensive Colab GPU training guide
- [**QUICK_START.md**](QUICK_START.md) - Quick start guide
- [**TRAINING_SUMMARY.md**](TRAINING_SUMMARY.md) - Training details and results
- [`colab_commands.sh`](colab_commands.sh) - Quick reference commands

## 🔐 Git LFS

This project uses Git LFS for large model files. Supported file types:
- Model weights: `.pt`, `.pth`, `.bin`, `.ckpt`, `.safetensors`
- Data files: `.parquet`, `.arrow`

See [.gitattributes](.gitattributes) for full configuration.

## ⚙️ Configuration

Dependencies are managed through:
- `requirements.txt` - Pip dependencies with version pins
- `environment.yml` - Conda environment specification
- `setup.py` - Package setup for editable install

## 🚦 CI/CD

Automated validation runs on push/PR:
- Python syntax validation
- Module import tests
- Basic functionality tests

See [.github/workflows/ci-validation.yml](.github/workflows/ci-validation.yml)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

Cyberbullying Detection Project Team

## 🙏 Acknowledgments

- IndicBERT: [ai4bharat/indic-bert](https://huggingface.co/ai4bharat/indic-bert)
- Hugging Face Transformers
- PyTorch
- scikit-learn

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Ready to start training?** 🎯

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-hn/Cyberbullying-detection/blob/main/Cyberbullying_Detection_Colab_Training.ipynb) - Click to open in Colab
2. Read [COLAB_TRAINING.md](COLAB_TRAINING.md) for detailed instructions
3. Follow the notebook cells to train your models with GPU acceleration!
