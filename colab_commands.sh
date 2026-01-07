#!/bin/bash
# Quick Reference Commands for Colab GPU Training
# Copy and paste these commands into Colab cells as needed

# ============================================================
# 1. MOUNT GOOGLE DRIVE
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')

# ============================================================
# 2. SETUP ENVIRONMENT VARIABLES
# ============================================================
export HF_HOME=/content/drive/MyDrive/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# ============================================================
# 3. INSTALL GIT LFS
# ============================================================
apt-get update && apt-get install -y git-lfs
git lfs install

# ============================================================
# 4. CLONE REPOSITORY
# ============================================================
git clone https://github.com/vinod-hn/Cyberbullying-detection.git
cd Cyberbullying-detection
git lfs pull

# ============================================================
# 5. INSTALL DEPENDENCIES
# ============================================================
pip install -U pip
pip install -r requirements.txt
pip install -e .

# ============================================================
# 6. CONFIGURE GIT LFS TRACKING
# ============================================================
git lfs track "*.pt" "*.bin" "*.ckpt" "*.safetensors"
git add .gitattributes

# ============================================================
# 7. CONFIGURE GIT (Replace with your details)
# ============================================================
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# ============================================================
# 8. SETUP NBSTRIPOUT (Strip notebook outputs)
# ============================================================
pip install nbstripout
nbstripout --install

# ============================================================
# 9. CONFIGURE REMOTE WITH TOKEN (Replace <TOKEN>)
# ============================================================
# git remote set-url origin https://<TOKEN>@github.com/vinod-hn/Cyberbullying-detection.git

# ============================================================
# 10. COMMIT AND PUSH
# ============================================================
# git add -A
# git commit -m "Colab changes"
# git push origin HEAD

# ============================================================
# TRAINING COMMANDS
# ============================================================

# Train baseline models
# python 03_models/baseline/train_baseline.py --model all --tune

# Train LSTM context model
# python 03_models/context_aware/train_context_model.py --model lstm --epochs 50

# Train Transformer
# python 03_models/context_aware/train_context_model.py --model transformer --batch-size 32

# ============================================================
# USEFUL DEBUGGING COMMANDS
# ============================================================

# Check GPU
# !nvidia-smi

# Check PyTorch + CUDA
# python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check disk space
# !df -h

# Clear PyTorch cache
# python -c "import torch; torch.cuda.empty_cache()"

# List LFS files
# git lfs ls-files

# Check environment
# !env | grep HF_
