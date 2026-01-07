# Colab GPU Training Guide

This guide explains how to train the Cyberbullying Detection models using Google Colab with GPU acceleration.

## Quick Start

### Option 1: Pure Colab (Browser)

1. **Open the notebook in Colab**
   - Click the "Open in Colab" badge in `Cyberbullying_Detection_Colab_Training.ipynb`
   - Or go to [Google Colab](https://colab.research.google.com/) and upload the notebook

2. **Enable GPU Runtime**
   - Runtime → Change runtime type → GPU (T4 recommended)
   - Verify GPU: Run `!nvidia-smi`

3. **Mount Google Drive** (for persisting data/models)
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Set HuggingFace Cache to Drive** (to avoid re-downloading)
   ```bash
   export HF_HOME=/content/drive/MyDrive/hf_cache
   export HF_DATASETS_CACHE=$HF_HOME/datasets
   export TRANSFORMERS_CACHE=$HF_HOME/transformers
   ```

5. **Install Git LFS**
   ```bash
   apt-get update && apt-get install -y git-lfs
   git lfs install
   ```

6. **Clone Repository**
   ```bash
   git clone https://github.com/vinod-hn/Cyberbullying-detection.git
   cd Cyberbullying-detection
   git lfs pull
   ```

7. **Install Dependencies**
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   pip install -e .
   ```

8. **Start Training**
   - Follow the notebook cells to train models
   - Models will be saved to Google Drive

### Option 2: VS Code + Colab Extension

1. Install [VS Code Colab extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-jupyter-colab)
2. Open notebook in VS Code
3. Connect to Colab backend (select kernel)
4. Enable GPU runtime
5. Follow same setup steps as Option 1 in notebook cells

## Key Features

### GPU Acceleration
- **PyTorch AMP** (Automatic Mixed Precision) for faster training
- **Hugging Face Accelerate** for distributed training (if needed)
- DataLoader optimization with multi-workers and pin_memory

### Persistent Storage
- **Google Drive mount** for datasets and model checkpoints
- **Git LFS** for tracking large model files
- **HuggingFace cache** on Drive to avoid re-downloading models

### Dependencies
All dependencies are pinned in `requirements.txt`:
- PyTorch 2.0+
- Transformers 4.30+
- Accelerate 0.20+
- scikit-learn, pandas, numpy
- And more...

## Training Commands

### Baseline Models
```bash
python 03_models/baseline/train_baseline.py --model all --tune
```

### Context-Aware Models (LSTM)
```bash
python 03_models/context_aware/train_context_model.py --model lstm --epochs 50
```

### Transformer Models (IndicBERT)
```bash
# Coming soon - use notebook for now
```

## Saving Models

### To Google Drive
```python
import torch
torch.save(model.state_dict(), 
          '/content/drive/MyDrive/experiments/run1/model.pt')
```

### Track with Git LFS
```bash
git lfs track "*.pt" "*.bin" "*.ckpt" "*.safetensors"
git add .gitattributes
git add path/to/model.pt
git commit -m "Add trained model"
```

## Pushing Changes to GitHub

1. **Strip notebook outputs** (to avoid storing large outputs)
   ```bash
   pip install nbstripout
   nbstripout --install
   nbstripout Cyberbullying_Detection_Colab_Training.ipynb
   ```

2. **Configure Git** (with your details)
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```

3. **Set remote with token** (use Personal Access Token)
   ```bash
   git remote set-url origin https://<TOKEN>@github.com/vinod-hn/Cyberbullying-detection.git
   ```

4. **Commit and push**
   ```bash
   git add -A
   git commit -m "Training logs + artifacts"
   git push origin HEAD
   ```

## Reproducibility

### Freeze Dependencies
```bash
pip freeze > requirements-lock.txt
```

### Save Training Config
Always save:
- Random seeds
- Hyperparameters
- Model architecture details
- PyTorch/CUDA versions
- Training timestamps

### Example Config
```python
config = {
    'seed': 42,
    'model': 'lstm-attention',
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 50,
    'pytorch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'timestamp': datetime.now().isoformat()
}
```

## Common Issues & Solutions

### Issue: LFS pointer files instead of actual files
**Solution:** Install git-lfs and run `git lfs pull`
```bash
apt-get install -y git-lfs
git lfs install
git lfs pull
```

### Issue: Runtime disconnects/resets
**Solution:** 
- Save checkpoints to Drive frequently
- Re-run setup cells after reconnection
- Use persistent Drive paths

### Issue: CUDA out of memory
**Solution:**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision (AMP)
- Clear cache: `torch.cuda.empty_cache()`

### Issue: Module import errors
**Solution:** Install in editable mode
```bash
pip install -e .
# OR add to path
import sys
sys.path.append('/content/Cyberbullying-detection')
```

### Issue: HuggingFace models downloading repeatedly
**Solution:** Set cache to Drive
```bash
export HF_HOME=/content/drive/MyDrive/hf_cache
```

## Advanced: Conda on Colab

Use CondaColab for conda environments:
```python
pip install -q condacolab
import condacolab
condacolab.install()
# Kernel will restart
```

Then:
```bash
mamba env update -f environment.yml
```

## Performance Tips

1. **Use DataLoader efficiently**
   ```python
   DataLoader(dataset, 
             batch_size=32, 
             num_workers=2,  # Use 2-4 workers
             pin_memory=True)  # Faster GPU transfer
   ```

2. **Enable Mixed Precision**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       output = model(input)
       loss = criterion(output, target)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Use Accelerate for simplicity**
   ```python
   from accelerate import Accelerator
   accelerator = Accelerator(mixed_precision='fp16')
   model, optimizer, dataloader = accelerator.prepare(
       model, optimizer, dataloader)
   ```

## Resources

- [Google Colab Documentation](https://colab.research.google.com/)
- [PyTorch AMP Tutorial](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/)
- [Git LFS Documentation](https://git-lfs.github.com/)

## Next Steps

1. Run the provided notebook: `Cyberbullying_Detection_Colab_Training.ipynb`
2. Experiment with hyperparameters
3. Try IndicBERT/XLM-R variants
4. Track experiments with weights & biases (optional)
5. Share results and trained models

---

**Happy Training! 🚀**
