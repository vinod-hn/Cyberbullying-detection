import importlib.util
from pathlib import Path
import torch
import json

# Load module from file
module_path = Path(__file__).parent / 'bert_message_classifier.py'
spec = importlib.util.spec_from_file_location('bm', str(module_path))
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

# Initialize classifier
clf = bm.BertMessageClassifier()

# Load data and dataloaders
train_df, val_df, test_df = clf.load_data()
train_loader, val_loader, test_loader = clf.prepare_dataloaders(train_df, val_df, test_df)

# Build model and load checkpoint if available
clf.build_model()
checkpoint_path = bm.PROJECT_ROOT / '03_models' / 'saved_models' / 'transformer' / 'best_model.pt'
if not checkpoint_path.exists():
    # fallback to bert_classifier/model.pt
    fallback = bm.PROJECT_ROOT / '03_models' / 'saved_models' / 'transformer' / 'bert_classifier' / 'model.pt'
    if fallback.exists():
        chk = torch.load(fallback, map_location=clf.device)
        # assume it's a state_dict
        state = chk if isinstance(chk, dict) and not ('model_state_dict' in chk) else chk
        if isinstance(state, dict) and 'model_state_dict' in state:
            clf.model.load_state_dict(state['model_state_dict'])
        else:
            clf.model.load_state_dict(state)
        print(f"Loaded fallback model from {fallback}")
    else:
        print(f"No checkpoint found at {checkpoint_path} or fallback; training will start from scratch")
else:
    chk = torch.load(checkpoint_path, map_location=clf.device)
    if isinstance(chk, dict) and 'model_state_dict' in chk:
        clf.model.load_state_dict(chk['model_state_dict'])
        start_best = chk.get('best_val_f1', 0.0)
    else:
        clf.model.load_state_dict(chk)
        start_best = 0.0
    print(f"Loaded checkpoint from {checkpoint_path} (best_val_f1={start_best})")

# Resume training: run one more epoch
# Configure to run 1 epoch (resume for remaining)
clf.config['epochs'] = 1
trainer = bm.BertTrainer(clf.model, clf.config, clf.device)
# preserve previous best if available
try:
    trainer.best_val_f1 = float(start_best)
except Exception:
    pass

print('Resuming training for 1 epoch...')
history = trainer.train(train_loader, val_loader)
print('Training finished. Running evaluation on test set...')
metrics = trainer.evaluate(test_loader)

# Save full classifier (tokenizer, label encoder, model)
clf.save()

# Print metrics JSON to stdout
print('\n=== TEST METRICS ===')
print(json.dumps({
    'loss': metrics['loss'],
    'accuracy': metrics['accuracy'],
    'f1_macro': metrics['f1_macro'],
    'f1_weighted': metrics['f1_weighted'],
    'precision': metrics['precision'],
    'recall': metrics['recall']
}, indent=2))

# Also write metrics to a file
out_path = bm.PROJECT_ROOT / '03_models' / 'saved_models' / 'transformer' / 'test_metrics.json'
with open(out_path, 'w') as f:
    json.dump({
        'loss': metrics['loss'],
        'accuracy': metrics['accuracy'],
        'f1_macro': metrics['f1_macro'],
        'f1_weighted': metrics['f1_weighted'],
        'precision': metrics['precision'],
        'recall': metrics['recall']
    }, f, indent=2)
print(f"Metrics written to {out_path}")
