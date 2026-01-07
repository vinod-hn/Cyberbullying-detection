import numpy as np
import sys
from pathlib import Path

# Configure import path to ensemble module
PROJECT_ROOT = Path(r"c:/Users/admin/Desktop/cyber bullyingggg/my guide(shwe)")
sys.path.insert(0, str(PROJECT_ROOT / "cyberbullying-detection" / "03_models" / "ensemble"))

from weight_calibrator import WeightCalibrator

# Dummy predictions (binary hard labels)
pred1 = np.array([0, 1, 1, 0, 1])
pred2 = np.array([0, 1, 0, 0, 1])
pred3 = np.array([1, 1, 1, 0, 1])
y_true = np.array([0, 1, 1, 0, 1])

cal = WeightCalibrator(metric='accuracy', method='random')
weights, score = cal.calibrate([pred1, pred2, pred3], y_true, n_iterations=50)
print("Weights:", weights)
print("Score:", f"{score:.4f}")
