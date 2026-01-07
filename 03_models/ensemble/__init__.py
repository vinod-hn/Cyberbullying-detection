"""
Ensemble Module for Cyberbullying Detection

Provides ensemble methods combining multiple models:
- Voting ensemble (hard/soft voting)
- Stacking ensemble (meta-learner)
- Weighted ensemble (optimized weights)
- Feature-level ensemble
- Multi-task learning

Author: Cyberbullying Detection Project Team
"""

from .ensemble_model import (
    VotingEnsemble,
    StackingEnsemble,
    WeightedEnsemble,
    FeatureEnsemble,
    CyberbullyingEnsemble,
    create_ensemble_from_baseline
)

from .weight_calibrator import (
    WeightCalibrator,
    TemperatureScaling
)

from .multi_task_learning import (
    MultiTaskLearner,
    combine_predictions
)

__all__ = [
    'VotingEnsemble',
    'StackingEnsemble',
    'WeightedEnsemble',
    'FeatureEnsemble',
    'CyberbullyingEnsemble',
    'create_ensemble_from_baseline',
    'WeightCalibrator',
    'TemperatureScaling',
    'MultiTaskLearner',
    'combine_predictions'
]

__version__ = '1.0.0'
