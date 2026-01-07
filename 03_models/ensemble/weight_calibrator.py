"""
Weight Calibrator for Ensemble Models

Optimizes model weights using various calibration strategies:
- Grid search
- Bayesian optimization
- Gradient descent
- Cross-validation based

Author: Cyberbullying Detection Project Team
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class WeightCalibrator:
    """
    Calibrate ensemble weights for optimal performance.
    
    Supports multiple optimization strategies:
    - Grid search (exhaustive)
    - Random search (faster)
    - Differential evolution (global optimization)
    - Gradient descent (local optimization)
    """
    
    def __init__(self, metric: str = 'f1_weighted', method: str = 'grid'):
        """
        Initialize calibrator.
        
        Args:
            metric: Optimization metric ('accuracy', 'f1_macro', 'f1_weighted')
            method: Optimization method ('grid', 'random', 'evolution', 'gradient')
        """
        self.metric = metric
        self.method = method
        self.best_weights = None
        self.best_score = 0.0
        self.optimization_history = []

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Safely normalize weight vector, avoiding division-by-zero and NaNs."""
        sum_w = float(np.sum(weights))
        if not np.isfinite(sum_w) or sum_w <= 0.0:
            # Fallback to uniform weights
            return np.ones_like(weights, dtype=float) / max(len(weights), 1)
        normalized = np.clip(weights, 0.0, 1.0) / sum_w
        sum_n = float(np.sum(normalized))
        if sum_n <= 0.0:
            return np.ones_like(weights, dtype=float) / max(len(weights), 1)
        return normalized
    
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute specified metric."""
        if self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'f1_macro':
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif self.metric == 'f1_weighted':
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _weighted_predict(self, predictions: List[np.ndarray], 
                          weights: np.ndarray) -> np.ndarray:
        """
        Compute weighted predictions.
        
        Args:
            predictions: List of prediction arrays from each model
            weights: Weight for each model
            
        Returns:
            Final weighted predictions
        """
        if predictions[0].ndim == 1:
            # Hard predictions - use majority voting
            weighted_preds = np.zeros(len(predictions[0]), dtype=float)
            for i, preds in enumerate(predictions):
                # Weighted voting (vectorized)
                weighted_preds += preds * weights[i]
            # Use explicit threshold to avoid banker's rounding at 0.5
            return (weighted_preds >= 0.5).astype(int)
        else:
            # Soft predictions (probabilities)
            weighted_proba = np.zeros_like(predictions[0], dtype=float)
            for i, proba in enumerate(predictions):
                weighted_proba += proba * weights[i]
            return np.argmax(weighted_proba, axis=1)
    
    def grid_search(self, predictions: List[np.ndarray], 
                   y_true: np.ndarray,
                   resolution: int = 10) -> Tuple[np.ndarray, float]:
        """
        Grid search for optimal weights.
        
        Args:
            predictions: List of predictions from each model
            y_true: True labels
            resolution: Number of weight values to try (higher = finer grid)
            
        Returns:
            Tuple of (best_weights, best_score)
        """
        logger.info(f"Starting grid search with resolution {resolution}...")
        
        n_models = len(predictions)
        weight_options = np.linspace(0, 1, resolution)
        
        best_weights = None
        best_score = 0.0
        
        # Generate all weight combinations
        from itertools import product
        total_combinations = resolution ** n_models
        
        for i, weights in enumerate(product(weight_options, repeat=n_models)):
            # Normalize weights safely
            norm_weights = self._normalize_weights(np.array(weights, dtype=float))
            
            # Get predictions
            preds = self._weighted_predict(predictions, norm_weights)
            
            # Compute score
            score = self._compute_metric(y_true, preds)
            
            # Track history
            self.optimization_history.append({
                'weights': norm_weights,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_weights = norm_weights
            
            # Progress
            if (i + 1) % 1000 == 0:
                logger.info(f"  Tested {i+1}/{total_combinations} combinations, best: {best_score:.4f}")
        
        logger.info(f"Grid search complete. Best score: {best_score:.4f}")
        return best_weights, best_score
    
    def random_search(self, predictions: List[np.ndarray],
                     y_true: np.ndarray,
                     n_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Random search for optimal weights.
        
        Args:
            predictions: List of predictions from each model
            y_true: True labels
            n_iterations: Number of random combinations to try
            
        Returns:
            Tuple of (best_weights, best_score)
        """
        logger.info(f"Starting random search with {n_iterations} iterations...")
        
        n_models = len(predictions)
        best_weights = None
        best_score = 0.0
        
        for i in range(n_iterations):
            # Generate random weights
            weights = np.random.random(n_models)
            weights = self._normalize_weights(weights)
            
            # Get predictions
            preds = self._weighted_predict(predictions, weights)
            
            # Compute score
            score = self._compute_metric(y_true, preds)
            
            # Track history
            self.optimization_history.append({
                'weights': weights,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_weights = weights
            
            # Progress
            if (i + 1) % 100 == 0:
                logger.info(f"  Iteration {i+1}/{n_iterations}, best: {best_score:.4f}")
        
        logger.info(f"Random search complete. Best score: {best_score:.4f}")
        return best_weights, best_score
    
    def evolution_search(self, predictions: List[np.ndarray],
                        y_true: np.ndarray,
                        max_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """
        Differential evolution for optimal weights.
        
        Args:
            predictions: List of predictions from each model
            y_true: True labels
            max_iterations: Maximum iterations for evolution
            
        Returns:
            Tuple of (best_weights, best_score)
        """
        logger.info("Starting differential evolution...")
        
        n_models = len(predictions)
        
        # Objective function to minimize (negative score)
        def objective(weights):
            # Normalize weights safely
            weights = self._normalize_weights(np.array(weights, dtype=float))
            
            # Get predictions
            preds = self._weighted_predict(predictions, weights)
            
            # Compute score (negative because we minimize)
            score = self._compute_metric(y_true, preds)
            
            # Track history
            self.optimization_history.append({
                'weights': weights,
                'score': score
            })
            
            return -score
        
        # Bounds for each weight
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Run optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iterations,
            seed=42,
            disp=True
        )
        
        # Normalize best weights safely
        best_weights = self._normalize_weights(np.array(result.x, dtype=float))
        best_score = -result.fun
        
        logger.info(f"Evolution complete. Best score: {best_score:.4f}")
        return best_weights, best_score
    
    def gradient_descent(self, predictions: List[np.ndarray],
                        y_true: np.ndarray,
                        learning_rate: float = 0.01,
                        n_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Gradient descent for optimal weights.
        
        Args:
            predictions: List of predictions from each model
            y_true: True labels
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations
            
        Returns:
            Tuple of (best_weights, best_score)
        """
        logger.info("Starting gradient descent...")
        
        n_models = len(predictions)
        
        # Initialize weights uniformly
        weights = np.ones(n_models) / n_models
        
        best_weights = weights.copy()
        best_score = 0.0
        
        for iteration in range(n_iterations):
            # Get current predictions
            preds = self._weighted_predict(predictions, weights)
            score = self._compute_metric(y_true, preds)
            
            # Track history
            self.optimization_history.append({
                'weights': weights.copy(),
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
            
            # Compute gradient (numerical approximation)
            gradients = np.zeros(n_models)
            epsilon = 1e-5
            
            for i in range(n_models):
                weights_plus = weights.copy()
                weights_plus[i] += epsilon
                weights_plus = weights_plus / np.sum(weights_plus)
                
                preds_plus = self._weighted_predict(predictions, weights_plus)
                score_plus = self._compute_metric(y_true, preds_plus)
                
                gradients[i] = (score_plus - score) / epsilon
            
            # Update weights
            weights += learning_rate * gradients
            weights = self._normalize_weights(np.clip(weights, 0.0, 1.0))
            
            # Progress
            if (iteration + 1) % 100 == 0:
                logger.info(f"  Iteration {iteration+1}/{n_iterations}, score: {score:.4f}, best: {best_score:.4f}")
        
        logger.info(f"Gradient descent complete. Best score: {best_score:.4f}")
        return best_weights, best_score
    
    def calibrate(self, predictions: List[np.ndarray],
                 y_true: np.ndarray,
                 **kwargs) -> Tuple[np.ndarray, float]:
        """
        Calibrate weights using specified method.
        
        Args:
            predictions: List of predictions from each model
            y_true: True labels
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Tuple of (best_weights, best_score)
        """
        if self.method == 'grid':
            self.best_weights, self.best_score = self.grid_search(
                predictions, y_true, 
                resolution=kwargs.get('resolution', 10)
            )
        elif self.method == 'random':
            self.best_weights, self.best_score = self.random_search(
                predictions, y_true,
                n_iterations=kwargs.get('n_iterations', 1000)
            )
        elif self.method == 'evolution':
            self.best_weights, self.best_score = self.evolution_search(
                predictions, y_true,
                max_iterations=kwargs.get('max_iterations', 100)
            )
        elif self.method == 'gradient':
            self.best_weights, self.best_score = self.gradient_descent(
                predictions, y_true,
                learning_rate=kwargs.get('learning_rate', 0.01),
                n_iterations=kwargs.get('n_iterations', 1000)
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info(f"\nCalibration Results:")
        logger.info(f"  Method: {self.method}")
        logger.info(f"  Best Score: {self.best_score:.4f}")
        logger.info(f"  Best Weights: {self.best_weights}")
        
        return self.best_weights, self.best_score
    
    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history."""
        return self.optimization_history


class TemperatureScaling:
    """
    Temperature scaling for probability calibration.
    
    Adjusts model confidence by scaling logits with a learned temperature.
    """
    
    def __init__(self):
        """Initialize temperature scaling."""
        self.temperature = 1.0
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray):
        """
        Learn optimal temperature.
        
        Args:
            logits: Raw model outputs (before softmax)
            y_true: True labels
        """
        # Objective: minimize negative log-likelihood
        def objective(temp):
            # Ensure scalar temperature for stable broadcasting
            t = float(np.squeeze(temp))
            scaled_proba = self._softmax(logits / t)
            nll = -np.mean(np.log(scaled_proba[np.arange(len(y_true)), y_true] + 1e-10))
            return nll
        
        # Optimize temperature
        result = minimize(objective, x0=1.0, bounds=[(0.1, 10.0)])
        self.temperature = result.x[0]
        
        logger.info(f"Learned temperature: {self.temperature:.4f}")
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling.
        
        Args:
            logits: Raw model outputs
            
        Returns:
            Calibrated probabilities
        """
        return self._softmax(logits / self.temperature)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


if __name__ == "__main__":
    print("Weight Calibrator Module")
    print("="*60)
    print("Available methods:")
    print("  - grid: Exhaustive grid search")
    print("  - random: Random weight sampling")
    print("  - evolution: Differential evolution (global)")
    print("  - gradient: Gradient descent (local)")
    print("\nUsage:")
    print("  from weight_calibrator import WeightCalibrator")
    print("  calibrator = WeightCalibrator(metric='f1_weighted', method='evolution')")
    print("  best_weights, best_score = calibrator.calibrate(predictions, y_true)")

