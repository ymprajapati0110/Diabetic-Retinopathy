import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

class OptimizedRounder:
    """
    Bayesian/Nelder-Mead Optimizer to find the exact classification thresholds
    that maximize the Quadratic Weighted Kappa metric.
    
    Why: Neural networks output uncalibrated continuous scores. Hard-rounding 
    (0.5, 1.5, 2.5) is rarely optimal for an imbalanced metric like QWK.
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Loss function to minimize (Negative QWK).
        X: continuous predictions
        y: true ordinal grades
        coef: array of 4 thresholds (Grade 0->1, 1->2, 2->3, 3->4)
        """
        # Vectorized: much faster than element-wise loop
        bins = [-np.inf] + list(coef) + [np.inf]
        X_p = np.digitize(np.array(X), bins) - 1
        X_p = np.clip(X_p, 0, 4)

        qwk = cohen_kappa_score(y, X_p, weights='quadratic')
        return -qwk

    def fit(self, X, y):
        """
        Find the optimal thresholds based on validation predictions.
        X: Continuous Regression scores from our Regression Head
        """
        # Sensible initial guesses for the 4 boundaries
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        
        # Nelder-Mead is an effective heuristic search without requiring gradients
        loss_partial = getattr(self, '_kappa_loss')
        
        result = minimize(
            loss_partial, 
            initial_coef, 
            args=(X, y), 
            method='nelder-mead',
            options={'maxiter': 500} # Cap iterations
        )
        
        # Sort thresholds to ensure monotonicity
        self.coef_ = np.sort(result.x)
        print(f"Optimized Thresholds: {self.coef_}")
        return self.coef_

    def predict(self, X, coef=None):
        """ Apply the optimal thresholds to raw test predictions. """
        if coef is None:
            coef = self.coef_
            
        # Vectorized: consistent with _kappa_loss
        bins = [-np.inf] + list(coef) + [np.inf]
        X_p = np.digitize(np.array(X), bins) - 1
        return np.clip(X_p, 0, 4)
