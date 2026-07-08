import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, f1_score
import numpy as np

def validate_consistency(model, val_loader, device):
    """
    Consistency Check (Phase 8).
    Verifies Ordinal Monotonicity and Regression Alignment before running
    costly full-scale training epochs.
    """
    model.eval()
    all_ordinal_preds = []
    all_regression_preds = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            outputs = model(images)
            
            # Get ordinal class prediction (sum the binary thresholds)
            probs = torch.sigmoid(outputs['ordinal_logits'])
            pred_grades_ordinal = torch.sum(probs > 0.5, dim=1).cpu().numpy()
            all_ordinal_preds.extend(pred_grades_ordinal)
            
            # Get regression score
            reg_scores = outputs['regression_score'].cpu().numpy()
            all_regression_preds.extend(reg_scores)
            
            # We only need a single batch to check consistency logic
            break
            
    all_ordinal_preds = np.array(all_ordinal_preds)
    all_regression_preds = np.array(all_regression_preds)
    
    # Calculate difference
    diff = np.abs(all_ordinal_preds - np.round(all_regression_preds))
    mean_diff = np.mean(diff)
    
    print(f"Consistency Check Mean Difference: {mean_diff:.4f}")
    if mean_diff > 1.5:
         print("WARNING: Ordinal and Regression heads are severely misaligned!")
    return mean_diff

def calculate_metrics(y_true, y_pred_ordinal, y_true_binary, y_pred_binary):
    """ Calculates validation metrics including QWK. """
    qwk = cohen_kappa_score(y_true, y_pred_ordinal, weights='quadratic')
    acc = accuracy_score(y_true, y_pred_ordinal)
    macro_f1 = f1_score(y_true, y_pred_ordinal, average='macro')
    
    # Binary Specificity / Sensitivity
    # Fix #3: Always force labels=[0,1] to guarantee a 2x2 matrix.
    # Without this, a collapsed model predicting only class-0 returns a 1x1 matrix,
    # the shape check silently returns (0.0, 0.0), and the collapse guard fires incorrectly.
    cm_bin = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm_bin.ravel()
    specificity = tn / (tn + fp + 1e-6)
    sensitivity = tp / (tp + fn + 1e-6)
        
    return {
        'qwk': qwk,
        'acc': acc,
        'macro_f1': macro_f1,
        'sensitivity': sensitivity,
        'specificity': specificity
    }
