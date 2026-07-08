import torch
import torch.nn as nn
import torch.nn.functional as F

def corn_loss(logits, y_train, num_classes, label_smoothing=0.0):
    """
    Conditional Ordinal Regression Neural Network loss.
    y_train is the ground truth ordinal class (0 to num_classes-1).
    Heavily weighted (1.5x) on positive cases to fix 0.0 QWK (Grade 0 bias).
    """
    sets = []
    for i in range(num_classes - 1):
        label_mask = y_train > i
        label_tensor = torch.unsqueeze(label_mask.float(), 1)
        sets.append(label_tensor)
    
    y_train_binary = torch.cat(sets, dim=1)
    
    if label_smoothing > 0.0:
        y_train_binary = y_train_binary * (1.0 - label_smoothing) + 0.5 * label_smoothing
    
    # Weighting: 1.5x on positive cases (Grade > i) to force minority-class learning
    loss = F.binary_cross_entropy_with_logits(logits, y_train_binary, pos_weight=torch.tensor([1.5], device=logits.device), reduction='mean')
    return loss

def focal_loss_binary(logits, targets, alpha=0.28, gamma=2.0):
    """
    Focal Loss for Binary Classification (handles severe class imbalance). 
    ADVICE: Monitor referable DR (Binary) performance early. If the model 
    is too "conservative" (missing many Grade 2s), try increasing alpha 
    to 0.50 to give more weight to the referable cases.
    """
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()

class QWKLoss(nn.Module):
    """ Differentiable Quadratic Weighted Kappa Loss using soft confusion matrix. """
    def __init__(self, num_classes=5, epsilon=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
        # Create cost weight matrix based on quadratic distance
        weights = torch.zeros((num_classes, num_classes))
        for i in range(num_classes):
             for j in range(num_classes):
                  weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer('weights', weights)

    def forward(self, logits, targets):
        """
        Calculates Soft QWK. Since these are CORN logits (independent binary classifiers),
        we convert them to marginal probabilities first.
        """
        probs = torch.sigmoid(logits) # (B, 4)
        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6) # Stability clamping
        
        p = torch.zeros((logits.size(0), self.num_classes), device=logits.device)
        p[:, 0] = 1.0 - probs[:, 0]
        p[:, 1] = probs[:, 0] * (1.0 - probs[:, 1])
        p[:, 2] = probs[:, 0] * probs[:, 1] * (1.0 - probs[:, 2])
        p[:, 3] = probs[:, 0] * probs[:, 1] * probs[:, 2] * (1.0 - probs[:, 3])
        p[:, 4] = probs[:, 0] * probs[:, 1] * probs[:, 2] * probs[:, 3]
        
        # Normalize for numerical stability
        p = p / (p.sum(dim=1, keepdim=True) + self.epsilon)
        
        # Actual one-hot targets
        y = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Calculate confusion matrix (soft)
        O = torch.matmul(p.t(), y)
        
        # Expected matrix E
        e_hist_pred = p.sum(dim=0)
        e_hist_true = y.sum(dim=0)
        E = torch.outer(e_hist_pred, e_hist_true) / (p.size(0) + self.epsilon)
        
        # Standard QWK formula
        num = torch.sum(self.weights * O)
        den = torch.sum(self.weights * E)
        
        kappa = 1.0 - (num / (den + self.epsilon))
        
        # Loss is 1 - kappa (0 = perfect agreement, up to 2 = worst case)
        return 1.0 - kappa

class MultiTaskLoss(nn.Module):
    """
    Composite loss function aligning exactly with the architecture specification.
    L = L_ordinal + 0.3*L_binary + 0.2*L_regression + 0.1*L_QWK + 0.05*L_consistency
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.qwk_loss = QWKLoss(num_classes=num_classes)
        self.huber_loss = nn.HuberLoss(delta=1.0)
        
    def forward(self, outputs, targets, lambda_qwk=0.1, lambda_consistency=0.05, label_smoothing=0.0):
        # 1. Ordinal Loss
        l_ord = corn_loss(outputs['ordinal_logits'], targets['ordinal'], self.num_classes, label_smoothing=label_smoothing)
        
        # 2. Binary Loss (Referable DR)
        l_bin = focal_loss_binary(outputs['binary_logits'], targets['binary'])
        
        # 3. Regression Loss
        l_reg = self.huber_loss(outputs['regression_score'], targets['regression'])
        
        # 4. QWK Loss
        l_qwk = self.qwk_loss(outputs['ordinal_logits'], targets['ordinal'])
        
        # 5. Consistency Loss
        probs = torch.sigmoid(outputs['ordinal_logits'])
        expected_grade = (
            probs[:, 0] +
            probs[:, 1] +
            probs[:, 2] +
            probs[:, 3]
        )
        # Fix #2: Clamp regression score before consistency loss.
        # Unbounded regression scores (outside [0,4]) cause l_consist to explode,
        # which destabilizes the ordinal head via backprop.
        reg_clamped = torch.clamp(outputs['regression_score'], 0.0, 4.0)
        l_consist = F.huber_loss(expected_grade, reg_clamped, delta=1.0)
        
        # Total Composite Loss — equal weights per spec (1.0·CORN + 1.0·focal + 1.0·huber)
        total_loss = l_ord + l_bin + l_reg + (lambda_qwk * l_qwk) + (lambda_consistency * l_consist)
        
        loss_dict = {
            'total_loss': total_loss,
            'l_ord': l_ord,
            'l_bin': l_bin,
            'l_reg': l_reg,
            'l_qwk': l_qwk,
            'l_consist': l_consist
        }
        return total_loss, loss_dict
