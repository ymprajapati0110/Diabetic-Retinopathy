import torch
import torch.nn as nn

class CORNOrdinalHead(nn.Module):
    """
    Conditional Ordinal Regression Neural Network (CORN) Head.
    For Num_Classes = 5 (Grades 0-4), we need 4 binary classifiers.
    LayerNorm used instead of BatchNorm1d — works at any batch size including 1.
    """
    def __init__(self, in_features, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),   # Fix #1: LayerNorm is batch-size agnostic (was BatchNorm1d)
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes - 1)
        )
        
    def forward(self, x):
        return self.fc(x)

class BinaryHead(nn.Module):
    """ Referable DR >= 2 """
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),   # Fix #1: LayerNorm (was BatchNorm1d)
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.fc(x)
        
class RegressionHead(nn.Module):
    """ Continuous 0-4 severity output """
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),   # Fix #1: LayerNorm (was BatchNorm1d)
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.fc(x)  # No clamp during training — preserves gradient flow. Clamp at inference only.

class MultiTaskHead(nn.Module):
    """
    Combines the Ordinal, Binary, and Regression heads.
    Takes globally pooled features as input.
    """
    def __init__(self, in_features, num_classes=5):
        super().__init__()
        self.ordinal_head = CORNOrdinalHead(in_features, num_classes)
        self.binary_head = BinaryHead(in_features)
        self.regression_head = RegressionHead(in_features)
        
    def forward(self, x):
        ordinal_logits = self.ordinal_head(x)
        binary_logits = self.binary_head(x)
        regression_score = self.regression_head(x)
        
        return {
            'ordinal_logits': ordinal_logits,
            'binary_logits': binary_logits.view(-1),
            'regression_score': regression_score.view(-1)
        }
