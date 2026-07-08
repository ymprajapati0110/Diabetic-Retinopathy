import torch
import torch.nn as nn
from src.models.backbones import DRBackbone
from src.models.attention import FeatureRefinementGate, CBAM_Module
from src.models.heads import MultiTaskHead

class SOTA_DR_Model(nn.Module):
    """
    Complete architecture combining Backbone -> Feature Refinement Gate -> CBAM -> Multi-Task Head.
    """
    def __init__(self, model_name='convnextv2_large', pretrained=True, drop_path_rate=0.3):
        super().__init__()
        self.backbone = DRBackbone(model_name=model_name, pretrained=pretrained, drop_path_rate=drop_path_rate)
        in_features = self.backbone.num_features
        
        self.frg = FeatureRefinementGate(in_channels=in_features)
        self.cbam = CBAM_Module(in_channels=in_features)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = MultiTaskHead(in_features=in_features)
        
    def forward(self, x, epoch=None):
        # 1. Extract features (B, C, H, W)
        features = self.backbone(x)
        
        # 2. Get scalar complexity weight
        frg_weight = self.frg(features)
        
        # 3. Apply CBAM, modulated by FRG weight
        refined_features = self.cbam(features, frg_weight=frg_weight, epoch=epoch)
        
        # 4. Global pooling
        pooled = self.global_pool(refined_features).flatten(1)
        
        # 5. Multi-task prediction (heads carry their own Dropout)
        outputs = self.head(pooled)
        return outputs
