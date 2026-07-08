import torch
import torch.nn as nn
import timm

class DRBackbone(nn.Module):
    """
    Wrapper for ConvNeXt-V2-Large and EfficientNetV2-L.
    Returns the feature maps before global pooling (Shape: B, C, H, W).
    """
    def __init__(self, model_name='convnextv2_large', pretrained=True, drop_path_rate=0.3):
        super().__init__()
        
        # Determine the correct model string for timm if shorthand is used
        if model_name == 'convnextv2_large':
            timm_name = 'convnextv2_large.fcmae_ft_in1k'
        elif model_name == 'efficientnetv2_l':
            timm_name = 'tf_efficientnetv2_l.in21k'
        else:
            timm_name = model_name
            
        # We only want the feature maps, no classification head and no global pooling yet
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool='', drop_path_rate=drop_path_rate)
        
        # Store feature dimension for downstream heads
        self.num_features = self.backbone.num_features
            
    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable=enable)
            
    def forward(self, x):
        # Output shape is (B, C, H, W)
        return self.backbone(x)
