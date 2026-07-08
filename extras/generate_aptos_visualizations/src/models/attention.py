import torch
import torch.nn as nn

class FeatureRefinementGate(nn.Module):
    """
    Lightweight MobileNet-style block.
    Outputs a scalar attention weight to modulate the attention layer based on image complexity/lesion density.
    Does NOT modify backbone features directly, just outputs a scalar scaler.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x is (B, C, H, W)
        # Output is (B, 1, 1, 1) representing scalar attention weight
        weight = self.gate(x)
        return weight

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM_Module(nn.Module):
    """
    Standard CBAM applied post-FRG.
    Optionally accepts a scalar weight from the FRG to scale the final output.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x, frg_weight=None, epoch=None):
        out = x * self.ca(x)
        out = out * self.sa(out)
        
        # Apply Feature Refinement Gate weight if provided
        if frg_weight is not None:
            # FRG Warmup Logic (Epochs 1-5: 1.0, Epochs 6-10: linear decay to network's output)
            if epoch is not None:
                if epoch <= 5:
                    frg_weight = torch.ones_like(frg_weight)
                elif epoch <= 10:
                    # Linearly decay the "force" weight from 1.0 down to the network's output
                    # Epoch 6: 0.8 * 1.0 + 0.2 * frg_weight
                    # Epoch 10: 0.0 * 1.0 + 1.0 * frg_weight
                    alpha = (11 - epoch) / 6.0  # Decays from 5/6 (0.83) to 1/6 (0.16)
                    frg_weight = alpha * torch.ones_like(frg_weight) + (1 - alpha) * frg_weight
                    
            out = x + frg_weight * (out - x)  # Stable residual scaling
            
        return out
