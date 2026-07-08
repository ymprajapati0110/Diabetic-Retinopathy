import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAMHooker:
    """
    Attaches to the final layers of ConvNeXt/EfficientNet just before global pooling.
    Generates a heatmap mapping the gradients of the regression score back 
    to the original spatial dimensions.
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Attach hooks to the backbone's final output layer
        self.target_layer = self.model.backbone
        
        # Fix #9: Store hook handles so they can be removed to prevent memory leaks
        self._hook_fwd = self.target_layer.register_forward_hook(self.save_activation)
        self._hook_bwd = self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        """Explicitly remove hooks. Call when done to free CUDA memory."""
        self._hook_fwd.remove()
        self._hook_bwd.remove()

    def __del__(self):
        """Auto-remove hooks when object goes out of scope."""
        try:
            self.remove_hooks()
        except Exception:
            pass
        
    def __call__(self, x_tensor, original_image_shape):
        """
        x_tensor: Model input (1, C, H, W)
        original_image_shape: tuple of (H, W) to resize the heatmap back
        """
        self.model.zero_grad()
        outputs = self.model(x_tensor)
        
        score = outputs['regression_score'][0]
        score.backward()
        
        # Pool the gradients across the spatial dimensions
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        activations = self.activations.detach()[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        heatmap = cv2.resize(heatmap, (original_image_shape[1], original_image_shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap

def apply_heatmap(original_bgr_img, heatmap_bgr, alpha=0.5):
    """ Overlays the GradCAM heatmap on the original image. """
    superimposed_img = cv2.addWeighted(heatmap_bgr, alpha, original_bgr_img, 1 - alpha, 0)
    return superimposed_img
