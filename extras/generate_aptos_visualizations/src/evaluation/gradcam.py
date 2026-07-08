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

def generate_masked_gradcam(hooker, x_tensor, original_bgr_img):
    """
    Mask-Based GradCAM logic to fix border artifacts.
    Generates a GradCAM heatmap, and masks out activations outside the fundus area.
    """
    hooker.model.zero_grad()
    outputs = hooker.model(x_tensor)
    
    score = outputs['regression_score'][0]
    score.backward()
    
    pooled_gradients = torch.mean(hooker.gradients, dim=[0, 2, 3])
    activations = hooker.activations.detach()[0]
    
    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(activations, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    
    # Create mask from original_bgr_img
    gray = cv2.cvtColor(original_bgr_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Resize mask to GradCAM resolution
    mask_resized = cv2.resize(mask, (heatmap.shape[1], heatmap.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask_resized > 127).astype(np.float32)
    
    # Apply Mask
    heatmap *= mask_binary
    
    # Renormalize
    heatmap_max = np.max(heatmap)
    if heatmap_max > 0:
        heatmap /= heatmap_max + 1e-8
        
    # Resize to original
    heatmap = cv2.resize(heatmap, (original_bgr_img.shape[1], original_bgr_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_bgr = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap_bgr

