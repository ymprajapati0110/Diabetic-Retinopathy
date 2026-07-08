import cv2
import numpy as np

def validate_image_quality(img):
    """
    Strong filtering: Removes stitched images, partial retina, and black frames.
    """
    if img is None:
        return False, "Failed to decode"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Too dark
    if gray.mean() < 15:
        return False, f"Too dark (Mean={gray.mean():.2f})"
        
    # Too low contrast
    if gray.std() < 8:
        return False, f"Low contrast (Std={gray.std():.2f})"
        
    # Check retina area
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return False, "No contours found"
        
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (img.shape[0]*img.shape[1])
    
    # Reject broken/partial retina
    if area_ratio < 0.3:
        return False, f"Partial retina (Area Ratio={area_ratio:.2f})"
        
    return True, "Valid"

def preprocess_image(img_path, output_size=512):
    """
    Minimal Preprocessing: 
    1. Resize.
    2. Light Y-Channel CLAHE (Optional, clipLimit=1.5).
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # Resize to target
    img = cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
    
    # LIGHT CLAHE (Contrast without color shift)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)) # Light CLAHE
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img

class EyePACSPreprocessor:
    """Standard interface for dataset usage."""
    def __init__(self, target_size=512):
        self.target_size = target_size
        
    def __call__(self, img_path):
        img = preprocess_image(img_path, output_size=self.target_size)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
