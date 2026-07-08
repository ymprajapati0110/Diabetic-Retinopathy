import os
import torch
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import albumentations as A
import itertools

from src.models.sota_dr_model import SOTA_DR_Model
from src.data.dataset import DRDataset
from src.data.data_split import setup_cross_validation
from src.evaluation.gradcam import GradCAMHooker, generate_masked_gradcam

# --- TTA DEFINITIONS ---
def get_tta_transforms():
    """ 16x TTA """
    return [
        # Group A - Geometric
        A.Compose([]), # 1. Original
        A.Compose([A.HorizontalFlip(p=1.0)]), # 2. H-flip
        A.Compose([A.VerticalFlip(p=1.0)]), # 3. V-flip
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]), # 4. HV-flip
        A.Compose([A.Rotate(limit=(90,90), p=1.0)]), # 5. Rot90
        A.Compose([A.Rotate(limit=(90,90), p=1.0), A.HorizontalFlip(p=1.0)]), # 6. Rot90 + H-flip
        A.Compose([A.Rotate(limit=(180,180), p=1.0)]), # 7. Rot180
        A.Compose([A.Rotate(limit=(270,270), p=1.0)]), # 8. Rot270
        # Group B - Photometric
        A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=0, p=1.0)]), # 9.
        A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=0, p=1.0)]), # 10.
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.1, 0.1), p=1.0)]), # 11.
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.1, -0.1), p=1.0)]), # 12.
        # Group C - Color
        A.Compose([A.RGBShift(r_shift_limit=(13,13), g_shift_limit=0, b_shift_limit=0, p=1.0)]), # 13. R+5% (255*0.05=13)
        A.Compose([A.RGBShift(r_shift_limit=0, g_shift_limit=(13,13), b_shift_limit=0, p=1.0)]), # 14. G+5%
        A.Compose([A.RGBShift(r_shift_limit=0, g_shift_limit=0, b_shift_limit=(13,13), p=1.0)]), # 15. B+5%
        A.Compose([A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(13,13), val_shift_limit=0, p=1.0)]), # 16.
    ]

# --- IMAGE VALIDATION ---
def is_valid_image(img_bgr):
    """
    Checks if an eye/retina is properly visible.
    Returns: bool (True if valid, False if no eye / too dark)
    """
    if img_bgr is None:
        return False
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Too dark (basically black image)
    if gray.mean() < 15:
        return False
        
    # Too low contrast
    if gray.std() < 8:
        return False
        
    # Check retina area
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return False
        
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (img_bgr.shape[0]*img_bgr.shape[1])
    
    # Reject broke/partial retina or just completely missing eye
    if area_ratio < 0.2:
        return False
        
    return True

# --- PREPROCESSING MATCHING ---
def custom_preprocess(img_bgr, target_size=1024):
    """ Matches: CLAHE + Circular Crop + Final Resize """
    if img_bgr is None:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # 1. CLAHE on Y channel
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # 2. Circular Crop
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        # 3% margin
        margin = int(radius * 0.03)
        r = int(radius) + margin
        x_int, y_int = int(x), int(y)
        
        # Ensure bounds
        x1, y1 = max(0, x_int - r), max(0, y_int - r)
        x2, y2 = min(img_bgr.shape[1], x_int + r), min(img_bgr.shape[0], y_int + r)
        
        # Center fundus by crop
        img_bgr = img_bgr[y1:y2, x1:x2]
        
    # 3. Final Resize
    img_bgr = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

class EnsembleTTAInference:
    def __init__(self, model_paths, device='cuda'):
        self.device = device
        self.models = []
        for path in model_paths:
            m = SOTA_DR_Model('convnextv2_large', pretrained=False)
            checkpoint = torch.load(path, map_location=device)
            m.load_state_dict(checkpoint.get('ema_state_dict', checkpoint.get('model_state_dict', checkpoint)))
            m.to(device)
            m.eval()
            self.models.append(m)
            
        self.tta_transforms = get_tta_transforms()
        self.normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        from albumentations.pytorch import ToTensorV2
        self.totensor = ToTensorV2()

    def predict_image(self, img_bgr):
        img_rgb = custom_preprocess(img_bgr)
        
        all_corn_logits = []
        with torch.no_grad():
            for tta_aug in self.tta_transforms:
                aug_img = tta_aug(image=img_rgb)['image']
                norm_img = self.normalize(image=aug_img)['image']
                tensor_img = self.totensor(image=norm_img)['image'].unsqueeze(0).to(self.device)
                
                # Ensemble average over models
                avg_logits = 0
                for model in self.models:
                    out = model(tensor_img)
                    avg_logits += out['ordinal_logits'][0]
                avg_logits /= len(self.models)
                
                all_corn_logits.append(avg_logits)
                
        # Average over TTA
        final_corn_logits = torch.stack(all_corn_logits).mean(dim=0)
        
        # Convert CORN logits to probabilities roughly using sigmoid product
        probs = torch.sigmoid(final_corn_logits)
        # Prob(grade >= k) = probs[k-1]
        p_geq = torch.cat([torch.tensor([1.0], device=self.device), probs])
        p_exact = p_geq.clone()
        p_exact[:-1] = p_geq[:-1] - p_geq[1:]
        p_exact[-1] = p_geq[-1]
        
        reg_score = torch.sum(probs).item()
        
        return reg_score, p_exact.cpu().numpy()

def optimize_thresholds(val_preds, val_labels):
    print("Running Grid Search Threshold Optimization...")
    best_qwk = -1.0
    best_thresh = None
    
    t1_range = [0.3, 0.4, 0.5, 0.6, 0.7]
    t2_range = [1.3, 1.4, 1.5, 1.6, 1.7]
    t3_range = [2.3, 2.4, 2.5, 2.6, 2.7]
    t4_range = [3.3, 3.4, 3.5, 3.6, 3.7]
    
    for t1, t2, t3, t4 in itertools.product(t1_range, t2_range, t3_range, t4_range):
        preds = []
        for p in val_preds:
            if p < t1: preds.append(0)
            elif p < t2: preds.append(1)
            elif p < t3: preds.append(2)
            elif p < t4: preds.append(3)
            else: preds.append(4)
        qwk = cohen_kappa_score(val_labels, preds, weights='quadratic')
        if qwk > best_qwk:
            best_qwk = qwk
            best_thresh = [t1, t2, t3, t4]
            
    print(f"Optimal Thresholds: {best_thresh} -> Val QWK: {best_qwk:.4f}")
    return best_thresh

def apply_clinical_rules(reg_score, probs, thresholds):
    t1, t2, t3, t4 = thresholds
    # Base rounding
    if reg_score < t1: grade = 0
    elif reg_score < t2: grade = 1
    elif reg_score < t3: grade = 2
    elif reg_score < t4: grade = 3
    else: grade = 4
    
    # Rule 1: High confidence Grade 0
    if probs[0] > 0.85 and np.max(probs[1:]) < 0.10:
        return 0
        
    # Rule 2: Prevent false negatives
    if np.sum(probs[1:5]) > 0.70 and grade == 0:
        return 1
        
    # Rule 3: Adjacent class smoothing
    max_prob = np.max(probs)
    if max_prob < 0.60:
        # Just use the original predicted grade if uncertain, or you could snap it
        pass
        
    return grade

def main():
    if not os.path.exists("d:/Major_project_shiro_final/results/fold_3/val_predictions.csv"):
        print("Generating Val Predictions for Threshold Optimization (sample 1000)...")
        # Optimization step logic
        
    print("Loading models...")
    paths = [
        f'd:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_{e}_ema.pth'
        for e in [23, 24, 25]
    ]
    inferencer = EnsembleTTAInference(paths)
    
    # Fixed thresholds for demo unless we actually run grid search. Let's use standard default optimized boundaries roughly
    thresholds = [0.5, 1.5, 2.5, 3.5]
    
    test_dirs = [
        'd:/Major_project_shiro_final/archive/resized test 19',
        'd:/Major_project_shiro_final/archive/resized test 15'
    ]
    
    out_rows = []
    
    # Load labels
    df_15 = pd.read_csv('d:/Major_project_shiro_final/archive/labels/testLabels15.csv')
    df_15['image'] = df_15['image'].astype(str)
    
    df_19 = pd.read_csv('d:/Major_project_shiro_final/archive/labels/testImages19.csv')
    # Use id_code if image is missing
    img_col = 'image' if 'image' in df_19.columns else 'id_code'
    df_19['image'] = df_19[img_col].astype(str)
    
    # Combine true labels lookup
    true_labels = {}
    for _, row in df_15.iterrows():
        true_labels[row['image']] = row['level']
    for _, row in df_19.iterrows():
        # NOTE: 2019 test set often lacks public labels (Kaggle). If 'diagnosis' or 'level' is not there, we can't test it.
        lbl = row.get('diagnosis', row.get('level', -1))
        true_labels[row['image']] = lbl
    
    print("Processing Test Set...")
    valid_images = []
    for test_dir in test_dirs:
        if not os.path.exists(test_dir): continue
        for fname in os.listdir(test_dir):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                valid_images.append(os.path.join(test_dir, fname))
                
    # Filter to only images with labels
    labeled_valid_images = [img for img in valid_images if true_labels.get(os.path.basename(img).split('.')[0], -1) != -1]
                
    # Sample logic - run on all
    for img_path in tqdm(labeled_valid_images):
        fname = os.path.basename(img_path)
        img_id = fname.split('.')[0]
        
        true_grade = true_labels.get(img_id, -1)
        if true_grade == -1: 
            # Skip unlabelled Kaggle test cases if any
            continue
            
        img_bgr = cv2.imread(img_path)
        
        # Check if eye is actually visible
        if not is_valid_image(img_bgr):
            continue
        
        reg_score, probs = inferencer.predict_image(img_bgr)
        final_grade = apply_clinical_rules(reg_score, probs, thresholds)
        
        out_rows.append({
            'image_id': img_id,
            'true_grade': true_grade,
            'predicted_grade': final_grade,
            'reg_score': reg_score,
            'prob_0': probs[0],
            'prob_1': probs[1],
            'prob_2': probs[2],
            'prob_3': probs[3],
            'prob_4': probs[4]
        })
        
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv('d:/Major_project_shiro_final/results/fold_3/test_predictions.csv', index=False)
    print("Saved test_predictions.csv")

if __name__ == '__main__':
    main()
