import os
import torch
import cv2
import pandas as pd
from tqdm import tqdm

from src.models.sota_dr_model import SOTA_DR_Model
from src.evaluation.ensemble import SoftVotingEnsemble
from src.evaluation.optimizer import OptimizedRounder
from src.evaluation.gradcam import GradCAMHooker, apply_heatmap
from src.data.dataset import get_validation_augmentations
import numpy as np

class FinalInferencer:
    """
    Ties together the complete inference architecture specified in the SOTA document.
    """
    def __init__(self, model_paths, target_resolution=768, temperature=1.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_resolution = target_resolution
        self.temperature = temperature
        
        # Load all models into the 10-model ensemble
        models = []
        for path in model_paths:
            # Infer architecture from filename
            arch = 'convnextv2_large' if 'convnext' in path.lower() else 'efficientnetv2_l'
            m = SOTA_DR_Model(model_name=arch, pretrained=False)
            
            # Note: During training we saved the EMA shadows, so we load those 
            # for maximum generalization stability
            state_dict = torch.load(path, map_location='cpu')
            m.load_state_dict(state_dict)
            m.to(self.device).eval()
            models.append(m)
            
        # Initialize the Soft Voting predictor
        # For simplicity in this script we assume equal weighting if validation QWKs 
        # aren't explicitly passed, but the class supports it.
        self.ensemble = SoftVotingEnsemble(models)
        
        # Initialize Albumentations only (no EyePACSPreprocessor needed for preprocessed images)
        self.alb_transform = get_validation_augmentations(target_resolution)
        
        # Initialize Threshold Optimizer (must be fitted on validation data before use)
        self.optimizer = OptimizedRounder()

    def process_image(self, image_path):
        """ Runs pure inference pipeline on a single preprocessed image file. """
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            img_np = np.zeros((self.target_resolution, self.target_resolution, 3), dtype=np.uint8)
            img_np[:] = 128
        else:
            img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Albumentations to Tensor
        augmented = self.alb_transform(image=img_np)
        img_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Soft Voting + TTA + Temperature Scaling
        with torch.no_grad():
            preds = self.ensemble.forward(img_tensor, temperature=self.temperature)
        
        return preds['ensemble_regression'].item(), img_tensor

    def fit_thresholds(self, df_val, img_dir):
        """ Fits the Bayesian optimizer using a validation set. """
        print("Fitting Threshold Optimizer on validation set...")
        all_preds = []
        all_trues = []
        
        for _, row in tqdm(df_val.iterrows(), total=len(df_val)):
            img_id = str(row['image_id']).split('.')[0]
            img_path = os.path.join(img_dir, img_id + '.jpg')
            if not os.path.exists(img_path):
                 continue
            reg_score, _ = self.process_image(img_path)
            all_preds.append(reg_score)
            all_trues.append(row['level'])
            
        # Fit optimal thresholds
        self.optimizer.fit(all_preds, all_trues)
        print("Thresholds fitted and saved internally.")
        
    def predict(self, image_path):
        """ Full prediction returning the final 0-4 Optimized Grade. """
        reg_score, tensor = self.process_image(image_path)
        # Apply optimal boundaries
        final_grade = self.optimizer.predict([reg_score])[0]
        return final_grade
        
    def generate_clinical_report(self, image_path, save_heatmap_path=None):
        """ 
        Runs inference and uses the first model in the ensemble to generate
        a GradCAM heatmap for clinical interpretability.
        """
        reg_score, tensor = self.process_image(image_path)
        grade = self.optimizer.predict([reg_score])[0]
        
        # Hook into first model for visualization
        hooker = GradCAMHooker(self.ensemble.models[0].model)
        
        # Load image directly using OpenCV (no preprocessor)
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None
        h, w = (self.target_resolution, self.target_resolution)
        if img_np is not None:
            h, w = img_np.shape[:2]
        
        # Compute heatmap
        heatmap_bgr = hooker(tensor, (h, w))
        
        # Overlay — use img_np loaded from disk above
        if img_np is not None:
            orig_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            orig_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        overlay = apply_heatmap(orig_bgr, heatmap_bgr, alpha=0.5)
        
        if save_heatmap_path:
             cv2.imwrite(save_heatmap_path, overlay)
             
        # Generate diagnostic text mapping
        diagnoses = {
            0: "No DR detected",
            1: "Mild NPDR - Annual screening recommended",
            2: "Moderate NPDR - Refer to ophthalmologist",
            3: "Severe NPDR - Urgent referral needed",
            4: "PDR - Immediate treatment required"
        }
             
        return {
             'score': reg_score,
             'grade': grade,
             'diagnosis': diagnoses[grade],
             'heatmap_path': save_heatmap_path
        }
