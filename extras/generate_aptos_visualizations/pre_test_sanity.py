import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sota_dr_model import SOTA_DR_Model
from src.data.dataset import DRDataset, get_validation_augmentations
from src.data.data_split import setup_cross_validation

def run_sanity_check():
    print("Running Pre-test Sanity Check on Fold 3 Validation Set (Sample)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SOTA_DR_Model('convnextv2_large', pretrained=False)
    weights_path = 'd:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_25_ema.pth'
    if not os.path.exists(weights_path):
        weights_path = 'd:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_25.pth'

    checkpoint = torch.load(weights_path, map_location=device)
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Load validation data
    csv_path = 'd:/Major_project_shiro_final/trainLabels.csv'
    df = pd.read_csv(csv_path)
    df = setup_cross_validation(df, n_splits=5)
    
    val_df = df[df['fold'] == 3].copy()
    
    # To save time and just verify a batch
    val_df_sample = val_df.sample(n=min(len(val_df), 200), random_state=42)
    
    # Initialize Dataset
    img_dir = 'd:/Major_project_shiro_final/preprocessed_1024'
    val_dataset = DRDataset(val_df_sample, img_dir, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, prefetch_factor=2)
    
    all_preds_reg = []
    all_labels = []
    
    with torch.no_grad():
        for batch_images, batch_targets in tqdm(val_loader):
            batch_images = batch_images.to(device)
            labels = batch_targets['ordinal'].numpy()
            all_labels.extend(labels)
            
            outputs = model(batch_images)
            preds_reg = outputs['regression_score'].detach().cpu().numpy()
            all_preds_reg.extend(preds_reg)
            
    # Naive rounding just for sanity bounds
    preds_rounded = np.clip(np.round(all_preds_reg), 0, 4).astype(int)
    val_qwk = cohen_kappa_score(all_labels, preds_rounded, weights='quadratic')
    
    print(f"Validation Sample QWK (naive rounding): {val_qwk:.4f}")
    if val_qwk > 0.84:
        print("SANITY CHECK PASSED: Model is outputting high-quality predictions as expected.")
    else:
        print("WARNING: QWK lower than expected. However note this is naive rounding.")
    
if __name__ == '__main__':
    run_sanity_check()
