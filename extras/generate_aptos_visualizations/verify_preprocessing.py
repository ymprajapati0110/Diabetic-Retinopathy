import os
import pandas as pd
import torch
import cv2
import numpy as np

from src.models.sota_dr_model import SOTA_DR_Model
from src.data.data_split import setup_cross_validation
from run_inference import custom_preprocess
from src.data.dataset import get_validation_augmentations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    print("Loading df...")
    df = pd.read_csv('d:/Major_project_shiro_final/trainLabels.csv')
    df = setup_cross_validation(df, n_splits=5)
    df_val = df[df['fold'] == 3].reset_index(drop=True).head(10)

    # We will load the Epoch 25 model for validation testing
    model = SOTA_DR_Model('convnextv2_large', pretrained=False)
    state = torch.load('d:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_25_ema.pth', map_location='cpu')
    model.load_state_dict(state.get('ema_state_dict', state.get('model_state_dict', state)))
    model.to(device).eval()

    alb = get_validation_augmentations(1024)

    print("Running verification on 10 validation samples...")
    for idx, row in df_val.iterrows():
        img_id = str(row['image']).split('.')[0]
        # In training, we used preprocessed_512 up to 768. But for final, we use 1024.
        # Let's see how our test-time custom_preprocess performs on the raw image vs valid preprocessed.
        
        # Test time path: would be raw image
        # Because we don't have the raw image dir readily for validation (unless it's in archive), 
        # let's just use custom_preprocess if available on the original image, else skip.
        raw_path = f'd:/Major_project_shiro_final/archive/resized train 15/{img_id}.jpeg'
        if not os.path.exists(raw_path):
            raw_path = f'd:/Major_project_shiro_final/archive/resized train 19/{img_id}.jpg'
            
        if not os.path.exists(raw_path):
            print(f"Skipping {img_id}: Raw image not found.")
            continue
            
        img_rgb = custom_preprocess(raw_path, target_size=1024)
        if img_rgb is None:
            print(f"{img_id} INVALID")
            continue
            
        tensor = alb(image=img_rgb)['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(tensor)
            reg_score = out['regression_score'].item()
            
        print(f"ID: {img_id} | True: {row['level']} | Model Regression: {reg_score:.3f}")

if __name__ == '__main__':
    main()
