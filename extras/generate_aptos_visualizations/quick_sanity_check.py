import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sota_dr_model import SOTA_DR_Model
from src.evaluation.ensemble import SoftVotingEnsemble
from test_with_ensemble import PerfectTestDataset

MODEL_PATHS = [
    "d:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_25_ema.pth",
    "d:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_24_ema.pth",
    "d:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_23_ema.pth",
]

IMG_DIR = "d:/Major_project_shiro_final/preprocessed_1024"
CSV_PATH = "d:/Major_project_shiro_final/trainLabels.csv"

def main():
    print("Loading test split...")
    df = pd.read_csv(CSV_PATH)
    if 'fold' not in df.columns:
        from src.data.data_split import setup_cross_validation
        df = setup_cross_validation(df, n_splits=5)
        
    df_test = df[df['fold'] != 3].reset_index(drop=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    print("Loading models...")
    for p in MODEL_PATHS:
        m = SOTA_DR_Model('convnextv2_large', pretrained=False)
        chkpt = torch.load(p, map_location=device)
        m.load_state_dict(chkpt.get('ema_state_dict', chkpt.get('model_state_dict', chkpt)))
        m.to(device)
        m.eval()
        models.append(m)

    # use_tta=True to match final test conditions
    ensemble = SoftVotingEnsemble(models, use_tta=True)
    
    sample_paths = []
    for _, row in df_test.iterrows():
        img_id = str(row['image']).split('.')[0]
        img_path = os.path.join(IMG_DIR, img_id + ".jpg")
        if os.path.exists(img_path):
            sample_paths.append(img_path)
            if len(sample_paths) >= 200:
                break
                
    dataset = PerfectTestDataset(sample_paths, target_size=1024)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=True, shuffle=False)
    
    sample_preds = []
    print("Running Lightning Fast 200-Image Sanity Check...")
    with torch.no_grad():
        for batch_i, (batch_ids, batch_tensors, valid_flags) in enumerate(tqdm(loader)):
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=True):
                preds = ensemble.forward(batch_tensors, temperature=1.5)
                
            reg_scores = preds['ensemble_regression'].cpu().numpy()
            for b_i in range(len(reg_scores)):
                if valid_flags[b_i]:
                    score = (reg_scores[b_i] - 0.5737) / 0.8560
                    sample_preds.append(score)

    print("\n" + "="*50)
    print("SANITY CHECK RESULTS:")
    print(f"Mean: {np.mean(sample_preds):.4f}  (Expected ≈ 1.7 - 2.0)")
    print(f"Std:  {np.std(sample_preds):.4f}  (Expected ≈ 0.9 - 1.2)")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
