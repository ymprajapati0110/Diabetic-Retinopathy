import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch
from torch.utils.data import DataLoader

from src.models.sota_dr_model import SOTA_DR_Model
from src.evaluation.ensemble import SoftVotingEnsemble
from src.evaluation.optimizer import OptimizedRounder
from test_with_ensemble import PerfectTestDataset

# ===== CONFIG =====
MODEL_PATHS = [
    "d:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_25_ema.pth",
    "d:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_24_ema.pth",
    "d:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_23_ema.pth",
]

IMG_DIR = "d:/Major_project_shiro_final/preprocessed_1024"
CSV_PATH = "d:/Major_project_shiro_final/trainLabels.csv"
FOLD = 3
# ==================

def main():
    print("Loading Validation Dataset...")
    df = pd.read_csv(CSV_PATH)
    
    if 'fold' not in df.columns:
        from src.data.data_split import setup_cross_validation
        df = setup_cross_validation(df, n_splits=5)
    
    df_val = df[df['fold'] == FOLD].reset_index(drop=True)

    print("Initializing Multi-checkpoint Ensemble WITHOUT TTA...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    
    for p in MODEL_PATHS:
        m = SOTA_DR_Model('convnextv2_large', pretrained=False)
        checkpoint = torch.load(p, map_location=device)
        m.load_state_dict(checkpoint.get('ema_state_dict', checkpoint.get('model_state_dict', checkpoint)))
        m.to(device)
        m.eval()
        models.append(m)

    ensemble = SoftVotingEnsemble(models, use_tta=False)
    
    valid_paths = []
    valid_trues = []
    
    for _, row in df_val.iterrows():
        img_id = str(row['image']).split('.')[0]
        img_path = os.path.join(IMG_DIR, img_id + ".jpg")
        if os.path.exists(img_path):
            valid_paths.append(img_path)
            valid_trues.append(row['level'])
            
    print(f"Running highly optimized BATCH inference on {len(valid_paths)} images...")
    dataset = PerfectTestDataset(valid_paths, target_size=1024)
    # Reduced workers to 0 since we run on Windows, avoiding potential spawn deadlocks
    loader = DataLoader(dataset, batch_size=6, num_workers=0, pin_memory=True, shuffle=False)
    
    all_preds = []
    all_trues_aligned = []
    
    with torch.no_grad():
        for batch_idx, (batch_ids, batch_tensors, batch_valid_flags) in enumerate(tqdm(loader, miniters=10)):
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=True):
                preds = ensemble.forward(batch_tensors, temperature=1.5)
                
            reg_scores = preds['ensemble_regression'].cpu().numpy()
            flags = batch_valid_flags.numpy()
            
            for i in range(len(reg_scores)):
                if flags[i]:
                    # Collect pure raw scores first
                    all_preds.append(reg_scores[i])
                    
                    global_idx = batch_idx * 6 + i
                    all_trues_aligned.append(valid_trues[global_idx])

    print("\nFitting NELDER-MEAD Bayesian Optimizer (Maximizing QWK)...")
    all_preds_raw = np.array(all_preds)
    all_trues = np.array(all_trues_aligned)
    
    # 1. Compute Raw Stats
    raw_mean = np.mean(all_preds_raw)
    raw_std = np.std(all_preds_raw)
    
    # 2. Normalize ONLY (NO SHIFT)
    all_preds_scaled = (all_preds_raw - raw_mean) / raw_std
    
    # 3. Compute Scaled Stats
    calib_mean = np.mean(all_preds_scaled)
    calib_std = np.std(all_preds_scaled)
    
    print("\n" + "-"*30)
    print("DISTRIBUTION STATISTICS:")
    print(f"RAW Mean: {raw_mean:.4f}")
    print(f"RAW Std:  {raw_std:.4f}")
    print(f"SCALED Mean: {calib_mean:.4f}")
    print(f"SCALED Std:  {calib_std:.4f}")
    print("-" * 30 + "\n")
    
    # Baseline comparison (on raw unshifted outputs)
    baseline_preds = np.clip(np.round(all_preds_raw), 0, 4)
    from sklearn.metrics import cohen_kappa_score
    baseline_qwk = cohen_kappa_score(all_trues, baseline_preds, weights='quadratic')
    print(f"[BEFORE OPTIMIZATION] Hard Rounding QWK: {baseline_qwk:.4f}")
    
    optimizer = OptimizedRounder()
    coefs = optimizer.fit(all_preds_scaled, all_trues)
    
    print(f"\n[FINAL TUNED THRESHOLDS]: {list(coefs)}")
    
    # Predict to see validation QWK
    preds_rounded = optimizer.predict(all_preds_scaled, coefs)
    val_qwk = cohen_kappa_score(all_trues, preds_rounded, weights='quadratic')
    print(f"[AFTER OPTIMIZATION] QWK ON THESE THRESHOLDS: {val_qwk:.4f}")
    print(f"[GAIN]: +{(val_qwk - baseline_qwk):.4f}")
    
    print("\n" + "="*50)
    print("COPY AND VERIFY THESE COEFFICIENTS INTO test_with_ensemble.py:")
    print(f"self.optimizer.coef_ = {list(coefs)}")
    print("="*50 + "\n")
    
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    main()
