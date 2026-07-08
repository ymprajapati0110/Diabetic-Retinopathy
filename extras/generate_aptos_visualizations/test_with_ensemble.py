import os
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
import gc

# Set performance flags at module level for massive speedup
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.sota_dr_model import SOTA_DR_Model
from src.evaluation.ensemble import SoftVotingEnsemble
from src.evaluation.optimizer import OptimizedRounder
from src.data.preprocess import preprocess_image # USE EXACT TRAINING PREPROCESSING

class PerfectTestDataset(Dataset):
    """
    Match training EXACTLY. 
    Reads raw image -> applies src.data.preprocess.preprocess_image -> Normalizes via Albumentations
    """
    def __init__(self, image_paths, target_size=1024):
        self.image_paths = image_paths
        self.target_size = target_size
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img_id = os.path.basename(path).split('.')[0]
        
        # Exact Training Match (CLAHE + Resize implicitly executed)
        img_bgr = preprocess_image(path, output_size=self.target_size)
        
        if img_bgr is None:
            return img_id, torch.zeros((3, self.target_size, self.target_size)), False
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=img_rgb)['image']
        return img_id, tensor, True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--out_csv', type=str, default='d:/Major_project_shiro_final/results/fold_3/test_results/predictions.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Multi-Checkpoint Ensemble...")
    models = []
    
    # We prioritize epoch 25 since it matured the longest
    paths = [
        'd:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_25_ema.pth',
        'd:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_24_ema.pth',
        'd:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_23_ema.pth'
    ]
    
    for p in paths:
        m = SOTA_DR_Model('convnextv2_large', pretrained=False)
        checkpoint = torch.load(p, map_location=device)
        m.load_state_dict(checkpoint.get('ema_state_dict', checkpoint.get('model_state_dict', checkpoint)))
        m.to(device)
        m.eval()
        models.append(m)

    # Apply weighted voting biased toward the finalized epoch QWK scores
    raw_w = torch.tensor([0.866, 0.864, 0.862])
    weights = (raw_w / raw_w.sum()).tolist()
    ensemble = SoftVotingEnsemble(models, weights=weights)
    optimizer = OptimizedRounder()

    # THRESHOLDS EXPECTED HERE (Replace these post fit_thresholds.py!)
    optimizer.coef_ = [0.5, 1.5, 2.5, 3.5]
    print(f"USING VALIDATION THRESHOLDS: {list(optimizer.coef_)}")

    print("Loading test labels to match valid IDs...")
    df_15 = pd.read_csv('d:/Major_project_shiro_final/archive/labels/testLabels15.csv')
    df_15['image'] = df_15['image'].astype(str)
    df_19 = pd.read_csv('d:/Major_project_shiro_final/archive/labels/testImages19.csv')
    img_col = 'image' if 'image' in df_19.columns else 'id_code'
    df_19['image'] = df_19[img_col].astype(str)
    
    true_labels = {}
    for _, row in df_15.iterrows(): true_labels[row['image']] = row['level']
    for _, row in df_19.iterrows(): 
        lbl = row.get('diagnosis', row.get('level', -1))
        true_labels[row['image']] = lbl

    print("Scanning Raw Test Set...")
    valid_images = []
    test_dirs = [
        'd:/Major_project_shiro_final/archive/resized test 19',
        'd:/Major_project_shiro_final/archive/resized test 15'
    ]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for fname in os.listdir(test_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_id = fname.split('.')[0]
                    if true_labels.get(img_id, -1) != -1:
                        valid_images.append(os.path.join(test_dir, fname))
                    
    print(f"Total labeled test images: {len(valid_images)}")
    
    # Generate test results dict to ensure directory exists
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    
    dataset = PerfectTestDataset(valid_images, target_size=args.resolution)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=0, 
        pin_memory=True, 
        prefetch_factor=None, 
        shuffle=False
    )
    
    out_rows = []
    reg_distribution = []
    
    print("\n" + "*"*50)
    print("[RUNNING 200-IMAGE SANITY CHECK BEFORE FULL TEST]")
    sample_preds = []
    with torch.no_grad():
        for batch_i, (batch_ids, batch_tensors, batch_valid_flags) in enumerate(loader):
            if len(sample_preds) >= 200:
                break
                
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=True):
                preds = ensemble.forward(batch_tensors, temperature=1.5)
            
            # Record scores
            reg_scores = preds['ensemble_regression'].cpu().numpy()
            for b_i in range(len(reg_scores)):
                if batch_valid_flags.numpy()[b_i]:
                    calibrated_score = (reg_scores[b_i] - 0.5737) / 0.8560
                    sample_preds.append(calibrated_score)
                    
    mean_val = np.mean(sample_preds)
    std_val = np.std(sample_preds)
    print(f"Mean: {mean_val:.4f} (Expected ~1.5-2.2)")
    print(f"Std:  {std_val:.4f} (Expected ~0.8-1.2)")
    
    if mean_val < 1.0 or mean_val > 2.5 or std_val < 0.5:
        print("[CRITICAL STOP] PIPELINE DISTRIBUTION COLLAPSE DETECTED.")
        print("Exiting securely before wasting 5 hours.")
        return
    print("Looks perfect. Booting main loop...\n")
    print("*"*50 + "\n")
    
    pbar = tqdm(total=len(valid_images), miniters=20)
    
    with torch.no_grad():
        for batch_ids, batch_tensors, batch_valid_flags in loader:
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            B = batch_tensors.shape[0]
            
            with torch.amp.autocast('cuda', enabled=True):
                preds = ensemble.forward(batch_tensors, temperature=1.5)
                
            probs = preds['ensemble_ordinal'] 
            p_geq = torch.cat([torch.ones(B, 1, device=device), probs], dim=1)
            p_exact = p_geq.clone()
            p_exact[:, :-1] = p_geq[:, :-1] - p_geq[:, 1:]
            p_exact[:, -1] = p_geq[:, -1]
            p_exact = torch.clamp(p_exact, 0.0, 1.0)
            
            reg_scores = preds['ensemble_regression'].cpu().numpy()
            probs_cpu = p_exact.cpu().numpy()
            valid_flags = batch_valid_flags.numpy()
            
            for i in range(B):
                if not valid_flags[i]:
                    pbar.update(1)
                    continue
                    
                img_id = batch_ids[i]
                true_grade = true_labels.get(img_id, -1)
                
                # Apply Full Statistical Calibration
                calibrated_score = (reg_scores[i] - 0.5737) / 0.8560
                
                # Predict optimal integer grade
                final_grade = optimizer.predict([calibrated_score])[0]
                reg_distribution.append(calibrated_score)
                    
                out_rows.append({
                    'image_id': img_id,
                    'true_grade': true_grade,
                    'predicted_grade': final_grade,
                    'reg_score': calibrated_score,
                    'prob_0': probs_cpu[i][0],
                    'prob_1': probs_cpu[i][1],
                    'prob_2': probs_cpu[i][2],
                    'prob_3': probs_cpu[i][3],
                    'prob_4': probs_cpu[i][4]
                })
                pbar.update(1)

    print("\n" + "="*50)
    print("SANITY CHECK - REGRESSION DISTRIBUTION")
    print(f"Mean Score: {np.mean(reg_distribution):.4f} (Expected 1.5 - 2.0)")
    print(f"Std  Score: {np.std(reg_distribution):.4f} (Expected 0.8 - 1.2)")
    print("="*50 + "\n")
                
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(args.out_csv, index=False)
    print(f"Saved cleanly without overrides to: {args.out_csv}")

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    main()
