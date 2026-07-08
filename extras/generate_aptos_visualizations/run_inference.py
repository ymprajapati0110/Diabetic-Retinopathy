import os
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.sota_dr_model import SOTA_DR_Model
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- IMAGE VALIDATION ---
def is_valid_image(img_bgr):
    """
    Checks if an eye/retina is properly visible.
    """
    if img_bgr is None: return False
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 15: return False
    if gray.std() < 8: return False
    
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return False
        
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (img_bgr.shape[0]*img_bgr.shape[1])
    if area_ratio < 0.2: return False
        
    return True

# --- PREPROCESSING MATCHING ---
def custom_preprocess(img_path, target_size=1024):
    img_bgr = cv2.imread(img_path)
    if not is_valid_image(img_bgr):
        return None
        
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
        margin = int(radius * 0.03)
        r = int(radius) + margin
        x_int, y_int = int(x), int(y)
        x1, y1 = max(0, x_int - r), max(0, y_int - r)
        x2, y2 = min(img_bgr.shape[1], x_int + r), min(img_bgr.shape[0], y_int + r)
        img_bgr = img_bgr[y1:y2, x1:x2]
        
    # 3. Final Resize
    img_bgr = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

class FastTestDataset(Dataset):
    def __init__(self, img_paths, target_size=1024):
        self.img_paths = img_paths
        self.target_size = target_size
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img_id = os.path.basename(path).split('.')[0]
        
        img_rgb = custom_preprocess(path, self.target_size)
        if img_rgb is None:
            # Return dummy but valid tensor to keep batch size stable, flag as invalid
            dummy = torch.zeros(3, self.target_size, self.target_size)
            return img_id, dummy, False
            
        tensor = self.transform(image=img_rgb)['image']
        return img_id, tensor, True

def apply_clinical_rules(reg_score, probs):
    thresholds = [0.5, 1.5, 2.5, 3.5]
    t1, t2, t3, t4 = thresholds
    if reg_score < t1: grade = 0
    elif reg_score < t2: grade = 1
    elif reg_score < t3: grade = 2
    elif reg_score < t4: grade = 3
    else: grade = 4
    
    if probs[0] > 0.85 and np.max(probs[1:]) < 0.10:
        return 0
    if np.sum(probs[1:5]) > 0.70 and grade == 0:
        return 1
    return grade

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--tta', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--data_csv', type=str, default='d:/Major_project_shiro_final/hospital_1/data.csv')
    parser.add_argument('--data_dir', type=str, default='d:/Major_project_shiro_final/hospital_1/data')
    parser.add_argument('--out_csv', type=str, default='d:/Major_project_shiro_final/results/fold_3/test_predictions_fast.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading models...")
    models = []
    paths = [
        f'd:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_{e}_ema.pth'
        for e in [23, 24, 25]
    ]
    for p in paths:
        m = SOTA_DR_Model('convnextv2_large', pretrained=False)
        checkpoint = torch.load(p, map_location=device)
        m.load_state_dict(checkpoint.get('ema_state_dict', checkpoint.get('model_state_dict', checkpoint)))
        m.to(device)
        m.eval()
        models.append(m)

    # Load labels
    df_15 = pd.read_csv('d:/Major_project_shiro_final/archive/labels/testLabels15.csv')
    df_15['image'] = df_15['image'].astype(str)
    
    df_19 = pd.read_csv('d:/Major_project_shiro_final/archive/labels/testImages19.csv')
    img_col = 'image' if 'image' in df_19.columns else 'id_code'
    df_19['image'] = df_19[img_col].astype(str)
    
    true_labels = {}
    for _, row in df_15.iterrows():
        true_labels[row['image']] = row['level']
    for _, row in df_19.iterrows():
        lbl = row.get('diagnosis', row.get('level', -1))
        true_labels[row['image']] = lbl

    print("Scanning Test Set...")
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
    
    dataset = FastTestDataset(valid_images, target_size=args.resolution)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=8, 
        pin_memory=True, 
        prefetch_factor=2, 
        shuffle=False
    )
    
    out_rows = []
    
    # Vectorized GPU TTA generator
    def apply_gpu_tta(img_batch):
        yield img_batch
        if args.tta == 8:
            yield img_batch.flip(-1) # H-flip
            yield img_batch.flip(-2) # V-flip
            yield img_batch.flip([-1, -2]) # HV-flip
            yield img_batch.rot90(1, [-2, -1]) # Rot90
            yield img_batch.rot90(1, [-2, -1]).flip(-1) # Rot90 + H-flip
            yield img_batch.rot90(2, [-2, -1]) # Rot180
            yield img_batch.rot90(3, [-2, -1]) # Rot270
            
    pbar = tqdm(total=len(valid_images), miniters=100)
    
    with torch.no_grad():
        for batch_ids, batch_tensors, batch_valid_flags in loader:
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            B = batch_tensors.shape[0]
            
            # shape: [B, num_models * num_ttas, num_classes] (Corn logits are num_classes-1, but let's accumulate)
            all_logits = torch.zeros((B, 4), device=device, dtype=torch.float32)
            aug_count = 0
            
            for tta_batch in apply_gpu_tta(batch_tensors):
                for model in models:
                    with torch.amp.autocast('cuda', enabled=args.use_amp):
                        out = model(tta_batch)
                        all_logits += out['ordinal_logits']
                    aug_count += 1
            
            all_logits /= aug_count
            
            probs = torch.sigmoid(all_logits)
            p_geq = torch.cat([torch.ones(B, 1, device=device), probs], dim=1)
            p_exact = p_geq.clone()
            p_exact[:, :-1] = p_geq[:, :-1] - p_geq[:, 1:]
            p_exact[:, -1] = p_geq[:, -1]
            
            reg_scores = torch.sum(probs, dim=1)
            
            probs_cpu = p_exact.cpu().numpy()
            reg_scores_cpu = reg_scores.cpu().numpy()
            valid_flags = batch_valid_flags.numpy()
            
            for i in range(B):
                if not valid_flags[i]:
                    pbar.update(1)
                    continue
                    
                img_id = batch_ids[i]
                true_grade = true_labels.get(img_id, -1)
                final_grade = apply_clinical_rules(reg_scores_cpu[i], probs_cpu[i])
                
                out_rows.append({
                    'image_id': img_id,
                    'true_grade': true_grade,
                    'predicted_grade': final_grade,
                    'reg_score': reg_scores_cpu[i],
                    'prob_0': probs_cpu[i][0],
                    'prob_1': probs_cpu[i][1],
                    'prob_2': probs_cpu[i][2],
                    'prob_3': probs_cpu[i][3],
                    'prob_4': probs_cpu[i][4]
                })
                pbar.update(1)
                
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(args.out_csv, index=False)
    print(f"Saved {args.out_csv}")

if __name__ == '__main__':
    main()
