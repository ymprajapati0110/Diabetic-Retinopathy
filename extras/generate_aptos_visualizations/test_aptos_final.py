import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    cohen_kappa_score, classification_report,
    confusion_matrix, f1_score, accuracy_score
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    "checkpoint_dir": "d:/Major_project_shiro_final/results/fold_3",
    "checkpoints": [
        "convnextv2_large_epoch_23_ema.pth",
        "convnextv2_large_epoch_24_ema.pth",
        "convnextv2_large_epoch_25_ema.pth"
    ],
    "weights": [0.3314, 0.3337, 0.3349],
    "thresholds": np.array([0.3012, 1.4639, 2.0536, 3.1606]),

    "labels_csv": "d:/Major_project_shiro_final/archive/labels/trainLabels19.csv",
    "img_dir": "d:/Major_project_shiro_final/archive/resized train 19",
    
    "out_dir": "d:/Major_project_shiro_final/results/fold_3/aptos_results",
    "out_csv": "d:/Major_project_shiro_final/results/fold_3/aptos_results/predictions_aptos.csv",

    "resolution": 1024,
    "batch_size": 8,
    "num_workers": 0,
    "use_amp": True,
}

# ═══════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════

def preprocess_image(img_path, output_size=1024):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_bgr = cv2.resize(img_bgr, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

IMAGENET_NORM = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

class AptosDataset(Dataset):
    def __init__(self, img_paths, true_labels=None, resolution=1024):
        self.img_paths = img_paths
        self.true_labels = true_labels
        self.resolution = resolution

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img_id = os.path.splitext(os.path.basename(path))[0]
        label = self.true_labels[idx] if self.true_labels is not None else -1

        img_rgb = preprocess_image(path, self.resolution)
        if img_rgb is None:
            dummy = torch.zeros(3, self.resolution, self.resolution)
            return img_id, dummy, label, False

        tensor = IMAGENET_NORM(image=img_rgb)['image']
        return img_id, tensor, label, True

def _test_collate(batch):
    ids     = [b[0] for b in batch]
    tensors = torch.stack([b[1] for b in batch])
    labels  = [b[2] for b in batch]
    valids  = [b[3] for b in batch]
    return ids, tensors, labels, valids

# ═══════════════════════════════════════════
# MODEL & TTA
# ═══════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device):
    from src.models.sota_dr_model import SOTA_DR_Model
    model = SOTA_DR_Model('convnextv2_large', pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict') or ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def tta_views(x):
    yield x                                      # original
    yield torch.flip(x, dims=[-1])               # H-flip
    yield torch.flip(x, dims=[-2])               # V-flip
    yield torch.rot90(x, k=1, dims=[-2, -1])     # rot 90
    yield torch.rot90(x, k=2, dims=[-2, -1])     # rot 180
    yield torch.rot90(x, k=3, dims=[-2, -1])     # rot 270
    yield torch.flip(torch.rot90(x, k=1, dims=[-2, -1]), dims=[-1])  # rot90 + H-flip
    yield torch.flip(torch.rot90(x, k=3, dims=[-2, -1]), dims=[-1])  # rot270 + H-flip

@torch.no_grad()
def predict_batch(models, weights, batch: torch.Tensor, device: torch.device, use_amp: bool):
    B = batch.shape[0]
    acc_ordinal = torch.zeros(B, 4, device=device)
    acc_regression = torch.zeros(B, device=device)
    total_w = 0.0

    for model, w in zip(models, weights):
        for view in tta_views(batch):
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(view)
            
            ord_probs = torch.sigmoid(out['ordinal_logits'])
            acc_ordinal += ord_probs * w
            
            reg = torch.clamp(out['regression_score'], 0.0, 4.0)
            acc_regression += reg * w
            
            total_w += w   # ← MUST be inside the TTA loop, not outside

    acc_ordinal /= total_w
    acc_regression /= total_w

    p_geq = torch.cat([torch.ones(B, 1, device=device), acc_ordinal], dim=1)
    p_exact = torch.zeros(B, 5, device=device)
    p_exact[:, 0] = 1.0 - p_geq[:, 1]
    for k in range(1, 4):
        p_exact[:, k] = p_geq[:, k] - p_geq[:, k + 1]
    p_exact[:, 4] = p_geq[:, 4]
    p_exact = torch.clamp(p_exact, 0.0, 1.0)

    return acc_regression.cpu().numpy(), p_exact.cpu().numpy()

def apply_thresholds(reg_scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    bins = [-np.inf] + list(thresholds) + [np.inf]
    return np.clip(np.digitize(reg_scores, bins) - 1, 0, 4)

# ═══════════════════════════════════════════
# SANITY CHECK & INFERENCE
# ═══════════════════════════════════════════

def sanity_check(models, weights, img_paths, true_labels, device, cfg):
    print("\n── Sanity check (50 images) ──")
    
    paths = img_paths[:50]
    labels = true_labels[:50]
    
    dataset = AptosDataset(paths, labels, cfg['resolution'])
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, collate_fn=_test_collate)

    reg_scores, actual_labels = [], []

    for batch_ids, batch_imgs, batch_labels, batch_valids in tqdm(loader, desc='sanity check'):
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        reg, _ = predict_batch(models, weights, batch_imgs, device, cfg['use_amp'])
        
        for i in range(len(batch_valids)):
            if batch_valids[i]:
                reg_scores.append(reg[i])
                actual_labels.append(batch_labels[i])

    reg_scores = np.array(reg_scores)
    actual_labels = np.array(actual_labels)

    mean_r = reg_scores.mean()
    std_r = reg_scores.std()
    
    naive_preds = np.clip(np.round(reg_scores).astype(int), 0, 4)
    naive_qwk = cohen_kappa_score(actual_labels, naive_preds, weights='quadratic')

    print(f"  reg_score mean = {mean_r:.4f}")
    print(f"  reg_score std  = {std_r:.4f}")
    print(f"  Naive QWK      = {naive_qwk:.4f}")

    if mean_r < 0.3 or mean_r > 2.0:
        print("  ❌ FAILURE: reg_score mean < 0.3 or mean > 2.0 → pipeline broken")
        sys.exit(1)
    if std_r < 0.3:
        print("  ❌ FAILURE: reg_score std < 0.3 → predictions collapsed")
        sys.exit(1)
    if naive_qwk < 0.70:
        print("  ❌ FAILURE: Sanity naive QWK < 0.70 → something wrong")
        sys.exit(1)

    print("  ✅ Sanity check passed!")

def run_inference(models, weights, img_paths, true_labels, device, cfg):
    print("\n── Running Full Inference ──")
    
    done_ids = set()
    if os.path.exists(cfg['out_csv']):
        try:
            df_done = pd.read_csv(cfg['out_csv'])
            done_ids = set(df_done['id_code'].astype(str).tolist())
            print(f"  Resuming — {len(done_ids)} images already done")
        except Exception:
            pass

    valid_paths = []
    valid_labels = []
    for path, label in zip(img_paths, true_labels):
        img_id = os.path.splitext(os.path.basename(path))[0]
        if img_id not in done_ids:
            valid_paths.append(path)
            valid_labels.append(label)

    total_remaining = len(valid_paths)
    print(f"  Total remaining: {total_remaining}")

    if total_remaining == 0:
        print("  All images already processed.")
        return pd.read_csv(cfg['out_csv'])

    dataset = AptosDataset(valid_paths, valid_labels, cfg['resolution'])
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, collate_fn=_test_collate)

    rows = []
    saved_count = len(done_ids)
    SAVE_EVERY = 500
    write_header = not os.path.exists(cfg['out_csv'])

    pbar = tqdm(total=total_remaining, desc='inference', miniters=10)

    for batch_ids, batch_tensors, batch_labels, batch_valids in loader:
        batch_tensors = batch_tensors.to(device, non_blocking=True)
        B = batch_tensors.shape[0]

        reg, probs = predict_batch(models, weights, batch_tensors, device, cfg['use_amp'])

        for i in range(B):
            pbar.update(1)
            if not batch_valids[i]:
                continue

            img_id = batch_ids[i]
            true_grade = batch_labels[i]
            pred_grade = int(apply_thresholds(np.array([reg[i]]), cfg['thresholds'])[0])

            rows.append({
                'id_code': img_id,
                'true_grade': true_grade,
                'predicted_grade': pred_grade,
                'reg_score': float(reg[i]),
                'prob_0': float(probs[i, 0]),
                'prob_1': float(probs[i, 1]),
                'prob_2': float(probs[i, 2]),
                'prob_3': float(probs[i, 3]),
                'prob_4': float(probs[i, 4]),
            })
            saved_count += 1

        if len(rows) >= SAVE_EVERY:
            pd.DataFrame(rows).to_csv(cfg['out_csv'], mode='a', header=write_header, index=False)
            write_header = False
            rows = []
            pbar.set_postfix({'saved': saved_count})

    if rows:
        pd.DataFrame(rows).to_csv(cfg['out_csv'], mode='a', header=write_header, index=False)

    pbar.close()
    return pd.read_csv(cfg['out_csv'])

def print_metrics(df: pd.DataFrame):
    y_true = df['true_grade'].values
    y_pred = df['predicted_grade'].values

    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    acc = accuracy_score(y_true, y_pred)
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    print("\n" + "═" * 55)
    print(f"  FINAL APTOS RESULTS")
    print("═" * 55)
    print(f"  QWK            = {qwk:.4f}")
    print(f"  Accuracy       = {acc:.4f}")
    print(f"  Per-class F1   = [{', '.join([f'{f:.4f}' for f in f1s])}]")
    print("\n  Confusion matrix:")
    print("       G0    G1    G2    G3    G4")
    for i, row in enumerate(cm):
        print(f"  G{i}  " + "  ".join(f"{v:5d}" for v in row))
    print("═" * 55)

def main():
    cfg = CFG
    os.makedirs(cfg['out_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n── Loading models ──")
    models = []
    for ckpt_name in cfg['checkpoints']:
        ckpt_path = os.path.join(cfg['checkpoint_dir'], ckpt_name)
        models.append(load_model(ckpt_path, device))
    weights = cfg['weights']

    print("\n── Loading APTOS labels ──")
    df = pd.read_csv(cfg['labels_csv'])
    
    img_paths = []
    true_labels = []
    
    # Shuffle for sanity check to get a mix of classes
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    for _, row in df.iterrows():
        img_id = str(row['id_code'])
        img_path = os.path.join(cfg['img_dir'], f"{img_id}.jpg")
        if os.path.exists(img_path):
            img_paths.append(img_path)
            true_labels.append(int(row['diagnosis']))

    print(f"Found {len(img_paths)} valid images out of {len(df)}.")
    
    if len(img_paths) < 50:
        print("Not enough images for sanity check (need 50)!")
        return

    # Sanity Check
    sanity_check(models, weights, img_paths, true_labels, device, cfg)

    # Full inference
    df_results = run_inference(models, weights, img_paths, true_labels, device, cfg)

    # Final Metrics
    print_metrics(df_results)

if __name__ == '__main__':
    # Ensure correct start method on Windows
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    main()
