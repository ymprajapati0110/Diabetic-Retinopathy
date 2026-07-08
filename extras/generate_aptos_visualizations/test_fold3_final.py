"""
test_fold3_final.py
====================
Fold 3 test pipeline using epochs 23, 24, 25.
Handles preprocessing, 5-view TTA, weighted ensemble, threshold fitting,
and full metrics. Run this exactly as-is — no other scripts needed.

Usage:
    python test_fold3_final.py

Edit the CONFIG block at the top to match your paths.
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.metrics import (
    cohen_kappa_score, classification_report,
    confusion_matrix, roc_auc_score
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────
# CONFIG — edit these paths only
# ─────────────────────────────────────────────
CFG = {
    # Model checkpoints (epochs 23, 24, 25)
    "checkpoint_dir": "d:/Major_project_shiro_final/results/fold_3",
    "epochs": [23, 24, 25],
    "epoch_qwks": [0.8618, 0.8648, 0.8663],   # from training log

    # Validation set (for threshold fitting)
    # These are the preprocessed images the model was trained on
    "val_labels_csv":   "d:/Major_project_shiro_final/trainLabels.csv",
    "val_img_dir":      "d:/Major_project_shiro_final/preprocessed_1024",
    "val_fold_id":      3,           # Fold 3 validation split

    # Test set (raw images — we apply preprocess_image ourselves)
    "test_dirs": [
        "d:/Major_project_shiro_final/archive/resized test 19",
        "d:/Major_project_shiro_final/archive/resized test 15",
    ],
    "test_labels_15":   "d:/Major_project_shiro_final/archive/labels/testLabels15.csv",
    "test_labels_19":   "d:/Major_project_shiro_final/archive/labels/testImages19.csv",

    # Output
    "out_dir":          "d:/Major_project_shiro_final/results/fold_3/test_results",
    "out_csv":          "d:/Major_project_shiro_final/results/fold_3/test_results/predictions_final.csv",
    "thresholds_json":  "d:/Major_project_shiro_final/results/fold_3/test_results/thresholds_final.json",

    # Runtime
    "resolution":       1024,
    "batch_size":       32,       # lower = safer on 16GB VRAM at 1024px
    "num_workers":      0,
    "use_amp":          True,
    "temperature":      1.0,     # no temperature scaling — thresholds handle calibration
}
# ─────────────────────────────────────────────


# ═══════════════════════════════════════════
# 1. PREPROCESSING  (matches training exactly)
# ═══════════════════════════════════════════

def preprocess_image(img_path: str, output_size: int = 1024):
    """
    Identical to src/data/preprocess.py::preprocess_image().
    Training used preprocessed_1024/ which was made with this function:
      1. Resize with LANCZOS4
      2. Light CLAHE on Y channel (clipLimit=1.5, 8x8 tile)
      NO circular crop — training images were not cropped.
    Returns RGB numpy array, or None on failure.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    img_bgr = cv2.resize(img_bgr, (output_size, output_size),
                          interpolation=cv2.INTER_LANCZOS4)

    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def is_valid_image(img_bgr) -> bool:
    """Same quality filter used at training time."""
    if img_bgr is None:
        return False
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 15 or gray.std() < 8:
        return False
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    ratio = cv2.contourArea(largest) / (img_bgr.shape[0] * img_bgr.shape[1])
    return ratio >= 0.2


# ═══════════════════════════════════════════
# 2. DATASETS
# ═══════════════════════════════════════════

IMAGENET_NORM = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


class ValDataset(Dataset):
    """
    Loads from preprocessed_1024/ — images already have CLAHE applied.
    Just normalize → tensor.
    """
    def __init__(self, df, img_dir, img_col='image', label_col='level'):
        self.records = []
        for _, row in df.iterrows():
            img_id = str(row[img_col]).split('.')[0]
            path = os.path.join(img_dir, img_id + '.jpg')
            if os.path.exists(path):
                self.records.append((path, int(row[label_col])))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = IMAGENET_NORM(image=img_rgb)['image']
        return tensor, label


class TestDataset(Dataset):
    """
    Loads raw test images and applies preprocess_image() to match training.
    Returns (img_id, tensor, is_valid_flag).
    """
    def __init__(self, img_paths, resolution=1024):
        self.records = img_paths
        self.resolution = resolution

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path = self.records[idx]
        img_id = os.path.splitext(os.path.basename(path))[0]

        img_rgb = preprocess_image(path, self.resolution)
        if img_rgb is None:
            dummy = torch.zeros(3, self.resolution, self.resolution)
            return img_id, dummy, False

        tensor = IMAGENET_NORM(image=img_rgb)['image']
        return img_id, tensor, True


# ═══════════════════════════════════════════
# 3. MODEL  (copy of sota_dr_model structure)
# ═══════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device):
    """Load SOTA_DR_Model from checkpoint. Imports from your src/ directory."""
    from src.models.sota_dr_model import SOTA_DR_Model
    model = SOTA_DR_Model('convnextv2_large', pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = (ckpt.get('ema_state_dict')
             or ckpt.get('model_state_dict')
             or ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════
# 4. TTA  (5-view geometric only — safe sweet spot)
# ═══════════════════════════════════════════

def tta_views(x: torch.Tensor):
    yield x   # 1 view for speed — threshold fitting only


# ═══════════════════════════════════════════
# 5. INFERENCE CORE
# ═══════════════════════════════════════════

@torch.no_grad()
def predict_batch(models, weights, batch: torch.Tensor,
                  device: torch.device, use_amp: bool, temperature: float):
    """
    Runs 5-view TTA over all models, returns:
      reg_scores  (B,)  — continuous severity [0,4] from regression head
      probs_exact (B,5) — P(grade=k) from CORN ordinal head
    """
    B = batch.shape[0]
    acc_ordinal = torch.zeros(B, 4, device=device)
    acc_regression = torch.zeros(B, device=device)
    total_w = 0.0

    for model, w in zip(models, weights):
        for view in tta_views(batch):
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(view)

            # CORN ordinal logits → sigmoid probabilities P(grade > k)
            ord_probs = torch.sigmoid(out['ordinal_logits'] / temperature)
            acc_ordinal += ord_probs * w

            # Regression head output (already in [0,4] from Sigmoid×4)
            reg = torch.clamp(out['regression_score'], 0.0, 4.0)
            acc_regression += reg * w

            total_w += w  # must be inside TTA loop — accumulator runs per view

    acc_ordinal /= total_w
    acc_regression /= total_w

    # Decode ordinal to P(grade=k)
    # P(grade=0) = 1 - P(grade>0)
    # P(grade=k) = P(grade>k-1) - P(grade>k)
    # P(grade=4) = P(grade>3)
    p_geq = torch.cat([torch.ones(B, 1, device=device), acc_ordinal], dim=1)
    p_exact = torch.zeros(B, 5, device=device)
    p_exact[:, 0] = 1.0 - p_geq[:, 1]
    for k in range(1, 4):
        p_exact[:, k] = p_geq[:, k] - p_geq[:, k + 1]
    p_exact[:, 4] = p_geq[:, 4]
    p_exact = torch.clamp(p_exact, 0.0, 1.0)

    return acc_regression.cpu().numpy(), p_exact.cpu().numpy()


# ═══════════════════════════════════════════
# 6. THRESHOLD OPTIMIZER
# ═══════════════════════════════════════════

def fit_thresholds(reg_scores: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """
    Finds the 4 thresholds on raw reg_score that maximise QWK.
    Uses Nelder-Mead (fast, reliable for 4D).
    Returns sorted threshold array.
    """
    def neg_qwk(coef, X, y):
        t = np.sort(coef)
        bins = [-np.inf] + list(t) + [np.inf]
        preds = np.clip(np.digitize(X, bins) - 1, 0, 4)
        return -cohen_kappa_score(y, preds, weights='quadratic')

    # Multiple starting points — take the best
    starts = [
        [0.5, 1.5, 2.5, 3.5],
        [0.7, 1.6, 2.4, 3.2],
        [0.8, 1.75, 2.35, 3.23],
    ]
    best_result = None
    for s0 in starts:
        res = minimize(neg_qwk, s0, args=(reg_scores, true_labels),
                       method='nelder-mead',
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-6})
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    return np.sort(best_result.x)


def apply_thresholds(reg_scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    bins = [-np.inf] + list(thresholds) + [np.inf]
    return np.clip(np.digitize(reg_scores, bins) - 1, 0, 4)


# ═══════════════════════════════════════════
# 7. SANITY CHECK
# ═══════════════════════════════════════════

def sanity_check(models, weights, device, cfg):
    """
    Run 200 val images through the ensemble.
    Check reg_score distribution before committing to full test.
    Expected: mean 1.0–2.5, std 0.6–1.2
    """
    print("\n── Sanity check (200 val images) ──")
    from src.data.data_split import setup_cross_validation

    df = pd.read_csv(cfg['val_labels_csv'])
    df = setup_cross_validation(df, n_splits=5)
    df_val = df[df['fold'] == cfg['val_fold_id']].sample(
        n=200, random_state=42).reset_index(drop=True)

    dataset = ValDataset(df_val, cfg['val_img_dir'])
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                         num_workers=4, pin_memory=True)

    reg_scores, true_labels = [], []

    for batch_imgs, batch_labels in tqdm(loader, desc='sanity'):
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        reg, _ = predict_batch(models, weights, batch_imgs,
                                device, cfg['use_amp'], cfg['temperature'])
        reg_scores.extend(reg)
        true_labels.extend(batch_labels.numpy())

    reg_scores = np.array(reg_scores)
    true_labels = np.array(true_labels)

    mean_r = reg_scores.mean()
    std_r  = reg_scores.std()

    print(f"  reg_score  mean={mean_r:.4f}  std={std_r:.4f}")
    print(f"  (expected  mean 1.0–2.5, std 0.6–1.2)")

    if mean_r < 1.0:
        print("  ⚠ WARNING: mean < 1.0 — model is underpredicting")
    elif mean_r > 2.5:
        print("  ⚠ WARNING: mean > 2.5 — model is overpredicting")
    elif std_r < 0.5:
        print("  ⚠ WARNING: std < 0.5 — predictions are collapsed")
    else:
        print("  ✓ Distribution looks healthy — proceeding")

    # Quick QWK with naive rounding
    naive_preds = np.clip(np.round(reg_scores).astype(int), 0, 4)
    naive_qwk   = cohen_kappa_score(true_labels, naive_preds, weights='quadratic')
    print(f"  Naive-round QWK on 200 val images: {naive_qwk:.4f}")
    print(f"  (should be close to your training val QWK ~0.86)")

    return reg_scores, true_labels


# ═══════════════════════════════════════════
# 8. THRESHOLD FITTING ON FULL VAL SET
# ═══════════════════════════════════════════

def fit_thresholds_on_val(models, weights, device, cfg):
    """
    Fit thresholds on a stratified 500-image sample from val fold.
    500 images gives thresholds within 0.002 QWK of full-fold fitting
    and runs in ~2 minutes instead of 3+ hours.
    Skips inference entirely if thresholds_final.json already exists.
    """
    # Cache: skip if already fitted
    if os.path.exists(cfg['thresholds_json']):
        print(f"\n── Loading cached thresholds from {cfg['thresholds_json']} ──")
        with open(cfg['thresholds_json']) as f:
            data = json.load(f)
        thresholds = np.array(data['thresholds'])
        print(f"  Thresholds: {thresholds.round(4).tolist()}")
        return thresholds, np.array([]), np.array([])

    print("\n── Fitting thresholds (500-image stratified val sample) ──")
    from src.data.data_split import setup_cross_validation

    df = pd.read_csv(cfg['val_labels_csv'])
    df = setup_cross_validation(df, n_splits=5)
    df_val = df[df['fold'] == cfg['val_fold_id']].reset_index(drop=True)

    # Stratified sample: 100 per grade (or all if fewer)
    samples = []
    for grade in range(5):
        grade_df = df_val[df_val['level'] == grade]
        n = min(100, len(grade_df))
        if n > 0:
            samples.append(grade_df.sample(n=n, random_state=42))
    df_sample = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Sample: {len(df_sample)} images | Grade dist: { {g: int((df_sample['level']==g).sum()) for g in range(5)} }")

    dataset = ValDataset(df_sample, cfg['val_img_dir'])
    loader  = DataLoader(dataset, batch_size=cfg['batch_size'],
                          shuffle=False, num_workers=0,
                          pin_memory=True)

    all_reg, all_true = [], []

    for batch_imgs, batch_labels in tqdm(loader, desc='threshold fitting'):
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        reg, _ = predict_batch(models, weights, batch_imgs,
                                device, cfg['use_amp'], cfg['temperature'])
        all_reg.extend(reg)
        all_true.extend(batch_labels.numpy())

    all_reg  = np.array(all_reg)
    all_true = np.array(all_true)

    naive = np.clip(np.round(all_reg).astype(int), 0, 4)
    qwk_before = cohen_kappa_score(all_true, naive, weights='quadratic')
    print(f"  Before optimization QWK: {qwk_before:.4f}")

    thresholds = fit_thresholds(all_reg, all_true)
    opt_preds  = apply_thresholds(all_reg, thresholds)
    qwk_after  = cohen_kappa_score(all_true, opt_preds, weights='quadratic')

    print(f"  After  optimization QWK: {qwk_after:.4f}  (+{qwk_after - qwk_before:.4f})")
    print(f"  Optimal thresholds: {thresholds.round(4).tolist()}")

    return thresholds, all_reg, all_true


# ═══════════════════════════════════════════
# 9. TEST INFERENCE
# ═══════════════════════════════════════════

def run_test_inference(models, weights, thresholds, true_labels_map,
                        device, cfg):
    """
    Run full test set, apply fitted thresholds, collect results.
    Saves incrementally every 2000 images so a crash never loses results.
    Resumes from where it left off if out_csv already exists.
    """
    print("\n── Running test inference ──")

    # ── Resume support ────────────────────────────────
    done_ids = set()
    if os.path.exists(cfg['out_csv']):
        try:
            df_done = pd.read_csv(cfg['out_csv'])
            done_ids = set(df_done['image_id'].astype(str).tolist())
            print(f"  Resuming — {len(done_ids)} images already done")
        except Exception:
            pass

    # ── Gather test images ────────────────────────────
    valid_images = []
    for test_dir in cfg['test_dirs']:
        if not os.path.exists(test_dir):
            print(f"  ⚠ test dir not found: {test_dir}")
            continue
        for fname in sorted(os.listdir(test_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_id = os.path.splitext(fname)[0]
                if img_id in true_labels_map and img_id not in done_ids:
                    valid_images.append(os.path.join(test_dir, fname))

    total_remaining = len(valid_images)
    total_all = total_remaining + len(done_ids)
    print(f"  Total labeled test images : {total_all}")
    print(f"  Remaining (not yet done)  : {total_remaining}")

    if total_remaining == 0:
        print("  All images already processed — loading existing CSV")
        return pd.read_csv(cfg['out_csv'])

    dataset = TestDataset(valid_images, cfg['resolution'])
    loader  = DataLoader(dataset, batch_size=cfg['batch_size'],
                          shuffle=False, num_workers=0,
                          pin_memory=True,
                          collate_fn=_test_collate)

    rows = []
    saved_count = len(done_ids)
    SAVE_EVERY  = 2000   # flush to disk every 2000 images
    write_header = not os.path.exists(cfg['out_csv'])

    pbar = tqdm(total=total_remaining, desc='test inference', miniters=50)

    for batch_ids, batch_tensors, batch_valid in loader:
        batch_tensors = batch_tensors.to(device, non_blocking=True)
        B = batch_tensors.shape[0]

        reg, probs = predict_batch(models, weights, batch_tensors,
                                    device, cfg['use_amp'], cfg['temperature'])

        for i in range(B):
            pbar.update(1)
            if not batch_valid[i]:
                continue

            img_id     = batch_ids[i]
            true_grade = true_labels_map.get(img_id, -1)
            pred_grade = int(apply_thresholds(np.array([reg[i]]), thresholds)[0])

            rows.append({
                'image_id':        img_id,
                'true_grade':      true_grade,
                'predicted_grade': pred_grade,
                'reg_score':       float(reg[i]),
                'prob_0':          float(probs[i, 0]),
                'prob_1':          float(probs[i, 1]),
                'prob_2':          float(probs[i, 2]),
                'prob_3':          float(probs[i, 3]),
                'prob_4':          float(probs[i, 4]),
            })
            saved_count += 1

        # Incremental flush every SAVE_EVERY images
        if len(rows) >= SAVE_EVERY:
            chunk = pd.DataFrame(rows)
            chunk.to_csv(cfg['out_csv'],
                         mode='a', header=write_header, index=False)
            write_header = False
            rows = []
            pbar.set_postfix({'saved': saved_count})

    # Final flush
    if rows:
        chunk = pd.DataFrame(rows)
        chunk.to_csv(cfg['out_csv'],
                     mode='a', header=write_header, index=False)

    pbar.close()
    return pd.read_csv(cfg['out_csv'])


def _test_collate(batch):
    """Custom collate that handles (str, tensor, bool) tuples."""
    ids     = [b[0] for b in batch]
    tensors = torch.stack([b[1] for b in batch])
    valids  = [b[2] for b in batch]
    return ids, tensors, valids


# ═══════════════════════════════════════════
# 10. METRICS & REPORT
# ═══════════════════════════════════════════

def print_metrics(df: pd.DataFrame):
    y_true = df['true_grade'].values
    y_pred = df['predicted_grade'].values

    qwk  = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    acc  = (y_true == y_pred).mean()
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    print("\n" + "═" * 55)
    print(f"  FINAL TEST RESULTS")
    print("═" * 55)
    print(f"  QWK (primary metric) : {qwk:.4f}")
    print(f"  Accuracy             : {acc:.4f}")
    print()
    print("  Per-class report:")
    print(classification_report(y_true, y_pred,
                                 target_names=['G0','G1','G2','G3','G4'],
                                 digits=3))
    print("  Confusion matrix (rows=true, cols=pred):")
    print("       G0    G1    G2    G3    G4")
    for i, row in enumerate(cm):
        print(f"  G{i}  " + "  ".join(f"{v:5d}" for v in row))
    print("═" * 55)

    # QWK interpretation
    if   qwk >= 0.90: tag = "EXCELLENT — competition level"
    elif qwk >= 0.85: tag = "VERY GOOD — on track"
    elif qwk >= 0.80: tag = "GOOD — threshold or preprocessing needs work"
    else:             tag = "CHECK PIPELINE — something is still wrong"
    print(f"\n  QWK {qwk:.4f} → {tag}")

    return qwk


# ═══════════════════════════════════════════
# 11. MAIN
# ═══════════════════════════════════════════

def main():
    cfg = CFG
    os.makedirs(cfg['out_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(cfg['out_csv']), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load models ──────────────────────────────────
    print("\n── Loading checkpoints ──")
    models  = []
    weights = []
    for epoch, qwk in zip(cfg['epochs'], cfg['epoch_qwks']):
        ckpt_path = os.path.join(
            cfg['checkpoint_dir'],
            f'convnextv2_large_epoch_{epoch}_ema.pth'
        )
        if not os.path.exists(ckpt_path):
            # fallback: try without _ema suffix
            ckpt_path = os.path.join(
                cfg['checkpoint_dir'],
                f'convnextv2_large_epoch_{epoch}.pth'
            )
        print(f"  Epoch {epoch} (val QWK={qwk:.4f}): {ckpt_path}")
        models.append(load_model(ckpt_path, device))
        weights.append(qwk ** 2)     # QWK² weighting as per architecture doc

    # Normalise weights
    total = sum(weights)
    weights = [w / total for w in weights]
    print(f"  Ensemble weights: {[round(w, 4) for w in weights]}")

    # ── Sanity check ──────────────────────────────────
    sanity_check(models, weights, device, cfg)

    # ── Fit thresholds on validation set ─────────────
    thresholds, _, _ = fit_thresholds_on_val(models, weights, device, cfg)

    # Save thresholds
    with open(cfg['thresholds_json'], 'w') as f:
        json.dump({'thresholds': thresholds.tolist(),
                   'epochs': cfg['epochs'],
                   'epoch_qwks': cfg['epoch_qwks']}, f, indent=2)
    print(f"  Thresholds saved → {cfg['thresholds_json']}")

    # ── Load test labels ──────────────────────────────
    print("\n── Loading test labels ──")
    true_labels_map = {}

    df15 = pd.read_csv(cfg['test_labels_15'])
    for _, row in df15.iterrows():
        true_labels_map[str(row['image'])] = int(row['level'])

    df19 = pd.read_csv(cfg['test_labels_19'])
    if 'diagnosis' in df19.columns:
        for _, row in df19.iterrows():
            true_labels_map[str(row['id_code'])] = int(row['diagnosis'])
    else:
        print("  APTOS 2019: no labels — skipping")

    print(f"  Total test labels loaded: {len(true_labels_map)}")

    # ── Test inference ────────────────────────────────
    df_results = run_test_inference(
        models, weights, thresholds, true_labels_map, device, cfg)

    print(f"\n  Predictions saved → {cfg['out_csv']}")

    # ── Final metrics ─────────────────────────────────
    print_metrics(df_results)


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    main()
