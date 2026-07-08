import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
# Set performance flags at module level so DataLoader workers inherit them
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
import pandas as pd
import cv2
import json

from src.training.train import run_curriculum_training
from src.evaluation.optimizer import OptimizedRounder
from src.evaluation.gradcam import GradCAMHooker, apply_heatmap
from src.models.sota_dr_model import SOTA_DR_Model
from src.data.dataset import get_validation_augmentations

def run_post_fold_optimization(df_val, img_dir, model_path, model_name, fold_num, target_res=768):
    """ Post-Fold Phase 8 & 9 (Optimization and Clinical Sanity) """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = f"results/fold_{fold_num}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load Model
    model = SOTA_DR_Model(model_name=model_name, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.to(device).eval()
    
    alb_transform = get_validation_augmentations(target_res)
    
    # 2. Extract Validation Scores for Optimizer
    all_preds = []
    all_trues = []
    severe_images = []
    
    print(f"--- Running Post-Fold Optimization for Fold {fold_num} ---")
    for _, row in df_val.iterrows():
        img_name = str(row['image']).strip()
        img_id = img_name.split('.')[0]
        img_path = os.path.join(img_dir, img_id + '.jpg')
            
        if not os.path.exists(img_path):
            continue
            
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_rgb, (target_res, target_res))
        img_tensor = alb_transform(image=img_np)['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            reg_score = outputs['regression_score'].item()
            
        all_preds.append(reg_score)
        all_trues.append(row['level'])
        
        if row['level'] >= 3:
            severe_images.append({'path': img_path, 'score': reg_score, 'true': row['level'], 'tensor': img_tensor})
            
    # Fit thresholds
    optimizer = OptimizedRounder()
    coef = optimizer.fit(all_preds, all_trues)
    
    with open(f"{save_dir}/fold_{fold_num}_thresholds.json", "w") as f:
        json.dump({"thresholds": coef.tolist()}, f)
        
    print(f"Optimal Thresholds for Fold {fold_num}: {coef.tolist()} Saved.")
    
    # 3. Clinical Sanity Check (Top 5 GradCAMs)
    severe_images.sort(key=lambda x: x['score'], reverse=True)
    top_5 = severe_images[:5]
    
    print("Generating GradCAM verification samples for Top 5 active predictions...")
    hooker = GradCAMHooker(model)
    
    for i, item in enumerate(top_5):
        img_bgr = cv2.imread(item['path'])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(img_rgb, (target_res, target_res))
        h, w, _ = orig_img.shape
        
        heatmap_bgr = hooker(item['tensor'], (h, w))
        orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        overlay = apply_heatmap(orig_bgr, heatmap_bgr, alpha=0.5)
        
        cam_path = f"{save_dir}/gradcam_rank{i+1}_true{item['true']}_pred{item['score']:.2f}.jpg"
        cv2.imwrite(cam_path, overlay)


def main():
    import argparse
    # A4000 Performance Flags (global)
    import torch as _torch
    _torch.backends.cudnn.benchmark = True
    _torch.backends.cuda.matmul.allow_tf32 = True
    _torch.backends.cudnn.allow_tf32 = True
    parser = argparse.ArgumentParser(description="Diabetic Retinopathy Training Pipeline")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--folds", type=int, default=3, help="Number of folds to run")
    parser.add_argument("--stabilize", type=bool, default=True, help="Enable training stability fixes (NaN guard, grad clipping)")
    args = parser.parse_args()

    csv_path = "D:/Major_project_shiro_final/trainLabels.csv"
    img_dir = "D:/Major_project_shiro_final/preprocessed_512" # Corrected to align with max 768px stage
    os.makedirs('results', exist_ok=True)
    
    # Load metadata
    df = pd.read_csv(csv_path)
    
    # Run GroupKFold to assign folds (Ensures Phase 1 isolation)
    from src.data.data_split import setup_cross_validation
    df = setup_cross_validation(df, n_splits=5)
    
    # The models rotate architectures to build an ensemble set later (Phase 10)
    models_to_run = {
        1: 'convnextv2_large',
        2: 'efficientnetv2_l',
        3: 'convnextv2_large',
        4: 'efficientnetv2_l',
        5: 'convnextv2_large'
    }
    
    try:
        # Loop Continuously without pausing
        for fold in range(3, 6):
            arch = models_to_run[fold]
            print(f"=========================================")
            print(f"  EXECUTING FOLD {fold} -> {arch.upper()}  ")
            print(f"  Config: Epochs={args.epochs}, Stabilize={args.stabilize}")
            print(f"=========================================")
            
            # Phase 2-7: Train the fold
            best_qwk = run_curriculum_training(df, img_dir, fold_num=fold, model_name=arch, epochs=args.epochs, save_dir='results', stabilize=args.stabilize)
            print(f"[Fold {fold}] Completed. Highest Validation QWK: {best_qwk:.4f}")
            
            # Phase 8-9: Optimization & Visualization
            best_model_path = f"results/fold_{fold}/{arch}_best_ema.pth"
            df_val = df[df['fold'] == fold].reset_index(drop=True)
            
            run_post_fold_optimization(df_val, img_dir, best_model_path, arch, fold, target_res=768)
            
            # --- CRITICAL: HARDWARE SAFETY PROTOCOL (Clear Memory Between Folds) ---
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"--- Memory Cleared for Fold {fold+1} ---")
            
        print(f"ALL {args.folds} FOLDS COMPLETED AUTONOMOUSLY.")
        
        # Post-Pipeline Phase: K-Fold Threshold Averaging
        try:
            from src.evaluation.average_thresholds import average_k_fold_thresholds
            print("\n=========================================")
            print("  POST-PIPELINE: K-FOLD THRESHOLD AVERAGING")
            print("=========================================")
            average_k_fold_thresholds(results_dir='results', num_folds=args.folds)
        except Exception as avg_e:
            print(f"Warning: Failed to compute average thresholds: {avg_e}")
        
    except Exception as e:
         import traceback
         print(f"CRITICAL PIPELINE FAILURE: {e}")
         traceback.print_exc()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)  # CRITICAL: required on Windows for DataLoader workers
    mp.freeze_support()
    main()
