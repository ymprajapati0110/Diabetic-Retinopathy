import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
import pandas as pd
from tqdm import tqdm
import time

from src.data.data_split import setup_cross_validation
from src.data.dataset import DRDataset, mixup_cutmix_collate, mixup_collate, cutmix_collate
from src.models.sota_dr_model import SOTA_DR_Model
from src.training.losses import MultiTaskLoss, corn_loss, focal_loss_binary
from src.training.ema import EMA
from src.training.metrics import validate_consistency, calculate_metrics

def get_img_dir(base_img_dir, resolution):
    """
    Routes each curriculum stage to its pre-generated folder.
    Eliminates 8.7M runtime resize operations (35k imgs × 50 epochs × 5 folds).
    Falls back to base_img_dir with a warning if the folder doesn't exist.
    """
    base   = os.path.dirname(os.path.abspath(base_img_dir))   # e.g. d:/HYLr
    prefix = os.path.basename(base_img_dir).rsplit('_', 1)[0]  # 'preprocessed'

    if resolution == 512:
        candidate = os.path.join(base, f"{prefix}_512")
    elif resolution == 768:
        candidate = os.path.join(base, f"{prefix}_768")
    elif resolution == 1024:
        candidate = os.path.join(base, f"{prefix}_1024")
    else:
        candidate = base_img_dir

    if not os.path.isdir(candidate):
        print(f"WARNING: {candidate} not found. Falling back to {base_img_dir} (runtime resize active).")
        return base_img_dir
    return candidate

def get_dataloaders(df_train, df_val, img_dir, fold_num, target_size, batch_size, num_workers=10, override_collate='auto'):
    """ Returns train/val dataloaders for the specific progressive resolution & fold. """
    
    train_dataset = DRDataset(df_train, img_dir, fold_num=fold_num, is_train=True)
    val_dataset   = DRDataset(df_val,   img_dir, fold_num=fold_num, is_train=False)
    
    # NO SAMPLER - back to regular shuffle
    # (Sampler causing too much instability)
    
    # Collate function for augmentation
    if override_collate == 'auto':
        collate_fn = None
        if fold_num == 1:
             collate_fn = mixup_collate
        elif fold_num == 2:
             collate_fn = cutmix_collate
    else:
        collate_fn = override_collate
    
    # Train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,              # Regular shuffle
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # Val loader unchanged
    val_batch_size = max(batch_size, 8)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        drop_last=False
    )
    
    return train_loader, val_loader

def _compute_loss(outputs, targets, criterion, lambda_qwk, lambda_consistency, label_smoothing=0.0):
    """
    Unified loss computation for both mixed (MixUp/CutMix) and standard batches.
    Uses criterion.huber_loss for regression to stay consistent with MultiTaskLoss.
    """
    if 'lam' in targets:
        # Mixed batch (Fold 1 MixUp / Fold 2 CutMix)
        lam    = targets['lam']
        y_ord  = targets['ordinal']
        y_perm = targets['ordinal_perm']
        y_bin  = targets['binary']
        y_reg  = targets['regression']

        loss_ord1 = corn_loss(outputs['ordinal_logits'], y_ord,  num_classes=5, label_smoothing=label_smoothing)
        loss_ord2 = corn_loss(outputs['ordinal_logits'], y_perm, num_classes=5, label_smoothing=label_smoothing)
        l_ord   = lam * loss_ord1 + (1 - lam) * loss_ord2
        l_bin   = focal_loss_binary(outputs['binary_logits'], y_bin)
        l_reg   = criterion.huber_loss(outputs['regression_score'], y_reg)
        l_qwk   = criterion.qwk_loss(outputs['ordinal_logits'], y_ord)
        probs   = torch.sigmoid(outputs['ordinal_logits'])
        l_cons  = torch.nn.functional.huber_loss(torch.sum(probs, dim=1), outputs['regression_score'], delta=1.0)
        return l_ord + l_bin + l_reg + (lambda_qwk * l_qwk) + (lambda_consistency * l_cons)
    else:
        # Standard batch
        loss, _ = criterion(outputs, targets, lambda_qwk=lambda_qwk,
                            lambda_consistency=lambda_consistency, label_smoothing=label_smoothing)
        return loss


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, ema, device, lambda_qwk, lambda_consistency, grad_accum_steps=1, max_norm=1.0, stabilize=True, label_smoothing=0.0, epoch=None):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(dataloader, desc="Training")
    last_monitor_time = time.time()
    last_monitor_step = 0
    
    for step, (images, targets) in enumerate(pbar):
        # Optimized GPU Transfer
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
        
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images, epoch=epoch)
            loss = _compute_loss(outputs, targets, criterion, lambda_qwk, lambda_consistency, label_smoothing=label_smoothing)
        
        # Skip unstable batches (Bug Fix 2)
        if stabilize and (torch.isnan(loss) or torch.isinf(loss)):
            print("Skipping unstable batch")
            optimizer.zero_grad(set_to_none=True)
            continue
            
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            # Always unscale before step (required for AMP). Clip only when stabilize=True
            scaler.unscale_(optimizer)
            if stabilize:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Thermal Protection: Mandatory cooling pause (Spec) removed            
            optimizer.zero_grad(set_to_none=True)
            
            if ema is not None:
                ema.update()
        
        running_loss += loss.item() * grad_accum_steps
        
        # Speed & VRAM Monitoring — log every 250 steps (reduced from 100 to cut console I/O)
        if (step + 1) % 250 == 0 or (step + 1) == 1:
            current_time = time.time()
            steps_since_last = (step + 1) - last_monitor_step
            s_per_it = (current_time - last_monitor_time) / steps_since_last if steps_since_last > 0 else 0
            vram_gb = torch.cuda.memory_allocated() / (1024**3)
            pbar.set_postfix(loss=loss.item() * grad_accum_steps, speed=f"{s_per_it:.2f}s/it", vram=f"{vram_gb:.1f}GB")
            if (step + 1) % 250 == 0:
                pbar.write(f"[Monitor] Step {step+1}: {s_per_it:.2f} s/it | VRAM: {vram_gb:.2f} GB | Loss: {loss.item() * grad_accum_steps:.4f}")
            last_monitor_time = current_time
            last_monitor_step = step + 1
        else:
            pbar.set_postfix(loss=loss.item() * grad_accum_steps)
        
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device, lambda_qwk, lambda_consistency, epoch=None):
    model.eval()
    running_loss = 0.0
    
    y_true_ord = []
    y_pred_ord = []
    y_true_bin = []
    y_pred_bin = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
            
            with autocast(device_type="cuda"):
                outputs = model(images, epoch=epoch)
                loss, _ = criterion(outputs, targets, lambda_qwk=lambda_qwk, lambda_consistency=lambda_consistency)
                running_loss += loss.item()
                
            logits = outputs['ordinal_logits']
            s = torch.sigmoid(logits)
            p0 = 1.0 - s[:, 0]
            p1 = s[:, 0] * (1.0 - s[:, 1])
            p2 = s[:, 0] * s[:, 1] * (1.0 - s[:, 2])
            p3 = s[:, 0] * s[:, 1] * s[:, 2] * (1.0 - s[:, 3])
            p4 = s[:, 0] * s[:, 1] * s[:, 2] * s[:, 3]
            p = torch.stack([p0, p1, p2, p3, p4], dim=1)
            preds = torch.argmax(p, dim=1)
            
            y_pred_ord.extend(preds.cpu().numpy())
            y_true_ord.extend(targets['ordinal'].cpu().numpy())
            
            bin_preds = (torch.sigmoid(outputs['binary_logits']) > 0.5).float()
            y_pred_bin.extend(bin_preds.cpu().numpy())
            y_true_bin.extend(targets['binary'].cpu().numpy())
            
    metrics = calculate_metrics(y_true_ord, y_pred_ord, y_true_bin, y_pred_bin)
    metrics['val_loss'] = running_loss / len(dataloader)
    
    return metrics

def get_optimizer(model, model_name):
    if 'efficientnet' in model_name:
        params = [
            {'params': model.backbone.parameters(), 'lr': 5.0e-6},   # Backbone: slow & stable
            {'params': model.frg.parameters(), 'lr': 2.0e-5},        # FRG: normal
            {'params': model.cbam.parameters(), 'lr': 2.0e-5},       # CBAM: normal
            {'params': model.head.parameters(), 'lr': 2.0e-5},       # Heads: normal
        ]
    elif 'convnext' in model_name:
        params = [
            {'params': model.backbone.parameters(), 'lr': 8.0e-6},   # Backbone: slow
            {'params': model.frg.parameters(), 'lr': 2.5e-5},        # FRG: normal
            {'params': model.cbam.parameters(), 'lr': 2.5e-5},       # CBAM: normal
            {'params': model.head.parameters(), 'lr': 2.5e-5},       # Heads: normal
        ]
    else:
        params = [
            {'params': model.backbone.parameters(), 'lr': 7.0e-6},
            {'params': model.frg.parameters(), 'lr': 2.2e-5},
            {'params': model.cbam.parameters(), 'lr': 2.2e-5},
            {'params': model.head.parameters(), 'lr': 2.2e-5},
        ]
    return AdamW(params, weight_decay=1e-4)

def load_checkpoint_safely(ckpt_path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    is_dict = isinstance(ckpt, dict) and 'model_state_dict' in ckpt
    if is_dict:
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if optimizer and 'optimizer_state_dict' in ckpt:
            try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e: print(f"    Skipped optimizer state restore: {e}")
        if scheduler and 'scheduler_state_dict' in ckpt:
            try: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception: pass
        if scaler and 'scaler_state_dict' in ckpt:
            try: scaler.load_state_dict(ckpt['scaler_state_dict'])
            except Exception: pass
    else:
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
    return missing, unexpected, is_dict


def run_curriculum_training(df, img_dir, fold_num, model_name='convnextv2_large', epochs=25, save_dir='results', stabilize=True):
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(f"{save_dir}/fold_{fold_num}", exist_ok=True)
    save_path = f"{save_dir}/fold_{fold_num}/{model_name}_best_ema.pth"
    log_path = f"{save_dir}/fold_{fold_num}/training_log.csv"
    
    df_train = df[df['fold'] != fold_num].reset_index(drop=True)
    df_val = df[df['fold'] == fold_num].reset_index(drop=True)
    
    model = SOTA_DR_Model(model_name=model_name, pretrained=True, drop_path_rate=0.3).to(device)
    model.backbone.set_grad_checkpointing(enable=False)  # Disabled at 512/768px for speed; enabled at 1024px
    ema = EMA(model, decay=0.999)  # Faster adaptation
    criterion = MultiTaskLoss().to(device)
    scaler = torch.amp.GradScaler(device="cuda")
    
    best_qwk = -1.0
    log_data = []
    start_epoch = 1
    patience_counter = 0  # Early Stopping Counter
    early_stopping_patience = 10
    
    # Pre-initialize optimizer, scheduler, and scaler so their states can be resumed
    optimizer = get_optimizer(model, model_name)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    loaded_from_dict = False

    # Auto-Resume Persistence Logic
    latest_ckpt = f"{save_dir}/fold_{fold_num}/{model_name}_latest.pth"
    best_ckpt   = f"{save_dir}/fold_{fold_num}/{model_name}_best_ema.pth"
    if os.path.exists(latest_ckpt):
        if os.path.exists(log_path):
            df_log = pd.read_csv(log_path)
            if len(df_log) > 0:
                last_qwk = float(df_log.iloc[-1]['qwk'])
                # ===== EMA CONTAMINATION GUARD =====
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if last_qwk < 0.05 and os.path.exists(best_ckpt):
                    print(f"!!! COLLAPSE DETECTED in log (last QWK={last_qwk:.4f}). Loading best_ema.pth instead of latest.pth !!!")
                    missing, unexpected, loaded_from_dict = load_checkpoint_safely(best_ckpt, model, optimizer, scheduler, scaler)
                    if unexpected: print(f"    Skipped checkpoint keys (architecture change): {len(unexpected)}")
                    df_log = df_log[df_log['qwk'] > 0.05].reset_index(drop=True)
                else:
                    print(f">>> Persistence: Resuming from {latest_ckpt} <<<")
                    missing, unexpected, loaded_from_dict = load_checkpoint_safely(latest_ckpt, model, optimizer, scheduler, scaler)
                    if unexpected: print(f"    Skipped checkpoint keys (architecture change): {len(unexpected)}")

                if len(df_log) > 0:
                    start_epoch = int(df_log['epoch'].max()) + 1
                    best_qwk = float(df_log['qwk'].max())
                    log_data = df_log.to_dict('records')
                    df_log.to_csv(log_path, index=False)  # re-save cleaned log
        else:
            print(f">>> Persistence: Resuming from {latest_ckpt} <<<")
            missing, unexpected, loaded_from_dict = load_checkpoint_safely(latest_ckpt, model, optimizer, scheduler, scaler)
            if unexpected: print(f"    Skipped checkpoint keys (architecture change): {len(unexpected)}")
        print(f"Resuming at Epoch {start_epoch} with previous Best QWK: {best_qwk:.4f}")
    # (Start epoch override removed to allow natural curriculum flow)

    batch_size = 16   # 512px
    grad_accum = 2    
    current_size = 512 

    if start_epoch > 6 and start_epoch <= 16:  # 768px (Stage 2: epochs 7-16)
        current_size = 768
        batch_size = 12
        grad_accum = 2
        model.backbone.set_grad_checkpointing(enable=True)  # ENABLED NOW TO FIX TIMM GRN 45GB OOM BUG
    elif start_epoch > 16:  # 1024px (Stage 3: epochs 17-25)
        current_size = 1024
        batch_size = 8   # Effective BS = 8×3 = 24
        grad_accum = 3
        model.backbone.set_grad_checkpointing(enable=True)  # Enable only at 1024px (VRAM tight)

    print(f"\n--- Starting Fold {fold_num} | Model {model_name} ---")
    
    train_loader, val_loader = get_dataloaders(df_train, df_val, get_img_dir(img_dir, current_size), fold_num, current_size, batch_size)
    
    # Catch up scheduler if resuming and it wasn't natively loaded from a dict checkpoint
    if start_epoch > 1 and not loaded_from_dict:
        for _ in range(start_epoch - 1):
            scheduler.step()
    
    # print("Performing pre-flight consistency check...")
    # validate_consistency(model, val_loader, device)

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Curriculum: Dynamic Lambdas — lowered to prevent QWK pressure overpowering stability
        lamb_qwk  = 0.0  if epoch <= 10 else 0.05  # Active from epoch 11
        lamb_cons = 0.01 if epoch <= 10 else 0.03  # Increased from epoch 11
        
        # EMA active from epoch 3 (aligned with unfreeze)
        current_ema = ema if epoch >= 3 else None
        
        # Stage Transitions
        if epoch == 7:  # Transition to Stage 2
            print(">>> Scaling to Stage 2: 768px Resolution <<<")
            current_size = 768
            batch_size = 12  # Reduced from 14 to avoid OOM
            grad_accum = 2
            
            train_loader, val_loader = get_dataloaders(df_train, df_val, get_img_dir(img_dir, current_size), fold_num, current_size, batch_size)
            
            # Update learning rates WITHOUT recreating optimizer (preserves Adam momentum)
            # Stage 2 LR = same as Stage 1 per spec (no raise at 768px)
            if 'efficientnet' in model_name:
                optimizer.param_groups[0]['lr'] = 5.0e-6   # Backbone stays slow
                optimizer.param_groups[1]['lr'] = 2.0e-5   # Heads
                optimizer.param_groups[2]['lr'] = 2.0e-5   # CBAM
                optimizer.param_groups[3]['lr'] = 2.0e-5   # FRG
            elif 'convnext' in model_name:
                optimizer.param_groups[0]['lr'] = 8.0e-6  # Per spec: same as Stage 1
                optimizer.param_groups[1]['lr'] = 2.5e-5  # Per spec: same as Stage 1
                optimizer.param_groups[2]['lr'] = 2.5e-5
                optimizer.param_groups[3]['lr'] = 2.5e-5
            else:
                optimizer.param_groups[0]['lr'] = 7.0e-6
                optimizer.param_groups[1]['lr'] = 2.2e-5
                optimizer.param_groups[2]['lr'] = 2.2e-5
                optimizer.param_groups[3]['lr'] = 2.2e-5
                
            patience_counter = 0 # Resolution Reset
            
        elif epoch == 17:  # Transition to Stage 3 at Epoch 17 (per spec: Stage 2 = ep 7-16)
            print(">>> Scaling to Stage 3: 1024px Resolution <<<")
            current_size = 1024
            batch_size = 8   # Per spec: BS=8, accum=3 → effective BS=24
            grad_accum = 3
            model.backbone.set_grad_checkpointing(enable=True)  # Enable NOW - VRAM gets tight at 1024px
            
            train_loader, val_loader = get_dataloaders(df_train, df_val, get_img_dir(img_dir, current_size), fold_num, current_size, batch_size)
            
            # Higher LR at 1024px per spec
            if 'efficientnet' in model_name:
                optimizer.param_groups[0]['lr'] = 6.0e-6
                optimizer.param_groups[1]['lr'] = 2.5e-5
                optimizer.param_groups[2]['lr'] = 2.5e-5
                optimizer.param_groups[3]['lr'] = 2.5e-5
            elif 'convnext' in model_name:
                optimizer.param_groups[0]['lr'] = 1.0e-5  # Per spec
                optimizer.param_groups[1]['lr'] = 3.0e-5  # Per spec
                optimizer.param_groups[2]['lr'] = 3.0e-5
                optimizer.param_groups[3]['lr'] = 3.0e-5
            else:
                 optimizer.param_groups[0]['lr'] = 4.0e-6
                 optimizer.param_groups[1]['lr'] = 1.6e-5
                 optimizer.param_groups[2]['lr'] = 1.6e-5
                 optimizer.param_groups[3]['lr'] = 1.6e-5
                
            patience_counter = 0 # Resolution Reset
            
        # Training
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, current_ema, device, 
            lamb_qwk, lamb_cons, grad_accum_steps=grad_accum, stabilize=stabilize,
            label_smoothing=0.15, epoch=epoch
        )
        scheduler.step()  # Bug 20: CosineAnnealingWarmRestarts expects per-call step, not epoch index
        
        # Validation
        val_with_ema = (epoch >= 3)
        if val_with_ema:
            ema.apply_shadow()
        
        metrics = validate(model, val_loader, criterion, device, lamb_qwk, lamb_cons, epoch=epoch)
        
        if val_with_ema:
            ema.restore()
        
        qwk = metrics['qwk']
        sensitivity = metrics['sensitivity']
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f}")
        print(f"Val QWK: {qwk:.4f} | Acc: {metrics['acc']:.4f} | Sens: {sensitivity:.4f} | Spec: {metrics['specificity']:.4f}")

        # === AUTO COLLAPSE GUARD DISABLED ===
        # (Commented out to prevent LR death spiral - let training continue naturally)
        # if sensitivity < 0.10 and metrics['specificity'] > 0.99 and epoch > 5:
        #     print(f"!!! MODE COLLAPSE DETECTED at Epoch {epoch} !!!")
        #     print(f"!!! Auto-reverting to best checkpoint: {save_path} !!!")
        #     if os.path.exists(save_path):
        #         # strict=False: gracefully skips BatchNorm-specific keys if architecture changed
        #         model.load_state_dict(torch.load(save_path, map_location=device), strict=False)
        #         ema = EMA(model, decay=0.9999)  # Reset EMA to match reverted weights
        #         new_lr = optimizer.param_groups[0]['lr'] * 0.5
        #         for g in optimizer.param_groups:
        #             g['lr'] = new_lr
        #         print(f"!!! LR halved to {new_lr:.2e} after collapse revert !!!")
        #     else:
        #         print("!!! No best checkpoint to revert to. Continuing. !!!")
        
        def _get_state_dict(model_state):
            return {
                'epoch': epoch,
                'best_qwk': best_qwk,
                'qwk': qwk,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }

        # Checkpoint — EMA from epoch >= 3
        if qwk > best_qwk:
            print(f"+++ New best QWK! Saving to {save_path} +++")
            best_qwk = qwk
            if epoch >= 3:
                ema.apply_shadow()
                torch.save(_get_state_dict(model.state_dict()), save_path)
                ema.restore()
            else:
                torch.save(_get_state_dict(model.state_dict()), save_path)
                
        # Epoch EMA save — keep only last 3
        epoch_save_path = f"{save_dir}/fold_{fold_num}/{model_name}_epoch_{epoch}_ema.pth"
        if epoch >= 3:
            ema.apply_shadow()
            torch.save(_get_state_dict(model.state_dict()), epoch_save_path)
            ema.restore()
        else:
            torch.save(_get_state_dict(model.state_dict()), epoch_save_path)
        # Delete checkpoint from 3 epochs ago
        old_epoch_path = f"{save_dir}/fold_{fold_num}/{model_name}_epoch_{epoch - 3}_ema.pth"
        if os.path.exists(old_epoch_path):
            os.remove(old_epoch_path)
                
        # Latest weights
        torch.save(_get_state_dict(model.state_dict()), f"{save_dir}/fold_{fold_num}/{model_name}_latest.pth")
                
        # Logging
        log_data.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': metrics['val_loss'],
            'qwk': qwk,
            'accuracy': metrics['acc'],
            'macro_f1': metrics['macro_f1'],
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity']
        })
        pd.DataFrame(log_data).to_csv(log_path, index=False)

        # Early Stopping DISABLED — Model must complete all 25 epochs
        # (Oscillations at 1024px are normal; forced completion maximises final QWK)

        # GPU Cooling Phase REMOVED — A4000 has active cooling, no thermal throttle observed
            
    return best_qwk
