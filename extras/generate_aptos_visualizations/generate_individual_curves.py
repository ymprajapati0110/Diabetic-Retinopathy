import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_curves(fold_num):
    log_path = f'd:/Major_project_shiro_final/results/fold_{fold_num}/training_log.csv'
    if not os.path.exists(log_path):
        print(f"No training log found for fold {fold_num}")
        return

    df = pd.read_csv(log_path)
    out_dir = f'd:/Major_project_shiro_final/results/fold_{fold_num}/'

    # 1. Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Fold {fold_num} - Loss Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    
    # 2. Accuracy Curve
    plt.figure(figsize=(8, 6))
    if 'accuracy' in df.columns:
        plt.plot(df['epoch'], df['accuracy'], 'purple', label='Accuracy', linewidth=2)
    if 'qwk' in df.columns:
        plt.plot(df['epoch'], df['qwk'], 'g-', label='QWK', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Fold {fold_num} - Accuracy & QWK Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_curve.png'), dpi=300)
    plt.close()

    print(f"Generated standalone loss_curve.png and accuracy_curve.png in fold_{fold_num}")

def main():
    generate_curves(3)
    generate_curves(4)

if __name__ == '__main__':
    main()
