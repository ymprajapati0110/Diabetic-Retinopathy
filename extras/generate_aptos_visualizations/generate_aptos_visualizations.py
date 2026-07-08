import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support, 
    classification_report
)

def main():
    base_dir = 'd:/Major_project_shiro_final/results/fold_3'
    aptos_dir = os.path.join(base_dir, 'aptos_results')
    
    os.makedirs(os.path.join(aptos_dir, 'confusion_matrices'), exist_ok=True)
    os.makedirs(os.path.join(aptos_dir, 'visualizations'), exist_ok=True)

    # Load results
    df_results = pd.read_csv(os.path.join(aptos_dir, 'predictions_aptos.csv'))
    y_true = df_results['true_grade'].values
    y_pred = df_results['predicted_grade'].values
    y_probs = df_results[['prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4']].values

    # Load training log
    df_train = pd.read_csv(os.path.join(base_dir, 'training_log.csv'))

    #──────────────────────────────────────────────────────────────────────────────
    # VISUALIZATION 1: Confusion Matrix
    #──────────────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Raw counts
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Grade', fontsize=12)
    axes[0].set_xlabel('Predicted Grade', fontsize=12)

    # Normalized by row (Recall)
    cm_recall = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_recall, annot=True, fmt='.2f', cmap='Greens', ax=axes[1],
                xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    axes[1].set_title('Normalized by Row (Recall)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Grade', fontsize=12)
    axes[1].set_xlabel('Predicted Grade', fontsize=12)

    # Normalized by column (Precision)
    cm_precision = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    sns.heatmap(cm_precision, annot=True, fmt='.2f', cmap='Oranges', ax=axes[2],
                xticklabels=[0,1,2,3,4], yticklabels=[0,1,2,3,4])
    axes[2].set_title('Normalized by Column (Precision)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('True Grade', fontsize=12)
    axes[2].set_xlabel('Predicted Grade', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(aptos_dir, 'confusion_matrices', 'confusion_matrix_all.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #──────────────────────────────────────────────────────────────────────────────
    # VISUALIZATION 2: Training Curves (using identical training log)
    #──────────────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss curves
    axes[0, 0].plot(df_train['epoch'], df_train['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(df_train['epoch'], df_train['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].axvline(x=7, color='gray', linestyle='--', alpha=0.5, label='768px Stage')
    axes[0, 0].axvline(x=17, color='gray', linestyle='-.', alpha=0.5, label='1024px Stage')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # QWK curve
    axes[0, 1].plot(df_train['epoch'], df_train['qwk'], 'g-', linewidth=2, marker='o', markersize=4)
    axes[0, 1].axhline(y=0.90, color='red', linestyle='--', alpha=0.7, label='Target (0.90)')
    axes[0, 1].axvline(x=7, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(x=17, color='gray', linestyle='-.', alpha=0.5)
    best_epoch = df_train['qwk'].idxmax() + 1
    best_qwk = df_train['qwk'].max()
    axes[0, 1].scatter([best_epoch], [best_qwk], color='red', s=100, zorder=5, label=f'Best: Ep{best_epoch} (QWK={best_qwk:.4f})')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Quadratic Weighted Kappa', fontsize=12)
    axes[0, 1].set_title('Validation QWK Progress', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Sensitivity & Specificity
    axes[1, 0].plot(df_train['epoch'], df_train['sensitivity'], 'b-', label='Sensitivity', linewidth=2)
    axes[1, 0].plot(df_train['epoch'], df_train['specificity'], 'r-', label='Specificity', linewidth=2)
    axes[1, 0].axvline(x=7, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=17, color='gray', linestyle='-.', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Sensitivity & Specificity', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Accuracy & F1
    axes[1, 1].plot(df_train['epoch'], df_train['accuracy'], 'purple', label='Accuracy', linewidth=2)
    axes[1, 1].plot(df_train['epoch'], df_train['macro_f1'], 'orange', label='Macro F1', linewidth=2)
    axes[1, 1].axvline(x=7, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=17, color='gray', linestyle='-.', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Accuracy & Macro F1', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(aptos_dir, 'visualizations', 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #──────────────────────────────────────────────────────────────────────────────
    # VISUALIZATION 3: ROC Curves (Per-Class)
    #──────────────────────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for grade in range(5):
        y_binary = (y_true == grade).astype(int)
        y_score = y_probs[:, grade]
        
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[grade], lw=2,
                label=f'Grade {grade} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('APTOS ROC Curves - Per-Class', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(aptos_dir, 'visualizations', 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #──────────────────────────────────────────────────────────────────────────────
    # VISUALIZATION 4: Precision-Recall Curves
    #──────────────────────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 8))

    for grade in range(5):
        y_binary = (y_true == grade).astype(int)
        y_score = y_probs[:, grade]
        
        precision, recall, _ = precision_recall_curve(y_binary, y_score)
        ap = average_precision_score(y_binary, y_score)
        
        ax.plot(recall, precision, color=colors[grade], lw=2,
                label=f'Grade {grade} (AP = {ap:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('APTOS Precision-Recall Curves - Per-Class', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(aptos_dir, 'visualizations', 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("-> APTOS visualizations generated successfully!")

    #──────────────────────────────────────────────────────────────────────────────
    # VISUALIZATION 5: Per-Class Metrics Table
    #──────────────────────────────────────────────────────────────────────────────

    report = classification_report(y_true, y_pred, 
                                  target_names=[f'Grade {i}' for i in range(5)],
                                  digits=4, output_dict=True)

    df_metrics = pd.DataFrame(report).T

    specificities = []
    for grade in range(5):
        y_binary_true = (y_true == grade).astype(int)
        y_binary_pred = (y_pred == grade).astype(int)
        cm_bin = confusion_matrix(y_binary_true, y_binary_pred, labels=[0, 1])
        tn, fp, fn, tp = cm_bin.ravel()
        spec = tn / (tn + fp + 1e-6) if (tn + fp) > 0 else 0.0
        specificities.append(spec)

    df_metrics['specificity'] = specificities + [np.nan] * (len(df_metrics) - 5)

    df_metrics.to_csv(os.path.join(aptos_dir, 'per_class_metrics.csv'))

    print("\n-> APTOS Per-class metrics table saved!")
    print("\nMetrics Summary:")
    print(df_metrics[['precision', 'recall', 'f1-score', 'specificity', 'support']].round(3))

if __name__ == '__main__':
    main()
