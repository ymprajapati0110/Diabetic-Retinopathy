import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
import cv2
from src.models.sota_dr_model import SOTA_DR_Model
from src.evaluation.gradcam import GradCAMHooker, generate_masked_gradcam
import torch
import glob
from src.data.dataset import get_validation_augmentations

def generate_confusion_matrix(df):
    cm = confusion_matrix(df['true_grade'], df['predicted_grade'], labels=[0,1,2,3,4])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.ylabel('True Grade')
    plt.xlabel('Predicted Grade')
    plt.title('Confusion Matrix')
    plt.savefig('d:/Major_project_shiro_final/results/fold_3/confusion_matrix.png', bbox_inches='tight')
    plt.close()

def generate_roc_curves(df):
    plt.figure(figsize=(10, 8))
    for i in range(5):
        y_true = (df['true_grade'] == i).astype(int)
        y_prob = df[f'prob_{i}']
        if y_true.sum() == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Grade {i} (AUC = {roc_auc:.3f})')
        except ValueError:
            pass
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('d:/Major_project_shiro_final/results/fold_3/roc_curves.png', bbox_inches='tight')
    plt.close()

def generate_pr_curves(df):
    plt.figure(figsize=(10, 8))
    for i in range(5):
        y_true = (df['true_grade'] == i).astype(int)
        y_prob = df[f'prob_{i}']
        if y_true.sum() == 0:
            continue
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            plt.plot(recall, precision, label=f'Grade {i} (AP = {ap:.3f})')
        except ValueError:
            pass
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.savefig('d:/Major_project_shiro_final/results/fold_3/precision_recall_curves.png', bbox_inches='tight')
    plt.close()

def generate_training_curves():
    log_path = 'd:/Major_project_shiro_final/results/fold_3/training_log.csv'
    if not os.path.exists(log_path): return
    df = pd.read_csv(log_path)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    # Loss
    axes[0].plot(df['epoch'], df['train_loss'], label='Train Loss')
    axes[0].plot(df['epoch'], df['val_loss'], label='Val Loss')
    axes[0].axvline(x=7, color='r', linestyle='--', label='768px')
    axes[0].axvline(x=17, color='g', linestyle='--', label='1024px')
    axes[0].set_title('Loss vs Epoch')
    axes[0].legend()
    
    # QWK
    axes[1].plot(df['epoch'], df['qwk'], label='Val QWK')
    best_idx = df['qwk'].idxmax()
    axes[1].scatter(df['epoch'][best_idx], df['qwk'][best_idx], color='red')
    axes[1].axhline(y=0.90, color='grey', linestyle='--', label='Target 0.90')
    axes[1].set_title('QWK vs Epoch')
    axes[1].legend()
    
    # Sens/Spec
    axes[2].plot(df['epoch'], df['sensitivity'], label='Sensitivity')
    axes[2].plot(df['epoch'], df['specificity'], label='Specificity')
    axes[2].set_title('Sensitivity/Specificity vs Epoch')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('d:/Major_project_shiro_final/results/fold_3/training_curves.png')
    plt.close()

def error_analysis(df):
    os.makedirs('d:/Major_project_shiro_final/results/fold_3/error_analysis', exist_ok=True)
    df['error_mag'] = np.abs(df['true_grade'] - df['predicted_grade'])
    worst = df.sort_values(by='error_mag', ascending=False).head(20)
    
    transform = get_validation_augmentations()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SOTA_DR_Model('convnextv2_large', pretrained=False)
    checkpoint = torch.load('d:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_25_ema.pth', map_location=device)
    model.load_state_dict(checkpoint.get('ema_state_dict', checkpoint.get('model_state_dict', checkpoint)))
    model.to(device)
    model.eval()
    
    hooker = GradCAMHooker(model)
    
    for idx, row in worst.iterrows():
        img_id = row['image_id']
        path = glob.glob(f'd:/Major_project_shiro_final/archive/resized test */{img_id}.*')
        if not path: continue
        
        img_bgr = cv2.imread(path[0])
        from run_deployment import custom_preprocess
        img_rgb = custom_preprocess(img_bgr)
        tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(device)
        
        heatmap_bgr = generate_masked_gradcam(hooker, tensor, img_bgr)
        
        out_name = f"d:/Major_project_shiro_final/results/fold_3/error_analysis/rank{idx}_true{row['true_grade']}_pred{row['predicted_grade']}.jpg"
        cv2.imwrite(out_name, heatmap_bgr)

def generate_report(df):
    qwk = cohen_kappa_score(df['true_grade'], df['predicted_grade'], weights='quadratic')
    report = classification_report(df['true_grade'], df['predicted_grade'], output_dict=True)
    
    # Save CSV
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv('d:/Major_project_shiro_final/results/fold_3/per_class_metrics.csv')
    
    with open('d:/Major_project_shiro_final/results/fold_3/FINAL_RESULTS.md', 'w') as f:
        f.write("# Fold 3 Final Results\n\n")
        f.write("## Performance Summary\n")
        f.write(f"- Test QWK: {qwk:.4f}\n")
        f.write(f"- Macro F1: {report['macro avg']['f1-score']:.4f}\n")

if __name__ == '__main__':
    df = pd.read_csv('d:/Major_project_shiro_final/results/fold_3/test_predictions_fast.csv')
    generate_confusion_matrix(df)
    generate_roc_curves(df)
    generate_pr_curves(df)
    generate_training_curves()
    generate_report(df)
    error_analysis(df)
