import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report

out_file = 'd:/Major_project_shiro_final/results/fold_3/Fold_3_Final_Performance_Report.md'

with open(out_file, 'w') as f:
    f.write('# Fold 3 Final Performance Report\n\n')

    # 1. Training Metrics
    df_train = pd.read_csv('d:/Major_project_shiro_final/results/fold_3/training_log.csv')
    best_epoch = df_train.loc[df_train['qwk'].idxmax()]
    f.write('## 1. Validation (During Training)\n')
    f.write(f"- Best Epoch: {int(best_epoch['epoch'])}\n")
    f.write(f"- Val QWK: {best_epoch['qwk']:.4f}\n")
    f.write(f"- Val Loss: {best_epoch['val_loss']:.4f}\n\n")

    # 2. EyePACS Test Set (53,576 images)
    f.write('## 2. Test Set Evaluation (EyePACS 2015 - 53,576 images)\n')
    df_ep = pd.read_csv('d:/Major_project_shiro_final/results/fold_3/test_results/predictions_final.csv')
    y_true_ep = df_ep['true_grade'].values
    y_pred_ep = df_ep['predicted_grade'].values
    qwk_ep = cohen_kappa_score(y_true_ep, y_pred_ep, weights='quadratic')
    acc_ep = accuracy_score(y_true_ep, y_pred_ep)
    f.write(f'- QWK: {qwk_ep:.4f}\n')
    f.write(f'- Accuracy: {acc_ep:.4f}\n\n')
    f.write('### Classification Report (EyePACS)\n```text\n')
    f.write(classification_report(y_true_ep, y_pred_ep, digits=4))
    f.write('\n```\n\n')

    # 3. APTOS Test Set (3,662 images)
    f.write('## 3. Test Set Evaluation (APTOS 2019 - 3,662 images)\n')
    df_ap = pd.read_csv('d:/Major_project_shiro_final/results/fold_3/aptos_results/predictions_aptos.csv')
    y_true_ap = df_ap['true_grade'].values
    y_pred_ap = df_ap['predicted_grade'].values
    qwk_ap = cohen_kappa_score(y_true_ap, y_pred_ap, weights='quadratic')
    acc_ap = accuracy_score(y_true_ap, y_pred_ap)
    f.write(f'- QWK: {qwk_ap:.4f}\n')
    f.write(f'- Accuracy: {acc_ap:.4f}\n\n')
    f.write('### Classification Report (APTOS)\n```text\n')
    f.write(classification_report(y_true_ap, y_pred_ap, digits=4))
    f.write('\n```\n')

print(f'Report saved to {out_file}')
