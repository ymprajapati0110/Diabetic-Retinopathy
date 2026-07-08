import os
import pandas as pd
from src.evaluation.evaluate_final import FinalInferencer

def main():
    print("Loading FinalInferencer for APTOS error analysis...")
    best_checkpoint_path = 'd:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_25_ema.pth'
    inferencer = FinalInferencer([best_checkpoint_path], target_resolution=1024)

    # Note: Using the trained validation thresholds saved from earlier.
    import json
    with open('d:/Major_project_shiro_final/results/fold_3/fold_3_thresholds.json', 'r') as f:
        saved_thresholds = json.load(f)['thresholds']
    import numpy as np
    inferencer.optimizer.coef_ = np.array(saved_thresholds)

    # Find worst predictions for APTOS
    df_results = pd.read_csv('d:/Major_project_shiro_final/results/fold_3/aptos_results/predictions_aptos.csv')
    df_results['error'] = abs(df_results['true_grade'] - df_results['predicted_grade'])
    df_results['confidence'] = df_results[['prob_0', 'prob_1', 'prob_2', 'prob_3', 'prob_4']].max(axis=1)

    # Top 20 worst errors
    worst_errors = df_results.nlargest(20, 'error')

    error_dir = 'd:/Major_project_shiro_final/results/fold_3/aptos_results/error_analysis/'
    os.makedirs(error_dir, exist_ok=True)

    print("Generating Masked GradCAM for top 20 worst errors in APTOS...")
    for idx, row in worst_errors.iterrows():
        img_id = row['id_code'] # APTOS uses id_code
        
        # APTOS 2019 test images are usually from the train set of 2019, which are in `archive/resized train 19`
        img_path = f"d:/Major_project_shiro_final/archive/resized train 19/{img_id}.jpg"
        if not os.path.exists(img_path):
            img_path = f"d:/Major_project_shiro_final/archive/resized train 19/{img_id}.png"
            
        if not os.path.exists(img_path):
            print(f"Skipping {img_id}: Image not found at {img_path}.")
            continue
            
        save_path = f"{error_dir}/{img_id}_true{row['true_grade']}_pred{row['predicted_grade']}.jpg"
        
        # Generate GradCAM with masked version
        report = inferencer.generate_clinical_report(img_path, save_path)
        
    print(f"-> Saved {len(worst_errors)} worst error cases with GradCAM visualizations")

    # Analyze confusion patterns
    print("\nMost Confused Class Pairs in APTOS:")
    for i in range(5):
        for j in range(5):
            if i != j:
                count = len(df_results[(df_results['true_grade']==i) & (df_results['predicted_grade']==j)])
                if count > 10:
                    print(f"  Grade {i} -> {j}: {count} cases")

if __name__ == '__main__':
    main()
