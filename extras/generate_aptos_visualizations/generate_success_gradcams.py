import os
import pandas as pd
from src.evaluation.evaluate_final import FinalInferencer
import json
import numpy as np

def generate_success_cams(dataset_name, csv_path, img_dir_list, out_dir):
    print(f"\n--- Generating Success GradCAMs for {dataset_name} ---")
    best_checkpoint_path = 'd:/Major_project_shiro_final/results/fold_3/convnextv2_large_epoch_25_ema.pth'
    inferencer = FinalInferencer([best_checkpoint_path], target_resolution=1024)

    with open('d:/Major_project_shiro_final/results/fold_3/fold_3_thresholds.json', 'r') as f:
        saved_thresholds = json.load(f)['thresholds']
    inferencer.optimizer.coef_ = np.array(saved_thresholds)

    df = pd.read_csv(csv_path)
    
    # Filter for CORRECT predictions
    df_correct = df[df['true_grade'] == df['predicted_grade']].copy()
    
    # Calculate confidence based on the predicted class probability
    def get_confidence(row):
        grade = int(row['predicted_grade'])
        return row[f'prob_{grade}']
        
    df_correct['confidence'] = df_correct.apply(get_confidence, axis=1)
    
    os.makedirs(out_dir, exist_ok=True)
    
    total_saved = 0
    
    # Get top 5 most confident correct predictions for EACH of the 5 grades
    for grade in range(5):
        df_grade = df_correct[df_correct['true_grade'] == grade]
        top_confident = df_grade.nlargest(5, 'confidence')
        
        print(f"Generating for Grade {grade} (Found {len(top_confident)} images)")
        
        for idx, row in top_confident.iterrows():
            img_id = row['id_code'] if 'id_code' in row else row['image_id']
            
            # Try finding the image in the provided directories
            img_path = None
            for d in img_dir_list:
                for ext in ['.jpg', '.jpeg', '.png']:
                    p = os.path.join(d, f"{img_id}{ext}")
                    if os.path.exists(p):
                        img_path = p
                        break
                if img_path:
                    break
                    
            if not img_path:
                print(f"  Skipping {img_id}: Image not found.")
                continue
                
            conf_pct = row['confidence'] * 100
            save_path = f"{out_dir}/Grade{grade}_{img_id}_conf{conf_pct:.1f}.jpg"
            
            inferencer.generate_clinical_report(img_path, save_path)
            total_saved += 1
            
    print(f"-> Saved {total_saved} highly confident correct predictions for {dataset_name}.")


def main():
    # 1. APTOS Success GradCAMs
    aptos_csv = 'd:/Major_project_shiro_final/results/aptos_results/predictions_aptos.csv'
    aptos_img_dirs = ['d:/Major_project_shiro_final/archive/resized train 19']
    aptos_out = 'd:/Major_project_shiro_final/results/aptos_results/success_gradcams/'
    generate_success_cams("APTOS", aptos_csv, aptos_img_dirs, aptos_out)
    
    # 2. EyePACS Success GradCAMs
    eyepacs_csv = 'd:/Major_project_shiro_final/results/test_results/predictions_final.csv'
    eyepacs_img_dirs = [
        'd:/Major_project_shiro_final/archive/resized test 15',
        'd:/Major_project_shiro_final/archive/resized test 19'
    ]
    eyepacs_out = 'd:/Major_project_shiro_final/results/test_results/success_gradcams/'
    generate_success_cams("EyePACS", eyepacs_csv, eyepacs_img_dirs, eyepacs_out)


if __name__ == '__main__':
    main()
