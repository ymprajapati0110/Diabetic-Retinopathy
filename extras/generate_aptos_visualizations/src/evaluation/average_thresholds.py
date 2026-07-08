import os
import json
import numpy as np

def average_k_fold_thresholds(results_dir='results', num_folds=5, output_file='average_thresholds.json'):
    """
    Reads the optimal thresholds from all completed folds and calculates the median.
    This prevents overfitting to a single fold's validation distribution.
    """
    all_thresholds = []
    
    for fold in range(1, num_folds + 1):
        threshold_file = os.path.join(results_dir, f'fold_{fold}', f'fold_{fold}_thresholds.json')
        if os.path.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                data = json.load(f)
                all_thresholds.append(data['thresholds'])
        else:
            print(f"Warning: Could not find threshold file for fold {fold}: {threshold_file}")
            
    if not all_thresholds:
        print("Error: No threshold files found to average.")
        return None
        
    all_thresholds = np.array(all_thresholds)
    
    # Calculate median across the folds (more robust to outliers than mean)
    median_thresholds = np.median(all_thresholds, axis=0)
    
    output_path = os.path.join(results_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump({'thresholds': median_thresholds.tolist()}, f, indent=4)
        
    print(f"\n--- K-Fold Threshold Averaging Complete ---")
    print(f"Averaged over {len(all_thresholds)} folds.")
    print(f"Median Thresholds: {median_thresholds.tolist()}")
    print(f"Saved to: {output_path}")
    
    return median_thresholds.tolist()

if __name__ == '__main__':
    average_k_fold_thresholds()
