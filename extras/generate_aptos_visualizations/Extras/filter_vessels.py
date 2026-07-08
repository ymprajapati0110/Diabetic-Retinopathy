import os
import cv2
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import shutil

def has_vessel_structure(img_path):
    """
    Evaluates if an image contains learnable signal (vessels/structure).
    Returns (True, "Valid") if signal is present, (False, reason) otherwise.
    """
    img = cv2.imread(img_path)
    if img is None:
        return False, "Failed to load"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Extreme Blur / No Structure Check
    # Laplacian variance measures edge sharpness/focus
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 15: # Highly blurred 
        return False, f"Severe Blur (Var={laplacian_var:.1f})"

    # 2. Vessel / Contrast Check
    # Apply CLAHE to enhance vessels for detection only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Adaptive threshold to find thin structures (vessels)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 2)
    
    # Count vessel pixels inside the retinal area
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    vessel_pixels = cv2.bitwise_and(thresh, mask)
    vessel_ratio = np.sum(vessel_pixels > 0) / (np.sum(mask > 0) + 1e-6)
    
    # Normal retina has ~3-10% vessel pixels. Less than 0.5% means no vessels visible
    if vessel_ratio < 0.005: 
        return False, f"No Vessel Structure (Ratio={vessel_ratio:.4f})"
        
    # 3. Artifact / Washout Check (Too much high-intensity area)
    bright_ratio = np.sum(gray > 220) / (np.sum(mask > 0) + 1e-6)
    if bright_ratio > 0.15:
        return False, f"Severe Artifact/Reflection (Ratio={bright_ratio:.2f})"

    return True, "Valid"

def process_file(args):
    img_id, path = args
    is_valid, reason = has_vessel_structure(path)
    return img_id, path, is_valid, reason

def main():
    base_dir = r'D:\Major_project_shiro_final'
    src_dir = os.path.join(base_dir, 'preprocessed_1024')
    invalid_dir = os.path.join(base_dir, 'invalid_vessels')
    log_file = os.path.join(base_dir, 'invalid_images.log')
    csv_file = os.path.join(base_dir, 'trainLabels.csv')
    
    os.makedirs(invalid_dir, exist_ok=True)
    
    # Collect all existing valid images from the CSV
    df = pd.read_csv(csv_file)
    existing_images = set(df['image'].astype(str))
    
    tasks = []
    for f in os.listdir(src_dir):
        if f.endswith('.jpg'):
            img_id = os.path.splitext(f)[0]
            if img_id in existing_images:
                tasks.append((img_id, os.path.join(src_dir, f)))
                
    print(f"Scanning {len(tasks)} images for vessel structures...")
    
    invalid_results = []
    num_procs = max(1, os.cpu_count() - 1)
    
    with Pool(num_procs) as p:
        for img_id, path, is_valid, reason in tqdm(p.imap_unordered(process_file, tasks), total=len(tasks)):
            if not is_valid:
                invalid_results.append((img_id, reason, path))

    # Log and move invalid images
    print(f"\nFound {len(invalid_results)} newly invalid images.")
    if invalid_results:
        with open(log_file, 'a') as f:
            for img_id, reason, path in invalid_results:
                f.write(f"{img_id},{reason},{path}\n")
                
                # Move to invalid folder for inspection
                dst_path = os.path.join(invalid_dir, f"{img_id}.jpg")
                if not os.path.exists(dst_path):
                    shutil.copy2(path, dst_path)
                    
        # Update CSV
        print("Cleaning trainLabels.csv...")
        invalid_set = {str(item[0]) for item in invalid_results}
        
        # Also ensure previously logged invalid images are removed
        if os.path.exists(log_file):
            try:
                invalid_df = pd.read_csv(log_file, names=['image', 'reason', 'path'])
                invalid_set.update(invalid_df['image'].astype(str))
            except Exception:
                pass
                
        original_len = len(df)
        df = df[~df['image'].astype(str).isin(invalid_set)]
        df.to_csv(csv_file, index=False)
        
        print(f"Removed {original_len - len(df)} total invalid images from {csv_file}")
        
    print("Filtering Pipeline Complete.")

if __name__ == '__main__':
    main()
