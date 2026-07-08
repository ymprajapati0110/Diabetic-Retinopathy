import os
import pandas as pd
import cv2
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from src.data.preprocess import preprocess_image

def is_valid_image(img):
    """
    Strong filtering to remove problematic images (stitched, partial retina, etc.)
    """
    if img is None:
        return False, "Corrupted"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Too dark
    if gray.mean() < 15:
        return False, f"Too dark (Mean={gray.mean():.2f})"
        
    # Too low contrast
    if gray.std() < 8:
        return False, f"Low contrast (Std={gray.std():.2f})"
        
    # Check retina area
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return False, "No contours found"
        
    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (img.shape[0]*img.shape[1])
    
    # Reject broken/partial retina
    if area_ratio < 0.3:
        return False, f"Partial retina (Area Ratio={area_ratio:.2f})"
        
    return True, "Valid"

def process_image_at_resolutions(args):
    img_id, src_path, dst_dirs, sizes = args
    if not os.path.exists(src_path):
        return
    
    try:
        # Read and validate once per image
        img_raw = cv2.imread(src_path)
        is_valid, reason = is_valid_image(img_raw)
        
        if not is_valid:
            with open('invalid_images.log', 'a') as f:
                f.write(f"{img_id},{reason},{src_path}\n")
            
            # Save the invalid image to an invalid folder for manual review
            invalid_dir = os.path.join(os.path.dirname(dst_dirs[0]), 'invalid')
            os.makedirs(invalid_dir, exist_ok=True)
            invalid_path = os.path.join(invalid_dir, f"{img_id}.jpg")
            if not os.path.exists(invalid_path):
                cv2.imwrite(invalid_path, img_raw)
            return

        # Process and save for each resolution
        for size, dst_dir in zip(sizes, dst_dirs):
            dst_path = os.path.join(dst_dir, f"{img_id}.jpg")
            if os.path.exists(dst_path):
                continue
            
            # Use path-based preprocess_image
            img_processed = preprocess_image(src_path, output_size=size)
            if img_processed is not None:
                cv2.imwrite(dst_path, img_processed)
                
    except Exception as e:
        pass

def main():
    # Paths
    csv_15 = r'D:\Major_project_shiro_final\archive\labels\trainLabels15.csv'
    csv_19 = r'D:\Major_project_shiro_final\archive\labels\trainLabels19.csv'
    src_dir_15 = r'D:\Major_project_shiro_final\archive\resized train 15'
    src_dir_19 = r'D:\Major_project_shiro_final\archive\resized train 19'
    
    # SAVE IN CURRENT FOLDER
    dst_base = r'D:\Major_project_shiro_final'
    final_csv = os.path.join(dst_base, 'trainLabels.csv')
    
    sizes = [512, 768, 1024]
    dst_dirs = [os.path.join(dst_base, f'preprocessed_{s}') for s in sizes]
    
    for d in dst_dirs:
        os.makedirs(d, exist_ok=True)
    
    all_metadata = []
    tasks = []
    
    # Load 2015
    if os.path.exists(csv_15):
        print("Loading 2015 labels...")
        df15 = pd.read_csv(csv_15)
        df15 = df15[['image', 'level']]
        all_metadata.append(df15)
        for img_id in df15['image'].unique():
            img_id_str = str(img_id).strip()
            src_path = os.path.join(src_dir_15, f"{img_id_str}.jpeg")
            if not os.path.exists(src_path):
                src_path = os.path.join(src_dir_15, f"{img_id_str}.jpg")
            tasks.append((img_id_str, src_path, dst_dirs, sizes))
            
    # Load 2019
    if os.path.exists(csv_19):
        print("Loading 2019 labels...")
        df19 = pd.read_csv(csv_19)
        if 'id_code' in df19.columns:
            df19 = df19.rename(columns={'id_code': 'image', 'diagnosis': 'level'})
        df19 = df19[['image', 'level']]
        all_metadata.append(df19)
        for img_id in df19['image'].unique():
            img_id_str = str(img_id).strip()
            src_path = os.path.join(src_dir_19, f"{img_id_str}.png")
            if not os.path.exists(src_path):
                src_path = os.path.join(src_dir_19, f"{img_id_str}.jpg")
            tasks.append((img_id_str, src_path, dst_dirs, sizes))

    # Save merged metadata
    if all_metadata:
        merged_df = pd.concat(all_metadata)
        merged_df.to_csv(final_csv, index=False)
        print(f"Merged metadata saved to {final_csv}")

    print(f"Starting distribution: {len(tasks)} images @ {sizes} resolutions")
    
    if os.path.exists('invalid_images.log'):
        os.remove('invalid_images.log')
        
    num_procs = max(1, os.cpu_count() - 1)
    with Pool(num_procs) as p:
        list(tqdm(p.imap_unordered(process_image_at_resolutions, tasks), total=len(tasks)))
        
    # Clean up trainLabels.csv by removing invalid images
    if os.path.exists('invalid_images.log') and os.path.exists(final_csv):
        print("\nCleaning up trainLabels.csv...")
        try:
            invalid_df = pd.read_csv('invalid_images.log', names=['image', 'reason', 'path'])
            invalid_images = invalid_df['image'].astype(str).tolist()
            
            df = pd.read_csv(final_csv)
            original_len = len(df)
            df = df[~df['image'].astype(str).isin(invalid_images)]
            df.to_csv(final_csv, index=False)
            
            removed_count = original_len - len(df)
            print(f"Removed {removed_count} invalid images from {final_csv}")
        except Exception as e:
            print(f"Error cleaning up CSV: {e}")

    print("\nPreprocessing Mission Complete.")

if __name__ == "__main__":
    main()
