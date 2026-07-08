import os
import pandas as pd
import cv2
from src.data.preprocess import preprocess_image

def restore_severe_images():
    base_dir = r'D:\Major_project_shiro_final'
    csv_15 = os.path.join(base_dir, r'archive\labels\trainLabels15.csv')
    csv_19 = os.path.join(base_dir, r'archive\labels\trainLabels19.csv')
    src_dir_15 = os.path.join(base_dir, r'archive\resized train 15')
    src_dir_19 = os.path.join(base_dir, r'archive\resized train 19')
    final_csv = os.path.join(base_dir, 'trainLabels.csv')
    
    sizes = [512, 768, 1024]
    dst_dirs = [os.path.join(base_dir, f'preprocessed_{s}') for s in sizes]
    
    # Load original dataset to find all missing images
    all_metadata = []
    if os.path.exists(csv_15):
        df15 = pd.read_csv(csv_15)[['image', 'level']]
        all_metadata.append(df15)
        
    if os.path.exists(csv_19):
        df19 = pd.read_csv(csv_19)
        if 'id_code' in df19.columns:
            df19 = df19.rename(columns={'id_code': 'image', 'diagnosis': 'level'})
        df19 = df19[['image', 'level']]
        all_metadata.append(df19)
        
    df_orig = pd.concat(all_metadata) if all_metadata else pd.DataFrame()
    df_orig['image'] = df_orig['image'].astype(str)
    
    # Load current CSV to see what we are missing
    df_current = pd.read_csv(final_csv)
    df_current['image'] = df_current['image'].astype(str)
    
    # Find removed images
    removed_mask = ~df_orig['image'].isin(df_current['image'])
    df_removed = df_orig[removed_mask]
    
    # Target Severities 3 and 4
    df_target = df_removed[(df_removed['level'] == 3) | (df_removed['level'] == 4)]
    images_to_restore = df_target['image'].tolist()
    
    print(f"Found {len(images_to_restore)} severity 3/4 images to forcibly restore.")
    
    restored_count = 0
    # Process each image
    for img_id in images_to_restore:
        # Determine source path checking both 2015/2019 locations and the Invalid folder
        src_path = None
        extensions = ['.jpg', '.jpeg', '.png']
        
        for ext in extensions:
            p15 = os.path.join(src_dir_15, f"{img_id}{ext}")
            p19 = os.path.join(src_dir_19, f"{img_id}{ext}")
            p_invalid = os.path.join(base_dir, 'Invalid', f"{img_id}{ext}")
            
            if os.path.exists(p15):
                src_path = p15
                break
            if os.path.exists(p19):
                src_path = p19
                break
            if os.path.exists(p_invalid):
                src_path = p_invalid
                break
                
        if not src_path:
            print(f"Could not find source image for {img_id}")
            continue
            
        print(f"Processing missing image: {img_id}")
        
        # Bypassing the is_valid_image filter entirely as requested!
        success = True
        for size, dst_dir in zip(sizes, dst_dirs):
            dst_path = os.path.join(dst_dir, f"{img_id}.jpg")
            
            # Using the minimal pipeline logic (resize + light CLAHE)
            img = preprocess_image(src_path, output_size=size)
            if img is not None:
                cv2.imwrite(dst_path, img)
            else:
                # Fallback purely to resize if there's a complete processing failure
                raw = cv2.imread(src_path)
                if raw is not None:
                    raw_resized = cv2.resize(raw, (size, size), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(dst_path, raw_resized)
                else:
                    print(f"FATAL: OpenCV could not read {img_id}")
                    success = False
                    
        if success:
            restored_count += 1
            
    # Add back to CSV
    if restored_count > 0:
        df_restored = df_orig[df_orig['image'].isin(images_to_restore)]
        df_final = pd.concat([df_current, df_restored]).drop_duplicates(subset=['image'])
        df_final.to_csv(final_csv, index=False)
        print(f"\nSuccessfully preprocessed and restored {restored_count} images!")
        print(f"Final trainLabels.csv count: {len(df_final)}")
    else:
        print("\nNo images were restored. Please check paths.")

if __name__ == "__main__":
    restore_severe_images()
