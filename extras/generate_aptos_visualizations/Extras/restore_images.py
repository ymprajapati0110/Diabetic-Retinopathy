import os
import pandas as pd

def restore_images():
    base_dir = r'D:\Major_project_shiro_final'
    csv_15 = os.path.join(base_dir, r'archive\labels\trainLabels15.csv')
    csv_19 = os.path.join(base_dir, r'archive\labels\trainLabels19.csv')
    final_csv = os.path.join(base_dir, 'trainLabels.csv')
    log_file = os.path.join(base_dir, 'invalid_images.log')
    
    # 1. Get the list of the 205 images we need to restore
    # These are the ones we just removed via filter_vessels.py
    # We will look at the last 205 lines of invalid_images.log
    
    images_to_restore = set()
    try:
        df_log = pd.read_csv(log_file, names=['image', 'reason', 'path'])
        
        # We only want to restore the images filtered by "filter_vessels.py" 
        # which have reasons like "Severe Blur...", "No Vessel Structure...", "Severe Artifact..."
        vessel_reasons = df_log[df_log['reason'].str.contains('Blur|Vessel|Artifact|Reflection', na=False)]
        images_to_restore = set(vessel_reasons['image'].astype(str))
        print(f"Found {len(images_to_restore)} images to restore based on vessel filter reasons.")
        
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    if not images_to_restore:
        print("No images found to restore.")
        return

    # 2. Extract their original rows from the source CSVs
    all_metadata = []
    
    if os.path.exists(csv_15):
        df15 = pd.read_csv(csv_15)
        df15 = df15[['image', 'level']]
        df15['image'] = df15['image'].astype(str)
        restore_15 = df15[df15['image'].isin(images_to_restore)]
        all_metadata.append(restore_15)
        
    if os.path.exists(csv_19):
        df19 = pd.read_csv(csv_19)
        if 'id_code' in df19.columns:
            df19 = df19.rename(columns={'id_code': 'image', 'diagnosis': 'level'})
        df19 = df19[['image', 'level']]
        df19['image'] = df19['image'].astype(str)
        restore_19 = df19[df19['image'].isin(images_to_restore)]
        all_metadata.append(restore_19)
        
    df_to_restore = pd.concat(all_metadata) if all_metadata else pd.DataFrame()
    print(f"Recovered {len(df_to_restore)} rows from original CSVs.")

    # 3. Append them back to trainLabels.csv
    if os.path.exists(final_csv):
        df_current = pd.read_csv(final_csv)
        df_current['image'] = df_current['image'].astype(str)
        
        # Ensure we don't add duplicates
        images_already_in_csv = set(df_current['image'])
        df_to_restore_clean = df_to_restore[~df_to_restore['image'].isin(images_already_in_csv)]
        
        df_final = pd.concat([df_current, df_to_restore_clean])
        df_final.to_csv(final_csv, index=False)
        print(f"Restored {len(df_to_restore_clean)} images back into {final_csv}")
        print(f"Total images in {final_csv} is now: {len(df_final)}")
        
    # 4. Remove them from the invalid_images.log so we don't drop them again later
    try:
        # Keep rows that are NOT in our restore list
        df_log_clean = df_log[~df_log['image'].astype(str).isin(images_to_restore)]
        df_log_clean.to_csv(log_file, index=False, header=False)
        print("Cleaned restored images from invalid_images.log")
    except Exception as e:
        print(f"Error cleaning log file: {e}")

if __name__ == "__main__":
    restore_images()
