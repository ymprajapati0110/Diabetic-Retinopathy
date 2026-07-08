import os
import pandas as pd

def sync_and_report():
    base_dir = r'D:\Major_project_shiro_final'
    csv_15 = os.path.join(base_dir, r'archive\labels\trainLabels15.csv')
    csv_19 = os.path.join(base_dir, r'archive\labels\trainLabels19.csv')
    final_csv = os.path.join(base_dir, 'trainLabels.csv')
    dir_1024 = os.path.join(base_dir, 'preprocessed_1024')
    
    # 1. Load the COMPLETE original dataset for baseline comparison
    all_metadata = []
    
    if os.path.exists(csv_15):
        df15 = pd.read_csv(csv_15)[['image', 'level']]
        df15['image'] = df15['image'].astype(str)
        all_metadata.append(df15)
        
    if os.path.exists(csv_19):
        df19 = pd.read_csv(csv_19)
        if 'id_code' in df19.columns:
            df19 = df19.rename(columns={'id_code': 'image', 'diagnosis': 'level'})
        df19 = df19[['image', 'level']]
        df19['image'] = df19['image'].astype(str)
        all_metadata.append(df19)
        
    df_original = pd.concat(all_metadata) if all_metadata else pd.DataFrame()
    original_counts = df_original['level'].value_counts().sort_index()
    
    print("--- ORIGINAL DATASET SEVERITY COUNTS ---")
    print(original_counts)
    print(f"Total Original Images: {len(df_original)}\n")
    
    # 2. Find what is ACTUALLY in the preprocessed_1024 folder right now
    if not os.path.exists(dir_1024):
        print(f"Error: {dir_1024} not found.")
        return
        
    available_images = set()
    for f in os.listdir(dir_1024):
        if f.endswith('.jpg'):
            available_images.add(os.path.splitext(f)[0])
            
    # 3. Filter our dataframe to ONLY include what is physically present
    df_synced = df_original[df_original['image'].isin(available_images)]
    synced_counts = df_synced['level'].value_counts().sort_index()
    
    print("--- CURRENT (SYNCED) DATASET SEVERITY COUNTS ---")
    print(synced_counts)
    print(f"Total Synced Images: {len(df_synced)}\n")
    
    # 4. Calculate what was removed
    print("--- SUMMARY OF REMOVED IMAGES ---")
    total_removed = 0
    for level in range(5):
        orig_count = original_counts.get(level, 0)
        sync_count = synced_counts.get(level, 0)
        removed = orig_count - sync_count
        total_removed += removed
        print(f"Severity [{level}]: {removed} images removed ({(removed/orig_count*100) if orig_count else 0:.1f}%)")
        
    print(f"\nTotal Images Removed Across All Steps: {total_removed}")

    # 5. Overwrite trainLabels.csv to perfectly match current folder state
    df_synced.to_csv(final_csv, index=False)
    print(f"\nSuccessfully overwrote {final_csv} to exactly match the {len(df_synced)} images currently in preprocessed_1024.")

if __name__ == '__main__':
    sync_and_report()
