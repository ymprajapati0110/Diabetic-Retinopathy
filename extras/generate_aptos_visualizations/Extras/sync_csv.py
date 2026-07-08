import os
import pandas as pd

def sync_csv():
    base_dir = r'D:\Major_project_shiro_final'
    csv_file = os.path.join(base_dir, 'trainLabels.csv')
    
    dir_512 = os.path.join(base_dir, 'preprocessed_512')
    dir_768 = os.path.join(base_dir, 'preprocessed_768')
    dir_1024 = os.path.join(base_dir, 'preprocessed_1024')
    
    if not os.path.exists(csv_file):
        print("CSV not found.")
        return
        
    df = pd.read_csv(csv_file)
    original_len = len(df)
    
    valid_indices = []
    
    for idx, row in df.iterrows():
        img_id = str(row['image']).strip()
        img_name = f"{img_id}.jpg"
        
        path_512 = os.path.join(dir_512, img_name)
        path_768 = os.path.join(dir_768, img_name)
        path_1024 = os.path.join(dir_1024, img_name)
        
        if os.path.exists(path_512) and os.path.exists(path_768) and os.path.exists(path_1024):
            valid_indices.append(idx)
            
    df_synced = df.loc[valid_indices]
    df_synced.to_csv(csv_file, index=False)
    
    removed = original_len - len(df_synced)
    print(f"Sync complete. Removed {removed} orphaned entries from CSV.")
    print(f"Final synchronized CSV length: {len(df_synced)} images.")

if __name__ == '__main__':
    sync_csv()
