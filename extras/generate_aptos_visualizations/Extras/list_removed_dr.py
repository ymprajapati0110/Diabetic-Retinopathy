import os
import pandas as pd

def list_removed():
    base_dir = r'D:\Major_project_shiro_final'
    csv_15 = os.path.join(base_dir, r'archive\labels\trainLabels15.csv')
    csv_19 = os.path.join(base_dir, r'archive\labels\trainLabels19.csv')
    final_csv = os.path.join(base_dir, 'trainLabels.csv')
    
    # Load original dataset
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
    
    # Load current CSV
    df_current = pd.read_csv(final_csv)
    df_current['image'] = df_current['image'].astype(str)
    
    # Find removed images
    removed_mask = ~df_orig['image'].isin(df_current['image'])
    df_removed = df_orig[removed_mask]
    
    # Filter for severities 1 to 4
    df_removed_1_to_4 = df_removed[df_removed['level'] >= 1]
    
    print(f"List of removed images with Severity 1-4 ({len(df_removed_1_to_4)} total):")
    print("-" * 50)
    for idx, row in df_removed_1_to_4.sort_values(by='level', ascending=False).iterrows():
        print(f"Image: {row['image']:<25} | Severity: {row['level']}")

if __name__ == "__main__":
    list_removed()
