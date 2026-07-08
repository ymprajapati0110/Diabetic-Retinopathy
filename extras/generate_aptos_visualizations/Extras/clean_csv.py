import pandas as pd
import os
import argparse

def clean_csv(csv_path='d:/HYLr/trainLabels.csv', preprocessed_dir='d:/HYLr/preprocessed_512'):
    """
    Removes entries from the CSV where the corresponding preprocessed image does not exist.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Original entries in CSV: {len(df)}")
    
    # Use pre-cached file list for O(1) lookups
    if not os.path.exists(preprocessed_dir):
        print(f"Error: Preprocessed directory not found at {preprocessed_dir}")
        return
        
    existing_files = set(os.listdir(preprocessed_dir))
    print(f"Found {len(existing_files)} files in {preprocessed_dir}")
    
    def is_present(img_id):
        return f"{img_id}.jpg" in existing_files
        
    df['id_str'] = df['image'].astype(str)
    mask = df['id_str'].apply(is_present)
    
    df_clean = df[mask].drop(columns=['id_str'])
    missing_count = len(df) - len(df_clean)
    
    if missing_count == 0:
        print("No missing/invalid images found. CSV is already clean.")
        return
        
    print(f"Valid entries kept: {len(df_clean)}")
    print(f"Removed entries: {missing_count}")
    
    # Save the cleaned CSV
    backup_path = csv_path + ".backup"
    df.to_csv(backup_path, index=False)
    print(f"Created backup at {backup_path}")
        
    df_clean.to_csv(csv_path, index=False)
    print(f"Successfully cleaned CSV saved to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean CSV metadata based on valid preprocessed images.")
    parser.add_argument("--csv", type=str, default="d:/HYLr/trainLabels.csv", help="Path to original CSV")
    parser.add_argument("--dir", type=str, default="d:/HYLr/preprocessed_512", help="Path to preprocessed images directory to check against")
    args = parser.parse_args()
    
    clean_csv(args.csv, args.dir)
