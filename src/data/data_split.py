import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch

def setup_cross_validation(df: pd.DataFrame, n_splits: int = 5, img_col: str = 'image'):
    """
    Implements GroupKFold to ensure no patient overlap between train and validation splits.
    
    Args:
        df: DataFrame containing image names (e.g., '10_left') and 'level' (or 'grade')
        n_splits: Number of folds (default 5)
        
    Returns:
        df: DataFrame with a new 'fold' column.
    """
    # Ensure there is a dummy fold column
    df['fold'] = -1
    
    # Extract patient_id: split by '_' and take the first part
    if 'patient_id' not in df.columns:
        # Fix #10: Guard for image names without '_' (split returns full name if no '_' present)
        def extract_patient_id(name):
            parts = str(name).split('_')
            return parts[0] if len(parts) > 1 else str(name)
        df['patient_id'] = df[img_col].apply(extract_patient_id)
    
    gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # We yield splits based on patient_id isolation
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y=df['level'], groups=df['patient_id'])):
        df.loc[val_idx, 'fold'] = fold + 1  # 1-indexed to match pipeline (1-5)
        
    return df
