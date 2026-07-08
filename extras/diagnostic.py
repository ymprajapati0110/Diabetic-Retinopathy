import sys
import pandas as pd
import traceback

def main():
    try:
        from src.training.train import run_curriculum_training
        from src.data.data_split import setup_cross_validation
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        
        df = pd.read_csv("D:/Major_project_shiro_final/trainLabels.csv")
        df = setup_cross_validation(df, n_splits=5)
        run_curriculum_training(df, "D:/Major_project_shiro_final/preprocessed_512", fold_num=3, epochs=1, save_dir='results', stabilize=True)
    except Exception as e:
        traceback.print_exc()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
