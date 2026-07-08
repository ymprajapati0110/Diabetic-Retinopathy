import pandas as pd
import os

def setup_labels():
    print("--- Running Option B Data Setup: Merging Train 2015 and Train 2019 ---")
    
    # Load both datasets
    df15 = pd.read_csv('d:/archive/labels/train15.csv')
    df19 = pd.read_csv('d:/archive/labels/train19.csv')

    # Standardize column names
    df15 = df15.rename(columns={'image': 'image', 'level': 'level'})
    df19 = df19.rename(columns={'id_code': 'image', 'diagnosis': 'level'})

    # Add .png extension to 2019 images if needed
    if not str(df19['image'].iloc[0]).endswith('.png'):
        df19['image'] = df19['image'] + '.png'

    # Add source identifier (useful for debugging)
    df15['source'] = '2015'
    df19['source'] = '2019'

    # Combine
    df = pd.concat([df15, df19], ignore_index=True)
    print(f"Total images: {len(df)}")
    print(f"Class distribution:\n{df['level'].value_counts().sort_index()}")

    # Save
    os.makedirs('d:/HYLr', exist_ok=True)
    df.to_csv('d:/HYLr/trainLabels.csv', index=False)
    print("Successfully saved to d:/HYLr/trainLabels.csv")

if __name__ == "__main__":
    setup_labels()
