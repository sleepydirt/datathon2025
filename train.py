# Training script for the LightGBM model on the CTG dataset
import numpy as np
import pandas as pd
from sklearn.utils import compute_sample_weight
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib
import argparse
import os
import sys
from utils import calculate_deceleration_score, calculate_deviation, calculate_vmr, calculate_z_score

def prepare_dataset(X: pd.DataFrame, y: pd.Series):
    """
    Prepares the dataset by performing feature engineering.
    
    Args:
        X: Feature dataframe
        y: Target series
    
    Returns:
        Processed dataframe with engineered features and target
    """
    # Concatenate features and target
    df = pd.concat([X, y], axis=1)
    
    # Drop CLASS column if it exists
    if 'CLASS' in df.columns:
        df = df.drop(columns=['CLASS'])
    
    # Drop all duplicates
    df = df.drop_duplicates()
    
    # Feature Engineering
    
    # 1. DS_present: indicator for severe deceleration
    df['DS_present'] = (df['DS'] > 0).astype(int)
    
    # 2. DP_present: indicator for prolonged deceleration
    df['DP_present'] = (df['DP'] > 0).astype(int)
    
    # 3. Deceleration Score: weighted sum of decelerations
    df['Deceleration_Score'] = df.apply(calculate_deceleration_score, axis=1)
    
    # 4. Z-scores for Mode and Median
    df['Z_Mode'] = df.apply(
        lambda row: calculate_z_score(row['Mode'], row['Mean'], np.sqrt(row['Variance'])), 
        axis=1
    )
    df['Z_Median'] = df.apply(
        lambda row: calculate_z_score(row['Median'], row['Mean'], np.sqrt(row['Variance'])), 
        axis=1
    )
    
    # 5. Variance-to-Mean Ratio (Index of Dispersion)
    df['VMR'] = df.apply(lambda row: calculate_vmr(row['Mean'], row['Variance']), axis=1)
    
    # 6. Deviation from baseline heart rate
    df['Mean_Dev'] = df.apply(lambda row: calculate_deviation(row['Mean'], row['LB']), axis=1)
    df['Max_Dev'] = df.apply(lambda row: calculate_deviation(row['Max'], row['LB']), axis=1)
    df['Min_Dev'] = df.apply(lambda row: calculate_deviation(row['Min'], row['LB']), axis=1)
    
    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="train-lightgbm",
        description="Training script for a LightGBM model on the CTG dataset."
    )

    parser.add_argument("-o", "--output-dir",
                        type=str,
                        default=os.path.join('models'),
                        help="Output path of the trained model file (.pkl).",
                        required=True)

    parser.add_argument("-d", "--dataset",
                        type=str,
                        help="Path to the dataset. If not specified, will be automatically pulled from the internet.",)

    args = parser.parse_args()

    output_dir = args.output_dir
    dataset = args.dataset

    # Load dataset
    if not dataset:
        print("No dataset specified. Automatically pulling from UCI repository...")
        from ucimlrepo import fetch_ucirepo
        # fetch dataset
        cardiotocography = fetch_ucirepo(id=193)
        # data (as pandas dataframes)
        X = cardiotocography.data.features
        y = cardiotocography.data.targets
        
    else:
        try:
            print(f"Loading dataset from {dataset}...")
            df = pd.read_csv(dataset)
            if 'NSP' not in df.columns:
                print("Error: Dataset must contain 'NSP' column as target.")
                sys.exit(1)
            X = df.drop(columns=['NSP'])
            y = df['NSP']
        except Exception as e:
            print(f"Dataset load failed with error: {e}")
            print("Automatically pulling from UCI repository...")
            from ucimlrepo import fetch_ucirepo
            # fetch dataset
            cardiotocography = fetch_ucirepo(id=193)
            # data (as pandas dataframes)
            X = cardiotocography.data.features
            y = cardiotocography.data.targets
            # Handle targets (it might be a DataFrame, convert to Series)
            if isinstance(y, pd.DataFrame):
                y = y.squeeze()

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Prepare dataset with feature engineering
    print("Preparing dataset with feature engineering...")
    df_processed = prepare_dataset(X, y)
    
    # Separate features and target
    feature_columns = [col for col in df_processed.columns if col != "NSP"]
    X_full = df_processed[feature_columns].copy()
    y_full = df_processed['NSP'].copy()
    
    print(f"After feature engineering: {X_full.shape[1]} features")
    
    # Compute sample weights for handling class imbalance
    print("Computing sample weights for class imbalance...")
    sample_weight_full = compute_sample_weight(class_weight='balanced', y=y_full.values)
    
    # Initialize the LightGBM model with optimized hyperparameters
    print("Initializing LightGBM model...")
    lgbm_final = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=80,
        class_weight='balanced',
        n_jobs=-1,
        max_depth=31,
        num_leaves=60
    )
    
    # Train the model on the full dataset
    print("Training model on full dataset...")
    lgbm_final.fit(X_full, y_full, sample_weight=sample_weight_full)
    print("Training complete!")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the trained model
    model_path = os.path.join(output_dir, 'lgbm_final.pkl')
    joblib.dump(lgbm_final, model_path)
    print(f"Model saved to {model_path}")
    
    print("\n" + "="*50)
    print("Training Summary:")
    print(f"  - Training samples: {len(y_full)}")
    print(f"  - Features: {len(feature_columns)}")
    print(f"  - Model: LightGBM")
    print(f"  - Output: {model_path}")
    print("="*50)
