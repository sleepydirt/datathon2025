import os
import joblib
import numpy as np
import pandas as pd
from typing import List
from utils import calculate_deceleration_score, calculate_deviation, calculate_vmr, calculate_z_score, evaluate_model
import argparse
import sys


def inference_pipeline(input_df: pd.DataFrame, loaded_model) -> pd.Series:
    """
    Applies feature engineering steps and makes predictions using a loaded model.

    Args:
        input_df: Input DataFrame.
        loaded_model: Trained model object.
        feature_columns: List of feature column names used during training.

    Returns:
        A pandas Series containing predictions.
    """
    processed_df = input_df.copy()

    # Apply feature engineering steps
    processed_df['DS_present'] = (processed_df['DS'] > 0).astype(int)
    processed_df['DP_present'] = (processed_df['DP'] > 0).astype(int)
    processed_df['Deceleration_Score'] = processed_df.apply(calculate_deceleration_score, axis=1)
    processed_df['Z_Mode'] = processed_df.apply(
        lambda row: calculate_z_score(row['Mode'], row['Mean'], np.sqrt(row['Variance'])), axis=1
    )
    processed_df['Z_Median'] = processed_df.apply(
        lambda row: calculate_z_score(row['Median'], row['Mean'], np.sqrt(row['Variance'])), axis=1
    )
    processed_df['VMR'] = processed_df.apply(lambda row: calculate_vmr(row['Mean'], row['Variance']), axis=1)
    processed_df['Mean_Dev'] = processed_df.apply(lambda row: calculate_deviation(row['Mean'], row['LB']), axis=1)
    processed_df['Max_Dev'] = processed_df.apply(lambda row: calculate_deviation(row['Max'], row['LB']), axis=1)
    processed_df['Min_Dev'] = processed_df.apply(lambda row: calculate_deviation(row['Min'], row['LB']), axis=1)

    # Select the feature columns used during training
    X_processed = processed_df.drop(columns=["NSP"])
    # Exclude the index column
    X_processed = X_processed.iloc[:, 1:]
    # Make predictions
    predictions = loaded_model.predict(X_processed)

    return X_processed, pd.Series(predictions, index=input_df.index, name="Predicted_NSP")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="eval-ctg",
        description="Evaluation harness for the CTG dataset."
    )

    parser.add_argument("-m", "--model-path",
                        type=str,
                        default=os.path.join('models', 'lgbm_final.pkl'),
                        help="Path to the trained model file (.pkl).",
                        required=True)

    parser.add_argument("-e", "--eval-csv",
                        type=str,
                        default="path/to/eval.csv",
                        help="Path to the evaluation CSV file.",
                        required=True)

    args = parser.parse_args()

    model_path = args.model_path
    eval_csv_path = args.eval_csv

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}; quitting.")
        sys.exit(1)

    loaded_model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    if not os.path.exists(eval_csv_path):
        print(f"Evaluation CSV not found at {eval_csv_path}")
        sys.exit(1)

    eval_df = pd.read_csv(eval_csv_path)

    # Ground truth if available
    has_labels = "NSP" in eval_df.columns
    actual_nsp = eval_df["NSP"] if has_labels else None
    
    # Run inference
    features_for_evaluation, predictions = inference_pipeline(eval_df, loaded_model)
    eval_df["Predicted_NSP"] = predictions

    # Evaluate if labels exist
    if has_labels:
        print("Evaluating predictions...")
        evaluate_model(loaded_model, features_for_evaluation, actual_nsp)

    else:
        print("No ground truth column found for evaluation.")
