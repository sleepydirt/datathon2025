# Utility functions

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, balanced_accuracy_score
import numpy as np


def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    print(classification_report(y_test, y_pred))

def calculate_deceleration_score(row):
  """Calculates the deceleration severity score for a row."""
  return row['DL'] + 3.5 * row['DP'] + 5 * row['DS']

def calculate_z_score(value, mean, std):
  """Calculates the Z-score for a given value."""
  # Avoid division by zero
  if std == 0:
    return 0
  return (value - mean) / std

def calculate_vmr(mean, var):
  """Calculates the coefficient of variance."""
  # Avoid division by zero
  if mean == 0:
    return 0
  return var / mean

def calculate_deviation(value, baseline):
  """Calculates the deviation from the baseline heart rate."""
  return value - baseline