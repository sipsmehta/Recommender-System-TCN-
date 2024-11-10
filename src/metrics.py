import numpy as np
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate various metrics for model evaluation
    
    Args:
        y_true: True ratings/values
        y_pred: Predicted ratings/values
        threshold: Threshold for converting ratings to binary classes
    
    Returns:
        Dictionary containing RMSE, precision, recall, and F1-score
    """
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Convert ratings to binary classes for precision, recall, and F1
    # Assuming ratings above threshold are "positive" recommendations
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    return {
        'rmse': rmse,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    } 