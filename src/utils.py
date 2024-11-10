# src/utils.py
import numpy as np

def prepare_features(user_seq, trust_feat, user_id, item_id):
    """Prepare features for model input"""
    return [
        np.array([user_seq]),
        np.array([[trust_feat]]),
        np.array([[user_id, item_id]])
    ]

def inverse_transform_predictions(scaler, predictions):
    """Transform scaled predictions back to original rating scale"""
    return scaler.inverse_transform(predictions)