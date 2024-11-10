# main.py
import scipy.io as sio
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.trust_network import TrustNetwork
from src.model import HybridModel
from src.metrics import calculate_metrics
from sklearn.model_selection import train_test_split
from src.visualization import plot_training_history, plot_prediction_analysis
from datetime import datetime
import os
import json

def load_data(rating_path, trust_path):
    """Load data from .mat files"""
    try:
        rating_data = sio.loadmat(rating_path)
        trust_data = sio.loadmat(trust_path)
        
        ratings_df = pd.DataFrame(
            rating_data['rating'],
            columns=['userid', 'productid', 'categoryid', 'rating']
        )
        
        trust_df = pd.DataFrame(
            trust_data['trustnetwork'],
            columns=['trustor', 'trustee']
        )
        
        print(f"Loaded {len(ratings_df)} ratings and {len(trust_df)} trust relationships")
        return ratings_df, trust_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def prepare_model_data(data_processor, trust_network, min_sequence_length=5):
    """Prepare data for the model"""
    # Get temporal sequences
    sequences, labels = data_processor.create_temporal_sequences()
    
    # Get unique user IDs from sequences instead of all users
    user_ids = np.array([seq[0][0] for seq in sequences])  # Assuming first element of sequence contains user ID
    
    # Get trust features for these specific users
    trust_features = np.array([
        trust_network.get_trust_features(user_id)[0] 
        for user_id in user_ids
    ])
    
    # Get corresponding item IDs from sequences
    item_ids = np.array([seq[0][1] for seq in sequences])  # Assuming first element of sequence contains item ID
    
    # Ensure we have enough sequences
    if len(sequences) < min_sequence_length:
        raise ValueError(
            f"Not enough sequences. Need at least {min_sequence_length}, "
            f"but got {len(sequences)}"
        )
    
    return sequences, trust_features, user_ids, item_ids, labels

def main():
    # Load data
    try:
        ratings_df, trust_df = load_data(
            'data/rating.mat',
            'data/trustnetwork.mat'
        )
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Initialize components
    data_processor = DataProcessor(ratings_df, trust_df)
    processed_df = data_processor.preprocess_data()
    
    trust_network = TrustNetwork(trust_df)
    trust_graph = trust_network.build_trust_graph()
    
    try:
        # Prepare data for model
        sequences, trust_features, user_ids, item_ids, labels = prepare_model_data(
            data_processor,
            trust_network
        )
        
        # Split data into train and test sets
        train_sequences, test_sequences, \
        train_trust, test_trust, \
        train_users, test_users, \
        train_items, test_items, \
        train_labels, test_labels = train_test_split(
            sequences, trust_features, user_ids, item_ids, labels,
            test_size=0.2, random_state=42
        )
        
        # Create and train model
        model = HybridModel()
        hybrid_model = model.build_model()
        
        print("Model architecture:")
        hybrid_model.summary()
        
        # Train model with training data
        history = model.train_model(
            sequences=train_sequences,
            trust_features=train_trust,
            user_ids=train_users,
            item_ids=train_items,
            labels=train_labels,
            batch_size=32,
            epochs=20
        )
        
        # Make predictions on test set
        predictions = model.predict(
            test_sequences,
            test_trust,
            test_users,
            test_items
        )
        
        # Calculate metrics
        metrics = calculate_metrics(test_labels, predictions)
        
        # Store metrics in a log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_data = {
            "timestamp": timestamp,
            "metrics": {
                "rmse": float(metrics['rmse']),
                "precision": float(metrics['precision']),
                "recall": float(metrics['recall']),
                "f1_score": float(metrics['f1_score'])
            },
            "training_params": {
                "batch_size": 32,
                "epochs": 20,
                "test_size": 0.2
            },
            "sample_predictions": [
                {
                    "user_id": int(test_users[i]),
                    "item_id": int(test_items[i]),
                    "predicted": float(predictions[i][0]),
                    "actual": float(test_labels[i][0])
                }
                for i in range(5)
            ]
        }
        
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Save metrics to JSON file
        log_file = f'logs/metrics_{timestamp}.json'
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=4)
        
        print(f"\nMetrics saved to: {log_file}")
        
        # Print evaluation metrics
        print("\nModel Evaluation Metrics:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1_score']:.4f}")
        
        # Print sample predictions (optional)
        print("\nSample Predictions:")
        for i in range(5):
            print(f"User {test_users[i]}, Item {test_items[i]}: "
                  f"Predicted Rating = {predictions[i][0]:.2f}, "
                  f"Actual Rating = {test_labels[i][0]:.2f}")
        
        # Plot training history
        plot_training_history(history)
        
        # Plot prediction analysis
        plot_prediction_analysis(test_labels, predictions)
        
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    main()
