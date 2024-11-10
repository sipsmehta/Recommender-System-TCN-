# src/data_processor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, ratings_df, trust_df, sequence_length=5):
        self.ratings_df = ratings_df
        self.trust_df = trust_df
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self):
        """Preprocess ratings data"""
        # Create user and item mappings
        self.user_mapping = {uid: idx for idx, uid 
                           in enumerate(self.ratings_df['userid'].unique())}
        self.item_mapping = {iid: idx for idx, iid 
                           in enumerate(self.ratings_df['productid'].unique())}
        
        # Map IDs to indices
        self.ratings_df['user_idx'] = self.ratings_df['userid'].map(self.user_mapping)
        self.ratings_df['item_idx'] = self.ratings_df['productid'].map(self.item_mapping)
        
        # Scale ratings
        self.ratings_df['rating_scaled'] = self.scaler.fit_transform(
            self.ratings_df[['rating']]).ravel()
        
        return self.ratings_df
    
    def create_temporal_sequences(self):
        """Create temporal sequences for TCN"""
        sequences = []
        labels = []
        
        for user in self.ratings_df['user_idx'].unique():
            user_ratings = self.ratings_df[
                self.ratings_df['user_idx'] == user].sort_index()
            
            if len(user_ratings) >= self.sequence_length + 1:
                for i in range(len(user_ratings) - self.sequence_length):
                    seq = user_ratings.iloc[i:i+self.sequence_length]
                    target = user_ratings.iloc[i+self.sequence_length]
                    
                    seq_features = np.column_stack((
                        seq['item_idx'].values,
                        seq['rating_scaled'].values,
                        seq['categoryid'].values
                    ))
                    
                    sequences.append(seq_features)
                    labels.append([target['rating_scaled']])
        
        return np.array(sequences), np.array(labels)
    
    def split_data(self, test_size=0.2):
        """Split data into train and test sets"""
        return train_test_split(
            self.ratings_df,
            test_size=test_size,
            random_state=42
        )