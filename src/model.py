# src/model.py
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, Dropout, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np

class HybridModel:
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.model = None
        
    def build_model(self):
        """Build the hybrid recommendation model"""
        # Temporal branch (TCN)
        temporal_input = Input(shape=(self.sequence_length, 3), name='temporal_input')
        conv1 = Conv1D(64, 3, activation='relu', padding='causal', dilation_rate=1)(temporal_input)
        conv2 = Conv1D(64, 3, activation='relu', padding='causal', dilation_rate=2)(conv1)
        conv3 = Conv1D(64, 3, activation='relu', padding='causal', dilation_rate=4)(conv2)
        pool = GlobalMaxPooling1D()(conv3)
        
        # Trust network branch
        trust_input = Input(shape=(1,), name='trust_input')
        trust_embedding = Dense(32, activation='relu')(trust_input)
        
        # Collaborative filtering branch
        cf_input = Input(shape=(2,), name='cf_input')
        cf_embedding = Dense(32, activation='relu')(cf_input)
        
        # Combine all branches
        merged = tf.keras.layers.concatenate([pool, trust_embedding, cf_embedding])
        dense1 = Dense(128, activation='relu')(merged)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        output = Dense(1, activation='sigmoid')(dropout2)
        
        model = Model(
            inputs={
                'temporal_input': temporal_input,
                'trust_input': trust_input,
                'cf_input': cf_input
            },
            outputs=output
        )
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model

    def prepare_batch_data(self, sequences, trust_features, user_ids, item_ids):
        """Prepare data in the format expected by the model"""
        return {
            'temporal_input': np.array(sequences),
            'trust_input': np.array(trust_features).reshape(-1, 1),
            'cf_input': np.array([[u, i] for u, i in zip(user_ids, item_ids)])
        }

    def train_model(self, sequences, trust_features, user_ids, item_ids, labels, 
                   batch_size=32, epochs=5, validation_split=0.2):
        """
        Train the model with proper data splitting
        """
        # Ensure all inputs have the same length
        n_samples = len(sequences)
        assert len(trust_features) == n_samples
        assert len(user_ids) == n_samples
        assert len(item_ids) == n_samples
        assert len(labels) == n_samples

        # Split indices
        indices = np.arange(n_samples)
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_split,
            random_state=42
        )

        # Prepare training data
        train_data = self.prepare_batch_data(
            sequences[train_idx],
            trust_features[train_idx],
            user_ids[train_idx],
            item_ids[train_idx]
        )
        train_labels = labels[train_idx]

        # Prepare validation data
        val_data = self.prepare_batch_data(
            sequences[val_idx],
            trust_features[val_idx],
            user_ids[val_idx],
            item_ids[val_idx]
        )
        val_labels = labels[val_idx]

        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")

        # Train the model
        history = self.model.fit(
            train_data,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_data, val_labels),
            verbose=1
        )

        return history

    def predict(self, sequences, trust_features, user_ids, item_ids):
        """Make predictions using the trained model"""
        test_data = self.prepare_batch_data(
            sequences,
            trust_features,
            user_ids,
            item_ids
        )
        return self.model.predict(test_data)