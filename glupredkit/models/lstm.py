import numpy as np
import tensorflow as tf
import ast
from datetime import datetime
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense, Input, Masking, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit
from .base_model import BaseModel
from glupredkit.helpers.tf_keras import process_data

class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        timestamp = datetime.now().isoformat()
        safe_timestamp = timestamp.replace(':', '_').replace('.', '_')
        self.model_path = f"data/.keras_models/lstm_ph-{prediction_horizon}_{safe_timestamp}.h5"
        self.input_shape = None
        self.num_outputs = None
        
        # Define physiological/practical limits for each feature
        self.feature_limits = {
            'CGM': (40, 400),      # mg/dL - typical CGM range
            'insulin': (0, 15),     # Units - total insulin (bolus + basal)
            'carbs': (0, 100),      # grams
            'basal': (0, 0.5),      # Units per 5-min
            'bolus': (0, 15),       # Units
            'hour': (0, 23)         # Hour of day
        }

    def process_data(self, df, model_config_manager, real_time=False):
        """
        Process raw data into sequences with multiple prediction horizons
        """
        # Get configuration parameters
        num_lagged_features = model_config_manager.get_num_lagged_features()
        prediction_horizon = model_config_manager.get_prediction_horizon()
        numerical_features = model_config_manager.get_num_features()
        
        # Create empty lists to store sequences and targets
        sequences = []
        targets = []
        
        # Number of minutes per timestep
        minutes_per_step = 5
        
        # Calculate number of steps needed for max prediction horizon
        horizon_steps = prediction_horizon // minutes_per_step
        
        # Ensure df is sorted by datetime
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
        
        # Create sequences for each feature
        for i in range(len(df) - num_lagged_features - horizon_steps + 1):
            # Extract sequence
            sequence = df[numerical_features].iloc[i:(i + num_lagged_features)].values
            
            # Extract targets for multiple horizons (5, 10, 15... up to prediction_horizon)
            target_values = []
            for steps in range(1, horizon_steps + 1):
                target_idx = i + num_lagged_features + steps - 1
                if target_idx < len(df):
                    target_values.append(float(df['CGM'].iloc[target_idx]))
            
            # Only add if we have all target values
            if len(target_values) == horizon_steps:
                sequences.append(str(sequence.tolist()))
                targets.append(str(target_values))
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'sequence': sequences,
            'target': targets
        })
        
        return output_df

    def _calculate_tdd_from_df(self, df):
        """Calculate Total Daily Dose of insulin from raw DataFrame"""
        # Check for datetime column
        date_column = None
        for col in ['datetime', 'date', 'timestamp']:
            if col in df.columns:
                date_column = col
                break
        
        if date_column is None:
            print("Warning: No datetime column found. TDD statistics will not be calculated.")
            return {
                'mean_tdd': 0.0,
                'std_tdd': 0.0,
                'min_tdd': 0.0,
                'max_tdd': 0.0,
                'mean_basal': 0.0,
                'mean_bolus': 0.0,
                'basal_percent': 0.0
            }
        
        df[date_column] = pd.to_datetime(df[date_column])
        df['day'] = df[date_column].dt.date
        
        daily_totals = df.groupby('day').agg({
            'basal': 'sum',
            'bolus': 'sum',
        }).reset_index()
        
        daily_totals['tdd'] = daily_totals['basal'] + daily_totals['bolus']
        
        tdd_stats = {
            'mean_tdd': daily_totals['tdd'].mean(),
            'std_tdd': daily_totals['tdd'].std(),
            'min_tdd': daily_totals['tdd'].min(),
            'max_tdd': daily_totals['tdd'].max(),
            'mean_basal': daily_totals['basal'].mean(),
            'mean_bolus': daily_totals['bolus'].mean(),
            'basal_percent': (daily_totals['basal'].mean() / daily_totals['tdd'].mean()) * 100
        }
        
        return tdd_stats

    def _normalize_feature(self, data, feature_name):
        """Normalize feature based on predefined physiological/practical limits"""
        min_val, max_val = self.feature_limits[feature_name]
        return (data - min_val) / (max_val - min_val) * 2 - 1  # Scale to [-1, 1]
    
    def _denormalize_feature(self, data, feature_name):
        """Denormalize feature back to original scale"""
        min_val, max_val = self.feature_limits[feature_name]
        return ((data + 1) / 2) * (max_val - min_val) + min_val

    def _preprocess_sequences(self, sequences, feature_names):
        """Normalize each feature using domain-specific limits"""
        normalized_sequences = np.zeros_like(sequences, dtype=np.float32)
        
        for feature_idx, feature_name in enumerate(feature_names):
            if feature_name in self.feature_limits:
                feature_data = sequences[:, :, feature_idx]
                normalized_sequences[:, :, feature_idx] = self._normalize_feature(feature_data, feature_name)
        
        return normalized_sequences

    def _fit_model(self, x_train, y_train, epochs=20, *args):
        # Extract feature names from the configuration
        feature_names = ['CGM', 'insulin', 'carbs', 'basal', 'bolus']
        
        def process_sequence(seq):
            if isinstance(seq, str):
                return np.array(ast.literal_eval(seq))
            return np.array(seq)
        
        sequences = [process_sequence(seq) for seq in x_train['sequence']]
        targets = [process_sequence(target) for target in y_train['target']]
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"Sequence shape: {sequences.shape}")
        print(f"Target shape: {targets.shape}")
        
        # Normalize sequences using domain-specific limits
        sequences = self._preprocess_sequences(sequences, feature_names)
        # Normalize targets (CGM values)
        targets = self._normalize_feature(targets, 'CGM')

        self.input_shape = (sequences.shape[1], sequences.shape[2])
        self.num_outputs = targets.shape[1]  # Number of prediction horizons

        # Model architecture
        input_layer = Input(shape=self.input_shape)
        masked = Masking(mask_value=0.)(input_layer)
        
        # LSTM layers
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(masked)
        bn1 = BatchNormalization()(lstm1)
        drop1 = Dropout(0.3)(bn1)
        
        lstm2 = Bidirectional(LSTM(64, return_sequences=True))(drop1)
        bn2 = BatchNormalization()(lstm2)
        drop2 = Dropout(0.3)(bn2)
        
        lstm3 = Bidirectional(LSTM(32))(drop2)
        bn3 = BatchNormalization()(lstm3)
        drop3 = Dropout(0.3)(bn3)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(drop3)
        dense2 = Dense(32, activation='relu')(dense1)
        output_layer = Dense(self.num_outputs)(dense2)  # Multiple outputs for different horizons

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        
        # Simpler learning rate approach
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1
        )
        
        # Time series split for validation (20% validation)
        split_idx = int(len(sequences) * 0.8)
        
        train_X = sequences[:split_idx]
        train_Y = targets[:split_idx]
        val_X = sequences[split_idx:]
        val_Y = targets[split_idx:]

        print(f"Training shapes - X: {train_X.shape}, Y: {train_Y.shape}")
        print(f"Validation shapes - X: {val_X.shape}, Y: {val_Y.shape}")

        # Training history
        history = model.fit(
            train_X, train_Y,
            validation_data=(val_X, val_Y),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Print training summary
        print("\nTraining Summary:")
        print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
        print(f"Best training loss: {min(history.history['loss']):.4f}")
        
        if 'val_mae' in history.history:
            print(f"Best validation MAE: {min(history.history['val_mae']):.4f}")
        
        model.save(self.model_path)
        return self

    def _predict_model(self, x_test):
        feature_names = ['CGM', 'insulin', 'carbs', 'basal', 'bolus']
        
        def process_sequence(seq):
            if isinstance(seq, str):
                return np.array(ast.literal_eval(seq))
            return np.array(seq)
        
        sequences = [process_sequence(seq) for seq in x_test['sequence']]
        sequences = np.array(sequences)
        
        print(f"Test sequence shape: {sequences.shape}")
        
        # Normalize test sequences
        sequences = self._preprocess_sequences(sequences, feature_names)
        
        model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={"Adam": tf.keras.optimizers.legacy.Adam}
        )
        
        # Get predictions for all horizons
        predictions = model.predict(sequences)
        
        # Denormalize predictions back to CGM scale
        predictions = self._denormalize_feature(predictions, 'CGM')
        
        # Convert to list of lists, where each inner list contains predictions
        # for all horizons for one sequence
        return predictions.tolist()

    def get_tdd_stats(self):
        """Return the calculated TDD statistics"""
        return self.tdd_stats if hasattr(self, 'tdd_stats') else None

    def best_params(self):
        """Return the best parameters found during training"""
        return None