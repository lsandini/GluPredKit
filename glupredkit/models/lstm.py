import numpy as np
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
import ast
from datetime import datetime
from tensorflow.keras.layers import (
    LSTM, Dense, Input, Dropout, BatchNormalization,
    Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from .base_model import BaseModel
from glupredkit.helpers.tf_keras import process_data


@register_keras_serializable(package="GluPredKit")
class ClinicalLoss(tf.keras.losses.Loss):
    def __init__(self, name="clinical_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Basic error
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Get raw glucose values (assuming scaled 0-1 represents 0-400 mg/dL)
        y_true_glucose = y_true * 400
        y_pred_glucose = y_pred * 400
        
        # Clinical weights
        hypo_mask = tf.cast(y_true_glucose < 70, tf.float32)
        hyper_mask = tf.cast(y_true_glucose > 180, tf.float32)
        
        # Weighted errors with stronger emphasis on dangerous ranges
        clinical_error = (
            tf.reduce_mean(hypo_mask * tf.square(y_true - y_pred)) * 2.5 +
            tf.reduce_mean(hyper_mask * tf.square(y_true - y_pred)) * 1.75
        )
        
        return mse + clinical_error

    def get_config(self):
        return {"name": self.name}


class Model(BaseModel):
    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        # Initialize scalers
        self.scalers = {
            'glucose': RobustScaler(with_centering=True, with_scaling=True),
            'insulin': RobustScaler(with_centering=True, with_scaling=True),
            'carbs': RobustScaler(with_centering=True, with_scaling=True)
        }
        self.model = None
        self.input_shape = None
        self.num_outputs = None
        self.loss_fn = ClinicalLoss()
        
        # Feature mapping
        self.feature_indices = {
            'glucose': 0,
            'insulin': [1, 3, 4],  # insulin, basal, bolus
            'carbs': 2
        }
        
        # Validation thresholds
        self.valid_ranges = {
            'glucose': (30, 500),    # mg/dL
            'insulin': (0, 25),      # units
            'carbs': (0, 150)        # grams
        }

    def _validate_data(self, sequences):
        """Validate input data ranges"""
        glucose_data = sequences[:, :, self.feature_indices['glucose']]
        insulin_indices = self.feature_indices['insulin']
        insulin_data = sequences[:, :, insulin_indices]
        carbs_data = sequences[:, :, self.feature_indices['carbs']]
        
        # Check ranges
        if np.any(glucose_data < self.valid_ranges['glucose'][0]) or \
           np.any(glucose_data > self.valid_ranges['glucose'][1]):
            print(f"Warning: Glucose values outside valid range {self.valid_ranges['glucose']}")
        
        if np.any(insulin_data < self.valid_ranges['insulin'][0]) or \
           np.any(insulin_data > self.valid_ranges['insulin'][1]):
            print(f"Warning: Insulin values outside valid range {self.valid_ranges['insulin']}")
        
        if np.any(carbs_data < self.valid_ranges['carbs'][0]) or \
           np.any(carbs_data > self.valid_ranges['carbs'][1]):
            print(f"Warning: Carbs values outside valid range {self.valid_ranges['carbs']}")

    def _prepare_sequences(self, sequences_str, is_training=True):
        """Prepare and scale input sequences"""
        try:
            # Convert strings to arrays
            sequences = np.array([np.array(ast.literal_eval(seq)) for seq in sequences_str])
            
            # Validate data
            self._validate_data(sequences)
            
            # Initialize output array
            scaled_sequences = np.zeros_like(sequences)
            
            # Scale glucose
            glucose_data = sequences[:, :, self.feature_indices['glucose']]
            if is_training:
                scaled_sequences[:, :, self.feature_indices['glucose']] = self.scalers['glucose'].fit_transform(
                    glucose_data.reshape(-1, 1)
                ).reshape(glucose_data.shape)
            else:
                scaled_sequences[:, :, self.feature_indices['glucose']] = self.scalers['glucose'].transform(
                    glucose_data.reshape(-1, 1)
                ).reshape(glucose_data.shape)
            
            # Scale insulin features
            insulin_indices = self.feature_indices['insulin']
            insulin_data = sequences[:, :, insulin_indices]
            if is_training:
                scaled_insulin = self.scalers['insulin'].fit_transform(
                    insulin_data.reshape(-1, len(insulin_indices))
                ).reshape(insulin_data.shape)
            else:
                scaled_insulin = self.scalers['insulin'].transform(
                    insulin_data.reshape(-1, len(insulin_indices))
                ).reshape(insulin_data.shape)
            for idx, insulin_idx in enumerate(insulin_indices):
                scaled_sequences[:, :, insulin_idx] = scaled_insulin[:, :, idx]
            
            # Scale carbs
            carbs_data = sequences[:, :, self.feature_indices['carbs']]
            if is_training:
                scaled_sequences[:, :, self.feature_indices['carbs']] = self.scalers['carbs'].fit_transform(
                    carbs_data.reshape(-1, 1)
                ).reshape(carbs_data.shape)
            else:
                scaled_sequences[:, :, self.feature_indices['carbs']] = self.scalers['carbs'].transform(
                    carbs_data.reshape(-1, 1)
                ).reshape(carbs_data.shape)
            
            return scaled_sequences
            
        except Exception as e:
            raise ValueError(f"Error preparing sequences: {str(e)}")

    def _build_model(self, input_shape, num_outputs):
        """Build LSTM model architecture"""
        inputs = Input(shape=input_shape)
        
        # First LSTM layer
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Second LSTM layer
        x = Bidirectional(LSTM(32))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(num_outputs, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
        # Compile with gradient clipping
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=1.0,
                clipvalue=0.5
            ),
            loss=self.loss_fn,
            metrics=['mae']
        )
        
        return model

    def _fit_model(self, x_train, y_train, epochs=30, batch_size=32, *args):
        """Train the model"""
        try:
            # Prepare sequences
            print("Preparing training sequences...")
            sequences = self._prepare_sequences(x_train['sequence'], is_training=True)
            
            # Prepare targets
            print("Preparing target values...")
            targets = np.array([np.array(ast.literal_eval(target)) for target in y_train['target']])
            
            # Scale targets to [0,1] for sigmoid output
            targets = targets / 400.0  # Assuming max glucose is 400 mg/dL
            
            # Set shapes
            self.input_shape = (sequences.shape[1], sequences.shape[2])
            self.num_outputs = targets.shape[1]
            
            print(f"Input shape: {self.input_shape}, Output shape: {targets.shape}")
            
            # Build model
            print("Building model...")
            self.model = self._build_model(self.input_shape, self.num_outputs)
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    min_delta=0.001
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_delta=0.001,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            print("Starting training...")
            history = self.model.fit(
                sequences,
                targets,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"Error during model training: {str(e)}")

    def _predict_model(self, x_test):
        """Generate predictions"""
        try:
            if self.model is None:
                raise RuntimeError("Model not trained. Call fit() first.")
            
            # Prepare sequences
            sequences = self._prepare_sequences(x_test['sequence'], is_training=False)
            
            # Generate predictions
            predictions_scaled = self.model.predict(sequences)
            
            # Convert back to glucose values
            predictions = predictions_scaled * 400.0  # Convert from [0,1] to mg/dL
            
            return predictions.tolist()
            
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")

    def best_params(self):
        return None

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)