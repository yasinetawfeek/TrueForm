#!/usr/bin/env python3
"""
Model Loader Utility

Loads different types of workout classifier models and handles predictions.
"""

import pickle
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


# Define custom layers for Transformer model (must match training notebook)
# Note: These are registered via custom_objects when loading, not via decorator
class PositionalEncoding(layers.Layer):
    """Positional encoding layer for Transformer model."""
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
    def build(self, input_shape):
        # Learnable positional embeddings
        self.pos_encoding = self.add_weight(
            name='pos_encoding',
            shape=(self.max_len, self.d_model),
            initializer='uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model
        })
        return config


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention and feed-forward network."""
    def __init__(self, d_model, num_heads, dff, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        # Create sub-layers
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='gelu'),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        # Multi-head attention with residual connection
        attn_output = self.mha(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config


class ModelLoader:
    """Loads and manages workout classifier models."""
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the model loader.
        
        Args:
            models_dir: Directory containing model files (default: AI/workout_classifier/models)
        """
        if models_dir is None:
            # Default to the models directory relative to this file
            base_dir = Path(__file__).parent.parent
            models_dir = base_dir / 'AI' / 'workout_classifier' / 'models'
        self.models_dir = Path(models_dir)
        
        self.models = {}
        self.model_metadata = {}
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get list of available models and their metadata."""
        available = {}
        
        # Check for RandomForest
        rf_model = self.models_dir / 'randomforest_workout_classifier.pkl'
        if rf_model.exists():
            available['RandomForest'] = {
                'type': 'sklearn',
                'model_path': str(rf_model),
                'class_names_path': str(self.models_dir / 'class_names_rf.json'),
                'metadata_path': str(self.models_dir / 'model_metadata_rf.json')
            }
        
        # Check for XGBoost
        xgb_model = self.models_dir / 'xgboost_workout_classifier.pkl'
        if xgb_model.exists():
            available['XGBoost'] = {
                'type': 'sklearn',
                'model_path': str(xgb_model),
                'class_names_path': str(self.models_dir / 'class_names.json'),
                'metadata_path': str(self.models_dir / 'model_metadata.json')
            }
        
        # Check for Transformer
        transformer_model = self.models_dir / 'transformer_workout_classifier.keras'
        if transformer_model.exists():
            available['Transformer'] = {
                'type': 'keras',
                'model_path': str(transformer_model),
                'class_names_path': str(self.models_dir / 'class_names_transformer.json'),
                'normalization_path': str(self.models_dir / 'normalization_params_transformer.json'),
                'metadata_path': str(self.models_dir / 'model_metadata_transformer.json')
            }
        
        # Check for BiLSTM
        bilstm_model = self.models_dir / 'bilstm_workout_classifier.keras'
        if bilstm_model.exists():
            available['BiLSTM'] = {
                'type': 'keras',
                'model_path': str(bilstm_model),
                'class_names_path': str(self.models_dir / 'class_names_bilstm.json'),
                'normalization_path': str(self.models_dir / 'normalization_params_bilstm.json'),
                'metadata_path': str(self.models_dir / 'model_metadata_bilstm.json')
            }
        
        # Check for GRU
        gru_model = self.models_dir / 'gru_workout_classifier.keras'
        if gru_model.exists():
            available['GRU'] = {
                'type': 'keras',
                'model_path': str(gru_model),
                'class_names_path': str(self.models_dir / 'class_names_gru.json'),
                'normalization_path': str(self.models_dir / 'normalization_params_gru.json'),
                'metadata_path': str(self.models_dir / 'model_metadata_gru.json')
            }
        
        return available
    
    def load_model(self, model_name: str) -> Tuple:
        """
        Load a model and return (model, class_names, metadata, normalization_params).
        
        Args:
            model_name: Name of the model ('RandomForest', 'XGBoost', 'Transformer', 'BiLSTM', 'GRU')
            
        Returns:
            Tuple of (model, class_names, metadata, normalization_params)
        """
        available = self.get_available_models()
        
        if model_name not in available:
            raise ValueError(f"Model '{model_name}' not available. Available: {list(available.keys())}")
        
        model_info = available[model_name]
        
        # Load class names
        with open(model_info['class_names_path'], 'r') as f:
            class_names = json.load(f)
        
        # Load metadata
        metadata = {}
        if Path(model_info['metadata_path']).exists():
            with open(model_info['metadata_path'], 'r') as f:
                metadata = json.load(f)
        
        # Load normalization params for Keras models
        normalization_params = None
        if model_info['type'] == 'keras' and 'normalization_path' in model_info:
            norm_path = Path(model_info['normalization_path'])
            if norm_path.exists():
                with open(norm_path, 'r') as f:
                    normalization_params = json.load(f)
        
        # Load model
        if model_info['type'] == 'sklearn':
            with open(model_info['model_path'], 'rb') as f:
                model = pickle.load(f)
        else:  # keras
            # For Transformer model, use custom_objects to handle custom layers
            if model_name == 'Transformer':
                # Provide custom objects - these classes must match exactly
                # what was used during training
                custom_objects = {
                    'PositionalEncoding': PositionalEncoding,
                    'TransformerBlock': TransformerBlock,
                }
                try:
                    model = keras.models.load_model(
                        model_info['model_path'],
                        custom_objects=custom_objects,
                        compile=False
                    )
                except Exception as e:
                    # Try alternative: load with safe_mode=False for older Keras versions
                    try:
                        model = keras.models.load_model(
                            model_info['model_path'],
                            custom_objects=custom_objects,
                            compile=False,
                            safe_mode=False
                        )
                    except Exception as e2:
                        raise ValueError(
                            f"Failed to load Transformer model. "
                            f"Original error: {str(e)}. "
                            f"Alternative error: {str(e2)}. "
                            f"Make sure the model was saved with compatible Keras version."
                        ) from e2
            else:
                try:
                    model = keras.models.load_model(model_info['model_path'], compile=False)
                except Exception as e:
                    raise ValueError(f"Failed to load {model_name} model: {str(e)}") from e
        
        return model, np.array(class_names), metadata, normalization_params
    
    def predict(self, model_name: str, sequence: np.ndarray) -> Dict:
        """
        Make a prediction using the specified model.
        
        Args:
            model_name: Name of the model to use
            sequence: Pose sequence array of shape (15, 12, 3) or (15, 36)
            
        Returns:
            Dictionary with 'predicted_class', 'confidence', 'probabilities', 'top3'
        """
        if model_name not in self.models:
            model, class_names, metadata, norm_params = self.load_model(model_name)
            self.models[model_name] = {
                'model': model,
                'class_names': class_names,
                'metadata': metadata,
                'normalization_params': norm_params
            }
        
        model_data = self.models[model_name]
        model = model_data['model']
        class_names = model_data['class_names']
        norm_params = model_data['normalization_params']
        
        # Prepare sequence based on model type
        if model_name in ['RandomForest', 'XGBoost']:
            # Flatten to (1, 540) for sklearn models
            if sequence.ndim == 3:
                sequence_flat = sequence.reshape(1, -1)
            else:
                sequence_flat = sequence.reshape(1, -1)
            
            # Handle NaN and infinite values
            sequence_flat = np.nan_to_num(sequence_flat, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Get probabilities
            probabilities = model.predict_proba(sequence_flat)[0]
        
        else:  # Keras models (Transformer, BiLSTM, GRU)
            # Reshape to (1, 15, 36) for Keras models
            if sequence.ndim == 3:
                # Shape is (15, 12, 3) -> (1, 15, 36)
                sequence_reshaped = sequence.reshape(1, 15, -1)
            elif sequence.ndim == 2:
                # Shape is (15, 36) -> (1, 15, 36)
                sequence_reshaped = sequence.reshape(1, *sequence.shape)
            else:
                sequence_reshaped = sequence.reshape(1, 15, -1)
            
            # Handle NaN and infinite values
            sequence_reshaped = np.nan_to_num(sequence_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize if parameters provided
            if norm_params is not None:
                mean = np.array(norm_params['mean'])
                std = np.array(norm_params['std'])
                sequence_reshaped = (sequence_reshaped - mean) / std
            
            # Get probabilities
            probabilities = model.predict(sequence_reshaped, verbose=0)[0]
        
        # Get top prediction
        pred_class_idx = np.argmax(probabilities)
        pred_class = class_names[pred_class_idx]
        confidence = float(probabilities[pred_class_idx])
        
        # Get top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        top3_predictions = [
            {
                'class': class_names[idx],
                'probability': float(probabilities[idx])
            }
            for idx in top3_indices
        ]
        
        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': probabilities.tolist(),
            'top3': top3_predictions
        }
