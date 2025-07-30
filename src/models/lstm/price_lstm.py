"""
Price LSTM Model for Time Series Price Prediction

Advanced LSTM model with attention mechanism for predicting future prices.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import structlog
from pathlib import Path

from ..base import TimeSeriesModel, TimeSeriesConfig, ModelType

logger = structlog.get_logger(__name__)


@dataclass
class PriceLSTMConfig(TimeSeriesConfig):
    """Configuration for Price LSTM model - ADAPTIVE & ROBUST"""
    
    # Adaptive LSTM architecture - will scale based on data
    lstm_layers: int = 2  # Start with 2 layers, increase with more data
    lstm_hidden_size: int = 128  # Moderate hidden state
    lstm_dropout: float = 0.2  # Conservative dropout
    
    # Override parent class to disable duplicate feature engineering
    # when using UniversalFeatureGenerator
    use_external_features: bool = True  # Set to True when using UniversalFeatureGenerator
    
    # Multi-scale LSTM processing
    use_multiscale: bool = False  # Disabled by default
    scales: List[int] = None  # Will default to [1]
    scale_fusion_method: str = "concat"  # Simple concatenation
    
    # Attention mechanism (optional)
    use_attention: bool = False  # Disabled by default for small data
    attention_heads: int = 4  # Fewer attention heads
    attention_layers: int = 1  # Single attention layer
    attention_dropout: float = 0.1
    use_self_attention: bool = False
    use_cross_attention: bool = False
    
    # Residual connections and normalization
    use_residual_connections: bool = False  # Disabled for simplicity
    use_layer_norm: bool = True
    use_batch_norm: bool = False  # Can cause issues with small batches
    layer_norm_eps: float = 1e-6
    
    # FC layers
    fc_layers: List[int] = None  # Will default to simple network
    fc_dropout: float = 0.2
    use_highway_connections: bool = False  # Disabled for simplicity
    
    # Advanced training parameters
    optimizer: str = "adamw"  # AdamW optimizer
    weight_decay: float = 0.01  # Stronger regularization
    gradient_clip: float = 0.5
    gradient_accumulation_steps: int = 4  # Effective larger batch
    
    # Sophisticated learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # Keep supported scheduler type
    max_lr: float = 0.01  # Maximum learning rate for one-cycle
    warmup_epochs: int = 20
    scheduler_patience: int = 25
    cosine_restarts: bool = True  # Cosine annealing with restarts
    
    # Regularization techniques
    use_mixup: bool = False  # Disabled for stability
    mixup_alpha: float = 0.2
    use_cutmix: bool = False  # Disabled for stability
    cutmix_alpha: float = 0.3
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1
    
    # Stochastic depth for regularization
    use_stochastic_depth: bool = False
    stochastic_depth_prob: float = 0.1
    
    # Loss function
    loss_type: str = "mse"  # Simple MSE loss
    huber_delta: float = 1.0
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Multi-task learning
    use_auxiliary_tasks: bool = False
    aux_task_weight: float = 0.3
    
    # Advanced features for financial time series
    use_temporal_convolution: bool = False
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2
    
    # Uncertainty quantification
    use_monte_carlo_dropout: bool = False
    mc_samples: int = 10
    use_bayesian_layers: bool = False
    
    # Meta-learning capabilities
    use_meta_learning: bool = False
    meta_lr: float = 0.001
    meta_steps: int = 5
    
    def __post_init__(self):
        super().__post_init__()
        if self.fc_layers is None:
            self.fc_layers = [64, 32]  # Simple FC network
        if self.scales is None:
            self.scales = [1]  # Single scale by default
        self.model_type = ModelType.PRICE_PREDICTION
        self.name = "PriceLSTM"
        
        # Disable duplicate feature engineering when using external features
        if self.use_external_features:
            self.add_lag_features = False
            self.add_time_features = False
            self.add_rolling_features = False
            logger.info("Disabled internal feature engineering (using external features)")


class AttentionLayer(nn.Module):
    """Self-attention layer for LSTM outputs"""
    
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            
        Returns:
            Attention output tensor
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations and split into heads
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # Output projection
        output = self.output(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + self.dropout(output))
        
        return output


class PriceLSTMNetwork(nn.Module):
    """LSTM network with attention for price prediction"""
    
    def __init__(self, config: PriceLSTMConfig, input_size: int):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        lstm_output_size = config.lstm_hidden_size * (2 if config.bidirectional else 1)
        
        # Attention layer
        if config.use_attention:
            self.attention = AttentionLayer(lstm_output_size, config.attention_heads)
        else:
            self.attention = None
        
        # Fully connected layers
        fc_layers = []
        prev_size = lstm_output_size
        
        for fc_size in config.fc_layers:
            fc_layers.append(nn.Linear(prev_size, fc_size))
            if config.use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(fc_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(config.fc_dropout))
            prev_size = fc_size
        
        # Output layer
        fc_layers.append(nn.Linear(prev_size, len(config.output_features)))
        
        self.fc = nn.Sequential(*fc_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            
        Returns:
            Predictions tensor
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Apply attention if enabled
        if self.attention is not None:
            lstm_out = self.attention(lstm_out)
        
        # Take the last timestep output
        # TODO: Consider using all timesteps for multi-step prediction
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc(last_output)
        
        return output


class PriceLSTM(TimeSeriesModel):
    """Price prediction LSTM model"""
    
    def __init__(self, config: PriceLSTMConfig):
        super().__init__(config)
        self.config: PriceLSTMConfig = config
        self.device = torch.device(config.device)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = nn.MSELoss()
        
    def build_model(self) -> PriceLSTMNetwork:
        """Build the LSTM model with adaptive architecture"""
        if not self.feature_columns:
            raise ValueError("Must prepare data before building model to know input size")
        
        input_size = len(self.feature_columns)
        
        # Adapt architecture based on feature count and data size
        if hasattr(self, 'training_samples') and self.training_samples > 0:
            # Scale architecture based on data availability
            if self.training_samples < 1000:
                # Very small data - minimal architecture
                self.config.lstm_layers = 1
                self.config.lstm_hidden_size = 32
                self.config.fc_layers = [16]
                self.config.use_attention = False
                logger.info("Using minimal architecture for small dataset")
            elif self.training_samples < 5000:
                # Small data - simple architecture
                self.config.lstm_layers = 2
                self.config.lstm_hidden_size = 64
                self.config.fc_layers = [32, 16]
                self.config.use_attention = False
                logger.info("Using simple architecture for small dataset")
            elif self.training_samples < 20000:
                # Medium data - moderate architecture
                self.config.lstm_layers = 2
                self.config.lstm_hidden_size = 128
                self.config.fc_layers = [64, 32]
                self.config.use_attention = True
                self.config.attention_heads = 4
                logger.info("Using moderate architecture for medium dataset")
            else:
                # Large data - full architecture
                self.config.lstm_layers = 3
                self.config.lstm_hidden_size = 256
                self.config.fc_layers = [128, 64, 32]
                self.config.use_attention = True
                self.config.attention_heads = 8
                logger.info("Using full architecture for large dataset")
        
        self.model = PriceLSTMNetwork(self.config, input_size)
        self.model.to(self.device)
        
        # Initialize optimizer
        if self.config.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            # Default to Adam if unknown optimizer
            logger.warning(f"Unknown optimizer {self.config.optimizer}, defaulting to Adam")
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        # Initialize scheduler
        if self.config.use_scheduler:
            if self.config.scheduler_type == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.epochs
                )
            elif self.config.scheduler_type == "step":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=30,
                    gamma=0.1
                )
            elif self.config.scheduler_type == "exponential":
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=0.95
                )
        
        # Log model architecture
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info("LSTM model built",
                   total_params=total_params,
                   trainable_params=trainable_params,
                   input_size=input_size)
        
        return self.model
    
    def train(
        self,
        train_data: Union[pd.DataFrame, np.ndarray],
        train_labels: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[Any, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the LSTM model"""
        
        # Prepare data first to get feature dimensions
        X_train, y_train = self.prepare_data(train_data, train_labels, is_training=True)
        
        # Track training samples for adaptive architecture
        self.training_samples = len(X_train)
        
        logger.info(f"Training data prepared: sequences={X_train.shape}, features={X_train.shape[-1]}, feature_columns={len(self.feature_columns)}")
        
        # Build model after data preparation (so we know input size)
        if self.model is None:
            self.build_model()
        else:
            # Verify input size matches
            expected_size = self.model.lstm.input_size
            actual_size = X_train.shape[-1]
            if expected_size != actual_size:
                logger.error(f"Input size mismatch: expected {expected_size}, got {actual_size}")
                logger.error(f"Feature columns: {len(self.feature_columns)}")
                logger.error(f"This indicates a bug in feature generation consistency")
                raise ValueError(f"Input size mismatch: expected {expected_size}, got {actual_size}. "
                               "This should not happen with proper feature tracking.")
        
        # Prepare validation data
        X_val, y_val = None, None
        if validation_data:
            logger.info("Preparing validation data...")
            X_val, y_val = self.prepare_data(
                validation_data[0], 
                validation_data[1], 
                is_training=False
            )
            logger.info(f"Validation data prepared: sequences={X_val.shape}, features={X_val.shape[-1]}")
        else:
            # Create validation split
            X_train, y_train, X_val, y_val = self.create_train_val_split(
                X_train, y_train, self.config.validation_split
            )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for now to avoid multiprocessing issues
        )
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_X)
                
                # Reshape if needed
                if len(batch_y.shape) == 1:
                    batch_y = batch_y.unsqueeze(1)
                
                loss = self.loss_fn(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
                train_losses.append(loss.item())
            
            # Validation phase
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    predictions = self.model(batch_X)
                    
                    if len(batch_y.shape) == 1:
                        batch_y = batch_y.unsqueeze(1)
                    
                    loss = self.loss_fn(predictions, batch_y)
                    val_losses.append(loss.item())
            
            # Calculate epoch metrics
            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            # Early stopping check
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{self.config.epochs}",
                           train_loss=epoch_train_loss,
                           val_loss=epoch_val_loss,
                           lr=self.optimizer.param_groups[0]['lr'])
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Checkpoint saving
            if epoch % self.config.checkpoint_interval == 0:
                checkpoint_path = self.config.save_path / f"checkpoint_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss
                }, checkpoint_path)
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Update metadata
        self.is_trained = True
        self.metadata['last_trained'] = datetime.now().isoformat()
        self.metadata['training_samples'] = len(X_train)
        self.metadata['best_val_loss'] = best_val_loss
        self.training_history.append(history)
        
        logger.info("LSTM training completed",
                   epochs_trained=len(history['train_loss']),
                   best_val_loss=best_val_loss)
        
        return history
    
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        """Make predictions with the LSTM model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare data
        X, _ = self.prepare_data(data, labels=None, is_training=False)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()
        
        # Inverse transform if scaled
        if self.scaler is not None:
            predictions = self.inverse_transform_predictions(predictions)
        
        return predictions
    
    def evaluate(
        self,
        test_data: Union[pd.DataFrame, np.ndarray],
        test_labels: Union[pd.Series, np.ndarray],
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate LSTM model performance"""
        
        # Get predictions
        predictions = self.predict(test_data)
        
        # Prepare true values
        _, y_true = self.prepare_data(test_data, test_labels, is_training=False)
        
        # Ensure same shape
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.squeeze()
        if len(y_true.shape) > 1 and y_true.shape[1] == 1:
            y_true = y_true.squeeze()
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_true, predictions),
            'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
            'mae': mean_absolute_error(y_true, predictions),
            'r2': r2_score(y_true, predictions)
        }
        
        # Calculate directional accuracy
        if len(y_true) > 1:
            y_true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(predictions) > 0
            metrics['directional_accuracy'] = np.mean(y_true_direction == pred_direction)
        
        # Store in metadata
        self.metadata['validation_metrics'] = metrics
        
        logger.info("LSTM evaluation completed",
                   rmse=metrics['rmse'],
                   r2=metrics['r2'])
        
        return metrics
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save LSTM model"""
        # Save base model data
        base_path = super().save(path)
        
        # Save PyTorch model
        if self.model is not None:
            model_path = base_path.with_suffix('.pt')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'config': self.config.to_dict(),
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns
            }, model_path)
            
            logger.info("LSTM model saved", path=str(model_path))
        
        return base_path
    
    @classmethod
    def load(cls, path: Path) -> 'PriceLSTM':
        """Load LSTM model"""
        # Load base model data
        model_instance = super().load(path)
        
        # Load PyTorch model
        model_path = path.with_suffix('.pt')
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=model_instance.device)
            
            # Rebuild model
            model_instance.feature_columns = checkpoint['feature_columns']
            model_instance.target_columns = checkpoint['target_columns']
            model_instance.build_model()
            
            # Load weights
            model_instance.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer if available
            if checkpoint.get('optimizer_state_dict') and model_instance.optimizer:
                model_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logger.info("LSTM model loaded", path=str(model_path))
        
        return model_instance