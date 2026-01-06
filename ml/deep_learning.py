#!/usr/bin/env python3
"""
Deep Learning Models for Time Series

Advanced neural network architectures for stock price prediction:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Transformer (attention-based)
- Temporal Convolutional Networks (TCN)
- Hybrid models (CNN-LSTM, Attention-LSTM)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import json


@dataclass
class ModelConfig:
    """Configuration for deep learning models"""
    model_type: str  # 'lstm', 'gru', 'transformer', 'tcn', 'hybrid'
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    output_size: int = 3  # BUY, HOLD, SELL
    dropout: float = 0.2
    bidirectional: bool = False

    # Transformer specific
    num_heads: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512

    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10

    # Regularization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0


class TimeSeriesDataset(Dataset):
    """Dataset for time series data"""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 60,
    ):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction.

    Features:
    - Multiple LSTM layers
    - Dropout for regularization
    - Bidirectional option
    - Attention mechanism (optional)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

        # Calculate LSTM output size
        lstm_output_size = config.hidden_size
        if config.bidirectional:
            lstm_output_size *= 2

        # Attention layer (optional)
        self.attention = nn.Linear(lstm_output_size, 1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.output_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        use_attention: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            use_attention: Whether to use attention mechanism

        Returns:
            Output tensor [batch_size, output_size]
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch_size, seq_len, hidden_size * num_directions]

        if use_attention:
            # Attention mechanism
            attention_weights = torch.softmax(
                self.attention(lstm_out),
                dim=1
            )
            # attention_weights: [batch_size, seq_len, 1]

            # Apply attention
            context = torch.sum(attention_weights * lstm_out, dim=1)
            # context: [batch_size, hidden_size * num_directions]
        else:
            # Use last hidden state
            context = lstm_out[:, -1, :]

        # Fully connected
        output = self.fc(context)

        return output


class TransformerModel(nn.Module):
    """
    Transformer model for time series prediction.

    Features:
    - Multi-head self-attention
    - Positional encoding
    - Layer normalization
    - Feed-forward networks
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.hidden_size,
            dropout=config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]

        Returns:
            Output tensor [batch_size, output_size]
        """
        # Input projection
        x = self.input_projection(x)
        # x: [batch_size, seq_len, hidden_size]

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer encoder
        encoded = self.transformer_encoder(x)
        # encoded: [batch_size, seq_len, hidden_size]

        # Global average pooling
        pooled = torch.mean(encoded, dim=1)
        # pooled: [batch_size, hidden_size]

        # Output
        output = self.fc(pooled)

        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for time series.

    Features:
    - Dilated causal convolutions
    - Residual connections
    - Fast parallel training
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # TCN blocks
        num_channels = [config.hidden_size] * config.num_layers
        self.tcn = TemporalConvNet(
            config.input_size,
            num_channels,
            kernel_size=3,
            dropout=config.dropout,
        )

        # Output layer
        self.fc = nn.Linear(num_channels[-1], config.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]

        Returns:
            Output tensor [batch_size, output_size]
        """
        # TCN expects [batch_size, channels, seq_len]
        x = x.transpose(1, 2)

        # TCN
        y = self.tcn(x)

        # Take last time step
        y = y[:, :, -1]

        # Output
        output = self.fc(y)

        return output


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network"""

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TemporalBlock(nn.Module):
    """Single TCN block with residual connection"""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from the right side"""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class DeepLearningTrainer:
    """
    Trainer for deep learning models.

    Features:
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - Model checkpointing
    - Training metrics logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: Optional[Path] = None,
    ):
        """
        Train model with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_path: Path to save best model
        """
        logger.info(f"Training on {self.device}")
        logger.info(f"Model: {self.config.model_type}")

        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # Save best model
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Saved best model to {save_path}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Returns:
            predictions, probabilities
        """
        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                output = self.model(data)

                probs = torch.softmax(output, dim=1)
                _, preds = output.max(1)

                predictions.append(preds.cpu().numpy())
                probabilities.append(probs.cpu().numpy())

        return np.concatenate(predictions), np.concatenate(probabilities)

    def save_model(self, path: Path):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
        }, path)

    @classmethod
    def load_model(cls, path: Path, model_class: type) -> 'DeepLearningTrainer':
        """Load model checkpoint"""
        checkpoint = torch.load(path)

        config = checkpoint['config']
        model = model_class(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        trainer = cls(model, config)
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.history = checkpoint['history']

        return trainer


def create_model(config: ModelConfig) -> nn.Module:
    """Factory function to create models"""
    model_type = config.model_type.lower()

    if model_type == 'lstm':
        return LSTMModel(config)
    elif model_type == 'transformer':
        return TransformerModel(config)
    elif model_type == 'tcn':
        return TCNModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Example usage
    config = ModelConfig(
        model_type='lstm',
        input_size=10,
        hidden_size=128,
        num_layers=2,
        output_size=3,
        batch_size=32,
        num_epochs=50,
    )

    # Create model
    model = create_model(config)
    print(model)

    # Create dummy data
    X_train = np.random.randn(1000, 60, 10)  # 1000 samples, 60 timesteps, 10 features
    y_train = np.random.randint(0, 3, 1000)

    X_val = np.random.randn(200, 60, 10)
    y_val = np.random.randint(0, 3, 200)

    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Train
    trainer = DeepLearningTrainer(model, config)
    trainer.fit(train_loader, val_loader, save_path=Path('models/lstm_model.pt'))
