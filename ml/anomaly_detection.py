#!/usr/bin/env python3
"""
Anomaly Detection for Market Events

Detect unusual market behavior and events:
- Isolation Forest
- One-Class SVM
- Autoencoder-based detection
- Statistical methods (Z-score, IQR)
- Time series anomaly detection
- Change point detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection"""
    method: str  # 'isolation_forest', 'one_class_svm', 'autoencoder', 'statistical'
    contamination: float = 0.1  # Expected proportion of anomalies
    window_size: int = 20  # For time series methods
    threshold: float = 3.0  # Z-score threshold


class IsolationForestDetector:
    """
    Isolation Forest for anomaly detection.

    Fast tree-based anomaly detection that isolates anomalies
    instead of profiling normal points.
    """

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.model = IsolationForest(
            contamination=config.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        """Fit detector"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        logger.info("Fitted Isolation Forest detector")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.

        Returns:
            Array of -1 (anomaly) and 1 (normal)
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores.

        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        return scores


class OneClassSVMDetector:
    """
    One-Class SVM for anomaly detection.

    Learns boundary around normal data points.
    """

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.model = OneClassSVM(
            nu=config.contamination,
            kernel='rbf',
            gamma='auto',
        )
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        """Fit detector"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        logger.info("Fitted One-Class SVM detector")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores"""
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        return scores


class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection"""

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detection.

    Trains autoencoder on normal data. Anomalies have high
    reconstruction error.
    """

    def __init__(
        self,
        config: AnomalyConfig,
        input_dim: int,
        latent_dim: int = 8,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.config = config
        self.device = device
        self.model = Autoencoder(input_dim, latent_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = StandardScaler()
        self.threshold = None

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        """Fit autoencoder"""
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]

                # Forward pass
                self.optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = nn.MSELoss()(reconstructed, batch)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {total_loss:.4f}")

        # Calculate threshold based on training reconstruction errors
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            self.threshold = np.percentile(errors, (1 - self.config.contamination) * 100)

        logger.info(f"Fitted Autoencoder detector (threshold: {self.threshold:.4f})")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        scores = self.score_samples(X)
        predictions = np.where(scores > self.threshold, -1, 1)
        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction errors"""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

        return errors


class StatisticalDetector:
    """
    Statistical anomaly detection methods.

    Methods:
    - Z-score
    - IQR (Interquartile Range)
    - Moving average deviation
    """

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.mean = None
        self.std = None
        self.q1 = None
        self.q3 = None
        self.iqr = None

    def fit(self, X: np.ndarray):
        """Calculate statistics"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1

        logger.info("Fitted Statistical detector")

    def predict_zscore(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies using Z-score"""
        z_scores = np.abs((X - self.mean) / (self.std + 1e-10))
        anomalies = np.any(z_scores > self.config.threshold, axis=1)
        predictions = np.where(anomalies, -1, 1)
        return predictions

    def predict_iqr(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies using IQR"""
        lower_bound = self.q1 - 1.5 * self.iqr
        upper_bound = self.q3 + 1.5 * self.iqr

        anomalies = np.any((X < lower_bound) | (X > upper_bound), axis=1)
        predictions = np.where(anomalies, -1, 1)
        return predictions


class TimeSeriesAnomalyDetector:
    """
    Time series specific anomaly detection.

    Detects:
    - Sudden spikes
    - Level shifts
    - Trend changes
    - Seasonality violations
    """

    def __init__(self, config: AnomalyConfig):
        self.config = config

    def detect_spikes(self, series: np.ndarray) -> np.ndarray:
        """Detect sudden spikes in time series"""
        # Calculate moving statistics
        window = self.config.window_size
        rolling_mean = pd.Series(series).rolling(window=window, center=True).mean()
        rolling_std = pd.Series(series).rolling(window=window, center=True).std()

        # Z-score
        z_scores = np.abs((series - rolling_mean) / (rolling_std + 1e-10))

        # Anomalies
        anomalies = z_scores > self.config.threshold

        return anomalies.values

    def detect_level_shifts(
        self,
        series: np.ndarray,
        threshold: float = 2.0,
    ) -> np.ndarray:
        """Detect level shifts"""
        # Calculate differences
        diffs = np.diff(series)

        # Moving statistics of differences
        window = self.config.window_size
        rolling_mean = pd.Series(diffs).rolling(window=window, center=True).mean()
        rolling_std = pd.Series(diffs).rolling(window=window, center=True).std()

        # Z-score of differences
        z_scores = np.abs((diffs - rolling_mean) / (rolling_std + 1e-10))

        # Anomalies
        anomalies = np.zeros(len(series), dtype=bool)
        anomalies[1:] = z_scores > threshold

        return anomalies

    def detect_trend_changes(self, series: np.ndarray) -> np.ndarray:
        """Detect trend changes using second derivative"""
        # First derivative (slope)
        first_diff = np.diff(series)

        # Second derivative (change in slope)
        second_diff = np.diff(first_diff)

        # Moving statistics
        window = self.config.window_size
        rolling_mean = pd.Series(second_diff).rolling(window=window, center=True).mean()
        rolling_std = pd.Series(second_diff).rolling(window=window, center=True).std()

        # Z-score
        z_scores = np.abs((second_diff - rolling_mean) / (rolling_std + 1e-10))

        # Anomalies
        anomalies = np.zeros(len(series), dtype=bool)
        anomalies[2:] = z_scores > self.config.threshold

        return anomalies


class AnomalyDetector:
    """
    Unified anomaly detection interface.

    Supports multiple methods and automatic selection.
    """

    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.detector = None
        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize appropriate detector"""
        if self.config.method == 'isolation_forest':
            self.detector = IsolationForestDetector(self.config)
        elif self.config.method == 'one_class_svm':
            self.detector = OneClassSVMDetector(self.config)
        elif self.config.method == 'statistical':
            self.detector = StatisticalDetector(self.config)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

        logger.info(f"Initialized {self.config.method} detector")

    def fit(self, X: np.ndarray):
        """Fit detector"""
        self.detector.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        return self.detector.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores"""
        if hasattr(self.detector, 'score_samples'):
            return self.detector.score_samples(X)
        else:
            raise NotImplementedError(f"{self.config.method} does not support score_samples")

    def detect_market_events(
        self,
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect market events from price data.

        Returns DataFrame with anomaly flags and scores.
        """
        # Calculate features
        features = []

        # Returns
        returns = price_data['close'].pct_change()
        features.append(returns)

        # Volatility
        volatility = returns.rolling(window=20).std()
        features.append(volatility)

        # Volume
        if 'volume' in price_data.columns:
            volume_change = price_data['volume'].pct_change()
            features.append(volume_change)

        # Combine features
        X = pd.concat(features, axis=1).dropna().values

        # Detect anomalies
        predictions = self.predict(X)
        scores = self.score_samples(X)

        # Create results DataFrame
        results = pd.DataFrame({
            'timestamp': price_data['timestamp'].iloc[len(price_data) - len(predictions):].values,
            'close': price_data['close'].iloc[len(price_data) - len(predictions):].values,
            'is_anomaly': predictions == -1,
            'anomaly_score': scores,
        })

        return results


if __name__ == '__main__':
    # Example usage
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Normal data
    X_normal = np.random.randn(n_samples, n_features)

    # Add some anomalies
    n_anomalies = 50
    X_anomalies = np.random.randn(n_anomalies, n_features) * 5 + 10

    X = np.vstack([X_normal, X_anomalies])
    y_true = np.hstack([np.ones(n_samples), -np.ones(n_anomalies)])

    # Split data
    X_train = X_normal
    X_test = X

    # Test Isolation Forest
    config = AnomalyConfig(
        method='isolation_forest',
        contamination=0.1,
    )

    detector = AnomalyDetector(config)
    detector.fit(X_train)

    predictions = detector.predict(X_test)
    scores = detector.score_samples(X_test)

    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n=== Isolation Forest ===")
    print(classification_report(y_true, predictions))
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, predictions)}")

    # Count anomalies
    n_anomalies_detected = np.sum(predictions == -1)
    print(f"\nDetected {n_anomalies_detected} anomalies out of {len(X_test)} samples")
