#!/usr/bin/env python3
"""
Advanced Feature Engineering Pipeline

Automated feature generation and transformation for stock trading:
- Technical indicators (momentum, volatility, trend)
- Time-based features
- Statistical aggregations
- Interaction features
- Polynomial features
- Target encoding
- Feature scaling and normalization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from loguru import logger


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    technical_indicators: bool = True
    time_features: bool = True
    statistical_features: bool = True
    interaction_features: bool = False
    polynomial_features: bool = False
    polynomial_degree: int = 2
    pca_components: Optional[int] = None
    scaler_type: str = 'standard'  # 'standard', 'robust', 'minmax'


class TechnicalFeatures:
    """
    Technical indicator features for stock analysis.

    Generates:
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility indicators (Bollinger Bands, ATR)
    - Trend indicators (SMA, EMA, ADX)
    - Volume indicators (OBV, VWAP)
    """

    @staticmethod
    def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # Rate of Change
        df['roc'] = (df['close'] / df['close'].shift(10) - 1) * 100

        return df

    @staticmethod
    def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_middle'] = sma_20
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        # Historical Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)

        return df

    @staticmethod
    def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators"""
        # Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # Moving Average Crossovers
        df['ma_cross_50_200'] = df['sma_50'] - df['sma_200']
        df['ma_cross_20_50'] = df['sma_20'] - df['sma_50']

        # Price vs MA
        df['price_vs_sma_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma_50'] = (df['close'] - df['sma_50']) / df['sma_50']

        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators"""
        # Volume Moving Average
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(window=14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(window=14).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        df['mfi'] = mfi

        return df


class TimeFeatures:
    """Time-based features"""

    @staticmethod
    def add_time_features(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Add time-based features"""
        if timestamp_col not in df.columns:
            return df

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Date components
        df['year'] = df[timestamp_col].dt.year
        df['month'] = df[timestamp_col].dt.month
        df['day'] = df[timestamp_col].dt.day
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['quarter'] = df[timestamp_col].dt.quarter

        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # Market timing
        df['is_month_start'] = df[timestamp_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[timestamp_col].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[timestamp_col].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[timestamp_col].dt.is_quarter_end.astype(int)

        return df


class StatisticalFeatures:
    """Statistical aggregation features"""

    @staticmethod
    def add_statistical_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add rolling statistical features"""
        for window in windows:
            # Returns
            df[f'return_{window}'] = df['close'].pct_change(window)

            # Rolling statistics
            df[f'mean_{window}'] = df['close'].rolling(window).mean()
            df[f'std_{window}'] = df['close'].rolling(window).std()
            df[f'min_{window}'] = df['close'].rolling(window).min()
            df[f'max_{window}'] = df['close'].rolling(window).max()

            # Z-score
            df[f'zscore_{window}'] = (df['close'] - df[f'mean_{window}']) / df[f'std_{window}']

            # Skewness and Kurtosis
            df[f'skew_{window}'] = df['close'].rolling(window).skew()
            df[f'kurt_{window}'] = df['close'].rolling(window).kurt()

        return df


class FeatureEngineer:
    """
    Complete feature engineering pipeline.

    Automates:
    1. Technical indicator generation
    2. Time-based features
    3. Statistical aggregations
    4. Feature interactions
    5. Feature scaling
    6. Dimensionality reduction
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler = None
        self.pca = None
        self.feature_names: List[str] = []
        self.original_feature_names: List[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit and transform features.

        Args:
            df: Input dataframe
            target_col: Target column name (if any)

        Returns:
            Transformed features dataframe, target array
        """
        logger.info("Starting feature engineering...")

        # Store original features
        self.original_feature_names = list(df.columns)

        # Technical indicators
        if self.config.technical_indicators:
            logger.info("Adding technical indicators...")
            df = TechnicalFeatures.add_momentum_features(df)
            df = TechnicalFeatures.add_volatility_features(df)
            df = TechnicalFeatures.add_trend_features(df)
            df = TechnicalFeatures.add_volume_features(df)

        # Time features
        if self.config.time_features and 'timestamp' in df.columns:
            logger.info("Adding time features...")
            df = TimeFeatures.add_time_features(df)

        # Statistical features
        if self.config.statistical_features:
            logger.info("Adding statistical features...")
            df = StatisticalFeatures.add_statistical_features(df)

        # Drop NaN values
        df = df.dropna()

        # Extract target
        y = None
        if target_col and target_col in df.columns:
            y = df[target_col].values
            df = df.drop(columns=[target_col])

        # Store feature names
        self.feature_names = list(df.columns)

        # Interaction features
        if self.config.interaction_features:
            logger.info("Creating interaction features...")
            df = self._create_interactions(df)

        # Polynomial features
        if self.config.polynomial_features:
            logger.info(f"Creating polynomial features (degree={self.config.polynomial_degree})...")
            df = self._create_polynomial_features(df)

        # Convert to numpy
        X = df.values

        # Feature scaling
        logger.info(f"Scaling features ({self.config.scaler_type})...")
        X = self._fit_scaler(X)

        # PCA
        if self.config.pca_components:
            logger.info(f"Applying PCA (n_components={self.config.pca_components})...")
            X = self._fit_pca(X)

        logger.info(f"Feature engineering complete: {X.shape[1]} features")

        return df, X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted transformers"""
        # Apply same transformations
        if self.config.technical_indicators:
            df = TechnicalFeatures.add_momentum_features(df)
            df = TechnicalFeatures.add_volatility_features(df)
            df = TechnicalFeatures.add_trend_features(df)
            df = TechnicalFeatures.add_volume_features(df)

        if self.config.time_features and 'timestamp' in df.columns:
            df = TimeFeatures.add_time_features(df)

        if self.config.statistical_features:
            df = StatisticalFeatures.add_statistical_features(df)

        df = df.dropna()

        if self.config.interaction_features:
            df = self._create_interactions(df)

        if self.config.polynomial_features:
            df = self._create_polynomial_features(df)

        X = df[self.feature_names].values

        # Scale
        if self.scaler:
            X = self.scaler.transform(X)

        # PCA
        if self.pca:
            X = self.pca.transform(X)

        return X

    def _create_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature interactions"""
        # Select top numeric features for interactions
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

        return df

    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures

        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
        poly = PolynomialFeatures(degree=self.config.polynomial_degree, include_bias=False)

        poly_features = poly.fit_transform(df[numeric_cols])
        poly_feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]

        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        df = pd.concat([df, poly_df], axis=1)

        return df

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform scaler"""
        if self.config.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.config.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.config.scaler_type}")

        return self.scaler.fit_transform(X)

    def _fit_pca(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform PCA"""
        self.pca = PCA(n_components=self.config.pca_components)
        X_pca = self.pca.fit_transform(X)

        logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")

        return X_pca


if __name__ == '__main__':
    # Example usage
    # Generate sample OHLCV data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 1000),
        'target': np.random.choice([0, 1, 2], 1000),  # BUY, HOLD, SELL
    })

    # Create feature engineer
    config = FeatureConfig(
        technical_indicators=True,
        time_features=True,
        statistical_features=True,
        scaler_type='standard',
        pca_components=50,
    )

    engineer = FeatureEngineer(config)

    # Fit and transform
    df_features, X, y = engineer.fit_transform(df, target_col='target')

    print(f"\nOriginal features: {len(engineer.original_feature_names)}")
    print(f"Engineered features: {len(engineer.feature_names)}")
    print(f"Final features: {X.shape[1]}")
    print(f"\nSample features: {X[:5]}")
