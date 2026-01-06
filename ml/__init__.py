"""
Advanced Machine Learning Module

Enterprise-grade ML for stock trading with:
- Ensemble models (stacking, bagging, boosting)
- Deep learning (LSTM, Transformer, TCN)
- Reinforcement learning (DQN for trading)
- AutoML and hyperparameter optimization
- Model explainability (SHAP, LIME)
- Feature engineering
- Anomaly detection
- Distributed training (Multi-GPU, Multi-Node)
- Generative models (GANs for synthetic data)
- Meta-learning (MAML, Few-shot learning)
"""

from .ensemble import (
    EnsembleModel,
    AdaptiveEnsemble,
    ModelConfig,
    EnsembleConfig,
    create_default_ensemble,
)

from .deep_learning import (
    LSTMModel,
    TransformerModel,
    TCNModel,
    DeepLearningTrainer,
    TimeSeriesDataset,
    ModelConfig as DLModelConfig,
    create_model,
)

from .reinforcement_learning import (
    TradingEnvironment,
    DQNAgent,
    DQNNetwork,
    ReplayBuffer,
    TradingEnvironmentConfig,
    RLConfig,
    train_dqn_agent,
)

from .automl import (
    AutoML,
    HyperparameterOptimizer,
    FeatureSelector,
    AutoMLConfig,
)

from .explainability import (
    SHAPExplainer,
    LIMEExplainer,
    ModelExplainer,
    ExplainerConfig,
)

from .feature_engineering import (
    FeatureEngineer,
    TechnicalFeatures,
    TimeFeatures,
    StatisticalFeatures,
    FeatureConfig,
)

from .anomaly_detection import (
    AnomalyDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    AutoencoderDetector,
    StatisticalDetector,
    TimeSeriesAnomalyDetector,
    AnomalyConfig,
)

from .distributed_training import (
    DistributedTrainer,
    DistributedConfig,
    setup_distributed,
    cleanup_distributed,
    create_distributed_dataloader,
    launch_distributed_training,
)

from .advanced_models import (
    # GANs
    TimeGANGenerator,
    TimeGANDiscriminator,
    ConditionalGAN,
    WassersteinGAN,
    GANConfig,
    # Meta-Learning
    MAML,
    PrototypicalNetwork,
    FewShotLearner,
    MetaLearningConfig,
)

__version__ = '2.1.0'
__all__ = [
    # Ensemble
    'EnsembleModel',
    'AdaptiveEnsemble',
    'ModelConfig',
    'EnsembleConfig',
    'create_default_ensemble',

    # Deep Learning
    'LSTMModel',
    'TransformerModel',
    'TCNModel',
    'DeepLearningTrainer',
    'TimeSeriesDataset',
    'DLModelConfig',
    'create_model',

    # Reinforcement Learning
    'TradingEnvironment',
    'DQNAgent',
    'DQNNetwork',
    'ReplayBuffer',
    'TradingEnvironmentConfig',
    'RLConfig',
    'train_dqn_agent',

    # AutoML
    'AutoML',
    'HyperparameterOptimizer',
    'FeatureSelector',
    'AutoMLConfig',

    # Explainability
    'SHAPExplainer',
    'LIMEExplainer',
    'ModelExplainer',
    'ExplainerConfig',

    # Feature Engineering
    'FeatureEngineer',
    'TechnicalFeatures',
    'TimeFeatures',
    'StatisticalFeatures',
    'FeatureConfig',

    # Anomaly Detection
    'AnomalyDetector',
    'IsolationForestDetector',
    'OneClassSVMDetector',
    'AutoencoderDetector',
    'StatisticalDetector',
    'TimeSeriesAnomalyDetector',
    'AnomalyConfig',

    # Distributed Training
    'DistributedTrainer',
    'DistributedConfig',
    'setup_distributed',
    'cleanup_distributed',
    'create_distributed_dataloader',
    'launch_distributed_training',

    # Advanced Models (GANs)
    'TimeGANGenerator',
    'TimeGANDiscriminator',
    'ConditionalGAN',
    'WassersteinGAN',
    'GANConfig',

    # Advanced Models (Meta-Learning)
    'MAML',
    'PrototypicalNetwork',
    'FewShotLearner',
    'MetaLearningConfig',
]
