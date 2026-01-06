#!/usr/bin/env python3
"""
Ensemble Models Module

Advanced ensemble techniques for combining multiple models:
- Stacking: Train meta-learner on base model predictions
- Bagging: Bootstrap aggregating for variance reduction
- Boosting: Sequential learning for bias reduction
- Voting: Weighted ensemble predictions
- Model selection and optimization
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
from loguru import logger
import pickle
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb


@dataclass
class ModelConfig:
    """Configuration for individual model in ensemble"""
    name: str
    model_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    enabled: bool = True


@dataclass
class EnsembleConfig:
    """Configuration for ensemble system"""
    ensemble_type: str  # 'stacking', 'voting', 'bagging', 'boosting'
    base_models: List[ModelConfig] = field(default_factory=list)
    meta_learner: Optional[ModelConfig] = None
    voting_type: str = 'soft'  # 'hard' or 'soft'
    n_folds: int = 5
    use_probabilities: bool = True
    optimization_metric: str = 'f1'  # 'accuracy', 'f1', 'precision', 'recall'


class EnsembleModel:
    """
    Advanced ensemble model combining multiple base learners.

    Supports:
    - Stacking with cross-validation
    - Voting (hard/soft)
    - Bagging for variance reduction
    - Gradient boosting
    - XGBoost and LightGBM
    """

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.base_models: List[Any] = []
        self.meta_learner: Optional[Any] = None
        self.ensemble: Optional[Any] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.training_scores_: Dict[str, float] = {}

    def _create_model(self, model_config: ModelConfig) -> Any:
        """Create model instance from configuration"""
        model_type = model_config.model_type.lower()
        params = model_config.params

        if model_type == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
        elif model_type == 'xgboost':
            return xgb.XGBClassifier(**params)
        elif model_type == 'lightgbm':
            return lgb.LGBMClassifier(**params)
        elif model_type == 'logistic_regression':
            return LogisticRegression(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _build_stacking_ensemble(self) -> StackingClassifier:
        """Build stacking ensemble with meta-learner"""
        logger.info("Building stacking ensemble")

        # Create base estimators
        estimators = []
        for model_config in self.config.base_models:
            if not model_config.enabled:
                continue

            model = self._create_model(model_config)
            estimators.append((model_config.name, model))
            logger.info(f"Added base model: {model_config.name} ({model_config.model_type})")

        # Create meta-learner
        if self.config.meta_learner:
            final_estimator = self._create_model(self.config.meta_learner)
            logger.info(f"Meta-learner: {self.config.meta_learner.name}")
        else:
            # Default meta-learner: Logistic Regression
            final_estimator = LogisticRegression(max_iter=1000)
            logger.info("Using default meta-learner: LogisticRegression")

        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=self.config.n_folds,
            stack_method='auto',
            n_jobs=-1,
            verbose=1,
        )

    def _build_voting_ensemble(self) -> VotingClassifier:
        """Build voting ensemble"""
        logger.info(f"Building {self.config.voting_type} voting ensemble")

        estimators = []
        weights = []

        for model_config in self.config.base_models:
            if not model_config.enabled:
                continue

            model = self._create_model(model_config)
            estimators.append((model_config.name, model))
            weights.append(model_config.weight)
            logger.info(f"Added model: {model_config.name} (weight: {model_config.weight})")

        return VotingClassifier(
            estimators=estimators,
            voting=self.config.voting_type,
            weights=weights,
            n_jobs=-1,
            verbose=1,
        )

    def _build_bagging_ensemble(self) -> BaggingClassifier:
        """Build bagging ensemble"""
        logger.info("Building bagging ensemble")

        # Use first enabled model as base estimator
        base_config = next(
            (m for m in self.config.base_models if m.enabled),
            None
        )

        if not base_config:
            raise ValueError("No enabled models for bagging ensemble")

        base_estimator = self._create_model(base_config)
        logger.info(f"Base estimator: {base_config.name}")

        return BaggingClassifier(
            estimator=base_estimator,
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=False,
            n_jobs=-1,
            verbose=1,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> 'EnsembleModel':
        """
        Train ensemble model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Self
        """
        logger.info(f"Training {self.config.ensemble_type} ensemble")
        logger.info(f"Training samples: {len(X_train)}")

        # Build ensemble based on type
        if self.config.ensemble_type == 'stacking':
            self.ensemble = self._build_stacking_ensemble()
        elif self.config.ensemble_type == 'voting':
            self.ensemble = self._build_voting_ensemble()
        elif self.config.ensemble_type == 'bagging':
            self.ensemble = self._build_bagging_ensemble()
        else:
            raise ValueError(f"Unknown ensemble type: {self.config.ensemble_type}")

        # Train ensemble
        logger.info("Training ensemble...")
        self.ensemble.fit(X_train, y_train)

        # Calculate training scores
        y_train_pred = self.ensemble.predict(X_train)
        self.training_scores_['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        self.training_scores_['train_f1'] = f1_score(y_train, y_train_pred, average='weighted')

        logger.info(f"Training accuracy: {self.training_scores_['train_accuracy']:.4f}")
        logger.info(f"Training F1: {self.training_scores_['train_f1']:.4f}")

        # Validation scores
        if X_val is not None and y_val is not None:
            y_val_pred = self.ensemble.predict(X_val)
            self.training_scores_['val_accuracy'] = accuracy_score(y_val, y_val_pred)
            self.training_scores_['val_f1'] = f1_score(y_val, y_val_pred, average='weighted')

            logger.info(f"Validation accuracy: {self.training_scores_['val_accuracy']:.4f}")
            logger.info(f"Validation F1: {self.training_scores_['val_f1']:.4f}")

        # Extract feature importances if available
        self._extract_feature_importances()

        return self

    def _extract_feature_importances(self):
        """Extract feature importances from ensemble"""
        try:
            if hasattr(self.ensemble, 'feature_importances_'):
                self.feature_importances_ = self.ensemble.feature_importances_
            elif self.config.ensemble_type == 'stacking':
                # Average importances from base estimators
                importances = []
                for name, estimator in self.ensemble.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importances.append(estimator.feature_importances_)

                if importances:
                    self.feature_importances_ = np.mean(importances, axis=0)

            if self.feature_importances_ is not None:
                logger.info("Extracted feature importances")
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if self.ensemble is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.ensemble.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if self.ensemble is None:
            raise ValueError("Model not trained. Call fit() first.")

        if not hasattr(self.ensemble, 'predict_proba'):
            raise ValueError("Ensemble does not support probability predictions")

        return self.ensemble.predict_proba(X)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            cv: Number of folds

        Returns:
            Dictionary of scores
        """
        logger.info(f"Performing {cv}-fold cross-validation")

        if self.ensemble is None:
            # Build ensemble without training
            if self.config.ensemble_type == 'stacking':
                self.ensemble = self._build_stacking_ensemble()
            elif self.config.ensemble_type == 'voting':
                self.ensemble = self._build_voting_ensemble()
            elif self.config.ensemble_type == 'bagging':
                self.ensemble = self._build_bagging_ensemble()

        # Cross-validation
        cv_scores = cross_val_score(
            self.ensemble,
            X,
            y,
            cv=cv,
            scoring=self.config.optimization_metric,
            n_jobs=-1,
            verbose=1,
        )

        results = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores.tolist(),
        }

        logger.info(f"CV {self.config.optimization_metric}: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")

        return results

    def get_model_predictions(
        self,
        X: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual base models.

        Args:
            X: Features

        Returns:
            Dictionary mapping model name to predictions
        """
        if self.ensemble is None:
            raise ValueError("Model not trained. Call fit() first.")

        predictions = {}

        if self.config.ensemble_type in ['stacking', 'voting']:
            for name, estimator in self.ensemble.named_estimators_.items():
                predictions[name] = estimator.predict(X)

        return predictions

    def save(self, path: Path):
        """Save ensemble model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'ensemble': self.ensemble,
                'feature_importances': self.feature_importances_,
                'training_scores': self.training_scores_,
            }, f)

        logger.info(f"Saved ensemble model to {path}")

    @classmethod
    def load(cls, path: Path) -> 'EnsembleModel':
        """Load ensemble model"""
        path = Path(path)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(data['config'])
        model.ensemble = data['ensemble']
        model.feature_importances_ = data.get('feature_importances')
        model.training_scores_ = data.get('training_scores', {})

        logger.info(f"Loaded ensemble model from {path}")

        return model


class AdaptiveEnsemble:
    """
    Adaptive ensemble that dynamically adjusts model weights
    based on recent performance.
    """

    def __init__(
        self,
        base_models: List[ModelConfig],
        window_size: int = 100,
        update_frequency: int = 10,
    ):
        self.base_models_config = base_models
        self.base_models: List[Any] = []
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.model_weights: np.ndarray = np.ones(len(base_models)) / len(base_models)
        self.recent_predictions: List[Tuple[np.ndarray, np.ndarray]] = []
        self.update_counter = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train all base models"""
        logger.info("Training adaptive ensemble base models")

        self.base_models = []
        for config in self.base_models_config:
            if not config.enabled:
                continue

            model = self._create_model(config)
            model.fit(X_train, y_train)
            self.base_models.append(model)
            logger.info(f"Trained {config.name}")

        return self

    def _create_model(self, config: ModelConfig) -> Any:
        """Create model instance"""
        model_type = config.model_type.lower()
        params = config.params

        if model_type == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
        elif model_type == 'xgboost':
            return xgb.XGBClassifier(**params)
        elif model_type == 'lightgbm':
            return lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with weighted ensemble"""
        if not self.base_models:
            raise ValueError("Models not trained")

        # Get predictions from all models
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Weighted voting
        weighted_predictions = np.average(
            predictions,
            axis=0,
            weights=self.model_weights,
        )

        return np.round(weighted_predictions).astype(int)

    def update_weights(self, X: np.ndarray, y_true: np.ndarray):
        """Update model weights based on recent performance"""
        self.update_counter += 1

        # Store recent predictions
        self.recent_predictions.append((X, y_true))

        # Keep only recent window
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions = self.recent_predictions[-self.window_size:]

        # Update weights periodically
        if self.update_counter % self.update_frequency == 0:
            self._recalculate_weights()

    def _recalculate_weights(self):
        """Recalculate model weights based on recent performance"""
        if not self.recent_predictions:
            return

        # Concatenate recent data
        X_recent = np.vstack([x for x, _ in self.recent_predictions])
        y_recent = np.hstack([y for _, y in self.recent_predictions])

        # Calculate accuracy for each model
        accuracies = []
        for model in self.base_models:
            y_pred = model.predict(X_recent)
            acc = accuracy_score(y_recent, y_pred)
            accuracies.append(acc)

        accuracies = np.array(accuracies)

        # Update weights (softmax of accuracies)
        exp_acc = np.exp(accuracies * 10)  # Temperature scaling
        self.model_weights = exp_acc / exp_acc.sum()

        logger.info(f"Updated model weights: {self.model_weights}")


def create_default_ensemble() -> EnsembleConfig:
    """Create default ensemble configuration"""
    return EnsembleConfig(
        ensemble_type='stacking',
        base_models=[
            ModelConfig(
                name='random_forest',
                model_type='random_forest',
                params={
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42,
                },
                weight=1.0,
            ),
            ModelConfig(
                name='xgboost',
                model_type='xgboost',
                params={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                },
                weight=1.0,
            ),
            ModelConfig(
                name='lightgbm',
                model_type='lightgbm',
                params={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'verbose': -1,
                },
                weight=1.0,
            ),
        ],
        meta_learner=ModelConfig(
            name='meta_logistic',
            model_type='logistic_regression',
            params={'max_iter': 1000, 'random_state': 42},
        ),
        n_folds=5,
        optimization_metric='f1',
    )


if __name__ == '__main__':
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train ensemble
    config = create_default_ensemble()
    ensemble = EnsembleModel(config)

    # Cross-validation
    cv_results = ensemble.cross_validate(X_train, y_train, cv=5)
    print(f"\nCV Results: {cv_results}")

    # Train model
    ensemble.fit(X_train, y_train, X_test, y_test)

    # Predictions
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Save model
    ensemble.save(Path('models/ensemble_model.pkl'))
