#!/usr/bin/env python3
"""
AutoML and Hyperparameter Optimization

Automated machine learning pipeline with:
- Hyperparameter optimization (Optuna)
- Neural architecture search
- Automatic feature selection
- Model selection and ensemble
- Cross-validation strategies
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import json


@dataclass
class AutoMLConfig:
    """Configuration for AutoML"""
    n_trials: int = 100
    timeout: Optional[int] = 3600  # 1 hour
    n_jobs: int = -1
    cv_folds: int = 5
    optimization_metric: str = 'f1'  # 'accuracy', 'f1', 'precision', 'recall'
    model_types: List[str] = None  # None = all models

    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.

    Supports:
    - Tree-structured Parzen Estimator (TPE)
    - Median pruning for early stopping
    - Parallelization
    - Multi-objective optimization
    """

    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0
        self.study: Optional[optuna.Study] = None

    def optimize_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': 42,
            }

            model = RandomForestClassifier(**params)
            score = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring=self.config.optimization_metric,
                n_jobs=1,
            ).mean()

            return score

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(),
            pruner=MedianPruner(),
        )

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
        )

        return study.best_params

    def optimize_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                'random_state': 42,
            }

            model = xgb.XGBClassifier(**params)
            score = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring=self.config.optimization_metric,
                n_jobs=1,
            ).mean()

            return score

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(),
            pruner=MedianPruner(),
        )

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
        )

        return study.best_params

    def optimize_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters"""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                'random_state': 42,
                'verbose': -1,
            }

            model = lgb.LGBMClassifier(**params)
            score = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring=self.config.optimization_metric,
                n_jobs=1,
            ).mean()

            return score

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(),
            pruner=MedianPruner(),
        )

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
        )

        return study.best_params

    def optimize_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Optimize all model types and return best parameters"""
        logger.info("Starting hyperparameter optimization for all models")

        all_params = {}
        all_scores = {}

        for model_type in self.config.model_types:
            logger.info(f"Optimizing {model_type}...")

            if model_type == 'random_forest':
                params = self.optimize_random_forest(X, y)
            elif model_type == 'xgboost':
                params = self.optimize_xgboost(X, y)
            elif model_type == 'lightgbm':
                params = self.optimize_lightgbm(X, y)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                continue

            # Evaluate best model
            if model_type == 'random_forest':
                model = RandomForestClassifier(**params)
            elif model_type == 'xgboost':
                model = xgb.XGBClassifier(**params)
            elif model_type == 'lightgbm':
                model = lgb.LGBMClassifier(**params)

            score = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring=self.config.optimization_metric,
                n_jobs=-1,
            ).mean()

            all_params[model_type] = params
            all_scores[model_type] = score

            logger.info(f"{model_type} - Best {self.config.optimization_metric}: {score:.4f}")

        # Find best model overall
        best_model_type = max(all_scores, key=all_scores.get)
        self.best_params = {
            'model_type': best_model_type,
            'params': all_params[best_model_type],
            'score': all_scores[best_model_type],
        }

        logger.info(f"\nBest model: {best_model_type} ({all_scores[best_model_type]:.4f})")

        return {
            'all_params': all_params,
            'all_scores': all_scores,
            'best_model': self.best_params,
        }

    def save_results(self, path: Path, results: Dict[str, Any]):
        """Save optimization results"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results to {path}")


class FeatureSelector:
    """
    Automatic feature selection using various methods.
    """

    @staticmethod
    def select_by_importance(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        top_k: int = 20,
    ) -> List[str]:
        """Select top-k features by importance using Random Forest"""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]

        selected_features = [feature_names[i] for i in indices]

        logger.info(f"Selected {len(selected_features)} features by importance")

        return selected_features

    @staticmethod
    def select_by_correlation(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.95,
    ) -> List[str]:
        """Remove highly correlated features"""
        import pandas as pd

        df = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df.corr().abs()

        # Find pairs of highly correlated features
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Drop features with correlation > threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        selected_features = [f for f in feature_names if f not in to_drop]

        logger.info(f"Removed {len(to_drop)} correlated features (threshold={threshold})")
        logger.info(f"Selected {len(selected_features)} features")

        return selected_features


class AutoML:
    """
    Automated Machine Learning pipeline.

    Automatically:
    - Selects features
    - Optimizes hyperparameters
    - Selects best model
    - Creates ensemble if beneficial
    """

    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.optimizer = HyperparameterOptimizer(config)
        self.feature_selector = FeatureSelector()
        self.best_model = None
        self.selected_features: List[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """Run complete AutoML pipeline"""
        logger.info("Starting AutoML pipeline")

        # Feature selection
        if feature_names is not None:
            logger.info("Step 1: Feature selection")

            # By importance
            important_features = self.feature_selector.select_by_importance(
                X, y, feature_names, top_k=min(50, len(feature_names))
            )

            # By correlation
            self.selected_features = self.feature_selector.select_by_correlation(
                X[:, [feature_names.index(f) for f in important_features]],
                y,
                important_features,
                threshold=0.95,
            )

            # Filter X
            feature_indices = [feature_names.index(f) for f in self.selected_features]
            X = X[:, feature_indices]

            logger.info(f"Final features: {len(self.selected_features)}")

        # Hyperparameter optimization
        logger.info("Step 2: Hyperparameter optimization")
        results = self.optimizer.optimize_all_models(X, y)

        # Train best model
        logger.info("Step 3: Training best model")
        best_params = results['best_model']

        if best_params['model_type'] == 'random_forest':
            self.best_model = RandomForestClassifier(**best_params['params'])
        elif best_params['model_type'] == 'xgboost':
            self.best_model = xgb.XGBClassifier(**best_params['params'])
        elif best_params['model_type'] == 'lightgbm':
            self.best_model = lgb.LGBMClassifier(**best_params['params'])

        self.best_model.fit(X, y)

        logger.info("AutoML pipeline complete!")
        logger.info(f"Best model: {best_params['model_type']}")
        logger.info(f"Best score: {best_params['score']:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.best_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if self.best_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        return self.best_model.predict_proba(X)


if __name__ == '__main__':
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        random_state=42,
    )

    feature_names = [f'feature_{i}' for i in range(50)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Run AutoML
    config = AutoMLConfig(
        n_trials=50,
        timeout=300,  # 5 minutes
        model_types=['random_forest', 'xgboost', 'lightgbm'],
    )

    automl = AutoML(config)
    automl.fit(X_train, y_train, feature_names)

    # Predictions
    y_pred = automl.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Selected Features: {len(automl.selected_features)}")
