#!/usr/bin/env python3
"""
Model Explainability and Interpretability

Tools for understanding model predictions:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance visualization
- Partial dependence plots
- Individual prediction explanations
"""

import shap
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import pandas as pd


@dataclass
class ExplainerConfig:
    """Configuration for explainability tools"""
    explainer_type: str  # 'shap', 'lime', 'both'
    feature_names: List[str] = None
    class_names: List[str] = None
    n_samples: int = 100  # For SHAP background data
    n_features: int = 10  # Top features to show


class SHAPExplainer:
    """
    SHAP explainer for model interpretability.

    SHAP values quantify the contribution of each feature
    to the prediction for a specific instance.

    Features:
    - TreeExplainer for tree-based models
    - KernelExplainer for any model
    - DeepExplainer for neural networks
    - Summary plots
    - Dependence plots
    - Force plots
    - Waterfall plots
    """

    def __init__(
        self,
        model: Any,
        X_background: np.ndarray,
        config: ExplainerConfig,
    ):
        self.model = model
        self.X_background = X_background
        self.config = config
        self.explainer = None
        self.shap_values = None

        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer"""
        logger.info("Initializing SHAP explainer...")

        # Try TreeExplainer first (fastest for tree models)
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Using TreeExplainer")
            return
        except Exception as e:
            logger.debug(f"TreeExplainer not applicable: {e}")

        # Try DeepExplainer for neural networks
        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                self.explainer = shap.DeepExplainer(
                    self.model,
                    torch.FloatTensor(self.X_background),
                )
                logger.info("Using DeepExplainer")
                return
        except Exception as e:
            logger.debug(f"DeepExplainer not applicable: {e}")

        # Fall back to KernelExplainer (model-agnostic, slower)
        # Sample background data to speed up
        background = shap.sample(self.X_background, self.config.n_samples)

        def predict_fn(X):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                return self.model.predict(X)

        self.explainer = shap.KernelExplainer(predict_fn, background)
        logger.info("Using KernelExplainer")

    def explain(
        self,
        X: np.ndarray,
        check_additivity: bool = False,
    ) -> np.ndarray:
        """
        Calculate SHAP values for input data.

        Args:
            X: Input data [n_samples, n_features]
            check_additivity: Verify SHAP values sum to prediction

        Returns:
            SHAP values [n_samples, n_features] or [n_samples, n_features, n_classes]
        """
        logger.info(f"Calculating SHAP values for {len(X)} samples...")

        self.shap_values = self.explainer.shap_values(
            X,
            check_additivity=check_additivity,
        )

        return self.shap_values

    def plot_summary(
        self,
        X: np.ndarray,
        shap_values: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ):
        """
        Create summary plot showing feature importance.

        Args:
            X: Input data
            shap_values: SHAP values (if None, uses self.shap_values)
            save_path: Path to save plot
        """
        if shap_values is None:
            if self.shap_values is None:
                raise ValueError("No SHAP values available. Call explain() first.")
            shap_values = self.shap_values

        plt.figure(figsize=(10, 8))

        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.config.feature_names,
            show=False,
            max_display=self.config.n_features,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved summary plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_waterfall(
        self,
        X: np.ndarray,
        index: int = 0,
        shap_values: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ):
        """
        Create waterfall plot for single prediction.

        Shows how each feature contributes to pushing the prediction
        from the base value to the final prediction.

        Args:
            X: Input data
            index: Index of sample to explain
            shap_values: SHAP values
            save_path: Path to save plot
        """
        if shap_values is None:
            if self.shap_values is None:
                raise ValueError("No SHAP values available. Call explain() first.")
            shap_values = self.shap_values

        plt.figure(figsize=(10, 6))

        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values_sample = shap_values[0][index]
        else:
            shap_values_sample = shap_values[index]

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_sample,
                base_values=self.explainer.expected_value if not isinstance(self.explainer.expected_value, list) else self.explainer.expected_value[0],
                data=X[index],
                feature_names=self.config.feature_names,
            ),
            show=False,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved waterfall plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_force(
        self,
        X: np.ndarray,
        index: int = 0,
        shap_values: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ):
        """
        Create force plot for single prediction.

        Args:
            X: Input data
            index: Index of sample to explain
            shap_values: SHAP values
            save_path: Path to save plot
        """
        if shap_values is None:
            if self.shap_values is None:
                raise ValueError("No SHAP values available. Call explain() first.")
            shap_values = self.shap_values

        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values_sample = shap_values[0][index]
            expected_value = self.explainer.expected_value[0]
        else:
            shap_values_sample = shap_values[index]
            expected_value = self.explainer.expected_value

        shap.force_plot(
            expected_value,
            shap_values_sample,
            X[index],
            feature_names=self.config.feature_names,
            matplotlib=True,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved force plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_dependence(
        self,
        feature: str,
        X: np.ndarray,
        shap_values: Optional[np.ndarray] = None,
        interaction_feature: Optional[str] = None,
        save_path: Optional[Path] = None,
    ):
        """
        Create dependence plot for feature.

        Shows relationship between feature value and SHAP value,
        optionally colored by interaction feature.

        Args:
            feature: Feature name to plot
            X: Input data
            shap_values: SHAP values
            interaction_feature: Feature to color by
            save_path: Path to save plot
        """
        if shap_values is None:
            if self.shap_values is None:
                raise ValueError("No SHAP values available. Call explain() first.")
            shap_values = self.shap_values

        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        plt.figure(figsize=(10, 6))

        shap.dependence_plot(
            feature,
            shap_values,
            X,
            feature_names=self.config.feature_names,
            interaction_index=interaction_feature,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved dependence plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def get_feature_importance(
        self,
        shap_values: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Get feature importance from SHAP values.

        Returns:
            DataFrame with features and their importance
        """
        if shap_values is None:
            if self.shap_values is None:
                raise ValueError("No SHAP values available. Call explain() first.")
            shap_values = self.shap_values

        # Handle multi-class case
        if isinstance(shap_values, list):
            # Average absolute SHAP values across classes
            importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)

        df = pd.DataFrame({
            'feature': self.config.feature_names or [f'feature_{i}' for i in range(len(importance))],
            'importance': importance,
        })

        df = df.sort_values('importance', ascending=False).reset_index(drop=True)

        return df


class LIMEExplainer:
    """
    LIME explainer for local interpretability.

    LIME explains individual predictions by approximating
    the model locally with an interpretable model.

    Features:
    - Model-agnostic
    - Local fidelity
    - Feature perturbations
    - Individual explanations
    """

    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        config: ExplainerConfig,
        mode: str = 'classification',
    ):
        self.model = model
        self.config = config
        self.mode = mode

        # Create LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=config.feature_names,
            class_names=config.class_names,
            mode=mode,
            discretize_continuous=True,
        )

        logger.info("Initialized LIME explainer")

    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: Optional[int] = None,
    ) -> lime.explanation.Explanation:
        """
        Explain single prediction.

        Args:
            instance: Single instance to explain [n_features]
            num_features: Number of features to include

        Returns:
            LIME explanation object
        """
        if num_features is None:
            num_features = self.config.n_features

        # Prediction function
        def predict_fn(X):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                # Convert predictions to probabilities
                preds = self.model.predict(X)
                n_classes = len(self.config.class_names or [0, 1, 2])
                probs = np.zeros((len(preds), n_classes))
                probs[np.arange(len(preds)), preds] = 1.0
                return probs

        explanation = self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
        )

        return explanation

    def plot_explanation(
        self,
        explanation: lime.explanation.Explanation,
        label: int = 1,
        save_path: Optional[Path] = None,
    ):
        """
        Plot LIME explanation.

        Args:
            explanation: LIME explanation
            label: Class label to explain
            save_path: Path to save plot
        """
        fig = explanation.as_pyplot_figure(label=label)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved LIME plot to {save_path}")
        else:
            plt.show()

        plt.close()


class ModelExplainer:
    """
    Unified interface for model explainability.

    Supports both SHAP and LIME, automatically selecting
    the appropriate method based on model type.
    """

    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        config: ExplainerConfig,
    ):
        self.model = model
        self.X_train = X_train
        self.config = config

        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None

        if config.explainer_type in ['shap', 'both']:
            try:
                self.shap_explainer = SHAPExplainer(model, X_train, config)
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP: {e}")

        if config.explainer_type in ['lime', 'both']:
            try:
                self.lime_explainer = LIMEExplainer(model, X_train, config)
            except Exception as e:
                logger.warning(f"Failed to initialize LIME: {e}")

    def explain_global(
        self,
        X: np.ndarray,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate global explanations (feature importance).

        Args:
            X: Input data
            save_dir: Directory to save plots

        Returns:
            Dictionary with explanations
        """
        results = {}

        if self.shap_explainer:
            logger.info("Generating SHAP global explanations...")

            # Calculate SHAP values
            shap_values = self.shap_explainer.explain(X)

            # Summary plot
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                self.shap_explainer.plot_summary(
                    X, shap_values,
                    save_path=save_dir / 'shap_summary.png'
                )

            # Feature importance
            importance_df = self.shap_explainer.get_feature_importance(shap_values)
            results['shap_importance'] = importance_df

        return results

    def explain_local(
        self,
        instance: np.ndarray,
        index: int = 0,
        save_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate local explanations for single instance.

        Args:
            instance: Single instance
            index: Index for plot naming
            save_dir: Directory to save plots

        Returns:
            Dictionary with explanations
        """
        results = {}

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # SHAP
        if self.shap_explainer:
            logger.info("Generating SHAP local explanation...")

            # Calculate SHAP values
            shap_values = self.shap_explainer.explain(instance.reshape(1, -1))

            # Waterfall plot
            if save_dir:
                self.shap_explainer.plot_waterfall(
                    instance.reshape(1, -1), 0, shap_values,
                    save_path=save_dir / f'shap_waterfall_{index}.png'
                )

            results['shap_values'] = shap_values

        # LIME
        if self.lime_explainer:
            logger.info("Generating LIME local explanation...")

            # Explain instance
            explanation = self.lime_explainer.explain_instance(instance)

            # Plot
            if save_dir:
                self.lime_explainer.plot_explanation(
                    explanation,
                    save_path=save_dir / f'lime_explanation_{index}.png'
                )

            results['lime_explanation'] = explanation

        return results


if __name__ == '__main__':
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42,
    )

    feature_names = [f'feature_{i}' for i in range(20)]
    class_names = ['SELL', 'HOLD', 'BUY']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Create explainer
    config = ExplainerConfig(
        explainer_type='both',
        feature_names=feature_names,
        class_names=class_names,
        n_features=10,
    )

    explainer = ModelExplainer(model, X_train, config)

    # Global explanations
    global_results = explainer.explain_global(
        X_test[:100],
        save_dir=Path('outputs/explainability/global')
    )

    print("\nTop 10 Important Features:")
    print(global_results['shap_importance'].head(10))

    # Local explanation
    local_results = explainer.explain_local(
        X_test[0],
        index=0,
        save_dir=Path('outputs/explainability/local')
    )

    print("\nLocal explanations saved!")
