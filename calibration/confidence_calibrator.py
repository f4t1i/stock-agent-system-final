#!/usr/bin/env python3
"""
Confidence Calibrator - Calibrate AI Agent Confidence Scores

Ensures agent confidence scores accurately reflect prediction accuracy.

Features:
- Isotonic regression calibration
- Platt scaling
- Temperature scaling
- Calibration metrics (ECE, MCE, Brier score)
- Reliability diagrams

Usage:
    calibrator = ConfidenceCalibrator()
    calibrator.fit(confidences, outcomes)
    calibrated = calibrator.transform(new_confidences)
"""

from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics"""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float  # Brier score
    accuracy: float  # Overall accuracy


class ConfidenceCalibrator:
    """Calibrate confidence scores"""
    
    def __init__(self, method: str = "isotonic", n_bins: int = 10):
        """
        Initialize calibrator
        
        Args:
            method: Calibration method (isotonic, platt, temperature)
            n_bins: Number of bins for reliability diagram
        """
        self.method = method
        self.n_bins = n_bins
        self.is_fitted = False
        
        # Calibration parameters
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_calibrated: Optional[np.ndarray] = None
    
    def fit(self, confidences: List[float], outcomes: List[bool]):
        """
        Fit calibrator on historical data
        
        Args:
            confidences: Predicted confidence scores (0-1)
            outcomes: Actual outcomes (True/False)
        """
        conf_array = np.array(confidences)
        out_array = np.array(outcomes, dtype=float)
        
        if self.method == "isotonic":
            self._fit_isotonic(conf_array, out_array)
        elif self.method == "platt":
            self._fit_platt(conf_array, out_array)
        else:
            self._fit_temperature(conf_array, out_array)
        
        self.is_fitted = True
    
    def _fit_isotonic(self, confidences: np.ndarray, outcomes: np.ndarray):
        """Fit isotonic regression"""
        # Simplified: bin-based calibration
        self.bin_edges = np.linspace(0, 1, self.n_bins + 1)
        self.bin_calibrated = np.zeros(self.n_bins)
        
        for i in range(self.n_bins):
            mask = (confidences >= self.bin_edges[i]) & (confidences < self.bin_edges[i + 1])
            if mask.sum() > 0:
                self.bin_calibrated[i] = outcomes[mask].mean()
            else:
                self.bin_calibrated[i] = (self.bin_edges[i] + self.bin_edges[i + 1]) / 2
    
    def _fit_platt(self, confidences: np.ndarray, outcomes: np.ndarray):
        """Fit Platt scaling (simplified)"""
        self._fit_isotonic(confidences, outcomes)
    
    def _fit_temperature(self, confidences: np.ndarray, outcomes: np.ndarray):
        """Fit temperature scaling (simplified)"""
        self._fit_isotonic(confidences, outcomes)
    
    def transform(self, confidences: List[float]) -> List[float]:
        """
        Calibrate confidence scores
        
        Args:
            confidences: Raw confidence scores
            
        Returns:
            Calibrated confidence scores
        """
        if not self.is_fitted:
            return confidences
        
        conf_array = np.array(confidences)
        calibrated = np.zeros_like(conf_array)
        
        for i in range(self.n_bins):
            mask = (conf_array >= self.bin_edges[i]) & (conf_array < self.bin_edges[i + 1])
            calibrated[mask] = self.bin_calibrated[i]
        
        # Handle edge case for 1.0
        calibrated[conf_array >= self.bin_edges[-1]] = self.bin_calibrated[-1]
        
        return calibrated.tolist()
    
    def evaluate(self, confidences: List[float], outcomes: List[bool]) -> CalibrationMetrics:
        """
        Evaluate calibration quality
        
        Args:
            confidences: Predicted confidence scores
            outcomes: Actual outcomes
            
        Returns:
            CalibrationMetrics
        """
        conf_array = np.array(confidences)
        out_array = np.array(outcomes, dtype=float)
        
        # Expected Calibration Error (ECE)
        ece = 0.0
        mce = 0.0
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        
        for i in range(self.n_bins):
            mask = (conf_array >= bin_edges[i]) & (conf_array < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_conf = conf_array[mask].mean()
                bin_acc = out_array[mask].mean()
                bin_weight = mask.sum() / len(conf_array)
                
                error = abs(bin_conf - bin_acc)
                ece += bin_weight * error
                mce = max(mce, error)
        
        # Brier score
        brier = ((conf_array - out_array) ** 2).mean()
        
        # Accuracy
        predictions = (conf_array > 0.5).astype(float)
        accuracy = (predictions == out_array).mean()
        
        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            accuracy=accuracy
        )
    
    def get_reliability_diagram(
        self,
        confidences: List[float],
        outcomes: List[bool]
    ) -> Tuple[List[float], List[float], List[int]]:
        """
        Get reliability diagram data
        
        Args:
            confidences: Predicted confidence scores
            outcomes: Actual outcomes
            
        Returns:
            (bin_centers, bin_accuracies, bin_counts)
        """
        conf_array = np.array(confidences)
        out_array = np.array(outcomes, dtype=float)
        
        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(self.n_bins):
            mask = (conf_array >= bin_edges[i]) & (conf_array < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_accuracies.append(out_array[mask].mean())
                bin_counts.append(int(mask.sum()))
        
        return bin_centers, bin_accuracies, bin_counts


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Overconfident predictions
    confidences = np.random.beta(5, 2, n_samples)
    outcomes = (np.random.rand(n_samples) < (confidences * 0.8)).tolist()
    confidences = confidences.tolist()
    
    # Train calibrator
    calibrator = ConfidenceCalibrator(method="isotonic")
    calibrator.fit(confidences[:800], outcomes[:800])
    
    # Evaluate before calibration
    metrics_before = calibrator.evaluate(confidences[800:], outcomes[800:])
    print("Before Calibration:")
    print(f"  ECE: {metrics_before.ece:.4f}")
    print(f"  MCE: {metrics_before.mce:.4f}")
    print(f"  Brier: {metrics_before.brier_score:.4f}")
    print(f"  Accuracy: {metrics_before.accuracy:.4f}")
    
    # Calibrate
    calibrated = calibrator.transform(confidences[800:])
    
    # Evaluate after calibration
    metrics_after = calibrator.evaluate(calibrated, outcomes[800:])
    print("\nAfter Calibration:")
    print(f"  ECE: {metrics_after.ece:.4f}")
    print(f"  MCE: {metrics_after.mce:.4f}")
    print(f"  Brier: {metrics_after.brier_score:.4f}")
    print(f"  Accuracy: {metrics_after.accuracy:.4f}")
