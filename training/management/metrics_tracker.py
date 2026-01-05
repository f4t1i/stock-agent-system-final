#!/usr/bin/env python3
"""
Training Metrics Tracking - Task 7.3
Track and analyze training metrics over time.
Phase A1 Week 5-6: Task 7.3 COMPLETE
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class MetricPoint:
    """Single metric data point"""
    step: int
    epoch: Optional[float] = None
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class MetricsSummary:
    """Metrics summary statistics"""
    total_steps: int
    min_training_loss: Optional[float] = None
    max_training_loss: Optional[float] = None
    avg_training_loss: Optional[float] = None
    min_validation_loss: Optional[float] = None
    max_validation_loss: Optional[float] = None
    avg_validation_loss: Optional[float] = None
    final_training_loss: Optional[float] = None
    final_validation_loss: Optional[float] = None
    best_validation_loss: Optional[float] = None
    best_step: Optional[int] = None
    
    def __str__(self) -> str:
        return f"""Metrics Summary:
  Steps: {self.total_steps}
  Training Loss: {self.final_training_loss:.4f} (avg: {self.avg_training_loss:.4f})
  Validation Loss: {self.final_validation_loss:.4f} (best: {self.best_validation_loss:.4f} @ step {self.best_step})"""


class MetricsTracker:
    """Track and analyze training metrics"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.metrics: List[MetricPoint] = []
        logger.info(f"MetricsTracker initialized for job {job_id}")
    
    def add_metric(self, metric: MetricPoint):
        """Add metric point"""
        self.metrics.append(metric)
    
    def get_metrics(self, start_step: int = 0, end_step: Optional[int] = None) -> List[MetricPoint]:
        """Get metrics in range"""
        filtered = [m for m in self.metrics if m.step >= start_step]
        if end_step:
            filtered = [m for m in filtered if m.step <= end_step]
        return filtered
    
    def get_latest(self) -> Optional[MetricPoint]:
        """Get latest metric"""
        return self.metrics[-1] if self.metrics else None
    
    def calculate_summary(self) -> MetricsSummary:
        """Calculate summary statistics"""
        if not self.metrics:
            return MetricsSummary(total_steps=0)
        
        train_losses = [m.training_loss for m in self.metrics if m.training_loss is not None]
        val_losses = [m.validation_loss for m in self.metrics if m.validation_loss is not None]
        
        # Find best validation loss
        best_val_loss = None
        best_step = None
        if val_losses:
            best_val_loss = min(val_losses)
            for m in self.metrics:
                if m.validation_loss == best_val_loss:
                    best_step = m.step
                    break
        
        return MetricsSummary(
            total_steps=len(self.metrics),
            min_training_loss=min(train_losses) if train_losses else None,
            max_training_loss=max(train_losses) if train_losses else None,
            avg_training_loss=sum(train_losses) / len(train_losses) if train_losses else None,
            min_validation_loss=min(val_losses) if val_losses else None,
            max_validation_loss=max(val_losses) if val_losses else None,
            avg_validation_loss=sum(val_losses) / len(val_losses) if val_losses else None,
            final_training_loss=train_losses[-1] if train_losses else None,
            final_validation_loss=val_losses[-1] if val_losses else None,
            best_validation_loss=best_val_loss,
            best_step=best_step
        )
    
    def detect_overfitting(self, threshold: float = 0.1) -> bool:
        """Detect overfitting (validation loss >> training loss)"""
        latest = self.get_latest()
        if not latest or not latest.training_loss or not latest.validation_loss:
            return False
        
        diff = latest.validation_loss - latest.training_loss
        return diff > threshold
    
    def is_improving(self, window: int = 5) -> bool:
        """Check if metrics are improving"""
        if len(self.metrics) < window:
            return True
        
        recent = self.metrics[-window:]
        val_losses = [m.validation_loss for m in recent if m.validation_loss is not None]
        
        if len(val_losses) < 2:
            return True
        
        # Check if trending downward
        return val_losses[-1] < val_losses[0]


class MetricsAggregator:
    """Aggregate metrics across multiple jobs"""
    
    def __init__(self):
        self.trackers: Dict[str, MetricsTracker] = {}
    
    def add_tracker(self, tracker: MetricsTracker):
        """Add tracker"""
        self.trackers[tracker.job_id] = tracker
    
    def get_tracker(self, job_id: str) -> Optional[MetricsTracker]:
        """Get tracker by job ID"""
        return self.trackers.get(job_id)
    
    def compare_jobs(self, job_ids: List[str]) -> Dict[str, MetricsSummary]:
        """Compare metrics across jobs"""
        return {
            job_id: self.trackers[job_id].calculate_summary()
            for job_id in job_ids
            if job_id in self.trackers
        }
    
    def get_best_job(self, metric: str = 'validation_loss') -> Optional[str]:
        """Get job with best metric"""
        best_job = None
        best_value = float('inf')
        
        for job_id, tracker in self.trackers.items():
            summary = tracker.calculate_summary()
            if metric == 'validation_loss' and summary.best_validation_loss:
                if summary.best_validation_loss < best_value:
                    best_value = summary.best_validation_loss
                    best_job = job_id
        
        return best_job


if __name__ == "__main__":
    print("=== Metrics Tracker Test ===\n")
    
    # Test 1: Create tracker
    print("Test 1: Create tracker")
    tracker = MetricsTracker("job-123")
    print(f"✓ Tracker created for {tracker.job_id}\n")
    
    # Test 2: Add metrics
    print("Test 2: Add metrics")
    for i in range(5):
        tracker.add_metric(MetricPoint(
            step=i * 10,
            epoch=i * 0.5,
            training_loss=0.5 - i * 0.05,
            validation_loss=0.6 - i * 0.04
        ))
    print(f"✓ Added {len(tracker.metrics)} metrics\n")
    
    # Test 3: Calculate summary
    print("Test 3: Calculate summary")
    summary = tracker.calculate_summary()
    print(f"✓ {summary}\n")
    
    # Test 4: Check improvement
    print("Test 4: Check improvement")
    improving = tracker.is_improving(window=3)
    print(f"✓ Improving: {improving}\n")
    
    # Test 5: Detect overfitting
    print("Test 5: Detect overfitting")
    overfitting = tracker.detect_overfitting()
    print(f"✓ Overfitting: {overfitting}\n")
    
    print("=== Tests Complete ===")
