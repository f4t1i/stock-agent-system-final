#!/usr/bin/env python3
"""Training Monitoring - Task 11.4"""
from loguru import logger

class TrainingMonitor:
    def __init__(self):
        self.metrics = []
        logger.info("TrainingMonitor initialized")
    
    def log_metrics(self, step, metrics):
        self.metrics.append({'step': step, **metrics})
        if step % 10 == 0:
            logger.info(f"Step {step}: {metrics}")
    
    def get_summary(self):
        return {'total_steps': len(self.metrics)}

if __name__ == "__main__":
    print("✓ Training Monitor ready")
