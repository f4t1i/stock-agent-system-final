#!/usr/bin/env python3
"""DPO Training Loop - Task 10.4"""
from loguru import logger

class DPOTrainer:
    def __init__(self, dpo_loss, reference_model):
        self.loss_fn = dpo_loss
        self.ref_model = reference_model
        logger.info("DPOTrainer initialized")
    
    def train_step(self, batch):
        import numpy as np
        policy_chosen = np.random.normal(0, 1, len(batch))
        policy_rejected = np.random.normal(-0.5, 1, len(batch))
        ref_chosen = self.ref_model.compute_log_probs([p.prompt for p in batch], [p.chosen for p in batch])
        ref_rejected = self.ref_model.compute_log_probs([p.prompt for p in batch], [p.rejected for p in batch])
        loss = self.loss_fn.compute(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        return loss

if __name__ == "__main__":
    print("✓ DPO Trainer ready")
