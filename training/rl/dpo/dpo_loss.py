#!/usr/bin/env python3
"""DPO Loss Implementation - Task 10.2"""
import numpy as np
from loguru import logger

class DPOLoss:
    def __init__(self, beta=0.1):
        self.beta = beta
        logger.info(f"DPOLoss initialized: beta={beta}")
    
    def compute(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        loss = -np.log(1 / (1 + np.exp(-self.beta * logits))).mean()
        return loss

if __name__ == "__main__":
    print("✓ DPO Loss ready")
