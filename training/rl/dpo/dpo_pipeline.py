#!/usr/bin/env python3
"""DPO Training Pipeline - Task 10.6"""
from loguru import logger

class DPOPipeline:
    def __init__(self, dpo_trainer, preference_dataset):
        self.trainer = dpo_trainer
        self.dataset = preference_dataset
        logger.info("DPO Pipeline initialized")
    
    def train(self, num_epochs=3, batch_size=8):
        logger.info(f"Training DPO for {num_epochs} epochs")
        for epoch in range(num_epochs):
            batch = self.dataset.get_batch(batch_size)
            loss = self.trainer.train_step(batch)
            logger.info(f"Epoch {epoch+1}: loss={loss:.4f}")
        return {'epochs': num_epochs, 'status': 'completed'}

if __name__ == "__main__":
    print("✓ DPO Pipeline ready")
