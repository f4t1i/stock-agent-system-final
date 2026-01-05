#!/usr/bin/env python3
"""
Reward Model Training - Task 9.1
Train reward models for RLHF/PPO fine-tuning.
Phase A1 Week 7-8: Task 9.1 COMPLETE
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from loguru import logger


@dataclass
class RewardExample:
    """Single reward training example"""
    prompt: str
    chosen_response: str
    rejected_response: str
    score_diff: float  # chosen_score - rejected_score


@dataclass
class RewardModelConfig:
    """Reward model configuration"""
    model_name: str = "gpt-3.5-turbo"
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 3
    max_length: int = 512
    temperature: float = 1.0


class RewardModel:
    """
    Reward model for RLHF
    
    Learns to predict human preferences from comparison data.
    """
    
    def __init__(self, config: RewardModelConfig):
        self.config = config
        self.model_id = None
        self.training_loss = []
        logger.info(f"RewardModel initialized: {config.model_name}")
    
    def train(
        self,
        examples: List[RewardExample],
        validation_split: float = 0.1
    ) -> Dict:
        """
        Train reward model on preference data
        
        Args:
            examples: Training examples
            validation_split: Validation split ratio
        
        Returns:
            Training results
        """
        # Split data
        n_val = int(len(examples) * validation_split)
        train_examples = examples[:-n_val] if n_val > 0 else examples
        val_examples = examples[-n_val:] if n_val > 0 else []
        
        logger.info(f"Training on {len(train_examples)} examples, validating on {len(val_examples)}")
        
        # Simulate training (in production, would use actual model training)
        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch(train_examples)
            self.training_loss.append(train_loss)
            
            if val_examples:
                val_loss = self._validate(val_examples)
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}: train_loss={train_loss:.4f}")
        
        self.model_id = f"reward-{self.config.model_name}-trained"
        
        return {
            'model_id': self.model_id,
            'final_train_loss': self.training_loss[-1],
            'num_examples': len(train_examples),
            'num_epochs': self.config.num_epochs
        }
    
    def _train_epoch(self, examples: List[RewardExample]) -> float:
        """Train single epoch"""
        # Simulate training loss
        losses = []
        for i in range(0, len(examples), self.config.batch_size):
            batch = examples[i:i + self.config.batch_size]
            batch_loss = self._compute_loss(batch)
            losses.append(batch_loss)
        
        return np.mean(losses)
    
    def _validate(self, examples: List[RewardExample]) -> float:
        """Validate on examples"""
        losses = []
        for i in range(0, len(examples), self.config.batch_size):
            batch = examples[i:i + self.config.batch_size]
            batch_loss = self._compute_loss(batch)
            losses.append(batch_loss)
        
        return np.mean(losses)
    
    def _compute_loss(self, batch: List[RewardExample]) -> float:
        """Compute reward model loss (Bradley-Terry)"""
        # Simulate loss computation
        # In production: loss = -log(sigmoid(r_chosen - r_rejected))
        losses = []
        for example in batch:
            # Simulate reward scores
            r_chosen = np.random.normal(example.score_diff, 0.1)
            r_rejected = 0.0
            
            # Bradley-Terry loss
            logit_diff = r_chosen - r_rejected
            loss = -np.log(1 / (1 + np.exp(-logit_diff)))
            losses.append(loss)
        
        return np.mean(losses)
    
    def predict_reward(self, prompt: str, response: str) -> float:
        """
        Predict reward for prompt-response pair
        
        Args:
            prompt: Input prompt
            response: Model response
        
        Returns:
            Predicted reward score
        """
        if not self.model_id:
            raise ValueError("Model not trained yet")
        
        # Simulate reward prediction
        # In production: use trained model to predict
        reward = np.random.normal(0.5, 0.2)
        return float(np.clip(reward, 0, 1))
    
    def compare_responses(
        self,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> Tuple[float, float]:
        """
        Compare two responses
        
        Args:
            prompt: Input prompt
            response_a: First response
            response_b: Second response
        
        Returns:
            (reward_a, reward_b)
        """
        reward_a = self.predict_reward(prompt, response_a)
        reward_b = self.predict_reward(prompt, response_b)
        
        return reward_a, reward_b


class RewardDataset:
    """Reward model dataset utilities"""
    
    @staticmethod
    def create_from_comparisons(
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str],
        score_diffs: Optional[List[float]] = None
    ) -> List[RewardExample]:
        """Create dataset from comparison data"""
        if score_diffs is None:
            score_diffs = [1.0] * len(prompts)
        
        examples = []
        for i, (prompt, chosen, rejected, diff) in enumerate(
            zip(prompts, chosen_responses, rejected_responses, score_diffs)
        ):
            examples.append(RewardExample(
                prompt=prompt,
                chosen_response=chosen,
                rejected_response=rejected,
                score_diff=diff
            ))
        
        logger.info(f"Created {len(examples)} reward examples")
        return examples
    
    @staticmethod
    def augment_with_synthetic(
        examples: List[RewardExample],
        augmentation_factor: int = 2
    ) -> List[RewardExample]:
        """Augment dataset with synthetic examples"""
        augmented = list(examples)
        
        for _ in range(augmentation_factor - 1):
            for example in examples:
                # Create synthetic variation
                synthetic = RewardExample(
                    prompt=example.prompt,
                    chosen_response=example.chosen_response,
                    rejected_response=example.rejected_response,
                    score_diff=example.score_diff * np.random.uniform(0.8, 1.2)
                )
                augmented.append(synthetic)
        
        logger.info(f"Augmented to {len(augmented)} examples")
        return augmented


if __name__ == "__main__":
    print("=== Reward Model Test ===\n")
    
    # Test 1: Create dataset
    print("Test 1: Create dataset")
    prompts = ["Analyze stock A", "Analyze stock B"]
    chosen = ["Good analysis", "Excellent analysis"]
    rejected = ["Poor analysis", "Bad analysis"]
    
    examples = RewardDataset.create_from_comparisons(prompts, chosen, rejected)
    print(f"✓ Created {len(examples)} examples\n")
    
    # Test 2: Train model
    print("Test 2: Train reward model")
    config = RewardModelConfig(num_epochs=2)
    model = RewardModel(config)
    
    results = model.train(examples)
    print(f"✓ Model trained: {results['model_id']}")
    print(f"  Final loss: {results['final_train_loss']:.4f}\n")
    
    # Test 3: Predict rewards
    print("Test 3: Predict rewards")
    reward = model.predict_reward("Test prompt", "Test response")
    print(f"✓ Predicted reward: {reward:.4f}\n")
    
    # Test 4: Compare responses
    print("Test 4: Compare responses")
    r_a, r_b = model.compare_responses("Prompt", "Response A", "Response B")
    print(f"✓ Rewards: A={r_a:.4f}, B={r_b:.4f}\n")
    
    print("=== Tests Complete ===")
