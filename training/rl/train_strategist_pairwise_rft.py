"""
Reinforcement Fine-Tuning (RFT) with Pairwise Rewards

Trains strategist using pairwise comparisons instead of absolute scores.

Key Differences from Standard RL:
1. Generate pairs of outputs for same input
2. Judge compares pairs (which is better?)
3. Winner gets positive reward, loser gets negative
4. More stable than absolute scoring

Based on:
- RLHF (Reinforcement Learning from Human Feedback)
- DPO (Direct Preference Optimization)
- RLAIF (RL from AI Feedback)

Training Flow:
1. Generate pair of strategies for market state
2. Judge compares and picks winner
3. Compute pairwise rewards
4. Update model to prefer winner
5. Repeat

Advantages:
- Easier for judge (compare vs score)
- More consistent judgments
- Reduces reward hacking
- Better for complex reasoning
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
from tqdm import tqdm
import wandb

from training.pairwise.pairwise_comparison import (
    PairwiseJudge,
    PairwiseDataGenerator,
    PairwiseTrainingDataset,
    ComparisonResult,
    StrategyOutput
)


class PairwiseRFTConfig:
    """Configuration for Pairwise RFT Training"""
    
    def __init__(self):
        # Model
        self.model_name = "gpt-4.1-mini"
        self.use_lora = True
        self.lora_rank = 16
        self.lora_alpha = 32
        
        # Training
        self.num_epochs = 3
        self.batch_size = 4
        self.learning_rate = 5e-5
        self.warmup_steps = 100
        self.max_grad_norm = 1.0
        
        # Pairwise
        self.temperature_a = 0.7  # Lower temp for strategy A
        self.temperature_b = 0.9  # Higher temp for strategy B
        self.margin = 0.5  # Margin for ranking loss
        
        # Data
        self.num_comparison_pairs = 1000
        self.train_split = 0.8
        
        # Logging
        self.log_interval = 10
        self.eval_interval = 100
        self.save_interval = 500
        
        # Paths
        self.output_dir = "models/strategist_pairwise_rft"
        self.data_dir = "data/pairwise_comparisons"
        
        # WandB
        self.use_wandb = True
        self.wandb_project = "stock-agent-pairwise-rft"


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss
    
    Encourages model to rank winner higher than loser
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        score_winner: torch.Tensor,
        score_loser: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ranking loss
        
        Loss = max(0, margin - (score_winner - score_loser))
        
        Args:
            score_winner: Scores for winning strategies
            score_loser: Scores for losing strategies
        
        Returns:
            Loss value
        """
        # Hinge loss: encourage winner to be higher by at least margin
        loss = F.relu(self.margin - (score_winner - score_loser))
        return loss.mean()


class PairwiseDataCollector:
    """
    Collects pairwise comparison data
    
    Generates strategy pairs and gets judge comparisons
    """
    
    def __init__(
        self,
        strategist_agent,
        judge: PairwiseJudge,
        data_generator: PairwiseDataGenerator,
        config: PairwiseRFTConfig
    ):
        self.strategist = strategist_agent
        self.judge = judge
        self.data_generator = data_generator
        self.config = config
        
        self.dataset = PairwiseTrainingDataset(
            save_path=os.path.join(config.data_dir, "comparisons.jsonl")
        )
    
    def collect(
        self,
        market_states: List[Dict],
        agent_outputs_list: List[Dict]
    ) -> PairwiseTrainingDataset:
        """
        Collect pairwise comparison data
        
        Args:
            market_states: List of market states
            agent_outputs_list: List of agent outputs for each state
        
        Returns:
            Dataset with comparisons
        """
        logger.info(f"Collecting {len(market_states)} pairwise comparisons...")
        
        for market_state, agent_outputs in tqdm(
            zip(market_states, agent_outputs_list),
            total=len(market_states),
            desc="Collecting comparisons"
        ):
            try:
                # Generate pair of strategies
                strategy_a, strategy_b = self.data_generator.generate_pair(
                    market_state,
                    agent_outputs
                )
                
                # Judge comparison
                comparison = self.judge.compare(
                    market_state,
                    strategy_a,
                    strategy_b
                )
                
                # Add to dataset
                self.dataset.add_comparison(comparison)
                
                # Log
                if len(self.dataset.comparisons) % 10 == 0:
                    logger.info(f"Collected {len(self.dataset.comparisons)} comparisons")
            
            except Exception as e:
                logger.error(f"Error collecting comparison: {e}")
                continue
        
        # Save dataset
        self.dataset.save()
        
        # Log statistics
        stats = self.dataset.get_statistics()
        logger.info(f"Dataset statistics: {stats}")
        
        return self.dataset


class PairwiseRFTTrainer:
    """
    Trainer for Pairwise RFT
    
    Trains strategist using pairwise comparisons
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: PairwiseRFTConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Loss
        self.ranking_loss = PairwiseRankingLoss(margin=config.margin)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # WandB
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=vars(config)
            )
        
        # Metrics
        self.global_step = 0
        self.best_eval_loss = float('inf')
    
    def train(
        self,
        train_dataset: PairwiseTrainingDataset,
        eval_dataset: Optional[PairwiseTrainingDataset] = None
    ):
        """
        Train model on pairwise comparisons
        
        Args:
            train_dataset: Training comparisons
            eval_dataset: Evaluation comparisons (optional)
        """
        logger.info(f"Starting Pairwise RFT training...")
        logger.info(f"Train size: {len(train_dataset.comparisons)}")
        
        if eval_dataset:
            logger.info(f"Eval size: {len(eval_dataset.comparisons)}")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self._train_epoch(train_dataset)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Eval
            if eval_dataset:
                eval_loss = self._eval_epoch(eval_dataset)
                logger.info(f"Eval loss: {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint('best')
            
            # Save checkpoint
            self._save_checkpoint(f'epoch_{epoch + 1}')
            
            # Step scheduler
            self.scheduler.step()
        
        logger.info("Training completed!")
    
    def _train_epoch(self, dataset: PairwiseTrainingDataset) -> float:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle comparisons
        comparisons = dataset.comparisons.copy()
        np.random.shuffle(comparisons)
        
        for i in tqdm(range(0, len(comparisons), self.config.batch_size), desc="Training"):
            batch = comparisons[i:i + self.config.batch_size]
            
            # Skip unclear comparisons
            batch = [c for c in batch if c.winner != ComparisonResult.UNCLEAR]
            
            if not batch:
                continue
            
            # Compute loss
            loss = self._compute_batch_loss(batch)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Step
            self.optimizer.step()
            
            # Log
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if self.global_step % self.config.log_interval == 0:
                if self.config.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/step': self.global_step
                    })
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _eval_epoch(self, dataset: PairwiseTrainingDataset) -> float:
        """Evaluate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(dataset.comparisons), self.config.batch_size):
                batch = dataset.comparisons[i:i + self.config.batch_size]
                
                # Skip unclear
                batch = [c for c in batch if c.winner != ComparisonResult.UNCLEAR]
                
                if not batch:
                    continue
                
                # Compute loss
                loss = self._compute_batch_loss(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if self.config.use_wandb:
            wandb.log({
                'eval/loss': avg_loss,
                'eval/step': self.global_step
            })
        
        return avg_loss
    
    def _compute_batch_loss(self, batch: List) -> torch.Tensor:
        """
        Compute loss for a batch of comparisons
        
        For each comparison:
        1. Get model scores for winner and loser
        2. Compute ranking loss
        """
        winner_scores = []
        loser_scores = []
        
        for comparison in batch:
            # Determine winner and loser
            if comparison.winner == ComparisonResult.A_BETTER:
                winner = comparison.strategy_a
                loser = comparison.strategy_b
            elif comparison.winner == ComparisonResult.B_BETTER:
                winner = comparison.strategy_b
                loser = comparison.strategy_a
            elif comparison.winner == ComparisonResult.TIE:
                # For ties, treat both as winners (no loss)
                continue
            else:
                continue
            
            # Get model scores
            winner_score = self._score_strategy(winner, comparison.market_state)
            loser_score = self._score_strategy(loser, comparison.market_state)
            
            winner_scores.append(winner_score)
            loser_scores.append(loser_score)
        
        if not winner_scores:
            return torch.tensor(0.0, requires_grad=True)
        
        # Stack scores
        winner_scores = torch.stack(winner_scores)
        loser_scores = torch.stack(loser_scores)
        
        # Compute ranking loss
        loss = self.ranking_loss(winner_scores, loser_scores)
        
        return loss
    
    def _score_strategy(
        self,
        strategy: StrategyOutput,
        market_state: Dict
    ) -> torch.Tensor:
        """
        Score a strategy using the model
        
        In a real implementation, this would:
        1. Format strategy as text
        2. Tokenize
        3. Get model logits
        4. Compute reward score
        
        For now, use a simple heuristic
        """
        # Simple heuristic scoring
        # In production, use model forward pass
        
        score = 0.0
        
        # Reasoning length
        score += min(len(strategy.reasoning) / 500, 1.0) * 0.3
        
        # Has stop loss
        if strategy.stop_loss is not None:
            score += 0.2
        
        # Has take profit
        if strategy.take_profit is not None:
            score += 0.2
        
        # Confidence calibration
        score += (1.0 - abs(strategy.confidence - 0.7)) * 0.3
        
        return torch.tensor(score, requires_grad=True)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(self.config.output_dir, f"{name}.pt")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'config': vars(self.config)
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")


def main():
    """Main training script"""
    
    # Config
    config = PairwiseRFTConfig()
    
    # Load model and tokenizer
    # In production, load actual model
    # For now, use mock
    model = nn.Linear(10, 1)  # Mock model
    tokenizer = None  # Mock tokenizer
    
    # Load dataset
    dataset = PairwiseTrainingDataset(
        save_path=os.path.join(config.data_dir, "comparisons.jsonl")
    )
    dataset.load()
    
    if not dataset.comparisons:
        logger.error("No comparisons found. Run data collection first.")
        return
    
    # Split train/eval
    split_idx = int(len(dataset.comparisons) * config.train_split)
    
    train_dataset = PairwiseTrainingDataset()
    train_dataset.comparisons = dataset.comparisons[:split_idx]
    
    eval_dataset = PairwiseTrainingDataset()
    eval_dataset.comparisons = dataset.comparisons[split_idx:]
    
    # Trainer
    trainer = PairwiseRFTTrainer(model, tokenizer, config)
    
    # Train
    trainer.train(train_dataset, eval_dataset)


if __name__ == '__main__':
    main()
