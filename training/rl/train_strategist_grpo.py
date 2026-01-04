"""
GRPO (Group Relative Policy Optimization) Training für Senior Strategist

Speichereffizientere Alternative zu PPO ohne Value-Model
"""

import os
from pathlib import Path
from typing import Dict, List
import json

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
import wandb

from utils.config_loader import load_config
from loguru import logger


class StrategistGRPOTrainer:
    """GRPO Trainer für Senior Strategist"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        
        # Initialize wandb
        if self.config.get('use_wandb', True):
            wandb.init(
                project=self.config.get('wandb_project', 'stock-agent-system'),
                name=f"strategist-grpo-{self.config['model']['base_model'].split('/')[-1]}",
                config=self.config
            )
        
        # Load SFT checkpoint als Ausgangspunkt
        self.model, self.tokenizer = self._load_model()
        
        # Reward function
        self.reward_function = self._build_reward_function()
        
    def _load_model(self):
        """Load SFT checkpoint"""
        
        model_config = self.config['model']
        
        # Load SFT model als Basis
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['sft_checkpoint'],
            max_seq_length=model_config.get('max_seq_length', 2048),
            dtype=None,
            load_in_4bit=model_config.get('load_in_4bit', True),
        )
        
        # Enable gradient checkpointing für GRPO
        model.gradient_checkpointing_enable()
        
        return model, tokenizer
    
    def _build_reward_function(self):
        """
        Konstruiere Reward-Funktion
        
        Reward = 0.5 * sharpe_contribution + 0.3 * logic_score + 0.2 * drawdown_penalty
        """
        
        from judge.llm_judge import LLMJudge
        
        reward_config = self.config['reward']
        
        # LLM Judge für Logic Score
        judge = LLMJudge(reward_config.get('judge_config', {}))
        
        def compute_reward(
            prediction: str,
            actual_outcome: Dict,
            context: Dict
        ) -> float:
            """
            Compute reward für eine Prediction
            
            Args:
                prediction: Model-Output (JSON-String)
                actual_outcome: {
                    'returns': float,  # Prozentuale Rendite
                    'volatility': float,
                    'max_drawdown': float
                }
                context: Original-Kontext
                
            Returns:
                Reward-Score (0 bis 1)
            """
            
            try:
                pred_dict = json.loads(prediction)
            except:
                # Parsing-Fehler = hohe Strafe
                return -1.0
            
            # 1. Financial Performance (Sharpe Contribution)
            returns = actual_outcome.get('returns', 0.0)
            volatility = actual_outcome.get('volatility', 0.01)
            sharpe_contribution = returns / max(volatility, 0.01)
            
            # Normalize zu [0, 1]
            sharpe_normalized = (sharpe_contribution + 2) / 4  # Assume range [-2, 2]
            sharpe_normalized = max(0, min(1, sharpe_normalized))
            
            # 2. Logic Quality (from LLM Judge)
            logic_score = judge.evaluate_reasoning(
                prediction=pred_dict,
                context=context
            )
            
            # 3. Drawdown Penalty
            max_dd = actual_outcome.get('max_drawdown', 0.0)
            dd_threshold = reward_config.get('drawdown_threshold', 0.10)
            
            if max_dd > dd_threshold:
                dd_penalty = -(max_dd - dd_threshold) * 5  # Skalierungsfaktor
            else:
                dd_penalty = 0
            
            dd_normalized = max(0, min(1, 1 + dd_penalty))
            
            # Kombiniere Komponenten
            weights = reward_config.get('weights', {
                'sharpe': 0.5,
                'logic': 0.3,
                'drawdown': 0.2
            })
            
            total_reward = (
                weights['sharpe'] * sharpe_normalized +
                weights['logic'] * logic_score +
                weights['drawdown'] * dd_normalized
            )
            
            return total_reward
        
        return compute_reward
    
    def prepare_dataset(self) -> Dataset:
        """
        Prepare RL dataset
        
        Format:
        {
            "query": "...",  # Input prompt
            "context": {...},  # Market context
            "reference": "...",  # Optional reference answer
            "actual_outcome": {...}  # Ground truth results
        }
        """
        
        data_config = self.config['data']
        dataset_path = data_config['dataset_path']
        
        logger.info(f"Loading RL dataset from {dataset_path}")
        
        examples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
        
        dataset = Dataset.from_list(examples)
        
        logger.info(f"Loaded {len(dataset)} RL examples")
        
        return dataset
    
    def train(self):
        """Run GRPO training"""
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        # GRPO Config
        training_config = self.config['training']
        
        grpo_config = GRPOConfig(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config.get('num_epochs', 1),
            per_device_train_batch_size=training_config.get('batch_size', 2),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
            learning_rate=training_config.get('learning_rate', 1e-5),
            
            # GRPO-specific
            num_generations=training_config.get('num_generations', 4),  # Anzahl Antworten pro Query
            kl_penalty=training_config.get('kl_penalty', 0.1),
            
            # Optimization
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            
            # Logging
            logging_steps=training_config.get('logging_steps', 5),
            save_steps=training_config.get('save_steps', 100),
            save_total_limit=2,
            report_to="wandb" if self.config.get('use_wandb', True) else "none",
        )
        
        # Reward wrapper
        def reward_wrapper(queries, responses, contexts):
            """
            Wrapper für reward function
            
            Args:
                queries: List of input queries
                responses: List of model responses
                contexts: List of context dicts
                
            Returns:
                List of rewards
            """
            rewards = []
            
            for query, response, context in zip(queries, responses, contexts):
                # Get actual outcome from context
                actual_outcome = context.get('actual_outcome', {})
                
                reward = self.reward_function(response, actual_outcome, context)
                rewards.append(reward)
            
            return rewards
        
        # Trainer
        trainer = GRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=grpo_config,
            train_dataset=dataset,
            reward_function=reward_wrapper,
        )
        
        # Train
        logger.info("Starting GRPO training...")
        trainer.train()
        
        # Save final model
        final_path = Path(training_config['output_dir']) / 'final'
        final_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving final model to {final_path}")
        trainer.save_model(final_path)
        
        logger.info("GRPO training complete!")
        
        return final_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    trainer = StrategistGRPOTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
