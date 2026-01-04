"""
Supervised Fine-Tuning für Technical Analysis Agent

Nutzt Unsloth für speichereffizientes Training
"""

import os
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import wandb

from utils.config_loader import load_config
from utils.logging_setup import setup_logging
from loguru import logger


class TechnicalSFTTrainer:
    """Trainer für Technical Analysis Agent"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        
        # Initialize wandb
        if self.config.get('use_wandb', True):
            wandb.init(
                project=self.config.get('wandb_project', 'stock-agent-system'),
                name=f"technical-sft-{self.config['model']['base_model'].split('/')[-1]}",
                config=self.config
            )
        
        # Load model mit Unsloth
        self.model, self.tokenizer = self._load_model()
        
    def _load_model(self):
        """Load base model mit Unsloth optimizations"""
        
        model_config = self.config['model']
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['base_model'],
            max_seq_length=model_config.get('max_seq_length', 2048),
            dtype=None,  # Auto-detect
            load_in_4bit=model_config.get('load_in_4bit', True),
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=model_config.get('lora_rank', 16),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=model_config.get('lora_alpha', 32),
            lora_dropout=model_config.get('lora_dropout', 0.05),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        return model, tokenizer
    
    def prepare_dataset(self) -> Dataset:
        """
        Prepare training dataset
        
        Erwartetes Format:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
        """
        
        data_config = self.config['data']
        dataset_path = data_config['dataset_path']
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        if dataset_path.endswith('.jsonl'):
            # Load JSONL
            examples = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    examples.append(json.loads(line))
            
            dataset = Dataset.from_list(examples)
        
        elif dataset_path.endswith('.json'):
            # Load JSON
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                dataset = Dataset.from_list(data)
            else:
                dataset = Dataset.from_dict(data)
        
        else:
            # Try loading from HuggingFace
            dataset = load_dataset(dataset_path, split='train')
        
        logger.info(f"Dataset loaded: {len(dataset)} examples")
        
        return dataset
    
    def train(self):
        """Run SFT training"""
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        # Training arguments
        training_config = self.config['training']
        
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            per_device_train_batch_size=training_config.get('batch_size', 2),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
            warmup_steps=training_config.get('warmup_steps', 100),
            max_steps=training_config.get('max_steps', 1000),
            learning_rate=training_config.get('learning_rate', 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=training_config.get('logging_steps', 10),
            optim="adamw_8bit",
            weight_decay=training_config.get('weight_decay', 0.01),
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
            seed=3407,
            save_strategy="steps",
            save_steps=training_config.get('save_steps', 100),
            report_to="wandb" if self.config.get('use_wandb', True) else "none",
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="messages",
            max_seq_length=self.config['model'].get('max_seq_length', 2048),
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        final_output_dir = Path(training_config['output_dir']) / 'final'
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving final model to {final_output_dir}")
        trainer.save_model(str(final_output_dir))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(final_output_dir))
        
        logger.info("Training complete!")
        
        if self.config.get('use_wandb', True):
            wandb.finish()


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Technical Analysis Agent")
    parser.add_argument(
        '--config',
        type=str,
        default='config/sft/technical_agent.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Train
    trainer = TechnicalSFTTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
