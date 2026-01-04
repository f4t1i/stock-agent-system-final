"""
Supervised Fine-Tuning für News Sentiment Agent

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


class NewsSFTTrainer:
    """Trainer für News Sentiment Agent"""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        
        # Initialize wandb
        if self.config.get('use_wandb', True):
            wandb.init(
                project=self.config.get('wandb_project', 'stock-agent-system'),
                name=f"news-sft-{self.config['model']['base_model'].split('/')[-1]}",
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
                examples = json.load(f)
            dataset = Dataset.from_list(examples)
        
        else:
            # Try Hugging Face datasets
            dataset = load_dataset(dataset_path, split='train')
        
        logger.info(f"Loaded {len(dataset)} examples")
        
        # Format für SFT
        def format_prompt(example):
            messages = example['messages']
            
            # Apply chat template
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            return {'text': formatted}
        
        dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
        
        # Train/val split
        if 'validation' not in dataset:
            split = dataset.train_test_split(
                test_size=data_config.get('val_split', 0.1),
                seed=42
            )
            train_dataset = split['train']
            eval_dataset = split['test']
        else:
            train_dataset = dataset['train']
            eval_dataset = dataset['validation']
        
        return train_dataset, eval_dataset
    
    def train(self):
        """Run training"""
        
        # Prepare data
        train_dataset, eval_dataset = self.prepare_dataset()
        
        # Training arguments
        training_config = self.config['training']
        
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config.get('num_epochs', 3),
            per_device_train_batch_size=training_config.get('batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
            learning_rate=training_config.get('learning_rate', 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=training_config.get('logging_steps', 10),
            optim="adamw_8bit",
            weight_decay=training_config.get('weight_decay', 0.01),
            lr_scheduler_type=training_config.get('lr_scheduler', 'linear'),
            warmup_steps=training_config.get('warmup_steps', 100),
            save_steps=training_config.get('save_steps', 500),
            save_total_limit=3,
            evaluation_strategy="steps",
            eval_steps=training_config.get('eval_steps', 500),
            load_best_model_at_end=True,
            report_to="wandb" if self.config.get('use_wandb', True) else "none",
        )
        
        # Trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config['model'].get('max_seq_length', 2048),
            args=training_args,
            packing=False,  # Für Chat-Templates
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        final_path = Path(training_config['output_dir']) / 'final'
        final_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving final model to {final_path}")
        
        # Save with Unsloth
        model_to_save = trainer.model
        model_to_save.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        # Optional: Merge LoRA weights
        if training_config.get('merge_lora', False):
            logger.info("Merging LoRA weights...")
            merged_path = final_path / 'merged'
            merged_path.mkdir(exist_ok=True)
            
            model_to_save.save_pretrained_merged(
                merged_path,
                self.tokenizer,
                save_method="merged_16bit"
            )
        
        logger.info("Training complete!")
        
        return final_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    setup_logging('INFO')
    
    trainer = NewsSFTTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
