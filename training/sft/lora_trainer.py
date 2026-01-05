"""
LoRA/QLoRA Trainer - Base class for SFT training of junior agents

Purpose:
    Unified training infrastructure for News, Technical, and Fundamental agents
    using LoRA (Low-Rank Adaptation) or QLoRA (Quantized LoRA) fine-tuning.

Features:
    - Multi-model support (Llama, Mistral, Gemma, Phi)
    - LoRA/QLoRA with 4-bit quantization
    - Gradient checkpointing and mixed precision
    - Model versioning and checkpointing
    - Evaluation gates and regression testing
    - Integration with Hugging Face Transformers

Usage:
    trainer = LoRATrainer(
        agent_name="news_agent",
        config_path="training/sft/sft_config.yaml"
    )

    trainer.train(
        train_dataset=train_data,
        eval_dataset=eval_data
    )

    trainer.save_model("models/sft/news_agent_v1.0.0")
"""

import os
import yaml
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger

# Hugging Face imports (will be installed)
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
    from datasets import Dataset
    import bitsandbytes as bnb
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Hugging Face libraries not installed. Install with: pip install transformers peft datasets bitsandbytes accelerate")


@dataclass
class TrainingResult:
    """Result from training run"""
    agent_name: str
    model_path: str
    version: str

    # Training metrics
    final_train_loss: float
    final_eval_loss: float
    best_eval_loss: float

    # Performance metrics
    eval_accuracy: Optional[float] = None
    eval_f1: Optional[float] = None

    # Training info
    total_epochs: int = 0
    total_steps: int = 0
    training_time_seconds: float = 0.0

    # Model info
    base_model: str = ""
    lora_r: int = 0
    lora_alpha: int = 0

    # Dataset info
    dataset_version: str = ""
    num_train_examples: int = 0
    num_eval_examples: int = 0

    # Gates
    passed_eval_gates: bool = False
    passed_regression_test: bool = False

    timestamp: str = ""


class LoRATrainer:
    """
    Base trainer for LoRA/QLoRA fine-tuning of language models

    Supports:
    - LoRA: Low-rank adaptation without quantization
    - QLoRA: 4-bit quantized LoRA (saves 75% memory)
    """

    def __init__(
        self,
        agent_name: str,
        config_path: Optional[Path] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize trainer

        Args:
            agent_name: Name of agent (news_agent, technical_agent, fundamental_agent)
            config_path: Path to YAML config file
            config: Config dict (overrides config_path)
        """
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face libraries not installed. Run: pip install transformers peft datasets bitsandbytes accelerate")

        self.agent_name = agent_name

        # Load configuration
        if config:
            self.config = config
        elif config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            # Load default config
            default_config = Path(__file__).parent / "sft_config.yaml"
            with open(default_config) as f:
                self.config = yaml.safe_load(f)

        # Get agent-specific config
        self.agent_config = self.config["agents"].get(agent_name, {})

        # Model and tokenizer (initialized in setup())
        self.model = None
        self.tokenizer = None
        self.trainer = None

        logger.info(f"LoRATrainer initialized for {agent_name}")
        logger.info(f"Base model: {self.agent_config.get('model_base', 'default')}")

    def setup(self, preset: Optional[str] = None):
        """
        Setup model, tokenizer, and LoRA config

        Args:
            preset: Training preset (quick_test, production, high_quality)
        """
        logger.info("Setting up model and tokenizer...")

        # Apply preset if specified
        if preset and preset in self.config.get("presets", {}):
            preset_config = self.config["presets"][preset]
            self.config["training"].update(preset_config)
            logger.info(f"Applied preset: {preset}")

        # Get model configuration
        model_base = self.agent_config.get("model_base", "llama3_8b")
        model_config = self.config["models"]["base_models"][model_base]

        model_name = model_config["model_name"]
        tokenizer_name = model_config["tokenizer_name"]

        logger.info(f"Loading model: {model_name}")

        # Setup quantization config for QLoRA
        use_qlora = self.config["lora"].get("use_qlora", True)

        if use_qlora:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config["lora"]["bnb_4bit_use_double_quant"],
                bnb_4bit_quant_type=self.config["lora"]["bnb_4bit_quant_type"],
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config["lora"]["bnb_4bit_compute_dtype"] == "bfloat16" else torch.float16
            )

            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=self.config["hardware"].get("device_map", "auto"),
                trust_remote_code=True
            )

            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)

            logger.info("Model loaded with 4-bit quantization (QLoRA)")
        else:
            # Load model without quantization (standard LoRA)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.config["hardware"].get("device_map", "auto"),
                torch_dtype=torch.bfloat16 if self.config["training"]["bf16"] else torch.float16,
                trust_remote_code=True
            )

            logger.info("Model loaded without quantization (LoRA)")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup LoRA
        self._setup_lora(model_base)

        logger.info("Setup complete")

    def _setup_lora(self, model_base: str):
        """Setup LoRA configuration"""
        logger.info("Configuring LoRA...")

        # Get target modules for this model architecture
        target_modules = self._get_target_modules(model_base)

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            target_modules=target_modules,
            lora_dropout=self.config["lora"]["lora_dropout"],
            bias=self.config["lora"]["bias"],
            task_type=TaskType.CAUSAL_LM
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = 100 * trainable_params / total_params

        logger.info(f"LoRA configured:")
        logger.info(f"  Rank: {self.config['lora']['r']}")
        logger.info(f"  Alpha: {self.config['lora']['lora_alpha']}")
        logger.info(f"  Target modules: {target_modules}")
        logger.info(f"  Trainable params: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)")

    def _get_target_modules(self, model_base: str) -> List[str]:
        """Get target modules for LoRA based on model architecture"""
        # Map model base to architecture
        if "llama" in model_base.lower():
            arch = "llama"
        elif "mistral" in model_base.lower():
            arch = "mistral"
        elif "gemma" in model_base.lower():
            arch = "gemma"
        elif "phi" in model_base.lower():
            arch = "phi"
        else:
            arch = "llama"  # Default

        return self.config["lora"]["target_modules"][arch]

    def prepare_dataset(
        self,
        dataset_path: Path,
        split: str = "train"
    ) -> Dataset:
        """
        Prepare dataset from JSONL file

        Args:
            dataset_path: Path to dataset file (train.jsonl, val.jsonl, test.jsonl)
            split: Dataset split name

        Returns:
            Hugging Face Dataset
        """
        logger.info(f"Preparing {split} dataset from {dataset_path}")

        # Load JSONL
        data = []
        with open(dataset_path) as f:
            for line in f:
                data.append(json.loads(line))

        logger.info(f"Loaded {len(data)} examples")

        # Convert to HF Dataset
        dataset = Dataset.from_list(data)

        # Tokenize
        def tokenize_function(examples):
            # Get messages (chat format)
            messages_list = examples.get("messages", [])

            # Apply chat template
            texts = []
            for messages in messages_list:
                if isinstance(messages, list):
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                else:
                    texts.append("")  # Handle malformed data

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config["dataset"]["max_seq_length"],
                padding=False  # Dynamic padding in collator
            )

            # Copy input_ids to labels for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split} dataset"
        )

        logger.info(f"Dataset prepared: {len(tokenized_dataset)} examples")

        return tokenized_dataset

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: Optional[Path] = None
    ) -> TrainingResult:
        """
        Train the model

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Output directory for checkpoints

        Returns:
            TrainingResult with metrics
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not setup. Call setup() first.")

        logger.info("Starting training...")
        start_time = datetime.now()

        # Setup output directory
        if output_dir is None:
            output_dir = Path(self.config["registry"]["models_dir"]) / f"{self.agent_name}_temp"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create training arguments
        training_args = self._create_training_arguments(output_dir)

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Create trainer
        callbacks = []

        # Early stopping
        if self.config["training"].get("early_stopping_patience"):
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.config["training"]["early_stopping_patience"],
                early_stopping_threshold=self.config["training"]["early_stopping_threshold"]
            ))

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks
        )

        # Train
        logger.info(f"Training for {self.config['training']['num_epochs']} epochs...")
        train_result = self.trainer.train()

        # Training complete
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        logger.info(f"Training complete in {training_time:.1f}s")
        logger.info(f"Final train loss: {train_result.training_loss:.4f}")

        # Evaluate
        eval_metrics = {}
        if eval_dataset is not None:
            logger.info("Running final evaluation...")
            eval_metrics = self.trainer.evaluate()
            logger.info(f"Final eval loss: {eval_metrics.get('eval_loss', 0):.4f}")

        # Create result
        result = TrainingResult(
            agent_name=self.agent_name,
            model_path=str(output_dir),
            version="1.0.0",  # Will be set by registry
            final_train_loss=train_result.training_loss,
            final_eval_loss=eval_metrics.get("eval_loss", 0.0),
            best_eval_loss=eval_metrics.get("eval_loss", 0.0),
            total_epochs=self.config["training"]["num_epochs"],
            total_steps=train_result.global_step,
            training_time_seconds=training_time,
            base_model=self.agent_config.get("model_base", "unknown"),
            lora_r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            num_train_examples=len(train_dataset),
            num_eval_examples=len(eval_dataset) if eval_dataset else 0,
            timestamp=datetime.now().isoformat()
        )

        # Check evaluation gates
        result.passed_eval_gates = self._check_eval_gates(eval_metrics)

        logger.info(f"Evaluation gates: {'✅ PASSED' if result.passed_eval_gates else '❌ FAILED'}")

        return result

    def _create_training_arguments(self, output_dir: Path) -> TrainingArguments:
        """Create Hugging Face TrainingArguments"""
        cfg = self.config["training"]

        return TrainingArguments(
            output_dir=str(output_dir),

            # Training
            num_train_epochs=cfg["num_epochs"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],

            # Optimizer
            optim=self.config["hardware"]["optim"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            adam_beta1=cfg["adam_beta1"],
            adam_beta2=cfg["adam_beta2"],
            adam_epsilon=cfg["adam_epsilon"],
            max_grad_norm=cfg["max_grad_norm"],

            # LR scheduler
            lr_scheduler_type=cfg["lr_scheduler_type"],
            warmup_ratio=cfg["warmup_ratio"],
            warmup_steps=cfg.get("warmup_steps"),

            # Mixed precision
            fp16=cfg["fp16"],
            bf16=cfg["bf16"],

            # Gradient checkpointing
            gradient_checkpointing=cfg["gradient_checkpointing"],

            # Logging
            logging_steps=cfg["logging_steps"],
            logging_first_step=cfg["logging_first_step"],
            logging_strategy=cfg["logging_strategy"],

            # Evaluation
            evaluation_strategy=cfg["evaluation_strategy"],
            eval_steps=cfg.get("eval_steps"),
            eval_accumulation_steps=cfg.get("eval_accumulation_steps"),

            # Checkpointing
            save_strategy=cfg["save_strategy"],
            save_steps=cfg.get("save_steps"),
            save_total_limit=cfg["save_total_limit"],
            load_best_model_at_end=cfg["load_best_model_at_end"],
            metric_for_best_model=cfg["metric_for_best_model"],
            greater_is_better=cfg["greater_is_better"],

            # Reproducibility
            seed=cfg["seed"],
            data_seed=cfg["data_seed"],

            # Reporting
            report_to="tensorboard" if self.config["monitoring"]["use_tensorboard"] else "none",

            # Other
            remove_unused_columns=False,
            dataloader_num_workers=self.config["hardware"]["dataloader_num_workers"],
            dataloader_pin_memory=self.config["hardware"]["dataloader_pin_memory"]
        )

    def _check_eval_gates(self, eval_metrics: Dict) -> bool:
        """Check if model passes evaluation gates"""
        gates_config = self.config["eval_gates"]

        # Check minimum thresholds
        if gates_config.get("min_eval_loss") and eval_metrics.get("eval_loss", float("inf")) > gates_config["min_eval_loss"]:
            logger.warning(f"Failed min_eval_loss gate: {eval_metrics.get('eval_loss')} > {gates_config['min_eval_loss']}")
            return False

        if gates_config.get("min_eval_accuracy") and eval_metrics.get("eval_accuracy", 0) < gates_config["min_eval_accuracy"]:
            logger.warning(f"Failed min_eval_accuracy gate: {eval_metrics.get('eval_accuracy')} < {gates_config['min_eval_accuracy']}")
            return False

        if gates_config.get("min_eval_f1") and eval_metrics.get("eval_f1", 0) < gates_config["min_eval_f1"]:
            logger.warning(f"Failed min_eval_f1 gate: {eval_metrics.get('eval_f1')} < {gates_config['min_eval_f1']}")
            return False

        logger.info("All evaluation gates passed ✅")
        return True

    def save_model(self, output_dir: Path, version: str = "1.0.0"):
        """
        Save fine-tuned model

        Args:
            output_dir: Output directory
            version: Model version
        """
        if self.trainer is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to {output_dir}")

        # Save model and tokenizer
        self.trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        # Save metadata
        metadata = {
            "agent_name": self.agent_name,
            "version": version,
            "base_model": self.agent_config.get("model_base"),
            "lora_r": self.config["lora"]["r"],
            "lora_alpha": self.config["lora"]["lora_alpha"],
            "timestamp": datetime.now().isoformat()
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved: {output_dir}")


# CLI Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA/QLoRA Trainer for SFT")
    parser.add_argument("--agent", required=True, choices=["news_agent", "technical_agent", "fundamental_agent"], help="Agent to train")
    parser.add_argument("--train-data", type=Path, required=True, help="Path to train.jsonl")
    parser.add_argument("--eval-data", type=Path, help="Path to val.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, help="Config YAML file")
    parser.add_argument("--preset", choices=["quick_test", "production", "high_quality"], help="Training preset")

    args = parser.parse_args()

    # Initialize trainer
    trainer = LoRATrainer(
        agent_name=args.agent,
        config_path=args.config
    )

    # Setup
    trainer.setup(preset=args.preset)

    # Prepare datasets
    train_dataset = trainer.prepare_dataset(args.train_data, split="train")
    eval_dataset = trainer.prepare_dataset(args.eval_data, split="val") if args.eval_data else None

    # Train
    result = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir
    )

    # Save
    trainer.save_model(args.output_dir)

    print(f"\n✅ Training complete!")
    print(f"   Final train loss: {result.final_train_loss:.4f}")
    print(f"   Final eval loss: {result.final_eval_loss:.4f}")
    print(f"   Training time: {result.training_time_seconds:.1f}s")
    print(f"   Eval gates: {'✅ PASSED' if result.passed_eval_gates else '❌ FAILED'}")
    print(f"   Model saved: {args.output_dir}")
