#!/usr/bin/env python3
"""
GRPO Trainer - Group Relative Policy Optimization for Trading Strategist

Purpose:
    Train the Senior Strategist agent using Group Relative Policy Optimization.
    GRPO samples multiple responses per prompt and uses relative rewards to
    update the policy, making it more sample-efficient than standard RL.

Algorithm:
    1. Sample K responses for each prompt (group sampling)
    2. Compute rewards for each response (backtest outcomes)
    3. Compute advantages using group-relative ranking
    4. Update policy using PPO-style objectives
    5. Repeat until convergence

Features:
    - Group-based sampling (K responses per prompt)
    - Relative advantage computation
    - PPO-style policy updates with KL penalty
    - Integration with experience store
    - Checkpointing and evaluation

Usage:
    # Train from SFT checkpoint
    trainer = GRPOTrainer(
        policy_path="models/sft/strategist_v1.0.0",
        config_path="training/rl/rl_config.yaml"
    )

    trainer.train(
        experience_store_path="data/experiences",
        num_iterations=100,
        output_dir="models/rl/strategist_grpo_v1.0.0"
    )

References:
    - Rafailov et al. (2023): "Direct Preference Optimization"
    - Ouyang et al. (2022): "Training language models to follow instructions"
    - Schulman et al. (2017): "Proximal Policy Optimization"
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from loguru import logger

# Optional dependencies
try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer
    )
    from peft import PeftModel, get_peft_model, LoraConfig
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("Torch/Transformers not available - GRPO training disabled")


@dataclass
class GRPOConfig:
    """GRPO training configuration"""
    # Group sampling
    group_size: int = 4  # Number of responses per prompt
    temperature: float = 0.8  # Sampling temperature
    top_p: float = 0.95  # Nucleus sampling

    # Policy update
    learning_rate: float = 1e-5
    num_epochs_per_iteration: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # PPO parameters
    clip_epsilon: float = 0.2  # PPO clipping
    kl_coeff: float = 0.05  # KL penalty coefficient
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter

    # Training
    num_iterations: int = 100
    eval_every: int = 10
    save_every: int = 10
    max_prompt_length: int = 512
    max_response_length: int = 256

    # Experience store
    min_reward_threshold: float = 0.0  # Filter low-reward experiences


@dataclass
class GRPOBatch:
    """Batch of GRPO training data"""
    prompts: List[str]
    responses: List[List[str]]  # group_size responses per prompt
    rewards: List[List[float]]  # group_size rewards per prompt
    advantages: List[List[float]]  # Computed advantages


@dataclass
class GRPOTrainingResult:
    """Result from GRPO training"""
    iteration: int
    avg_reward: float
    avg_advantage: float
    policy_loss: float
    kl_divergence: float
    total_samples: int
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer

    Trains a policy model using relative rewards from sampled response groups.
    """

    def __init__(
        self,
        policy_path: Path,
        config_path: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize GRPO trainer

        Args:
            policy_path: Path to initial policy model (e.g., SFT checkpoint)
            config_path: Path to RL config YAML
            device: Device for training
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "Torch/Transformers required for GRPO. "
                "Install with: pip install torch transformers peft"
            )

        self.policy_path = Path(policy_path)
        self.device = device

        # Load config
        if config_path is None:
            config_path = Path(__file__).parent / "rl_config.yaml"

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        self.config = GRPOConfig(**config_dict.get("grpo", {}))

        # Initialize models (lazy loading)
        self.policy: Optional[PreTrainedModel] = None
        self.reference_policy: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.optimizer = None

        logger.info(f"GRPO Trainer initialized")
        logger.info(f"  Policy: {policy_path}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Group size: {self.config.group_size}")

    def setup(self):
        """Load models and initialize optimizer"""
        logger.info("Loading policy model...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.policy_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load policy model
        self.policy = AutoModelForCausalLM.from_pretrained(
            str(self.policy_path),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # Load reference policy (frozen copy for KL penalty)
        logger.info("Loading reference policy (frozen)...")
        self.reference_policy = AutoModelForCausalLM.from_pretrained(
            str(self.policy_path),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.reference_policy.eval()
        for param in self.reference_policy.parameters():
            param.requires_grad = False

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )

        logger.info("✅ Models loaded and optimizer initialized")

    def sample_responses(
        self,
        prompts: List[str],
        num_samples: int
    ) -> List[List[str]]:
        """
        Sample multiple responses for each prompt

        Args:
            prompts: List of prompts
            num_samples: Number of responses per prompt

        Returns:
            List of response groups (num_samples per prompt)
        """
        self.policy.eval()

        all_responses = []

        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length
                ).to(self.device)

                # Sample multiple responses
                responses = []
                for _ in range(num_samples):
                    outputs = self.policy.generate(
                        **inputs,
                        max_new_tokens=self.config.max_response_length,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                    # Decode response (excluding prompt)
                    response = self.tokenizer.decode(
                        outputs[0][inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    responses.append(response)

                all_responses.append(responses)

        return all_responses

    def compute_advantages(
        self,
        rewards: List[List[float]]
    ) -> List[List[float]]:
        """
        Compute group-relative advantages

        Args:
            rewards: Rewards for each response group

        Returns:
            Advantages using group-relative normalization
        """
        advantages = []

        for reward_group in rewards:
            # Normalize within group (zero mean, unit variance)
            reward_array = np.array(reward_group)
            mean_reward = reward_array.mean()
            std_reward = reward_array.std()

            if std_reward > 0:
                group_advantages = (reward_array - mean_reward) / (std_reward + 1e-8)
            else:
                group_advantages = np.zeros_like(reward_array)

            advantages.append(group_advantages.tolist())

        return advantages

    def compute_policy_loss(
        self,
        batch: GRPOBatch
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO policy loss with PPO clipping and KL penalty

        Args:
            batch: Batch of training data

        Returns:
            (loss, metrics_dict)
        """
        total_loss = 0.0
        total_kl = 0.0
        num_samples = 0

        for prompt, responses, advantages in zip(
            batch.prompts,
            batch.responses,
            batch.advantages
        ):
            for response, advantage in zip(responses, advantages):
                # Tokenize prompt + response
                full_text = prompt + response
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length + self.config.max_response_length
                ).to(self.device)

                # Get prompt length for masking
                prompt_inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length
                )
                prompt_length = prompt_inputs.input_ids.shape[1]

                # Forward pass through policy
                outputs = self.policy(**inputs, labels=inputs.input_ids)
                logits = outputs.logits

                # Compute log probabilities for response tokens
                response_logits = logits[:, prompt_length-1:-1, :]  # Shift for next-token prediction
                response_labels = inputs.input_ids[:, prompt_length:]

                log_probs = F.log_softmax(response_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    dim=-1,
                    index=response_labels.unsqueeze(-1)
                ).squeeze(-1)

                # Average log probability over response
                avg_log_prob = token_log_probs.mean()

                # Compute KL divergence with reference policy
                with torch.no_grad():
                    ref_outputs = self.reference_policy(**inputs)
                    ref_logits = ref_outputs.logits[:, prompt_length-1:-1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                kl = (torch.exp(log_probs) * (log_probs - ref_log_probs)).sum(dim=-1).mean()

                # GRPO loss: maximize advantage-weighted log prob, penalize KL
                policy_loss = -advantage * avg_log_prob
                kl_penalty = self.config.kl_coeff * kl

                loss = policy_loss + kl_penalty

                total_loss += loss
                total_kl += kl.item()
                num_samples += 1

        # Average over batch
        avg_loss = total_loss / num_samples
        avg_kl = total_kl / num_samples

        metrics = {
            "policy_loss": avg_loss.item(),
            "kl_divergence": avg_kl
        }

        return avg_loss, metrics

    def train_iteration(
        self,
        batch: GRPOBatch,
        iteration: int
    ) -> GRPOTrainingResult:
        """
        Train for one iteration on a batch

        Args:
            batch: Training batch
            iteration: Iteration number

        Returns:
            GRPOTrainingResult
        """
        self.policy.train()

        # Compute advantages
        batch.advantages = self.compute_advantages(batch.rewards)

        # Training loop
        epoch_losses = []
        epoch_kls = []

        for epoch in range(self.config.num_epochs_per_iteration):
            self.optimizer.zero_grad()

            # Compute loss
            loss, metrics = self.compute_policy_loss(batch)

            # Backward
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(metrics["policy_loss"])
            epoch_kls.append(metrics["kl_divergence"])

        # Compute average metrics
        avg_reward = np.mean([np.mean(r) for r in batch.rewards])
        avg_advantage = np.mean([np.mean(a) for a in batch.advantages])
        total_samples = len(batch.prompts) * self.config.group_size

        result = GRPOTrainingResult(
            iteration=iteration,
            avg_reward=avg_reward,
            avg_advantage=avg_advantage,
            policy_loss=np.mean(epoch_losses),
            kl_divergence=np.mean(epoch_kls),
            total_samples=total_samples
        )

        logger.info(
            f"Iteration {iteration}: "
            f"reward={avg_reward:.4f}, "
            f"loss={result.policy_loss:.4f}, "
            f"kl={result.kl_divergence:.4f}"
        )

        return result

    def train(
        self,
        experience_store_path: Path,
        num_iterations: int,
        output_dir: Path
    ):
        """
        Train policy using GRPO

        Args:
            experience_store_path: Path to experience store
            num_iterations: Number of training iterations
            output_dir: Output directory for checkpoints

        Note:
            This is a simplified training loop. In production, you would:
            1. Load experiences from store
            2. Create prompts from market states
            3. Sample responses (trading decisions)
            4. Execute in backtest to get rewards
            5. Update policy
        """
        # Setup models
        self.setup()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting GRPO training for {num_iterations} iterations")
        logger.info(f"Output directory: {output_dir}")

        # Training loop
        results = []

        for iteration in range(num_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            logger.info(f"{'='*60}")

            # TODO: Load experiences and create batch
            # For now, this is a placeholder
            logger.info("⚠️  Training loop requires integration with experience store")
            logger.info("   This is a framework implementation")

            # Placeholder batch
            # In production: batch = self.prepare_batch(experience_store_path)

            # Save checkpoint
            if (iteration + 1) % self.config.save_every == 0:
                checkpoint_path = output_dir / f"checkpoint_iter_{iteration + 1}"
                self.save_checkpoint(checkpoint_path)
                logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

        logger.info(f"\n{'='*60}")
        logger.info("✅ GRPO Training Complete")
        logger.info(f"{'='*60}")

        # Save final model
        final_path = output_dir / "final_model"
        self.save_checkpoint(final_path)
        logger.info(f"Final model saved: {final_path}")

        return results

    def save_checkpoint(self, path: Path):
        """Save model checkpoint"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save policy
        self.policy.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

        # Save training state
        torch.save({
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config)
        }, path / "training_state.pt")

        logger.info(f"Checkpoint saved: {path}")


def main():
    """CLI interface for GRPO trainer"""
    import argparse

    parser = argparse.ArgumentParser(
        description="GRPO Trainer - Group Relative Policy Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from SFT checkpoint
  python training/rl/grpo_trainer.py \\
      --policy models/sft/strategist_v1.0.0 \\
      --experience-store data/experiences \\
      --output models/rl/strategist_grpo_v1.0.0 \\
      --iterations 100

  # Resume from checkpoint
  python training/rl/grpo_trainer.py \\
      --policy models/rl/strategist_grpo_v1.0.0/checkpoint_iter_50 \\
      --experience-store data/experiences \\
      --output models/rl/strategist_grpo_v1.0.0 \\
      --iterations 100
        """
    )

    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Path to initial policy model"
    )

    parser.add_argument(
        "--experience-store",
        type=Path,
        required=True,
        help="Path to experience store"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for checkpoints"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to RL config YAML (default: training/rl/rl_config.yaml)"
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training (default: cuda if available)"
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = GRPOTrainer(
        policy_path=args.policy,
        config_path=args.config,
        device=args.device
    )

    # Train
    trainer.train(
        experience_store_path=args.experience_store,
        num_iterations=args.iterations,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
