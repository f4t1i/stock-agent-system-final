"""
Dataset Synthesizer - Converts backtest experiences into SFT/RL training datasets

Purpose:
    Transform raw experiences (signal, action, outcome, reward) into structured
    training datasets for supervised fine-tuning (SFT) and reinforcement learning (RL).

Features:
    - Judge-approved filtering (quality gates)
    - Multi-format support (prompt/completion, chat, instruction-tuning)
    - Train/val/test splitting with stratification
    - Dataset versioning and metadata
    - Multiple synthesis strategies (positive-only, contrastive, full-spectrum)

Usage:
    synthesizer = DatasetSynthesizer(
        experience_store=store,
        config=config
    )

    dataset = synthesizer.synthesize(
        strategy="judge_approved",
        min_reward=0.5,
        output_format="chat"
    )

    synthesizer.save_dataset(dataset, "data/datasets/sft_v1.0")
"""

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal, Tuple
from loguru import logger
import hashlib

from training.data_synthesis.experience_store import Experience, ExperienceStore


SynthesisStrategy = Literal[
    "positive_only",      # Only high-reward experiences (reward > 0.5)
    "contrastive",        # Pairs of positive/negative examples
    "full_spectrum",      # All experiences (balanced by reward)
    "judge_approved"      # Only judge-approved signals
]

DatasetFormat = Literal[
    "prompt_completion",  # Simple prompt → completion pairs
    "chat",               # Chat format (user/assistant messages)
    "instruction"         # Instruction-tuning format (instruction, input, output)
]


@dataclass
class DatasetExample:
    """Single training example synthesized from experience"""

    example_id: str
    format: DatasetFormat

    # Content (format-dependent)
    prompt: Optional[str] = None
    completion: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None  # For chat format
    instruction: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None

    # Metadata
    source_experience_id: str = ""
    symbol: str = ""
    reward: float = 0.0
    judge_approved: bool = False
    split: Literal["train", "val", "test"] = "train"

    @staticmethod
    def generate_id(content: str) -> str:
        """Generate unique example ID"""
        return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class Dataset:
    """Complete synthesized dataset"""

    dataset_id: str
    version: str
    strategy: SynthesisStrategy
    format: DatasetFormat

    # Examples
    examples: List[DatasetExample]

    # Metadata
    num_examples: int
    num_train: int
    num_val: int
    num_test: int
    source_experiences: int
    avg_reward: float
    approval_rate: float

    created_at: str
    config: Dict[str, Any]

    @staticmethod
    def generate_id(strategy: str, version: str) -> str:
        """Generate dataset ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dataset_{strategy}_{version}_{timestamp}"


class DatasetSynthesizer:
    """
    Synthesizes training datasets from backtest experiences

    Synthesis Strategies:
    - positive_only: Only successful trades (reward > threshold)
    - contrastive: Pairs of good/bad examples for preference learning
    - full_spectrum: All experiences, balanced by reward distribution
    - judge_approved: Only signals that passed judge validation
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42
    ):
        self.store = experience_store
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed

        random.seed(random_seed)

        logger.info("DatasetSynthesizer initialized")
        logger.info(f"Splits: train={train_split:.1%}, val={val_split:.1%}, test={test_split:.1%}")

    def synthesize(
        self,
        strategy: SynthesisStrategy,
        output_format: DatasetFormat = "chat",
        min_reward: float = 0.0,
        min_judge_score: Optional[float] = None,
        symbol: Optional[str] = None,
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """
        Synthesize dataset from experiences

        Args:
            strategy: Synthesis strategy (positive_only, contrastive, full_spectrum, judge_approved)
            output_format: Dataset format (prompt_completion, chat, instruction)
            min_reward: Minimum reward threshold for filtering
            min_judge_score: Minimum judge score (0-10) for filtering
            symbol: Filter by symbol (optional)
            version: Dataset version string
            config: Additional configuration

        Returns:
            Complete dataset with train/val/test splits
        """
        logger.info(f"Synthesizing dataset: strategy={strategy}, format={output_format}")

        # Query experiences based on strategy
        experiences = self._query_experiences(
            strategy=strategy,
            min_reward=min_reward,
            min_judge_score=min_judge_score,
            symbol=symbol
        )

        if not experiences:
            raise ValueError("No experiences found matching criteria")

        logger.info(f"Found {len(experiences)} matching experiences")

        # Convert to examples
        examples = self._convert_to_examples(
            experiences=experiences,
            output_format=output_format,
            strategy=strategy
        )

        logger.info(f"Converted to {len(examples)} training examples")

        # Split into train/val/test
        examples = self._split_dataset(examples)

        # Compute statistics
        num_train = sum(1 for e in examples if e.split == "train")
        num_val = sum(1 for e in examples if e.split == "val")
        num_test = sum(1 for e in examples if e.split == "test")
        avg_reward = sum(e.reward for e in examples) / len(examples)
        approval_rate = sum(1 for e in examples if e.judge_approved) / len(examples)

        # Create dataset
        dataset = Dataset(
            dataset_id=Dataset.generate_id(strategy, version),
            version=version,
            strategy=strategy,
            format=output_format,
            examples=examples,
            num_examples=len(examples),
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            source_experiences=len(experiences),
            avg_reward=avg_reward,
            approval_rate=approval_rate,
            created_at=datetime.now().isoformat(),
            config=config or {}
        )

        logger.info(f"Dataset created: {dataset.dataset_id}")
        logger.info(f"  Examples: {len(examples)} (train={num_train}, val={num_val}, test={num_test})")
        logger.info(f"  Avg reward: {avg_reward:.3f}")
        logger.info(f"  Approval rate: {approval_rate:.1%}")

        return dataset

    def _query_experiences(
        self,
        strategy: SynthesisStrategy,
        min_reward: float,
        min_judge_score: Optional[float],
        symbol: Optional[str]
    ) -> List[Experience]:
        """Query experiences based on synthesis strategy"""

        if strategy == "positive_only":
            # Only high-reward experiences
            return self.store.query(
                symbol=symbol,
                min_reward=max(0.5, min_reward),
                min_judge_score=min_judge_score
            )

        elif strategy == "judge_approved":
            # Only judge-approved signals
            return self.store.query(
                symbol=symbol,
                judge_approved_only=True,
                min_reward=min_reward,
                min_judge_score=min_judge_score
            )

        elif strategy == "full_spectrum":
            # All experiences (will balance in conversion)
            return self.store.query(
                symbol=symbol,
                min_reward=min_reward,
                min_judge_score=min_judge_score
            )

        elif strategy == "contrastive":
            # Get both positive and negative examples
            positive = self.store.query(
                symbol=symbol,
                min_reward=0.5,
                min_judge_score=min_judge_score
            )
            negative = self.store.query(
                symbol=symbol,
                max_reward=0.0,
                min_judge_score=min_judge_score
            )

            # Balance positive/negative
            min_count = min(len(positive), len(negative))
            return random.sample(positive, min_count) + random.sample(negative, min_count)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _convert_to_examples(
        self,
        experiences: List[Experience],
        output_format: DatasetFormat,
        strategy: SynthesisStrategy
    ) -> List[DatasetExample]:
        """Convert experiences to dataset examples"""

        examples = []

        for exp in experiences:
            if output_format == "chat":
                example = self._to_chat_format(exp)
            elif output_format == "prompt_completion":
                example = self._to_prompt_completion(exp)
            elif output_format == "instruction":
                example = self._to_instruction_format(exp)
            else:
                raise ValueError(f"Unknown format: {output_format}")

            example.source_experience_id = exp.experience_id
            example.symbol = exp.symbol
            example.reward = exp.reward
            example.judge_approved = exp.judge_approved

            examples.append(example)

        return examples

    def _to_chat_format(self, exp: Experience) -> DatasetExample:
        """
        Convert experience to chat format

        Format:
        [
            {"role": "system", "content": "You are a stock trading strategist..."},
            {"role": "user", "content": "Analyze AAPL: <analysis data>"},
            {"role": "assistant", "content": "Decision: buy, Position: 10%, ..."}
        ]
        """
        # System message
        system_msg = (
            "You are an expert stock trading strategist. Analyze the provided signal data "
            "and make a trading decision with proper risk management."
        )

        # User message (signal analysis)
        signal = exp.signal
        user_msg = f"""Analyze this trading signal for {exp.symbol}:

**News Analysis:**
- Sentiment: {signal['analysis']['news']['sentiment_score']:.2f}
- Confidence: {signal['analysis']['news']['confidence']:.2f}
- Key Events: {', '.join(signal['analysis']['news']['key_events'])}

**Technical Analysis:**
- Signal: {signal['analysis']['technical']['signal']}
- Signal Strength: {signal['analysis']['technical']['signal_strength']:.2f}
- RSI: {signal['analysis']['technical']['indicators']['rsi']:.1f}

**Fundamental Analysis:**
- Valuation: {signal['analysis']['fundamental']['valuation']}
- Financial Health: {signal['analysis']['fundamental']['financial_health_score']:.2f}
- Growth Score: {signal['analysis']['fundamental']['growth_score']:.2f}

**Current Quote:**
- Price: ${signal['metadata']['quote']['price']:.2f}

What is your trading decision?"""

        # Assistant message (action)
        action = exp.action
        outcome = exp.outcome

        assistant_msg = f"""**Decision:** {action['decision'].upper()}

**Position Sizing:** {action['position_size']*100:.1f}% of portfolio

**Entry Target:** ${action.get('entry_target', 'market')}

**Risk Management:**
- Stop Loss: ${action.get('stop_loss', 'N/A')}
- Take Profit: ${action.get('take_profit', 'N/A')}

**Rationale:**
{signal['rationale']}

**Outcome:** This trade resulted in {outcome.get('return_pct', 0)*100:+.2f}% return (${outcome.get('pnl', 0):+.2f} P&L)."""

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]

        example_id = DatasetExample.generate_id(json.dumps(messages))

        return DatasetExample(
            example_id=example_id,
            format="chat",
            messages=messages
        )

    def _to_prompt_completion(self, exp: Experience) -> DatasetExample:
        """
        Convert experience to prompt/completion format

        Format:
        Prompt: "Given this analysis..., what should we do?"
        Completion: "Decision: buy, Position: 10%, ..."
        """
        signal = exp.signal
        action = exp.action

        # Compact prompt
        prompt = f"""Symbol: {exp.symbol}
News Sentiment: {signal['analysis']['news']['sentiment_score']:.2f}
Technical Signal: {signal['analysis']['technical']['signal']}
Valuation: {signal['analysis']['fundamental']['valuation']}
Price: ${signal['metadata']['quote']['price']:.2f}

Decision:"""

        # Completion
        completion = f""" {action['decision']}
Position Size: {action['position_size']*100:.1f}%
Stop Loss: ${action.get('stop_loss', 'N/A')}
Take Profit: ${action.get('take_profit', 'N/A')}"""

        example_id = DatasetExample.generate_id(prompt + completion)

        return DatasetExample(
            example_id=example_id,
            format="prompt_completion",
            prompt=prompt,
            completion=completion
        )

    def _to_instruction_format(self, exp: Experience) -> DatasetExample:
        """
        Convert experience to instruction-tuning format

        Format:
        Instruction: "Analyze this stock signal and make a trading decision"
        Input: <signal data>
        Output: <action with rationale>
        """
        instruction = (
            "Analyze the provided stock trading signal and make a decision with proper risk management. "
            "Provide your decision (buy/sell/hold), position size, entry/exit prices, and detailed rationale."
        )

        signal = exp.signal
        input_text = json.dumps({
            "symbol": exp.symbol,
            "analysis": signal['analysis'],
            "quote": signal['metadata']['quote']
        }, indent=2)

        action = exp.action
        output_text = json.dumps({
            "decision": action['decision'],
            "position_size": action['position_size'],
            "entry_target": action.get('entry_target'),
            "stop_loss": action.get('stop_loss'),
            "take_profit": action.get('take_profit'),
            "rationale": signal['rationale']
        }, indent=2)

        example_id = DatasetExample.generate_id(instruction + input_text + output_text)

        return DatasetExample(
            example_id=example_id,
            format="instruction",
            instruction=instruction,
            input=input_text,
            output=output_text
        )

    def _split_dataset(self, examples: List[DatasetExample]) -> List[DatasetExample]:
        """Split examples into train/val/test sets"""

        # Shuffle
        random.shuffle(examples)

        # Calculate split indices
        n = len(examples)
        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)

        # Assign splits
        for i, example in enumerate(examples):
            if i < train_end:
                example.split = "train"
            elif i < val_end:
                example.split = "val"
            else:
                example.split = "test"

        return examples

    def save_dataset(
        self,
        dataset: Dataset,
        output_dir: Path,
        save_splits_separately: bool = True
    ):
        """
        Save dataset to disk

        Args:
            dataset: Dataset to save
            output_dir: Output directory
            save_splits_separately: Save train/val/test as separate files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "dataset_id": dataset.dataset_id,
            "version": dataset.version,
            "strategy": dataset.strategy,
            "format": dataset.format,
            "num_examples": dataset.num_examples,
            "num_train": dataset.num_train,
            "num_val": dataset.num_val,
            "num_test": dataset.num_test,
            "source_experiences": dataset.source_experiences,
            "avg_reward": dataset.avg_reward,
            "approval_rate": dataset.approval_rate,
            "created_at": dataset.created_at,
            "config": dataset.config
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save examples
        if save_splits_separately:
            # Separate files for train/val/test
            for split in ["train", "val", "test"]:
                split_examples = [e for e in dataset.examples if e.split == split]
                if split_examples:
                    self._save_examples(split_examples, output_dir / f"{split}.jsonl", dataset.format)
                    logger.info(f"Saved {len(split_examples)} {split} examples to {output_dir}/{split}.jsonl")
        else:
            # Single file with all examples
            self._save_examples(dataset.examples, output_dir / "dataset.jsonl", dataset.format)
            logger.info(f"Saved {len(dataset.examples)} examples to {output_dir}/dataset.jsonl")

        logger.info(f"Dataset saved to {output_dir}")

    def _save_examples(
        self,
        examples: List[DatasetExample],
        output_path: Path,
        format: DatasetFormat
    ):
        """Save examples to JSONL file"""

        with open(output_path, "w") as f:
            for example in examples:
                if format == "chat":
                    data = {"messages": example.messages}
                elif format == "prompt_completion":
                    data = {"prompt": example.prompt, "completion": example.completion}
                elif format == "instruction":
                    data = {
                        "instruction": example.instruction,
                        "input": example.input,
                        "output": example.output
                    }

                # Add metadata
                data["metadata"] = {
                    "example_id": example.example_id,
                    "source_experience_id": example.source_experience_id,
                    "symbol": example.symbol,
                    "reward": example.reward,
                    "judge_approved": example.judge_approved
                }

                f.write(json.dumps(data) + "\n")


# CLI Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Synthesizer - Convert experiences to training datasets")
    parser.add_argument("--storage-dir", type=Path, default=Path("data/experiences"), help="Experience store directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for dataset")
    parser.add_argument("--strategy", choices=["positive_only", "contrastive", "full_spectrum", "judge_approved"], default="judge_approved", help="Synthesis strategy")
    parser.add_argument("--format", choices=["chat", "prompt_completion", "instruction"], default="chat", help="Dataset format")
    parser.add_argument("--min-reward", type=float, default=0.0, help="Minimum reward threshold")
    parser.add_argument("--min-judge-score", type=float, help="Minimum judge score (0-10)")
    parser.add_argument("--symbol", type=str, help="Filter by symbol")
    parser.add_argument("--version", default="1.0.0", help="Dataset version")

    args = parser.parse_args()

    # Initialize store
    from training.data_synthesis.experience_store import ExperienceStoreConfig
    store_config = ExperienceStoreConfig(storage_dir=args.storage_dir)
    store = ExperienceStore(store_config)

    # Initialize synthesizer
    synthesizer = DatasetSynthesizer(experience_store=store)

    # Synthesize dataset
    dataset = synthesizer.synthesize(
        strategy=args.strategy,
        output_format=args.format,
        min_reward=args.min_reward,
        min_judge_score=args.min_judge_score,
        symbol=args.symbol,
        version=args.version
    )

    # Save
    synthesizer.save_dataset(dataset, args.output_dir)

    print(f"\n✅ Dataset synthesized successfully!")
    print(f"   ID: {dataset.dataset_id}")
    print(f"   Examples: {dataset.num_examples} (train={dataset.num_train}, val={dataset.num_val}, test={dataset.num_test})")
    print(f"   Avg Reward: {dataset.avg_reward:.3f}")
    print(f"   Approval Rate: {dataset.approval_rate:.1%}")
    print(f"   Output: {args.output_dir}")

    store.close()
