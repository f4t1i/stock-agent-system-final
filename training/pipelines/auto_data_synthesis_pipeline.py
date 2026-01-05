"""
Auto Data-Synthesis Pipeline (Post-Backtest)

Automatically extracts successful trajectories from backtest results and generates
high-quality SFT training datasets.

Phase A1 Week 3-4: Task 1
- Monitors backtest completion
- Filters successful trades (reward > threshold)
- Applies Judge-approved quality scoring
- Generates versioned SFT datasets
- Integrates with Dataset Registry

Based on:
- TradingGroup auto-synthesis
- SIRIUS trajectory filtering
- PrimoAgent continuous learning loop
"""

import json
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger
import pandas as pd

from data_pipeline.data_synthesis import DataSynthesisModule, SFTExample
from data_pipeline.experience_library_postgres import ExperienceLibraryPostgres, Trajectory
from training.rl.backtester_v2 import BacktestConfig
from judge.llm_judge import LLMJudge


@dataclass
class AutoSynthesisConfig:
    """Configuration for auto data synthesis pipeline"""
    # Quality thresholds
    min_reward: float = 0.5  # Minimum reward for success
    min_confidence: float = 0.7  # Minimum confidence
    quality_threshold: float = 0.6  # Overall quality threshold
    
    # Judge filtering
    enable_judge_filtering: bool = True
    judge_min_score: float = 0.7  # Minimum judge score
    
    # Dataset generation
    max_examples_per_agent: int = 1000
    output_format: str = 'chatml'  # 'chatml' or 'alpaca'
    
    # Storage
    output_dir: str = "datasets/auto_synthesis"
    versioning: bool = True
    
    # Automation
    auto_trigger_on_backtest: bool = True
    min_new_trajectories: int = 100  # Minimum new trajectories to trigger
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SynthesisResult:
    """Result of auto synthesis pipeline"""
    dataset_version: str
    agent_type: str
    num_trajectories_processed: int
    num_examples_generated: int
    num_judge_filtered: int
    avg_quality_score: float
    avg_judge_score: float
    output_path: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AutoDataSynthesisPipeline:
    """
    Automatic Data Synthesis Pipeline
    
    Monitors backtest completion and automatically generates SFT datasets
    from successful trajectories.
    """
    
    def __init__(
        self,
        experience_library: ExperienceLibraryPostgres,
        judge: Optional[LLMJudge] = None,
        config: Optional[AutoSynthesisConfig] = None
    ):
        """
        Initialize Auto Data Synthesis Pipeline
        
        Args:
            experience_library: Experience library instance
            judge: LLM Judge for quality filtering (optional)
            config: Pipeline configuration
        """
        self.library = experience_library
        self.judge = judge
        self.config = config or AutoSynthesisConfig()
        
        # Initialize data synthesis module
        self.synthesis_module = DataSynthesisModule(
            experience_library=experience_library,
            min_reward=self.config.min_reward,
            min_confidence=self.config.min_confidence,
            quality_threshold=self.config.quality_threshold
        )
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Auto Data Synthesis Pipeline initialized with config: {self.config.to_dict()}")
    
    def run_post_backtest_synthesis(
        self,
        backtest_id: str,
        agent_types: Optional[List[str]] = None
    ) -> List[SynthesisResult]:
        """
        Run auto synthesis after backtest completion
        
        Args:
            backtest_id: Backtest run ID
            agent_types: List of agent types to synthesize for (None = all)
        
        Returns:
            List of synthesis results per agent type
        """
        logger.info(f"Starting post-backtest synthesis for backtest_id={backtest_id}")
        
        # Default agent types
        if agent_types is None:
            agent_types = ['news', 'technical', 'fundamental', 'strategist']
        
        results = []
        
        for agent_type in agent_types:
            try:
                result = self._synthesize_for_agent(
                    backtest_id=backtest_id,
                    agent_type=agent_type
                )
                results.append(result)
                
                logger.info(
                    f"✅ Synthesized {result.num_examples_generated} examples "
                    f"for {agent_type} (quality={result.avg_quality_score:.3f}, "
                    f"judge={result.avg_judge_score:.3f})"
                )
            
            except Exception as e:
                logger.error(f"❌ Failed to synthesize for {agent_type}: {e}")
                continue
        
        # Save summary
        self._save_synthesis_summary(backtest_id, results)
        
        logger.info(
            f"✅ Post-backtest synthesis complete: "
            f"{sum(r.num_examples_generated for r in results)} total examples generated"
        )
        
        return results
    
    def _synthesize_for_agent(
        self,
        backtest_id: str,
        agent_type: str
    ) -> SynthesisResult:
        """
        Synthesize SFT dataset for a single agent type
        
        Args:
            backtest_id: Backtest run ID
            agent_type: Agent type
        
        Returns:
            Synthesis result
        """
        logger.info(f"Synthesizing for {agent_type} agent...")
        
        # Step 1: Filter high-quality trajectories
        scored_trajectories = self.synthesis_module.filter_high_quality_trajectories(
            agent_type=agent_type,
            market_regime=None,
            limit=self.config.max_examples_per_agent
        )
        
        logger.info(f"Filtered {len(scored_trajectories)} high-quality trajectories")
        
        # Step 2: Apply Judge filtering (if enabled)
        if self.config.enable_judge_filtering and self.judge is not None:
            scored_trajectories = self._apply_judge_filtering(
                scored_trajectories,
                agent_type
            )
            logger.info(f"After judge filtering: {len(scored_trajectories)} trajectories")
        
        # Step 3: Convert to SFT examples
        sft_examples = []
        quality_scores = []
        judge_scores = []
        
        for trajectory, quality_score in scored_trajectories:
            try:
                example = self.synthesis_module.trajectory_to_sft_example(
                    trajectory=trajectory,
                    agent_type=agent_type
                )
                sft_examples.append(example)
                quality_scores.append(quality_score)
                
                # Get judge score from metadata if available
                judge_score = trajectory.metadata.get('judge_score', 0.0)
                judge_scores.append(judge_score)
            
            except Exception as e:
                logger.warning(f"Failed to convert trajectory {trajectory.trajectory_id}: {e}")
                continue
        
        # Step 4: Generate dataset version
        dataset_version = self._generate_dataset_version(agent_type)
        
        # Step 5: Save dataset
        output_path = self._save_dataset(
            sft_examples=sft_examples,
            agent_type=agent_type,
            dataset_version=dataset_version
        )
        
        # Step 6: Create result
        result = SynthesisResult(
            dataset_version=dataset_version,
            agent_type=agent_type,
            num_trajectories_processed=len(scored_trajectories),
            num_examples_generated=len(sft_examples),
            num_judge_filtered=len(scored_trajectories) - len(sft_examples) if self.config.enable_judge_filtering else 0,
            avg_quality_score=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            avg_judge_score=sum(judge_scores) / len(judge_scores) if judge_scores else 0.0,
            output_path=str(output_path),
            timestamp=datetime.now().isoformat()
        )
        
        return result
    
    def _apply_judge_filtering(
        self,
        scored_trajectories: List[Tuple[Trajectory, float]],
        agent_type: str
    ) -> List[Tuple[Trajectory, float]]:
        """
        Apply LLM Judge filtering to trajectories
        
        Args:
            scored_trajectories: List of (trajectory, quality_score) tuples
            agent_type: Agent type
        
        Returns:
            Filtered list of trajectories
        """
        logger.info("Applying Judge filtering...")
        
        filtered = []
        
        for trajectory, quality_score in scored_trajectories:
            try:
                # Evaluate with Judge
                judge_result = self.judge.evaluate_trajectory(
                    trajectory=trajectory,
                    agent_type=agent_type
                )
                
                # Store judge score in metadata
                trajectory.metadata['judge_score'] = judge_result.score
                trajectory.metadata['judge_feedback'] = judge_result.feedback
                
                # Filter by judge score
                if judge_result.score >= self.config.judge_min_score:
                    filtered.append((trajectory, quality_score))
            
            except Exception as e:
                logger.warning(f"Judge evaluation failed for trajectory {trajectory.trajectory_id}: {e}")
                continue
        
        logger.info(
            f"Judge filtered: {len(scored_trajectories)} → {len(filtered)} "
            f"({len(filtered)/len(scored_trajectories)*100:.1f}% pass rate)"
        )
        
        return filtered
    
    def _generate_dataset_version(self, agent_type: str) -> str:
        """
        Generate dataset version string
        
        Args:
            agent_type: Agent type
        
        Returns:
            Version string (e.g., "v1.0.0_news_20260105_120000")
        """
        if self.config.versioning:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get version number from registry (if exists)
            version_file = self.output_dir / f"{agent_type}_version.txt"
            if version_file.exists():
                with open(version_file, 'r') as f:
                    last_version = f.read().strip()
                    # Increment patch version
                    major, minor, patch = map(int, last_version.split('_')[0][1:].split('.'))
                    patch += 1
                    version = f"v{major}.{minor}.{patch}"
            else:
                version = "v1.0.0"
            
            # Save new version
            with open(version_file, 'w') as f:
                f.write(f"{version}_{agent_type}_{timestamp}")
            
            return f"{version}_{agent_type}_{timestamp}"
        else:
            return f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _save_dataset(
        self,
        sft_examples: List[SFTExample],
        agent_type: str,
        dataset_version: str
    ) -> Path:
        """
        Save SFT dataset to file
        
        Args:
            sft_examples: List of SFT examples
            agent_type: Agent type
            dataset_version: Dataset version
        
        Returns:
            Output file path
        """
        # Create agent-specific directory
        agent_dir = self.output_dir / agent_type
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Output file
        output_file = agent_dir / f"{dataset_version}.jsonl"
        
        # Save examples in JSONL format
        with open(output_file, 'w') as f:
            for example in sft_examples:
                if self.config.output_format == 'chatml':
                    f.write(example.to_chatml() + '\n')
                elif self.config.output_format == 'alpaca':
                    f.write(json.dumps(example.to_alpaca()) + '\n')
        
        logger.info(f"Saved {len(sft_examples)} examples to {output_file}")
        
        # Save metadata
        metadata_file = agent_dir / f"{dataset_version}_metadata.json"
        metadata = {
            'dataset_version': dataset_version,
            'agent_type': agent_type,
            'num_examples': len(sft_examples),
            'output_format': self.config.output_format,
            'config': self.config.to_dict(),
            'created_at': datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_file
    
    def _save_synthesis_summary(
        self,
        backtest_id: str,
        results: List[SynthesisResult]
    ):
        """
        Save synthesis summary
        
        Args:
            backtest_id: Backtest run ID
            results: List of synthesis results
        """
        summary_file = self.output_dir / f"synthesis_summary_{backtest_id}.json"
        
        summary = {
            'backtest_id': backtest_id,
            'timestamp': datetime.now().isoformat(),
            'total_examples_generated': sum(r.num_examples_generated for r in results),
            'results': [r.to_dict() for r in results]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved synthesis summary to {summary_file}")
    
    def monitor_and_trigger(
        self,
        check_interval: int = 300,  # 5 minutes
        max_iterations: Optional[int] = None
    ):
        """
        Monitor backtest completion and auto-trigger synthesis
        
        Args:
            check_interval: Check interval in seconds
            max_iterations: Maximum iterations (None = infinite)
        """
        logger.info(f"Starting auto-synthesis monitor (check_interval={check_interval}s)")
        
        iteration = 0
        last_processed_backtest = None
        
        while max_iterations is None or iteration < max_iterations:
            try:
                # Check for new completed backtests
                latest_backtest = self._get_latest_completed_backtest()
                
                if latest_backtest and latest_backtest != last_processed_backtest:
                    logger.info(f"New backtest detected: {latest_backtest}")
                    
                    # Check if enough new trajectories
                    new_trajectories = self._count_new_trajectories(latest_backtest)
                    
                    if new_trajectories >= self.config.min_new_trajectories:
                        logger.info(f"Triggering auto-synthesis ({new_trajectories} new trajectories)")
                        
                        # Run synthesis
                        results = self.run_post_backtest_synthesis(
                            backtest_id=latest_backtest
                        )
                        
                        last_processed_backtest = latest_backtest
                    else:
                        logger.info(
                            f"Not enough new trajectories ({new_trajectories} < {self.config.min_new_trajectories})"
                        )
                
                # Wait for next check
                time.sleep(check_interval)
                iteration += 1
            
            except KeyboardInterrupt:
                logger.info("Monitor stopped by user")
                break
            
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(check_interval)
                continue
    
    def _get_latest_completed_backtest(self) -> Optional[str]:
        """Get latest completed backtest ID"""
        # Query experience library for latest backtest
        # This is a placeholder - implement based on your backtest tracking
        return None
    
    def _count_new_trajectories(self, backtest_id: str) -> int:
        """Count new trajectories since last synthesis"""
        # Query experience library for new trajectories
        # This is a placeholder - implement based on your tracking
        return 0


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Data Synthesis Pipeline")
    parser.add_argument('--backtest-id', type=str, required=True, help="Backtest run ID")
    parser.add_argument('--agent-types', nargs='+', default=None, help="Agent types to synthesize")
    parser.add_argument('--config', type=str, default=None, help="Config JSON file")
    parser.add_argument('--monitor', action='store_true', help="Run in monitor mode")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config = AutoSynthesisConfig(**config_dict)
    else:
        config = AutoSynthesisConfig()
    
    # Initialize pipeline
    experience_library = ExperienceLibraryPostgres()
    pipeline = AutoDataSynthesisPipeline(
        experience_library=experience_library,
        config=config
    )
    
    # Run synthesis or monitor
    if args.monitor:
        pipeline.monitor_and_trigger()
    else:
        results = pipeline.run_post_backtest_synthesis(
            backtest_id=args.backtest_id,
            agent_types=args.agent_types
        )
        
        print(f"\n✅ Synthesis complete: {sum(r.num_examples_generated for r in results)} examples generated")
        for result in results:
            print(f"  - {result.agent_type}: {result.num_examples_generated} examples (quality={result.avg_quality_score:.3f})")
