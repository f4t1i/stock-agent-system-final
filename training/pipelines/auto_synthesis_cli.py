#!/usr/bin/env python3
"""
Auto Data-Synthesis Pipeline CLI - Task 1.6

Command-line interface for the complete auto-synthesis pipeline.

Integrates all components:
1. Backtester Completion Detection (Task 1.1)
2. Trajectory Extraction from Postgres (Task 1.2)
3. Quality Score Calculation (Task 1.3)
4. ChatML/Alpaca Conversion (Task 1.4)
5. File I/O + Versioning (Task 1.5)

Commands:
- run: Run full pipeline (backtest → dataset)
- extract: Extract trajectories from backtest
- score: Score trajectories by quality
- convert: Convert trajectories to SFT format
- save: Save dataset with versioning

Usage:
    # Full pipeline
    python auto_synthesis_cli.py run --backtest-id test_001 --agent-type technical --version 1.0.0
    
    # Individual steps
    python auto_synthesis_cli.py extract --backtest-id test_001 --output trajectories.json
    python auto_synthesis_cli.py score --input trajectories.json --output scored.json
    python auto_synthesis_cli.py convert --input scored.json --format chatml --output dataset.jsonl
    python auto_synthesis_cli.py save --input dataset.jsonl --agent-type technical --version 1.0.0

Phase A1 Week 3-4: Task 1.6 COMPLETE
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger

# Import pipeline components
from trajectory_extractor import TrajectoryExtractor
from quality_scorer import QualityScorer
from dataset_formatter import DatasetFormatter
from dataset_storage import DatasetStorage, DatasetMetadata


class AutoSynthesisPipeline:
    """
    Complete auto-synthesis pipeline
    
    Orchestrates all components to convert backtest results into SFT datasets
    """
    
    def __init__(
        self,
        db_config: Optional[Dict] = None,
        storage_dir: str = "datasets",
        quality_threshold: float = 0.6
    ):
        """
        Initialize pipeline
        
        Args:
            db_config: Database configuration for trajectory extraction (optional)
            storage_dir: Directory for dataset storage
            quality_threshold: Minimum quality score for inclusion
        """
        self.extractor = TrajectoryExtractor(db_config) if db_config else None
        self.scorer = QualityScorer()
        self.formatter = DatasetFormatter()
        self.storage = DatasetStorage(base_dir=storage_dir)
        self.quality_threshold = quality_threshold
        
        logger.info("AutoSynthesisPipeline initialized")
    
    def run_full_pipeline(
        self,
        backtest_id: str,
        agent_type: str,
        version: str,
        format: str = "chatml",
        min_reward: float = 0.0,
        min_confidence: float = 0.0,
        include_reasoning: bool = True,
        judge_filter: bool = False,
        judge_threshold: float = 0.6
    ) -> DatasetMetadata:
        """
        Run complete pipeline: backtest → dataset
        
        Args:
            backtest_id: Backtest ID to extract from
            agent_type: Agent type (technical, news, etc.)
            version: Dataset version (semantic versioning)
            format: Output format (chatml or alpaca)
            min_reward: Minimum reward threshold
            min_confidence: Minimum confidence threshold
            include_reasoning: Include reasoning in output
            judge_filter: Apply judge filtering
            judge_threshold: Minimum judge score (if judge_filter=True)
        
        Returns:
            DatasetMetadata object
        """
        logger.info(f"Starting full pipeline for backtest {backtest_id}")
        
        # Step 1: Extract trajectories
        logger.info("Step 1: Extracting trajectories from Postgres...")
        trajectories = list(self.extractor.extract_trajectories(
            backtest_id=backtest_id,
            agent_type=agent_type,
            min_reward=min_reward,
            min_confidence=min_confidence
        ))
        logger.info(f"Extracted {len(trajectories)} trajectories")
        
        if len(trajectories) == 0:
            raise ValueError(f"No trajectories found for backtest {backtest_id}")
        
        # Step 2: Score trajectories
        logger.info("Step 2: Scoring trajectories by quality...")
        scored_trajectories = self.scorer.batch_score(trajectories)
        logger.info(f"Scored {len(scored_trajectories)} trajectories")
        
        # Filter by quality threshold
        filtered_trajectories = [
            traj for traj in scored_trajectories
            if traj['quality_score'] >= self.quality_threshold
        ]
        logger.info(
            f"Filtered to {len(filtered_trajectories)} trajectories "
            f"(quality >= {self.quality_threshold})"
        )
        
        if len(filtered_trajectories) == 0:
            raise ValueError(
                f"No trajectories passed quality threshold {self.quality_threshold}"
            )
        
        # Step 3: Judge filtering (optional)
        if judge_filter:
            logger.info("Step 3: Applying judge filtering...")
            # TODO: Integrate with judge_filtering.py (Task 3)
            # For now, skip judge filtering
            logger.warning("Judge filtering not yet integrated (Task 3)")
        
        # Step 4: Convert to SFT format
        logger.info(f"Step 4: Converting to {format} format...")
        if format == "chatml":
            dataset = self.formatter.batch_convert_chatml(
                trajectories=filtered_trajectories,
                include_reasoning=include_reasoning
            )
        elif format == "alpaca":
            dataset = self.formatter.batch_convert_alpaca(
                trajectories=filtered_trajectories,
                include_reasoning=include_reasoning
            )
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Converted {len(dataset)} examples to {format} format")
        
        # Step 5: Calculate statistics
        quality_stats = self._calculate_quality_stats(filtered_trajectories)
        
        # Step 6: Save dataset with versioning
        logger.info(f"Step 5: Saving dataset as {agent_type} v{version}...")
        metadata = self.storage.save_dataset(
            agent_type=agent_type,
            version=version,
            data=dataset,
            format=format,
            quality_stats=quality_stats,
            lineage={
                'backtest_id': backtest_id,
                'parent_version': None
            }
        )
        
        logger.info(
            f"Pipeline complete! Saved {metadata.example_count} examples "
            f"as {agent_type} v{version}"
        )
        
        return metadata
    
    def extract_trajectories(
        self,
        backtest_id: str,
        agent_type: str,
        output_file: str,
        min_reward: float = 0.0,
        min_confidence: float = 0.0
    ) -> int:
        """
        Extract trajectories and save to file
        
        Args:
            backtest_id: Backtest ID
            agent_type: Agent type
            output_file: Output JSON file path
            min_reward: Minimum reward
            min_confidence: Minimum confidence
        
        Returns:
            Number of trajectories extracted
        """
        logger.info(f"Extracting trajectories from backtest {backtest_id}...")
        
        trajectories = list(self.extractor.extract_trajectories(
            backtest_id=backtest_id,
            agent_type=agent_type,
            min_reward=min_reward,
            min_confidence=min_confidence
        ))
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(trajectories, f, indent=2)
        
        logger.info(f"Saved {len(trajectories)} trajectories to {output_file}")
        
        return len(trajectories)
    
    def score_trajectories(
        self,
        input_file: str,
        output_file: str
    ) -> int:
        """
        Score trajectories from file
        
        Args:
            input_file: Input JSON file with trajectories
            output_file: Output JSON file with scored trajectories
        
        Returns:
            Number of trajectories scored
        """
        logger.info(f"Scoring trajectories from {input_file}...")
        
        # Load trajectories
        with open(input_file, 'r') as f:
            trajectories = json.load(f)
        
        # Score
        quality_scores = self.scorer.batch_score(trajectories)
        
        # Merge scores back into trajectories
        scored_trajectories = []
        for traj, score in zip(trajectories, quality_scores):
            scored_traj = traj.copy()
            scored_traj['quality_score'] = score.overall_score
            scored_traj['quality_breakdown'] = {
                'reward_score': score.reward_score,
                'confidence_score': score.confidence_score,
                'reasoning_score': score.reasoning_score,
                'consistency_score': score.consistency_score
            }
            scored_trajectories.append(scored_traj)
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(scored_trajectories, f, indent=2)
        
        logger.info(f"Saved {len(scored_trajectories)} scored trajectories to {output_file}")
        
        return len(scored_trajectories)
    
    def convert_trajectories(
        self,
        input_file: str,
        output_file: str,
        format: str,
        include_reasoning: bool = True
    ) -> int:
        """
        Convert trajectories to SFT format
        
        Args:
            input_file: Input JSON file with trajectories
            output_file: Output file (JSONL for chatml, JSON for alpaca)
            format: Output format (chatml or alpaca)
            include_reasoning: Include reasoning
        
        Returns:
            Number of examples converted
        """
        logger.info(f"Converting trajectories to {format} format...")
        
        # Load trajectories
        with open(input_file, 'r') as f:
            trajectories = json.load(f)
        
        # Convert
        if format == "chatml":
            dataset = self.formatter.batch_convert_chatml(
                trajectories=trajectories,
                include_reasoning=include_reasoning
            )
            # Save as JSONL
            self.formatter.save_chatml(dataset, output_file)
        elif format == "alpaca":
            dataset = self.formatter.batch_convert_alpaca(
                trajectories=trajectories,
                include_reasoning=include_reasoning
            )
            # Save as JSON
            self.formatter.save_alpaca(dataset, output_file)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved {len(dataset)} examples to {output_file}")
        
        return len(dataset)
    
    def save_dataset(
        self,
        input_file: str,
        agent_type: str,
        version: str,
        format: str,
        backtest_id: Optional[str] = None
    ) -> DatasetMetadata:
        """
        Save dataset with versioning
        
        Args:
            input_file: Input file (JSONL for chatml, JSON for alpaca)
            agent_type: Agent type
            version: Semantic version
            format: Dataset format
            backtest_id: Backtest ID for lineage (optional)
        
        Returns:
            DatasetMetadata object
        """
        logger.info(f"Saving dataset as {agent_type} v{version}...")
        
        # Load dataset
        if format == "chatml":
            # JSONL format
            data = []
            with open(input_file, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        elif format == "alpaca":
            # JSON format
            with open(input_file, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Save with metadata
        metadata = self.storage.save_dataset(
            agent_type=agent_type,
            version=version,
            data=data,
            format=format,
            quality_stats={},  # TODO: Calculate from data
            lineage={
                'backtest_id': backtest_id
            } if backtest_id else {}
        )
        
        logger.info(f"Saved {metadata.example_count} examples as {agent_type} v{version}")
        
        return metadata
    
    def _calculate_quality_stats(self, trajectories: List[Dict]) -> Dict:
        """Calculate quality statistics"""
        if not trajectories:
            return {}
        
        quality_scores = [t['quality_score'] for t in trajectories]
        
        return {
            'avg_quality_score': sum(quality_scores) / len(quality_scores),
            'min_quality_score': min(quality_scores),
            'max_quality_score': max(quality_scores),
            'total_trajectories': len(trajectories)
        }


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Auto Data-Synthesis Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Common arguments
    def add_db_args(p):
        p.add_argument('--db-host', default='localhost', help='Database host')
        p.add_argument('--db-port', type=int, default=5432, help='Database port')
        p.add_argument('--db-name', default='stock_agent', help='Database name')
        p.add_argument('--db-user', default='postgres', help='Database user')
        p.add_argument('--db-password', default='', help='Database password')
    
    # run command
    run_parser = subparsers.add_parser('run', help='Run full pipeline')
    add_db_args(run_parser)
    run_parser.add_argument('--backtest-id', required=True, help='Backtest ID')
    run_parser.add_argument('--agent-type', required=True, help='Agent type')
    run_parser.add_argument('--version', required=True, help='Dataset version')
    run_parser.add_argument('--format', choices=['chatml', 'alpaca'], default='chatml')
    run_parser.add_argument('--min-reward', type=float, default=0.0)
    run_parser.add_argument('--min-confidence', type=float, default=0.0)
    run_parser.add_argument('--quality-threshold', type=float, default=0.6)
    run_parser.add_argument('--no-reasoning', action='store_true')
    run_parser.add_argument('--storage-dir', default='datasets')
    
    # extract command
    extract_parser = subparsers.add_parser('extract', help='Extract trajectories')
    add_db_args(extract_parser)
    extract_parser.add_argument('--backtest-id', required=True)
    extract_parser.add_argument('--agent-type', required=True)
    extract_parser.add_argument('--output', required=True)
    extract_parser.add_argument('--min-reward', type=float, default=0.0)
    extract_parser.add_argument('--min-confidence', type=float, default=0.0)
    
    # score command
    score_parser = subparsers.add_parser('score', help='Score trajectories')
    score_parser.add_argument('--input', required=True)
    score_parser.add_argument('--output', required=True)
    
    # convert command
    convert_parser = subparsers.add_parser('convert', help='Convert to SFT format')
    convert_parser.add_argument('--input', required=True)
    convert_parser.add_argument('--output', required=True)
    convert_parser.add_argument('--format', choices=['chatml', 'alpaca'], required=True)
    convert_parser.add_argument('--no-reasoning', action='store_true')
    
    # save command
    save_parser = subparsers.add_parser('save', help='Save dataset with versioning')
    save_parser.add_argument('--input', required=True)
    save_parser.add_argument('--agent-type', required=True)
    save_parser.add_argument('--version', required=True)
    save_parser.add_argument('--format', choices=['chatml', 'alpaca'], required=True)
    save_parser.add_argument('--backtest-id', help='Backtest ID for lineage')
    save_parser.add_argument('--storage-dir', default='datasets')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Build DB config
    if args.command in ['run', 'extract']:
        db_config = {
            'host': args.db_host,
            'port': args.db_port,
            'database': args.db_name,
            'user': args.db_user,
            'password': args.db_password
        }
        storage_dir = getattr(args, 'storage_dir', 'datasets')
        quality_threshold = getattr(args, 'quality_threshold', 0.6)
        
        pipeline = AutoSynthesisPipeline(
            db_config=db_config,
            storage_dir=storage_dir,
            quality_threshold=quality_threshold
        )
    else:
        pipeline = AutoSynthesisPipeline(
            db_config=None,  # Not needed for other commands
            storage_dir=getattr(args, 'storage_dir', 'datasets')
        )
    
    # Execute command
    try:
        if args.command == 'run':
            metadata = pipeline.run_full_pipeline(
                backtest_id=args.backtest_id,
                agent_type=args.agent_type,
                version=args.version,
                format=args.format,
                min_reward=args.min_reward,
                min_confidence=args.min_confidence,
                include_reasoning=not args.no_reasoning
            )
            print(f"\n✅ Pipeline complete!")
            print(f"   Dataset: {metadata.agent_type} v{metadata.version}")
            print(f"   Examples: {metadata.example_count}")
            print(f"   Format: {metadata.format}")
            print(f"   Quality: {metadata.quality_stats.get('avg_quality_score', 'N/A')}")
        
        elif args.command == 'extract':
            count = pipeline.extract_trajectories(
                backtest_id=args.backtest_id,
                agent_type=args.agent_type,
                output_file=args.output,
                min_reward=args.min_reward,
                min_confidence=args.min_confidence
            )
            print(f"\n✅ Extracted {count} trajectories to {args.output}")
        
        elif args.command == 'score':
            count = pipeline.score_trajectories(
                input_file=args.input,
                output_file=args.output
            )
            print(f"\n✅ Scored {count} trajectories, saved to {args.output}")
        
        elif args.command == 'convert':
            count = pipeline.convert_trajectories(
                input_file=args.input,
                output_file=args.output,
                format=args.format,
                include_reasoning=not args.no_reasoning
            )
            print(f"\n✅ Converted {count} examples to {args.format}, saved to {args.output}")
        
        elif args.command == 'save':
            metadata = pipeline.save_dataset(
                input_file=args.input,
                agent_type=args.agent_type,
                version=args.version,
                format=args.format,
                backtest_id=args.backtest_id
            )
            print(f"\n✅ Saved {metadata.example_count} examples as {metadata.agent_type} v{metadata.version}")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
