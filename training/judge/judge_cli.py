#!/usr/bin/env python3
"""
Judge CLI Interface - Task 3.6

Command-line interface for judge-approved filtering.

Commands:
- evaluate: Evaluate trajectories with judge
- list-rubrics: List available rubric templates
- show-rubric: Show rubric details
- batch-evaluate: Batch evaluation with progress
- filter: Filter by quality scores
- export: Export evaluation results
- stats: Show evaluation statistics

Phase A1 Week 3-4: Task 3.6 COMPLETE
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

from training.judge.llm_judge import (
    LLMJudge,
    JudgeProvider,
    JudgeEvaluation,
    JudgeRubric
)
from training.judge.rubrics_loader import (
    RubricsLoader,
    load_rubrics
)
from training.judge.batch_processor import (
    BatchProcessor,
    BatchProgress
)
from training.judge.rate_limiter import (
    RateLimiter,
    RateLimitConfig
)
from training.judge.retry_handler import (
    RetryHandler,
    RetryConfig
)


class JudgeCLI:
    """
    Judge CLI interface
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "openai",
        model: Optional[str] = None
    ):
        """
        Initialize judge CLI
        
        Args:
            api_key: API key for LLM provider
            provider: LLM provider (openai or anthropic)
            model: Model name (optional)
        """
        self.api_key = api_key
        self.provider = JudgeProvider(provider.lower())
        self.model = model
        
        # Initialize components
        self.rubrics_loader = RubricsLoader()
        
        logger.info(f"JudgeCLI initialized: provider={provider}, model={model}")
    
    def _create_judge(
        self,
        rubrics: List[JudgeRubric],
        pass_threshold: float = 0.7
    ) -> LLMJudge:
        """
        Create LLM judge instance
        
        Args:
            rubrics: List of rubrics
            pass_threshold: Pass threshold
        
        Returns:
            LLMJudge instance
        """
        return LLMJudge(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            pass_threshold=pass_threshold
        )
    
    def evaluate_single(
        self,
        trajectory_file: str,
        rubric_template: Optional[str] = None,
        rubric_file: Optional[str] = None,
        output_file: Optional[str] = None,
        pass_threshold: float = 0.7
    ):
        """
        Evaluate single trajectory
        
        Args:
            trajectory_file: Path to trajectory JSON file
            rubric_template: Rubric template name (optional)
            rubric_file: Path to rubric file (optional)
            output_file: Path to output file (optional)
            pass_threshold: Pass threshold
        """
        logger.info(f"Evaluating trajectory: {trajectory_file}")
        
        # Load trajectory
        with open(trajectory_file, 'r') as f:
            trajectory = json.load(f)
        
        # Load rubrics
        if rubric_file:
            rubrics = load_rubrics(file_path=rubric_file)
        elif rubric_template:
            rubrics = load_rubrics(template_name=rubric_template)
        else:
            logger.error("Must specify either --rubric-template or --rubric-file")
            return
        
        logger.info(f"Loaded {len(rubrics)} rubrics")
        
        # Create judge
        judge = self._create_judge(rubrics, pass_threshold)
        
        # Evaluate
        logger.info("Evaluating...")
        evaluation = judge.evaluate_trajectory(trajectory, rubrics)
        
        # Print results
        self._print_evaluation(evaluation)
        
        # Save to file
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(evaluation.to_dict(), f, indent=2)
            logger.info(f"Saved evaluation to {output_file}")
    
    def batch_evaluate(
        self,
        trajectories_file: str,
        rubric_template: Optional[str] = None,
        rubric_file: Optional[str] = None,
        output_file: Optional[str] = None,
        pass_threshold: float = 0.7,
        chunk_size: int = 10,
        max_workers: Optional[int] = None,
        rate_limit_rpm: int = 60,
        max_concurrent: int = 5
    ):
        """
        Batch evaluate trajectories
        
        Args:
            trajectories_file: Path to trajectories JSON/JSONL file
            rubric_template: Rubric template name (optional)
            rubric_file: Path to rubric file (optional)
            output_file: Path to output file (optional)
            pass_threshold: Pass threshold
            chunk_size: Chunk size for batch processing
            max_workers: Max workers for parallel processing
            rate_limit_rpm: Rate limit (requests per minute)
            max_concurrent: Max concurrent requests
        """
        logger.info(f"Batch evaluating trajectories: {trajectories_file}")
        
        # Load trajectories
        trajectories = self._load_trajectories(trajectories_file)
        logger.info(f"Loaded {len(trajectories)} trajectories")
        
        # Load rubrics
        if rubric_file:
            rubrics = load_rubrics(file_path=rubric_file)
        elif rubric_template:
            rubrics = load_rubrics(template_name=rubric_template)
        else:
            logger.error("Must specify either --rubric-template or --rubric-file")
            return
        
        logger.info(f"Loaded {len(rubrics)} rubrics")
        
        # Create judge
        judge = self._create_judge(rubrics, pass_threshold)
        
        # Create rate limiter
        rate_limiter = RateLimiter(RateLimitConfig(
            requests_per_minute=rate_limit_rpm,
            max_concurrent=max_concurrent
        ))
        
        # Create batch processor
        processor = BatchProcessor(
            judge=judge,
            chunk_size=chunk_size,
            max_workers=max_workers
        )
        
        # Progress callback
        def progress_callback(progress: BatchProgress):
            print(
                f"\rProgress: {progress.completed}/{progress.total} "
                f"({progress.progress_percent:.1f}%) | "
                f"Passed: {progress.passed} | "
                f"Failed: {progress.failed} | "
                f"ETA: {progress.eta_seconds:.0f}s",
                end='',
                flush=True
            )
        
        # Evaluate
        logger.info("Batch evaluating...")
        result = processor.process_batch(
            trajectories,
            rubrics,
            progress_callback=progress_callback
        )
        
        print()  # New line after progress
        
        # Print summary
        self._print_batch_summary(result)
        
        # Save to file
        if output_file:
            output_data = {
                'evaluations': [e.to_dict() for e in result.evaluations],
                'progress': result.progress.to_dict(),
                'errors': result.errors
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Saved {len(result.evaluations)} evaluations to {output_file}")
    
    def list_rubrics(self, agent_type: Optional[str] = None):
        """
        List available rubric templates
        
        Args:
            agent_type: Filter by agent type (optional)
        """
        templates = self.rubrics_loader.list_templates(agent_type=agent_type)
        
        if not templates:
            print("No rubric templates found")
            return
        
        print(f"\nAvailable Rubric Templates ({len(templates)}):\n")
        
        for template in templates:
            print(f"  • {template.name}")
            print(f"    Agent Type: {template.agent_type}")
            print(f"    Version: {template.version}")
            print(f"    Rubrics: {len(template.rubrics)}")
            if template.description:
                print(f"    Description: {template.description}")
            print()
    
    def show_rubric(self, template_name: str):
        """
        Show rubric template details
        
        Args:
            template_name: Template name
        """
        template = self.rubrics_loader.get_template(template_name)
        
        if not template:
            logger.error(f"Template not found: {template_name}")
            return
        
        print(f"\nRubric Template: {template.name}\n")
        print(f"Agent Type: {template.agent_type}")
        print(f"Version: {template.version}")
        if template.description:
            print(f"Description: {template.description}")
        print(f"\nRubrics ({len(template.rubrics)}):\n")
        
        total_weight = sum(r.weight for r in template.rubrics)
        
        for rubric in template.rubrics:
            criterion_name = rubric.criterion.value if hasattr(rubric.criterion, 'value') else rubric.criterion
            print(f"  • {criterion_name}")
            print(f"    Description: {rubric.description}")
            print(f"    Weight: {rubric.weight} ({rubric.weight/total_weight*100:.1f}%)")
            print(f"    Score Range: [{rubric.min_score}, {rubric.max_score}]")
            print()
        
        print(f"Total Weight: {total_weight}")
    
    def filter_evaluations(
        self,
        evaluations_file: str,
        output_file: str,
        min_score: Optional[float] = None,
        passed_only: bool = False
    ):
        """
        Filter evaluations by score
        
        Args:
            evaluations_file: Path to evaluations JSON file
            output_file: Path to output file
            min_score: Minimum score threshold (optional)
            passed_only: Filter passed only
        """
        logger.info(f"Filtering evaluations: {evaluations_file}")
        
        # Load evaluations
        with open(evaluations_file, 'r') as f:
            data = json.load(f)
        
        evaluations = [
            JudgeEvaluation(**e) for e in data.get('evaluations', data)
        ]
        
        logger.info(f"Loaded {len(evaluations)} evaluations")
        
        # Filter
        filtered = evaluations
        
        if passed_only:
            filtered = [e for e in filtered if e.passed]
            logger.info(f"Filtered to {len(filtered)} passed evaluations")
        
        if min_score is not None:
            filtered = [e for e in filtered if e.overall_score >= min_score]
            logger.info(f"Filtered to {len(filtered)} with score >= {min_score}")
        
        # Save
        output_data = [e.to_dict() for e in filtered]
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved {len(filtered)} filtered evaluations to {output_file}")
        
        # Print summary
        print(f"\nFiltered: {len(filtered)}/{len(evaluations)} evaluations")
        if filtered:
            avg_score = sum(e.overall_score for e in filtered) / len(filtered)
            print(f"Average score: {avg_score:.3f}")
    
    def export_results(
        self,
        evaluations_file: str,
        output_file: str,
        format: str = "json"
    ):
        """
        Export evaluation results
        
        Args:
            evaluations_file: Path to evaluations JSON file
            output_file: Path to output file
            format: Output format (json, csv, markdown)
        """
        logger.info(f"Exporting evaluations: {evaluations_file}")
        
        # Load evaluations
        with open(evaluations_file, 'r') as f:
            data = json.load(f)
        
        evaluations = [
            JudgeEvaluation(**e) for e in data.get('evaluations', data)
        ]
        
        logger.info(f"Loaded {len(evaluations)} evaluations")
        
        # Export
        if format == "json":
            self._export_json(evaluations, output_file)
        elif format == "csv":
            self._export_csv(evaluations, output_file)
        elif format == "markdown":
            self._export_markdown(evaluations, output_file)
        else:
            logger.error(f"Unsupported format: {format}")
            return
        
        logger.info(f"Exported to {output_file}")
    
    def show_stats(self, evaluations_file: str):
        """
        Show evaluation statistics
        
        Args:
            evaluations_file: Path to evaluations JSON file
        """
        logger.info(f"Loading evaluations: {evaluations_file}")
        
        # Load evaluations
        with open(evaluations_file, 'r') as f:
            data = json.load(f)
        
        evaluations = [
            JudgeEvaluation(**e) for e in data.get('evaluations', data)
        ]
        
        logger.info(f"Loaded {len(evaluations)} evaluations")
        
        # Calculate statistics
        stats = self._calculate_stats(evaluations)
        
        # Print statistics
        self._print_stats(stats)
    
    def _load_trajectories(self, file_path: str) -> List[Dict]:
        """Load trajectories from file"""
        trajectories = []
        
        with open(file_path, 'r') as f:
            # Try JSONL first
            first_line = f.readline()
            f.seek(0)
            
            try:
                json.loads(first_line)
                # JSONL format
                for line in f:
                    if line.strip():
                        trajectories.append(json.loads(line))
            except:
                # JSON format
                f.seek(0)
                data = json.load(f)
                if isinstance(data, list):
                    trajectories = data
                else:
                    trajectories = [data]
        
        return trajectories
    
    def _print_evaluation(self, evaluation: JudgeEvaluation):
        """Print single evaluation"""
        print("\n" + "="*60)
        print(f"Trajectory ID: {evaluation.trajectory_id}")
        print("="*60)
        print(f"\nOverall Score: {evaluation.overall_score:.3f}")
        print(f"Passed: {'✓' if evaluation.passed else '✗'} (threshold: {evaluation.pass_threshold})")
        print(f"\nCriterion Scores:")
        
        for criterion, score in evaluation.criterion_scores.items():
            print(f"  • {criterion}: {score:.3f}")
        
        print(f"\nReasoning:")
        print(f"  {evaluation.reasoning}")
        print()
    
    def _print_batch_summary(self, result):
        """Print batch evaluation summary"""
        print("\n" + "="*60)
        print("Batch Evaluation Summary")
        print("="*60)
        
        progress = result.progress
        
        print(f"\nTotal: {progress.total}")
        print(f"Completed: {progress.completed}")
        print(f"Passed: {progress.passed} ({result.pass_rate:.1%})")
        print(f"Failed: {progress.failed}")
        print(f"Skipped: {progress.skipped}")
        
        print(f"\nElapsed Time: {progress.elapsed_time:.1f}s")
        print(f"Items/sec: {progress.items_per_second:.2f}")
        
        if result.evaluations:
            avg_score = result.average_score
            print(f"\nAverage Score: {avg_score:.3f}")
        
        if result.errors:
            print(f"\nErrors: {len(result.errors)}")
            for error in result.errors[:5]:  # Show first 5
                print(f"  • {error.get('trajectory_id')}: {error.get('error')}")
        
        print()
    
    def _calculate_stats(self, evaluations: List[JudgeEvaluation]) -> Dict:
        """Calculate evaluation statistics"""
        if not evaluations:
            return {}
        
        total = len(evaluations)
        passed = sum(1 for e in evaluations if e.passed)
        
        scores = [e.overall_score for e in evaluations]
        avg_score = sum(scores) / total
        min_score = min(scores)
        max_score = max(scores)
        
        # Criterion averages
        criterion_scores = {}
        for e in evaluations:
            for criterion, score in e.criterion_scores.items():
                if criterion not in criterion_scores:
                    criterion_scores[criterion] = []
                criterion_scores[criterion].append(score)
        
        criterion_averages = {
            criterion: sum(scores) / len(scores)
            for criterion, scores in criterion_scores.items()
        }
        
        return {
            'total': total,
            'passed': passed,
            'pass_rate': passed / total,
            'average_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'criterion_averages': criterion_averages
        }
    
    def _print_stats(self, stats: Dict):
        """Print statistics"""
        print("\n" + "="*60)
        print("Evaluation Statistics")
        print("="*60)
        
        print(f"\nTotal Evaluations: {stats['total']}")
        print(f"Passed: {stats['passed']} ({stats['pass_rate']:.1%})")
        
        print(f"\nOverall Scores:")
        print(f"  Average: {stats['average_score']:.3f}")
        print(f"  Min: {stats['min_score']:.3f}")
        print(f"  Max: {stats['max_score']:.3f}")
        
        print(f"\nCriterion Averages:")
        for criterion, avg in sorted(stats['criterion_averages'].items()):
            print(f"  • {criterion}: {avg:.3f}")
        
        print()
    
    def _export_json(self, evaluations: List[JudgeEvaluation], output_file: str):
        """Export to JSON"""
        data = [e.to_dict() for e in evaluations]
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_csv(self, evaluations: List[JudgeEvaluation], output_file: str):
        """Export to CSV"""
        import csv
        
        if not evaluations:
            return
        
        # Get all criterion names
        all_criteria = set()
        for e in evaluations:
            all_criteria.update(e.criterion_scores.keys())
        
        all_criteria = sorted(all_criteria)
        
        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['trajectory_id', 'overall_score', 'passed'] + all_criteria
            writer.writerow(header)
            
            # Rows
            for e in evaluations:
                row = [
                    e.trajectory_id,
                    f"{e.overall_score:.3f}",
                    e.passed
                ]
                
                for criterion in all_criteria:
                    score = e.criterion_scores.get(criterion, '')
                    if score != '':
                        score = f"{score:.3f}"
                    row.append(score)
                
                writer.writerow(row)
    
    def _export_markdown(self, evaluations: List[JudgeEvaluation], output_file: str):
        """Export to Markdown"""
        if not evaluations:
            return
        
        # Get all criterion names
        all_criteria = set()
        for e in evaluations:
            all_criteria.update(e.criterion_scores.keys())
        
        all_criteria = sorted(all_criteria)
        
        # Write Markdown
        with open(output_file, 'w') as f:
            # Title
            f.write("# Evaluation Results\n\n")
            
            # Summary
            stats = self._calculate_stats(evaluations)
            f.write("## Summary\n\n")
            f.write(f"- Total: {stats['total']}\n")
            f.write(f"- Passed: {stats['passed']} ({stats['pass_rate']:.1%})\n")
            f.write(f"- Average Score: {stats['average_score']:.3f}\n\n")
            
            # Table
            f.write("## Evaluations\n\n")
            
            # Header
            header = "| Trajectory ID | Overall Score | Passed |"
            for criterion in all_criteria:
                header += f" {criterion} |"
            f.write(header + "\n")
            
            # Separator
            separator = "|" + "---|" * (3 + len(all_criteria))
            f.write(separator + "\n")
            
            # Rows
            for e in evaluations:
                row = f"| {e.trajectory_id} | {e.overall_score:.3f} | {'✓' if e.passed else '✗'} |"
                
                for criterion in all_criteria:
                    score = e.criterion_scores.get(criterion, '-')
                    if score != '-':
                        score = f"{score:.3f}"
                    row += f" {score} |"
                
                f.write(row + "\n")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Judge CLI - Evaluate trajectories with LLM judge"
    )
    
    # Global options
    parser.add_argument(
        '--api-key',
        help='API key for LLM provider (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)'
    )
    parser.add_argument(
        '--provider',
        choices=['openai', 'anthropic'],
        default='openai',
        help='LLM provider'
    )
    parser.add_argument(
        '--model',
        help='Model name (optional, uses provider default)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate single trajectory')
    evaluate_parser.add_argument('trajectory_file', help='Path to trajectory JSON file')
    evaluate_parser.add_argument('--rubric-template', help='Rubric template name')
    evaluate_parser.add_argument('--rubric-file', help='Path to rubric file')
    evaluate_parser.add_argument('--output', help='Path to output file')
    evaluate_parser.add_argument('--pass-threshold', type=float, default=0.7, help='Pass threshold')
    
    # Batch evaluate command
    batch_parser = subparsers.add_parser('batch-evaluate', help='Batch evaluate trajectories')
    batch_parser.add_argument('trajectories_file', help='Path to trajectories JSON/JSONL file')
    batch_parser.add_argument('--rubric-template', help='Rubric template name')
    batch_parser.add_argument('--rubric-file', help='Path to rubric file')
    batch_parser.add_argument('--output', help='Path to output file')
    batch_parser.add_argument('--pass-threshold', type=float, default=0.7, help='Pass threshold')
    batch_parser.add_argument('--chunk-size', type=int, default=10, help='Chunk size')
    batch_parser.add_argument('--max-workers', type=int, help='Max workers for parallel processing')
    batch_parser.add_argument('--rate-limit-rpm', type=int, default=60, help='Rate limit (requests per minute)')
    batch_parser.add_argument('--max-concurrent', type=int, default=5, help='Max concurrent requests')
    
    # List rubrics command
    list_parser = subparsers.add_parser('list-rubrics', help='List available rubric templates')
    list_parser.add_argument('--agent-type', help='Filter by agent type')
    
    # Show rubric command
    show_parser = subparsers.add_parser('show-rubric', help='Show rubric template details')
    show_parser.add_argument('template_name', help='Template name')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter evaluations by score')
    filter_parser.add_argument('evaluations_file', help='Path to evaluations JSON file')
    filter_parser.add_argument('output_file', help='Path to output file')
    filter_parser.add_argument('--min-score', type=float, help='Minimum score threshold')
    filter_parser.add_argument('--passed-only', action='store_true', help='Filter passed only')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export evaluation results')
    export_parser.add_argument('evaluations_file', help='Path to evaluations JSON file')
    export_parser.add_argument('output_file', help='Path to output file')
    export_parser.add_argument('--format', choices=['json', 'csv', 'markdown'], default='json', help='Output format')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show evaluation statistics')
    stats_parser.add_argument('evaluations_file', help='Path to evaluations JSON file')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create CLI
    cli = JudgeCLI(
        api_key=args.api_key,
        provider=args.provider,
        model=args.model
    )
    
    # Execute command
    try:
        if args.command == 'evaluate':
            cli.evaluate_single(
                trajectory_file=args.trajectory_file,
                rubric_template=args.rubric_template,
                rubric_file=args.rubric_file,
                output_file=args.output,
                pass_threshold=args.pass_threshold
            )
        
        elif args.command == 'batch-evaluate':
            cli.batch_evaluate(
                trajectories_file=args.trajectories_file,
                rubric_template=args.rubric_template,
                rubric_file=args.rubric_file,
                output_file=args.output,
                pass_threshold=args.pass_threshold,
                chunk_size=args.chunk_size,
                max_workers=args.max_workers,
                rate_limit_rpm=args.rate_limit_rpm,
                max_concurrent=args.max_concurrent
            )
        
        elif args.command == 'list-rubrics':
            cli.list_rubrics(agent_type=args.agent_type)
        
        elif args.command == 'show-rubric':
            cli.show_rubric(template_name=args.template_name)
        
        elif args.command == 'filter':
            cli.filter_evaluations(
                evaluations_file=args.evaluations_file,
                output_file=args.output_file,
                min_score=args.min_score,
                passed_only=args.passed_only
            )
        
        elif args.command == 'export':
            cli.export_results(
                evaluations_file=args.evaluations_file,
                output_file=args.output_file,
                format=args.format
            )
        
        elif args.command == 'stats':
            cli.show_stats(evaluations_file=args.evaluations_file)
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
