#!/usr/bin/env python3
"""
Batch Processing for Judge Evaluations - Task 3.3

Efficient batch processing of trajectory evaluations with progress tracking.

Features:
- Batch evaluation with progress tracking
- Parallel processing support (optional)
- Chunk-based processing
- Progress callbacks
- Error handling per item
- Results aggregation
- Resume from checkpoint

Phase A1 Week 3-4: Task 3.3 COMPLETE
"""

import os
import json
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from loguru import logger

from training.judge.llm_judge import LLMJudge, JudgeRubric, JudgeEvaluation


@dataclass
class BatchProgress:
    """
    Batch processing progress
    """
    total: int
    completed: int
    failed: int
    passed: int
    skipped: int
    start_time: float
    end_time: Optional[float] = None
    
    @property
    def elapsed_time(self) -> float:
        """Elapsed time in seconds"""
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    @property
    def items_per_second(self) -> float:
        """Processing rate"""
        elapsed = self.elapsed_time
        return self.completed / elapsed if elapsed > 0 else 0.0
    
    @property
    def eta_seconds(self) -> float:
        """Estimated time to completion"""
        rate = self.items_per_second
        remaining = self.total - self.completed
        return remaining / rate if rate > 0 else 0.0
    
    @property
    def progress_percent(self) -> float:
        """Progress percentage"""
        return (self.completed / self.total * 100) if self.total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'total': self.total,
            'completed': self.completed,
            'failed': self.failed,
            'passed': self.passed,
            'skipped': self.skipped,
            'elapsed_time': self.elapsed_time,
            'items_per_second': self.items_per_second,
            'eta_seconds': self.eta_seconds,
            'progress_percent': self.progress_percent
        }


@dataclass
class BatchResult:
    """
    Batch processing result
    """
    evaluations: List[JudgeEvaluation]
    progress: BatchProgress
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def pass_rate(self) -> float:
        """Pass rate"""
        total = len(self.evaluations)
        return (self.progress.passed / total) if total > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Error rate"""
        total = self.progress.total
        return (self.progress.failed / total) if total > 0 else 0.0
    
    @property
    def average_score(self) -> float:
        """Average overall score"""
        if not self.evaluations:
            return 0.0
        return sum(e.overall_score for e in self.evaluations) / len(self.evaluations)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'evaluations': [e.to_dict() for e in self.evaluations],
            'progress': self.progress.to_dict(),
            'errors': self.errors,
            'pass_rate': self.pass_rate,
            'error_rate': self.error_rate,
            'average_score': self.average_score,
            'metadata': self.metadata
        }


class BatchProcessor:
    """
    Batch processor for judge evaluations
    """
    
    def __init__(
        self,
        judge: LLMJudge,
        chunk_size: int = 10,
        max_workers: Optional[int] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize batch processor
        
        Args:
            judge: LLMJudge instance
            chunk_size: Number of items per chunk
            max_workers: Max parallel workers (None = sequential)
            checkpoint_dir: Directory for checkpoints (optional)
        """
        self.judge = judge
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.checkpoint_dir = checkpoint_dir
        
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"BatchProcessor initialized: chunk_size={chunk_size}, "
            f"max_workers={max_workers}, checkpoint_dir={checkpoint_dir}"
        )
    
    def process_batch(
        self,
        trajectories: List[Dict],
        rubrics: List[JudgeRubric],
        context: Optional[Dict] = None,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
        checkpoint_name: Optional[str] = None,
        resume: bool = False
    ) -> BatchResult:
        """
        Process batch of trajectories
        
        Args:
            trajectories: List of trajectory data
            rubrics: Evaluation rubrics
            context: Additional context (optional)
            progress_callback: Progress callback function (optional)
            checkpoint_name: Checkpoint name for resume (optional)
            resume: Resume from checkpoint if exists
        
        Returns:
            BatchResult object
        """
        logger.info(f"Processing batch of {len(trajectories)} trajectories")
        
        # Initialize progress
        progress = BatchProgress(
            total=len(trajectories),
            completed=0,
            failed=0,
            passed=0,
            skipped=0,
            start_time=time.time()
        )
        
        evaluations: List[JudgeEvaluation] = []
        errors: List[Dict[str, Any]] = []
        
        # Load checkpoint if resuming
        completed_ids = set()
        if resume and checkpoint_name and self.checkpoint_dir:
            checkpoint_data = self._load_checkpoint(checkpoint_name)
            if checkpoint_data:
                evaluations = checkpoint_data['evaluations']
                completed_ids = set(e.trajectory_id for e in evaluations)
                progress.completed = len(evaluations)
                progress.passed = sum(1 for e in evaluations if e.passed)
                progress.failed = checkpoint_data.get('failed_count', 0)
                logger.info(f"Resumed from checkpoint: {len(completed_ids)} completed")
        
        # Process in chunks
        for chunk_start in range(0, len(trajectories), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(trajectories))
            chunk = trajectories[chunk_start:chunk_end]
            
            # Filter out completed items if resuming
            if resume:
                chunk = [t for t in chunk if t.get('id') not in completed_ids]
            
            if not chunk:
                progress.skipped += (chunk_end - chunk_start)
                continue
            
            logger.info(f"Processing chunk {chunk_start}-{chunk_end} ({len(chunk)} items)")
            
            # Process chunk
            if self.max_workers and self.max_workers > 1:
                chunk_results = self._process_chunk_parallel(
                    chunk, rubrics, context, progress, progress_callback
                )
            else:
                chunk_results = self._process_chunk_sequential(
                    chunk, rubrics, context, progress, progress_callback
                )
            
            # Collect results
            for result in chunk_results:
                if isinstance(result, JudgeEvaluation):
                    evaluations.append(result)
                    if result.passed:
                        progress.passed += 1
                else:
                    # Error
                    errors.append(result)
                    progress.failed += 1
                
                progress.completed += 1
            
            # Save checkpoint
            if checkpoint_name and self.checkpoint_dir:
                self._save_checkpoint(
                    checkpoint_name,
                    evaluations,
                    progress,
                    errors
                )
        
        # Finalize
        progress.end_time = time.time()
        
        logger.info(
            f"Batch processing complete: {progress.completed}/{progress.total} "
            f"({progress.passed} passed, {progress.failed} failed, "
            f"{progress.skipped} skipped)"
        )
        
        return BatchResult(
            evaluations=evaluations,
            progress=progress,
            errors=errors,
            metadata={
                'chunk_size': self.chunk_size,
                'max_workers': self.max_workers,
                'checkpoint_name': checkpoint_name
            }
        )
    
    def _process_chunk_sequential(
        self,
        chunk: List[Dict],
        rubrics: List[JudgeRubric],
        context: Optional[Dict],
        progress: BatchProgress,
        progress_callback: Optional[Callable[[BatchProgress], None]]
    ) -> List[Any]:
        """
        Process chunk sequentially
        
        Args:
            chunk: Chunk of trajectories
            rubrics: Evaluation rubrics
            context: Additional context
            progress: Progress tracker
            progress_callback: Progress callback
        
        Returns:
            List of results (JudgeEvaluation or error dict)
        """
        results = []
        
        for trajectory in chunk:
            try:
                evaluation = self.judge.evaluate_trajectory(
                    trajectory, rubrics, context
                )
                results.append(evaluation)
            except Exception as e:
                logger.error(f"Failed to evaluate {trajectory.get('id')}: {e}")
                results.append({
                    'trajectory_id': trajectory.get('id', 'unknown'),
                    'error': str(e),
                    'error_type': type(e).__name__
                })
            
            # Call progress callback
            if progress_callback:
                progress_callback(progress)
        
        return results
    
    def _process_chunk_parallel(
        self,
        chunk: List[Dict],
        rubrics: List[JudgeRubric],
        context: Optional[Dict],
        progress: BatchProgress,
        progress_callback: Optional[Callable[[BatchProgress], None]]
    ) -> List[Any]:
        """
        Process chunk in parallel
        
        Args:
            chunk: Chunk of trajectories
            rubrics: Evaluation rubrics
            context: Additional context
            progress: Progress tracker
            progress_callback: Progress callback
        
        Returns:
            List of results (JudgeEvaluation or error dict)
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_trajectory = {
                executor.submit(
                    self.judge.evaluate_trajectory,
                    trajectory,
                    rubrics,
                    context
                ): trajectory
                for trajectory in chunk
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_trajectory):
                trajectory = future_to_trajectory[future]
                
                try:
                    evaluation = future.result()
                    results.append(evaluation)
                except Exception as e:
                    logger.error(f"Failed to evaluate {trajectory.get('id')}: {e}")
                    results.append({
                        'trajectory_id': trajectory.get('id', 'unknown'),
                        'error': str(e),
                        'error_type': type(e).__name__
                    })
                
                # Call progress callback
                if progress_callback:
                    progress_callback(progress)
        
        return results
    
    def _save_checkpoint(
        self,
        checkpoint_name: str,
        evaluations: List[JudgeEvaluation],
        progress: BatchProgress,
        errors: List[Dict]
    ):
        """
        Save checkpoint
        
        Args:
            checkpoint_name: Checkpoint name
            evaluations: List of evaluations
            progress: Progress tracker
            errors: List of errors
        """
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = Path(self.checkpoint_dir) / f"{checkpoint_name}.json"
        
        data = {
            'evaluations': [e.to_dict() for e in evaluations],
            'progress': progress.to_dict(),
            'errors': errors,
            'failed_count': progress.failed,
            'timestamp': time.time()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_name: str) -> Optional[Dict]:
        """
        Load checkpoint
        
        Args:
            checkpoint_name: Checkpoint name
        
        Returns:
            Checkpoint data or None
        """
        if not self.checkpoint_dir:
            return None
        
        checkpoint_path = Path(self.checkpoint_dir) / f"{checkpoint_name}.json"
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct evaluations
            evaluations = []
            for eval_dict in data['evaluations']:
                # Reconstruct rubrics
                rubrics = []
                for rubric_dict in eval_dict.get('rubrics', []):
                    rubric = JudgeRubric(
                        criterion=rubric_dict['criterion'],
                        description=rubric_dict['description'],
                        weight=rubric_dict['weight'],
                        min_score=rubric_dict['min_score'],
                        max_score=rubric_dict['max_score']
                    )
                    rubrics.append(rubric)
                
                # Reconstruct evaluation
                evaluation = JudgeEvaluation(
                    trajectory_id=eval_dict['trajectory_id'],
                    overall_score=eval_dict['overall_score'],
                    criterion_scores=eval_dict['criterion_scores'],
                    reasoning=eval_dict['reasoning'],
                    pass_threshold=eval_dict['pass_threshold'],
                    passed=eval_dict['passed'],
                    model=eval_dict['model'],
                    provider=eval_dict['provider'],
                    rubrics=rubrics,
                    metadata=eval_dict.get('metadata', {})
                )
                evaluations.append(evaluation)
            
            data['evaluations'] = evaluations
            
            logger.info(f"Checkpoint loaded: {checkpoint_path} ({len(evaluations)} evaluations)")
            
            return data
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def filter_by_score(
        self,
        evaluations: List[JudgeEvaluation],
        min_score: float
    ) -> List[JudgeEvaluation]:
        """
        Filter evaluations by minimum score
        
        Args:
            evaluations: List of evaluations
            min_score: Minimum overall score
        
        Returns:
            Filtered list
        """
        filtered = [e for e in evaluations if e.overall_score >= min_score]
        logger.info(f"Filtered {len(filtered)}/{len(evaluations)} by score >= {min_score}")
        return filtered
    
    def filter_passed(
        self,
        evaluations: List[JudgeEvaluation]
    ) -> List[JudgeEvaluation]:
        """
        Filter evaluations that passed
        
        Args:
            evaluations: List of evaluations
        
        Returns:
            Filtered list
        """
        filtered = [e for e in evaluations if e.passed]
        logger.info(f"Filtered {len(filtered)}/{len(evaluations)} passed")
        return filtered
    
    def calculate_statistics(
        self,
        evaluations: List[JudgeEvaluation]
    ) -> Dict[str, Any]:
        """
        Calculate statistics for evaluations
        
        Args:
            evaluations: List of evaluations
        
        Returns:
            Statistics dict
        """
        if not evaluations:
            return {
                'count': 0,
                'pass_rate': 0.0,
                'average_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'criterion_averages': {}
            }
        
        scores = [e.overall_score for e in evaluations]
        passed_count = sum(1 for e in evaluations if e.passed)
        
        # Calculate criterion averages
        criterion_averages = {}
        all_criteria = set()
        for e in evaluations:
            all_criteria.update(e.criterion_scores.keys())
        
        for criterion in all_criteria:
            criterion_scores = [
                e.criterion_scores.get(criterion, 0.0)
                for e in evaluations
                if criterion in e.criterion_scores
            ]
            if criterion_scores:
                criterion_averages[criterion] = sum(criterion_scores) / len(criterion_scores)
        
        return {
            'count': len(evaluations),
            'pass_rate': passed_count / len(evaluations),
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'criterion_averages': criterion_averages
        }


# ============================================================================
# Helper Functions
# ============================================================================

def process_trajectories_batch(
    trajectories: List[Dict],
    rubrics: List[JudgeRubric],
    judge: LLMJudge,
    chunk_size: int = 10,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[BatchProgress], None]] = None
) -> BatchResult:
    """
    Process trajectories in batch (convenience function)
    
    Args:
        trajectories: List of trajectory data
        rubrics: Evaluation rubrics
        judge: LLMJudge instance
        chunk_size: Chunk size
        max_workers: Max parallel workers
        progress_callback: Progress callback
    
    Returns:
        BatchResult object
    """
    processor = BatchProcessor(
        judge=judge,
        chunk_size=chunk_size,
        max_workers=max_workers
    )
    
    return processor.process_batch(
        trajectories=trajectories,
        rubrics=rubrics,
        progress_callback=progress_callback
    )


if __name__ == "__main__":
    # Example usage
    print("=== Batch Processor Example ===\n")
    
    # Mock trajectories
    trajectories = [
        {
            'id': f'traj_{i:03d}',
            'agent_type': 'technical',
            'query': f'Analyze stock {i}',
            'response': f'Analysis for stock {i}',
            'reasoning': 'Based on technical indicators',
            'final_reward': 0.7 + (i % 3) * 0.1,
            'confidence_score': 0.8
        }
        for i in range(25)
    ]
    
    print(f"Created {len(trajectories)} mock trajectories\n")
    
    # Mock rubrics
    from training.judge.llm_judge import JudgeRubric
    
    rubrics = [
        JudgeRubric("accuracy", "Accuracy check", weight=2.0),
        JudgeRubric("clarity", "Clarity check", weight=1.5)
    ]
    
    print(f"Created {len(rubrics)} rubrics\n")
    
    # Progress callback
    def progress_callback(progress: BatchProgress):
        print(f"  Progress: {progress.completed}/{progress.total} "
              f"({progress.progress_percent:.1f}%) - "
              f"{progress.items_per_second:.2f} items/sec")
    
    print("Batch processing configuration:")
    print(f"  Chunk size: 10")
    print(f"  Max workers: None (sequential)")
    print(f"  Progress callback: Yes")
    print()
    
    print("✅ Example completed!")
    print("\nNote: Actual batch processing requires LLM API and will be tested separately.")
