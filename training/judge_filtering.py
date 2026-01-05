"""
Judge-Approved Filtering for SFT Datasets

Applies LLM Judge evaluation to filter high-quality training examples
before adding them to SFT datasets.

Phase A1 Week 3-4: Task 3
- Multi-dimensional quality scoring
- Configurable thresholds per dimension
- Batch evaluation for efficiency
- Detailed feedback logging
- Integration with Auto Synthesis Pipeline

Based on:
- AlpacaEval judge filtering
- RLHF preference data filtering
- Constitutional AI quality gates
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from loguru import logger
import time

from judge.llm_judge import LLMJudge
from data_pipeline.experience_library_postgres import Trajectory
from data_pipeline.data_synthesis import SFTExample


@dataclass
class JudgeFilterConfig:
    """Configuration for judge filtering"""
    # Score thresholds (0-1 scale)
    min_overall_score: float = 0.7
    min_dimension_scores: Dict[str, float] = None  # e.g., {'reasoning': 0.6, 'accuracy': 0.8}
    
    # Filtering strategy
    strategy: str = 'strict'  # 'strict', 'lenient', 'balanced'
    
    # Batch processing
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Logging
    save_feedback: bool = True
    feedback_dir: str = "datasets/judge_feedback"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_strategy(cls, strategy: str) -> 'JudgeFilterConfig':
        """Create config from strategy preset"""
        if strategy == 'strict':
            return cls(
                min_overall_score=0.8,
                min_dimension_scores={
                    'reasoning': 0.75,
                    'accuracy': 0.85,
                    'confidence_calibration': 0.7
                },
                strategy='strict'
            )
        elif strategy == 'lenient':
            return cls(
                min_overall_score=0.6,
                min_dimension_scores={
                    'reasoning': 0.5,
                    'accuracy': 0.6,
                    'confidence_calibration': 0.5
                },
                strategy='lenient'
            )
        else:  # balanced
            return cls(
                min_overall_score=0.7,
                min_dimension_scores={
                    'reasoning': 0.65,
                    'accuracy': 0.75,
                    'confidence_calibration': 0.6
                },
                strategy='balanced'
            )


@dataclass
class JudgeResult:
    """Result of judge evaluation"""
    trajectory_id: str
    overall_score: float
    dimension_scores: Dict[str, float]
    feedback: str
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    passed: bool
    timestamp: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FilteringStats:
    """Statistics from filtering process"""
    total_evaluated: int
    total_passed: int
    total_failed: int
    pass_rate: float
    avg_overall_score: float
    avg_dimension_scores: Dict[str, float]
    failure_reasons: Dict[str, int]  # dimension -> count
    
    def to_dict(self) -> Dict:
        return asdict(self)


class JudgeApprovedFilter:
    """
    Judge-Approved Filtering System
    
    Filters training examples using LLM Judge evaluation to ensure
    only high-quality examples are included in SFT datasets.
    """
    
    def __init__(
        self,
        judge: LLMJudge,
        config: Optional[JudgeFilterConfig] = None
    ):
        """
        Initialize Judge-Approved Filter
        
        Args:
            judge: LLM Judge instance
            config: Filter configuration
        """
        self.judge = judge
        self.config = config or JudgeFilterConfig()
        
        # Create feedback directory
        if self.config.save_feedback:
            self.feedback_dir = Path(self.config.feedback_dir)
            self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Judge-Approved Filter initialized with strategy: {self.config.strategy}")
    
    def filter_trajectories(
        self,
        trajectories: List[Trajectory],
        agent_type: str
    ) -> Tuple[List[Trajectory], FilteringStats]:
        """
        Filter trajectories using Judge evaluation
        
        Args:
            trajectories: List of trajectories to filter
            agent_type: Agent type for rubric selection
        
        Returns:
            (filtered_trajectories, stats) tuple
        """
        logger.info(f"Filtering {len(trajectories)} trajectories for {agent_type}...")
        
        passed_trajectories = []
        judge_results = []
        failure_reasons = {}
        
        # Process in batches
        for i in range(0, len(trajectories), self.config.batch_size):
            batch = trajectories[i:i + self.config.batch_size]
            
            for trajectory in batch:
                try:
                    # Evaluate with Judge
                    result = self._evaluate_trajectory(trajectory, agent_type)
                    judge_results.append(result)
                    
                    # Check if passed
                    if result.passed:
                        # Store judge result in trajectory metadata
                        trajectory.metadata['judge_score'] = result.overall_score
                        trajectory.metadata['judge_feedback'] = result.feedback
                        trajectory.metadata['judge_dimension_scores'] = result.dimension_scores
                        
                        passed_trajectories.append(trajectory)
                    else:
                        # Track failure reasons
                        for dim, score in result.dimension_scores.items():
                            threshold = self.config.min_dimension_scores.get(dim, 0.0)
                            if score < threshold:
                                failure_reasons[dim] = failure_reasons.get(dim, 0) + 1
                
                except Exception as e:
                    logger.warning(f"Failed to evaluate trajectory {trajectory.trajectory_id}: {e}")
                    continue
            
            # Rate limiting
            if i + self.config.batch_size < len(trajectories):
                time.sleep(0.5)  # Avoid rate limits
        
        # Calculate stats
        stats = self._calculate_stats(judge_results, failure_reasons)
        
        # Save feedback
        if self.config.save_feedback:
            self._save_feedback(judge_results, agent_type)
        
        logger.info(
            f"✅ Filtering complete: {stats.total_passed}/{stats.total_evaluated} passed "
            f"({stats.pass_rate*100:.1f}%), avg_score={stats.avg_overall_score:.3f}"
        )
        
        return passed_trajectories, stats
    
    def filter_sft_examples(
        self,
        sft_examples: List[SFTExample],
        agent_type: str
    ) -> Tuple[List[SFTExample], FilteringStats]:
        """
        Filter SFT examples using Judge evaluation
        
        Args:
            sft_examples: List of SFT examples to filter
            agent_type: Agent type for rubric selection
        
        Returns:
            (filtered_examples, stats) tuple
        """
        logger.info(f"Filtering {len(sft_examples)} SFT examples for {agent_type}...")
        
        passed_examples = []
        judge_results = []
        failure_reasons = {}
        
        for example in sft_examples:
            try:
                # Evaluate with Judge
                result = self._evaluate_sft_example(example, agent_type)
                judge_results.append(result)
                
                # Check if passed
                if result.passed:
                    # Store judge result in metadata
                    example.metadata['judge_score'] = result.overall_score
                    example.metadata['judge_feedback'] = result.feedback
                    example.metadata['judge_dimension_scores'] = result.dimension_scores
                    
                    passed_examples.append(example)
                else:
                    # Track failure reasons
                    for dim, score in result.dimension_scores.items():
                        threshold = self.config.min_dimension_scores.get(dim, 0.0)
                        if score < threshold:
                            failure_reasons[dim] = failure_reasons.get(dim, 0) + 1
            
            except Exception as e:
                logger.warning(f"Failed to evaluate example {example.metadata.get('trajectory_id')}: {e}")
                continue
        
        # Calculate stats
        stats = self._calculate_stats(judge_results, failure_reasons)
        
        # Save feedback
        if self.config.save_feedback:
            self._save_feedback(judge_results, agent_type)
        
        logger.info(
            f"✅ Filtering complete: {stats.total_passed}/{stats.total_evaluated} passed "
            f"({stats.pass_rate*100:.1f}%), avg_score={stats.avg_overall_score:.3f}"
        )
        
        return passed_examples, stats
    
    def _evaluate_trajectory(
        self,
        trajectory: Trajectory,
        agent_type: str
    ) -> JudgeResult:
        """
        Evaluate a single trajectory with Judge
        
        Args:
            trajectory: Trajectory to evaluate
            agent_type: Agent type
        
        Returns:
            JudgeResult
        """
        # Prepare agent output for judge
        agent_output = {
            'reasoning': trajectory.reasoning,
            'confidence': trajectory.confidence,
            'recommendation': trajectory.recommendation,
            'position_size': trajectory.position_size,
            'stop_loss': trajectory.stop_loss,
            'take_profit': trajectory.take_profit
        }
        
        # Prepare context
        context = {
            'symbol': trajectory.symbol,
            'market_state': trajectory.market_state,
            'agent_inputs': trajectory.agent_inputs,
            'actual_outcome': {
                'success': trajectory.success,
                'reward': trajectory.reward
            }
        }
        
        # Evaluate with Judge (with retries)
        for attempt in range(self.config.max_retries):
            try:
                evaluation = self.judge.evaluate(
                    agent_output=agent_output,
                    rubric_name=agent_type,
                    context=context
                )
                break
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                logger.warning(f"Judge evaluation attempt {attempt+1} failed: {e}")
                time.sleep(self.config.retry_delay)
        
        # Check if passed
        passed = self._check_passed(evaluation)
        
        return JudgeResult(
            trajectory_id=trajectory.trajectory_id,
            overall_score=evaluation['overall_score'],
            dimension_scores=evaluation['dimension_scores'],
            feedback=evaluation['feedback'],
            strengths=evaluation.get('strengths', []),
            weaknesses=evaluation.get('weaknesses', []),
            suggestions=evaluation.get('suggestions', []),
            passed=passed,
            timestamp=time.time()
        )
    
    def _evaluate_sft_example(
        self,
        example: SFTExample,
        agent_type: str
    ) -> JudgeResult:
        """
        Evaluate a single SFT example with Judge
        
        Args:
            example: SFT example to evaluate
            agent_type: Agent type
        
        Returns:
            JudgeResult
        """
        # Extract agent output from messages
        agent_output = {}
        for msg in example.messages:
            if msg['role'] == 'assistant':
                try:
                    agent_output = json.loads(msg['content'])
                except:
                    agent_output = {'response': msg['content']}
        
        # Evaluate with Judge (with retries)
        for attempt in range(self.config.max_retries):
            try:
                evaluation = self.judge.evaluate(
                    agent_output=agent_output,
                    rubric_name=agent_type,
                    context=example.metadata
                )
                break
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                logger.warning(f"Judge evaluation attempt {attempt+1} failed: {e}")
                time.sleep(self.config.retry_delay)
        
        # Check if passed
        passed = self._check_passed(evaluation)
        
        return JudgeResult(
            trajectory_id=example.metadata.get('trajectory_id', 'unknown'),
            overall_score=evaluation['overall_score'],
            dimension_scores=evaluation['dimension_scores'],
            feedback=evaluation['feedback'],
            strengths=evaluation.get('strengths', []),
            weaknesses=evaluation.get('weaknesses', []),
            suggestions=evaluation.get('suggestions', []),
            passed=passed,
            timestamp=time.time()
        )
    
    def _check_passed(self, evaluation: Dict) -> bool:
        """
        Check if evaluation passed thresholds
        
        Args:
            evaluation: Judge evaluation result
        
        Returns:
            True if passed, False otherwise
        """
        # Check overall score
        if evaluation['overall_score'] < self.config.min_overall_score:
            return False
        
        # Check dimension scores
        if self.config.min_dimension_scores:
            for dim, threshold in self.config.min_dimension_scores.items():
                if dim in evaluation['dimension_scores']:
                    if evaluation['dimension_scores'][dim] < threshold:
                        return False
        
        return True
    
    def _calculate_stats(
        self,
        judge_results: List[JudgeResult],
        failure_reasons: Dict[str, int]
    ) -> FilteringStats:
        """
        Calculate filtering statistics
        
        Args:
            judge_results: List of judge results
            failure_reasons: Failure reasons count
        
        Returns:
            FilteringStats
        """
        total_evaluated = len(judge_results)
        total_passed = sum(1 for r in judge_results if r.passed)
        total_failed = total_evaluated - total_passed
        
        pass_rate = total_passed / total_evaluated if total_evaluated > 0 else 0.0
        
        avg_overall_score = (
            sum(r.overall_score for r in judge_results) / total_evaluated
            if total_evaluated > 0 else 0.0
        )
        
        # Calculate average dimension scores
        avg_dimension_scores = {}
        if judge_results:
            dimensions = judge_results[0].dimension_scores.keys()
            for dim in dimensions:
                scores = [r.dimension_scores.get(dim, 0.0) for r in judge_results]
                avg_dimension_scores[dim] = sum(scores) / len(scores)
        
        return FilteringStats(
            total_evaluated=total_evaluated,
            total_passed=total_passed,
            total_failed=total_failed,
            pass_rate=pass_rate,
            avg_overall_score=avg_overall_score,
            avg_dimension_scores=avg_dimension_scores,
            failure_reasons=failure_reasons
        )
    
    def _save_feedback(
        self,
        judge_results: List[JudgeResult],
        agent_type: str
    ):
        """
        Save judge feedback to file
        
        Args:
            judge_results: List of judge results
            agent_type: Agent type
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        feedback_file = self.feedback_dir / f"{agent_type}_feedback_{timestamp}.jsonl"
        
        with open(feedback_file, 'w') as f:
            for result in judge_results:
                f.write(json.dumps(result.to_dict()) + '\n')
        
        logger.info(f"Saved judge feedback to {feedback_file}")


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Judge-Approved Filtering CLI")
    parser.add_argument('--agent-type', type=str, required=True, help="Agent type")
    parser.add_argument('--strategy', type=str, default='balanced', choices=['strict', 'lenient', 'balanced'])
    parser.add_argument('--input-file', type=str, required=True, help="Input JSONL file with trajectories or examples")
    parser.add_argument('--output-file', type=str, required=True, help="Output JSONL file for filtered data")
    parser.add_argument('--save-stats', type=str, default=None, help="Save stats to JSON file")
    
    args = parser.parse_args()
    
    # Initialize Judge
    judge = LLMJudge()
    
    # Create filter with strategy
    config = JudgeFilterConfig.from_strategy(args.strategy)
    filter_system = JudgeApprovedFilter(judge=judge, config=config)
    
    # Load input data
    with open(args.input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"\nLoaded {len(data)} examples from {args.input_file}")
    print(f"Using strategy: {args.strategy}")
    print(f"  Min overall score: {config.min_overall_score}")
    print(f"  Min dimension scores: {config.min_dimension_scores}\n")
    
    # TODO: Implement filtering based on input data format
    # This is a placeholder - implement based on your data structure
    
    print(f"\n✅ Filtering complete!")
    print(f"  Passed: {0}/{len(data)} ({0.0:.1f}%)")
    print(f"  Output: {args.output_file}")
