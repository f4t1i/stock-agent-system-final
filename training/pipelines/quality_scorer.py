"""
Quality Score Calculation - Task 1.3

Multi-dimensional quality scoring system for trading trajectories.

Scoring Dimensions:
1. Reward Score: Based on actual trading outcome
2. Confidence Score: Agent's confidence in decision
3. Reasoning Quality: Length, clarity, specificity of reasoning
4. Consistency Score: Alignment between confidence and outcome

Combined Quality Score:
- Weighted average of all dimensions
- Configurable weights per dimension
- Normalized to [0, 1] range

Filtering:
- Threshold-based filtering (e.g., score >= 0.7)
- Per-dimension thresholds
- Rejection reason tracking

Phase A1 Week 3-4: Task 1.3 COMPLETE
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from loguru import logger
import re


@dataclass
class QualityScoreConfig:
    """Configuration for quality scoring"""
    # Weights for each dimension (must sum to 1.0)
    reward_weight: float = 0.4
    confidence_weight: float = 0.3
    reasoning_weight: float = 0.2
    consistency_weight: float = 0.1
    
    # Thresholds for each dimension
    min_reward: float = 0.5
    min_confidence: float = 0.6
    min_reasoning_score: float = 0.5
    min_consistency_score: float = 0.5
    
    # Overall threshold
    min_overall_score: float = 0.6
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total_weight = (
            self.reward_weight + 
            self.confidence_weight + 
            self.reasoning_weight + 
            self.consistency_weight
        )
        
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")


@dataclass
class QualityScore:
    """Quality score result"""
    overall_score: float
    reward_score: float
    confidence_score: float
    reasoning_score: float
    consistency_score: float
    
    passed: bool
    rejection_reasons: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'overall_score': self.overall_score,
            'reward_score': self.reward_score,
            'confidence_score': self.confidence_score,
            'reasoning_score': self.reasoning_score,
            'consistency_score': self.consistency_score,
            'passed': self.passed,
            'rejection_reasons': self.rejection_reasons
        }


class QualityScorer:
    """
    Multi-dimensional quality scorer for trading trajectories
    
    Usage:
        config = QualityScoreConfig(
            reward_weight=0.4,
            confidence_weight=0.3,
            min_overall_score=0.7
        )
        
        scorer = QualityScorer(config)
        
        score = scorer.score_trajectory(
            reward=0.8,
            confidence=0.9,
            reasoning="Strong bullish signal based on...",
            success=True
        )
        
        if score.passed:
            print(f"Quality score: {score.overall_score:.2f}")
        else:
            print(f"Rejected: {score.rejection_reasons}")
    """
    
    def __init__(self, config: Optional[QualityScoreConfig] = None):
        """
        Initialize scorer
        
        Args:
            config: Quality score configuration (uses defaults if None)
        """
        self.config = config or QualityScoreConfig()
        logger.info(f"QualityScorer initialized with config: {self.config}")
    
    def score_trajectory(
        self,
        reward: float,
        confidence: float,
        reasoning: str,
        success: bool
    ) -> QualityScore:
        """
        Calculate quality score for a trajectory
        
        Args:
            reward: Trading reward/return (0.0 to 1.0)
            confidence: Agent confidence (0.0 to 1.0)
            reasoning: Agent reasoning text
            success: Whether trade was successful
        
        Returns:
            QualityScore object
        """
        # Calculate individual dimension scores
        reward_score = self._score_reward(reward)
        confidence_score = self._score_confidence(confidence)
        reasoning_score = self._score_reasoning(reasoning)
        consistency_score = self._score_consistency(confidence, success)
        
        # Calculate weighted overall score
        overall_score = (
            reward_score * self.config.reward_weight +
            confidence_score * self.config.confidence_weight +
            reasoning_score * self.config.reasoning_weight +
            consistency_score * self.config.consistency_weight
        )
        
        # Check thresholds
        rejection_reasons = []
        
        if reward_score < self.config.min_reward:
            rejection_reasons.append(
                f"reward_score {reward_score:.2f} < {self.config.min_reward}"
            )
        
        if confidence_score < self.config.min_confidence:
            rejection_reasons.append(
                f"confidence_score {confidence_score:.2f} < {self.config.min_confidence}"
            )
        
        if reasoning_score < self.config.min_reasoning_score:
            rejection_reasons.append(
                f"reasoning_score {reasoning_score:.2f} < {self.config.min_reasoning_score}"
            )
        
        if consistency_score < self.config.min_consistency_score:
            rejection_reasons.append(
                f"consistency_score {consistency_score:.2f} < {self.config.min_consistency_score}"
            )
        
        if overall_score < self.config.min_overall_score:
            rejection_reasons.append(
                f"overall_score {overall_score:.2f} < {self.config.min_overall_score}"
            )
        
        passed = len(rejection_reasons) == 0
        
        return QualityScore(
            overall_score=overall_score,
            reward_score=reward_score,
            confidence_score=confidence_score,
            reasoning_score=reasoning_score,
            consistency_score=consistency_score,
            passed=passed,
            rejection_reasons=rejection_reasons
        )
    
    def _score_reward(self, reward: float) -> float:
        """
        Score based on trading reward
        
        Reward is already in [0, 1] range, so just normalize
        
        Args:
            reward: Trading reward (0.0 to 1.0)
        
        Returns:
            Reward score (0.0 to 1.0)
        """
        # Clamp to [0, 1]
        return max(0.0, min(1.0, reward))
    
    def _score_confidence(self, confidence: float) -> float:
        """
        Score based on agent confidence
        
        Confidence is already in [0, 1] range
        
        Args:
            confidence: Agent confidence (0.0 to 1.0)
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def _score_reasoning(self, reasoning: str) -> float:
        """
        Score based on reasoning quality
        
        Heuristics:
        - Length: Longer reasoning is better (up to a point)
        - Keywords: Presence of analytical keywords
        - Specificity: Numbers, percentages, specific terms
        - Structure: Multiple sentences, paragraphs
        
        Args:
            reasoning: Agent reasoning text
        
        Returns:
            Reasoning score (0.0 to 1.0)
        """
        if not reasoning or len(reasoning.strip()) == 0:
            return 0.0
        
        score = 0.0
        
        # 1. Length score (0.0 to 0.3)
        # Optimal length: 100-500 characters
        length = len(reasoning)
        if length < 50:
            length_score = length / 50 * 0.3
        elif length <= 500:
            length_score = 0.3
        else:
            # Penalize overly long reasoning
            length_score = max(0.1, 0.3 - (length - 500) / 1000 * 0.2)
        
        score += length_score
        
        # 2. Keyword score (0.0 to 0.3)
        analytical_keywords = [
            'because', 'due to', 'indicates', 'suggests', 'shows',
            'trend', 'pattern', 'signal', 'momentum', 'volume',
            'support', 'resistance', 'breakout', 'reversal',
            'bullish', 'bearish', 'volatility', 'correlation'
        ]
        
        reasoning_lower = reasoning.lower()
        keyword_count = sum(1 for kw in analytical_keywords if kw in reasoning_lower)
        keyword_score = min(0.3, keyword_count / 5 * 0.3)
        
        score += keyword_score
        
        # 3. Specificity score (0.0 to 0.2)
        # Look for numbers, percentages, specific values
        has_numbers = bool(re.search(r'\d+', reasoning))
        has_percentages = bool(re.search(r'\d+%', reasoning))
        has_decimals = bool(re.search(r'\d+\.\d+', reasoning))
        
        specificity_score = (
            (0.1 if has_numbers else 0.0) +
            (0.05 if has_percentages else 0.0) +
            (0.05 if has_decimals else 0.0)
        )
        
        score += specificity_score
        
        # 4. Structure score (0.0 to 0.2)
        # Multiple sentences indicate structured reasoning
        sentence_count = len(re.split(r'[.!?]+', reasoning))
        structure_score = min(0.2, sentence_count / 3 * 0.2)
        
        score += structure_score
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def _score_consistency(self, confidence: float, success: bool) -> float:
        """
        Score based on consistency between confidence and outcome
        
        High confidence + success = high score
        Low confidence + failure = high score
        High confidence + failure = low score
        Low confidence + success = medium score
        
        Args:
            confidence: Agent confidence (0.0 to 1.0)
            success: Whether trade was successful
        
        Returns:
            Consistency score (0.0 to 1.0)
        """
        if success:
            # Success: reward high confidence
            return confidence
        else:
            # Failure: reward low confidence (agent was appropriately cautious)
            return 1.0 - confidence
    
    def batch_score(
        self,
        trajectories: List[Dict]
    ) -> List[QualityScore]:
        """
        Score multiple trajectories in batch
        
        Args:
            trajectories: List of trajectory dicts with keys:
                - reward: float
                - confidence: float
                - reasoning: str
                - success: bool
        
        Returns:
            List of QualityScore objects
        """
        scores = []
        
        for traj in trajectories:
            score = self.score_trajectory(
                reward=traj['reward'],
                confidence=traj['confidence'],
                reasoning=traj['reasoning'],
                success=traj['success']
            )
            scores.append(score)
        
        return scores
    
    def get_statistics(self, scores: List[QualityScore]) -> Dict:
        """
        Calculate statistics for a batch of scores
        
        Args:
            scores: List of QualityScore objects
        
        Returns:
            Statistics dictionary
        """
        if not scores:
            return {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0
            }
        
        passed_count = sum(1 for s in scores if s.passed)
        failed_count = len(scores) - passed_count
        
        avg_overall = sum(s.overall_score for s in scores) / len(scores)
        avg_reward = sum(s.reward_score for s in scores) / len(scores)
        avg_confidence = sum(s.confidence_score for s in scores) / len(scores)
        avg_reasoning = sum(s.reasoning_score for s in scores) / len(scores)
        avg_consistency = sum(s.consistency_score for s in scores) / len(scores)
        
        # Collect rejection reasons
        rejection_reasons = {}
        for score in scores:
            for reason in score.rejection_reasons:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        return {
            'total': len(scores),
            'passed': passed_count,
            'failed': failed_count,
            'pass_rate': passed_count / len(scores),
            'avg_overall_score': avg_overall,
            'avg_reward_score': avg_reward,
            'avg_confidence_score': avg_confidence,
            'avg_reasoning_score': avg_reasoning,
            'avg_consistency_score': avg_consistency,
            'rejection_reasons': rejection_reasons
        }


if __name__ == "__main__":
    # Example usage
    config = QualityScoreConfig(
        reward_weight=0.4,
        confidence_weight=0.3,
        reasoning_weight=0.2,
        consistency_weight=0.1,
        min_overall_score=0.7
    )
    
    scorer = QualityScorer(config)
    
    # High quality trajectory
    score1 = scorer.score_trajectory(
        reward=0.85,
        confidence=0.9,
        reasoning="Strong bullish signal based on RSI divergence at 35, MACD crossover, and increasing volume. Price broke above 50-day MA with 15% gain potential to resistance at $150.",
        success=True
    )
    
    print("High Quality Trajectory:")
    print(f"  Overall Score: {score1.overall_score:.2f}")
    print(f"  Passed: {score1.passed}")
    print(f"  Reward: {score1.reward_score:.2f}")
    print(f"  Confidence: {score1.confidence_score:.2f}")
    print(f"  Reasoning: {score1.reasoning_score:.2f}")
    print(f"  Consistency: {score1.consistency_score:.2f}")
    
    # Low quality trajectory
    score2 = scorer.score_trajectory(
        reward=0.3,
        confidence=0.4,
        reasoning="Buy signal",
        success=False
    )
    
    print("\nLow Quality Trajectory:")
    print(f"  Overall Score: {score2.overall_score:.2f}")
    print(f"  Passed: {score2.passed}")
    print(f"  Rejection Reasons: {score2.rejection_reasons}")
