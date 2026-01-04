"""
Multi-Judge Consensus System

Uses multiple judges to evaluate agent outputs and combines their scores
for more robust and reliable evaluation.

Key Features:
1. Multiple judge models (Claude, GPT-4, Gemini)
2. Consensus mechanisms (majority vote, weighted average, etc.)
3. Disagreement detection and resolution
4. Confidence calibration
5. Judge performance tracking

Based on:
- Constitutional AI (Anthropic)
- Multi-rater reliability research
- Ensemble methods
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import anthropic
import openai
from loguru import logger
import statistics
import yaml


@dataclass
class JudgeScore:
    """Score from a single judge"""
    judge_id: str
    score: float  # [0, 100]
    confidence: float  # [0, 1]
    reasoning: str
    criteria_scores: Dict[str, float]


@dataclass
class ConsensusResult:
    """Result of multi-judge consensus"""
    final_score: float  # [0, 100]
    confidence: float  # [0, 1]
    agreement_level: float  # [0, 1] - how much judges agree
    individual_scores: List[JudgeScore]
    consensus_method: str
    disagreement_detected: bool
    resolution_applied: bool


class MultiJudgeConsensus:
    """
    Multi-Judge Consensus System
    
    Evaluates agent outputs using multiple judges and combines scores.
    """
    
    def __init__(
        self,
        judge_configs: List[Dict],
        consensus_method: str = 'weighted_average',
        disagreement_threshold: float = 20.0  # Score difference threshold
    ):
        """
        Initialize multi-judge system
        
        Args:
            judge_configs: List of judge configurations
                [{"id": "claude", "model": "claude-3-5-sonnet-20241022", "weight": 1.0}, ...]
            consensus_method: Method to combine scores
                - 'weighted_average': Weighted average of scores
                - 'majority_vote': Majority vote (pass/fail)
                - 'median': Median score
                - 'min': Minimum score (conservative)
                - 'max': Maximum score (optimistic)
            disagreement_threshold: Threshold for detecting disagreement
        """
        self.judge_configs = judge_configs
        self.consensus_method = consensus_method
        self.disagreement_threshold = disagreement_threshold
        
        # Initialize clients
        self.clients = {}
        for config in judge_configs:
            judge_id = config['id']
            if 'claude' in judge_id:
                self.clients[judge_id] = anthropic.Anthropic()
            elif 'gpt' in judge_id or 'openai' in judge_id:
                self.clients[judge_id] = openai.OpenAI()
            # Add more providers as needed
    
    def evaluate(
        self,
        agent_type: str,
        agent_output: Dict,
        rubric_path: str,
        context: Optional[Dict] = None
    ) -> ConsensusResult:
        """
        Evaluate agent output using multiple judges
        
        Args:
            agent_type: Type of agent
            agent_output: Agent's output to evaluate
            rubric_path: Path to rubric file
            context: Additional context (market data, etc.)
        
        Returns:
            ConsensusResult with combined evaluation
        """
        # Load rubric
        with open(rubric_path, 'r') as f:
            rubric = yaml.safe_load(f)
        
        # Get scores from all judges
        individual_scores = []
        for config in self.judge_configs:
            try:
                score = self._evaluate_single_judge(
                    judge_config=config,
                    agent_type=agent_type,
                    agent_output=agent_output,
                    rubric=rubric,
                    context=context
                )
                individual_scores.append(score)
            except Exception as e:
                logger.error(f"Judge {config['id']} failed: {e}")
                # Continue with other judges
        
        if not individual_scores:
            raise RuntimeError("All judges failed")
        
        # Detect disagreement
        disagreement_detected = self._detect_disagreement(individual_scores)
        
        # Apply consensus
        final_score, confidence, resolution_applied = self._apply_consensus(
            individual_scores,
            disagreement_detected
        )
        
        # Calculate agreement level
        agreement_level = self._calculate_agreement(individual_scores)
        
        return ConsensusResult(
            final_score=final_score,
            confidence=confidence,
            agreement_level=agreement_level,
            individual_scores=individual_scores,
            consensus_method=self.consensus_method,
            disagreement_detected=disagreement_detected,
            resolution_applied=resolution_applied
        )
    
    def _evaluate_single_judge(
        self,
        judge_config: Dict,
        agent_type: str,
        agent_output: Dict,
        rubric: Dict,
        context: Optional[Dict]
    ) -> JudgeScore:
        """
        Evaluate with a single judge
        
        Args:
            judge_config: Judge configuration
            agent_type: Agent type
            agent_output: Agent output
            rubric: Evaluation rubric
            context: Additional context
        
        Returns:
            JudgeScore
        """
        judge_id = judge_config['id']
        model = judge_config['model']
        
        # Build prompt
        prompt = self._build_judge_prompt(
            agent_type=agent_type,
            agent_output=agent_output,
            rubric=rubric,
            context=context
        )
        
        # Call judge
        client = self.clients[judge_id]
        
        if 'claude' in judge_id:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text
        
        elif 'gpt' in judge_id or 'openai' in judge_id:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            response_text = response.choices[0].message.content
        
        else:
            raise ValueError(f"Unknown judge type: {judge_id}")
        
        # Parse response
        score, confidence, reasoning, criteria_scores = self._parse_judge_response(
            response_text,
            rubric
        )
        
        return JudgeScore(
            judge_id=judge_id,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            criteria_scores=criteria_scores
        )
    
    def _build_judge_prompt(
        self,
        agent_type: str,
        agent_output: Dict,
        rubric: Dict,
        context: Optional[Dict]
    ) -> str:
        """Build prompt for judge"""
        prompt = f"""You are an expert financial analyst evaluating a {agent_type} agent's output.

Agent Output:
{yaml.dump(agent_output, default_flow_style=False)}

"""
        
        if context:
            prompt += f"""Context:
{yaml.dump(context, default_flow_style=False)}

"""
        
        prompt += f"""Evaluation Rubric:
{yaml.dump(rubric, default_flow_style=False)}

Please evaluate the agent's output according to the rubric. For each criterion, provide:
1. A score (0-100)
2. Brief reasoning

Then provide:
- Overall score (0-100)
- Confidence in your evaluation (0-1)
- Overall reasoning

Format your response as:
CRITERION_1: <score> - <reasoning>
CRITERION_2: <score> - <reasoning>
...
OVERALL_SCORE: <score>
CONFIDENCE: <confidence>
REASONING: <reasoning>
"""
        
        return prompt
    
    def _parse_judge_response(
        self,
        response_text: str,
        rubric: Dict
    ) -> Tuple[float, float, str, Dict[str, float]]:
        """
        Parse judge response
        
        Returns:
            (overall_score, confidence, reasoning, criteria_scores)
        """
        lines = response_text.strip().split('\n')
        
        criteria_scores = {}
        overall_score = 50.0  # Default
        confidence = 0.5  # Default
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'OVERALL_SCORE':
                    try:
                        overall_score = float(value)
                    except:
                        pass
                
                elif key == 'CONFIDENCE':
                    try:
                        confidence = float(value)
                    except:
                        pass
                
                elif key == 'REASONING':
                    reasoning = value
                
                else:
                    # Criterion score
                    try:
                        # Extract score (first number)
                        score_str = value.split('-')[0].strip()
                        score = float(score_str)
                        criteria_scores[key] = score
                    except:
                        pass
        
        return overall_score, confidence, reasoning, criteria_scores
    
    def _detect_disagreement(self, scores: List[JudgeScore]) -> bool:
        """
        Detect if judges significantly disagree
        
        Args:
            scores: List of judge scores
        
        Returns:
            bool: Whether disagreement detected
        """
        if len(scores) < 2:
            return False
        
        score_values = [s.score for s in scores]
        
        # Check if range exceeds threshold
        score_range = max(score_values) - min(score_values)
        
        return score_range > self.disagreement_threshold
    
    def _apply_consensus(
        self,
        scores: List[JudgeScore],
        disagreement_detected: bool
    ) -> Tuple[float, float, bool]:
        """
        Apply consensus method to combine scores
        
        Args:
            scores: List of judge scores
            disagreement_detected: Whether disagreement was detected
        
        Returns:
            (final_score, confidence, resolution_applied)
        """
        resolution_applied = False
        
        # If disagreement, apply resolution strategy
        if disagreement_detected:
            # Use median (more robust to outliers)
            score_values = [s.score for s in scores]
            final_score = statistics.median(score_values)
            
            # Lower confidence due to disagreement
            confidences = [s.confidence for s in scores]
            confidence = statistics.mean(confidences) * 0.8  # Penalty
            
            resolution_applied = True
            
            logger.warning(
                f"Judge disagreement detected. Using median: {final_score:.1f}"
            )
        
        else:
            # Apply configured consensus method
            if self.consensus_method == 'weighted_average':
                # Weighted average by judge weight and confidence
                total_weight = 0
                weighted_sum = 0
                
                for score, config in zip(scores, self.judge_configs):
                    weight = config.get('weight', 1.0) * score.confidence
                    weighted_sum += score.score * weight
                    total_weight += weight
                
                final_score = weighted_sum / total_weight if total_weight > 0 else 50.0
                confidence = statistics.mean([s.confidence for s in scores])
            
            elif self.consensus_method == 'majority_vote':
                # Majority vote (pass/fail at 70 threshold)
                pass_count = sum(1 for s in scores if s.score >= 70)
                final_score = 80.0 if pass_count > len(scores) / 2 else 40.0
                confidence = abs(pass_count - len(scores) / 2) / (len(scores) / 2)
            
            elif self.consensus_method == 'median':
                score_values = [s.score for s in scores]
                final_score = statistics.median(score_values)
                confidence = statistics.mean([s.confidence for s in scores])
            
            elif self.consensus_method == 'min':
                # Conservative (minimum score)
                final_score = min(s.score for s in scores)
                confidence = min(s.confidence for s in scores)
            
            elif self.consensus_method == 'max':
                # Optimistic (maximum score)
                final_score = max(s.score for s in scores)
                confidence = max(s.confidence for s in scores)
            
            else:
                raise ValueError(f"Unknown consensus method: {self.consensus_method}")
        
        return final_score, confidence, resolution_applied
    
    def _calculate_agreement(self, scores: List[JudgeScore]) -> float:
        """
        Calculate agreement level between judges
        
        Args:
            scores: List of judge scores
        
        Returns:
            Agreement level [0, 1]
        """
        if len(scores) < 2:
            return 1.0
        
        score_values = [s.score for s in scores]
        
        # Calculate coefficient of variation (normalized std dev)
        mean_score = statistics.mean(score_values)
        std_score = statistics.stdev(score_values)
        
        if mean_score == 0:
            return 1.0
        
        cv = std_score / mean_score
        
        # Convert to agreement (1 = perfect agreement, 0 = high disagreement)
        agreement = max(0, 1 - cv)
        
        return agreement


if __name__ == '__main__':
    # Test
    judge_configs = [
        {"id": "claude", "model": "claude-3-5-sonnet-20241022", "weight": 1.0},
        {"id": "gpt4", "model": "gpt-4", "weight": 0.8}
    ]
    
    consensus = MultiJudgeConsensus(
        judge_configs=judge_configs,
        consensus_method='weighted_average',
        disagreement_threshold=20.0
    )
    
    # Example evaluation
    agent_output = {
        'sentiment_score': 1.5,
        'confidence': 0.85,
        'key_events': ['Earnings beat', 'Product launch'],
        'reasoning': 'Strong positive sentiment due to earnings beat and new product.'
    }
    
    result = consensus.evaluate(
        agent_type='news',
        agent_output=agent_output,
        rubric_path='config/judge/rubrics/news_rubric.yaml'
    )
    
    print(f"Final Score: {result.final_score:.1f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Agreement: {result.agreement_level:.2f}")
    print(f"Disagreement Detected: {result.disagreement_detected}")
