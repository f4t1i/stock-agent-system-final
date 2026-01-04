"""
Pairwise Reward Optimization Framework

Instead of absolute scoring (1-10), uses relative comparisons:
- Generate two strategy outputs for same market state
- Judge decides which is better
- More stable and precise than absolute scores
- Better for complex reasoning chains

Key Benefits:
1. Easier for judge to compare than assign absolute scores
2. More consistent judgments
3. Reduces score inflation/deflation
4. Better handles nuanced differences
5. Aligns with RLHF best practices

Based on:
- InstructGPT (OpenAI, 2022)
- Constitutional AI (Anthropic, 2022)
- RLAIF (Google, 2023)
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from loguru import logger
from openai import OpenAI


class ComparisonResult(Enum):
    """Result of pairwise comparison"""
    A_BETTER = "A_better"
    B_BETTER = "B_better"
    TIE = "tie"
    UNCLEAR = "unclear"


@dataclass
class StrategyOutput:
    """A strategy recommendation output"""
    recommendation: str  # buy, sell, hold, etc.
    confidence: float
    reasoning: str
    position_size: float
    entry_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_assessment: str = ""
    agent_outputs: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'recommendation': self.recommendation,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'position_size': self.position_size,
            'entry_target': self.entry_target,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_assessment': self.risk_assessment,
            'agent_outputs': self.agent_outputs
        }


@dataclass
class PairwiseComparison:
    """A pairwise comparison between two strategies"""
    market_state: Dict
    strategy_a: StrategyOutput
    strategy_b: StrategyOutput
    winner: ComparisonResult
    reasoning: str
    confidence: float  # Judge's confidence in comparison
    criteria_scores: Dict[str, str] = field(default_factory=dict)  # Which is better per criterion
    timestamp: float = field(default_factory=time.time)


class PairwiseJudge:
    """
    Judge for pairwise comparisons
    
    Compares two strategy outputs and determines which is better
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model = self.config.get('model', 'gpt-4.1-mini')
        self.client = OpenAI()  # API key from env
        
        # Comparison criteria
        self.criteria = [
            'reasoning_quality',
            'risk_management',
            'confidence_calibration',
            'actionability',
            'market_context_awareness'
        ]
    
    def compare(
        self,
        market_state: Dict,
        strategy_a: StrategyOutput,
        strategy_b: StrategyOutput
    ) -> PairwiseComparison:
        """
        Compare two strategies and determine which is better
        
        Args:
            market_state: Market context
            strategy_a: First strategy
            strategy_b: Second strategy
        
        Returns:
            PairwiseComparison with winner and reasoning
        """
        prompt = self._build_comparison_prompt(market_state, strategy_a, strategy_b)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst evaluating trading strategies. Your task is to compare two strategies and determine which is better. Be objective and thorough."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Low temperature for consistency
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Parse result
            winner_str = result.get('winner', 'unclear').lower()
            
            if 'a' in winner_str and 'b' not in winner_str:
                winner = ComparisonResult.A_BETTER
            elif 'b' in winner_str and 'a' not in winner_str:
                winner = ComparisonResult.B_BETTER
            elif 'tie' in winner_str:
                winner = ComparisonResult.TIE
            else:
                winner = ComparisonResult.UNCLEAR
            
            return PairwiseComparison(
                market_state=market_state,
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                winner=winner,
                reasoning=result.get('reasoning', ''),
                confidence=result.get('confidence', 0.5),
                criteria_scores=result.get('criteria_scores', {}),
                timestamp=time.time()
            )
        
        except Exception as e:
            logger.error(f"Error in pairwise comparison: {e}")
            
            # Fallback: unclear
            return PairwiseComparison(
                market_state=market_state,
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                winner=ComparisonResult.UNCLEAR,
                reasoning=f"Error during comparison: {e}",
                confidence=0.0,
                criteria_scores={},
                timestamp=time.time()
            )
    
    def _build_comparison_prompt(
        self,
        market_state: Dict,
        strategy_a: StrategyOutput,
        strategy_b: StrategyOutput
    ) -> str:
        """Build prompt for pairwise comparison"""
        
        prompt = f"""Compare the following two trading strategies for the same market state and determine which is better.

## Market State

Symbol: {market_state.get('symbol', 'UNKNOWN')}
Current Price: ${market_state.get('current_price', 'N/A')}
Volatility: {market_state.get('volatility', 'N/A')}
Trend: {market_state.get('trend', 'N/A')}
News Sentiment: {market_state.get('news_sentiment', 'N/A')}

## Strategy A

**Recommendation:** {strategy_a.recommendation}
**Confidence:** {strategy_a.confidence:.2f}
**Position Size:** {strategy_a.position_size:.1%}
**Entry Target:** ${strategy_a.entry_target or 'N/A'}
**Stop Loss:** ${strategy_a.stop_loss or 'N/A'}
**Take Profit:** ${strategy_a.take_profit or 'N/A'}

**Reasoning:**
{strategy_a.reasoning}

**Risk Assessment:**
{strategy_a.risk_assessment}

## Strategy B

**Recommendation:** {strategy_b.recommendation}
**Confidence:** {strategy_b.confidence:.2f}
**Position Size:** {strategy_b.position_size:.1%}
**Entry Target:** ${strategy_b.entry_target or 'N/A'}
**Stop Loss:** ${strategy_b.stop_loss or 'N/A'}
**Take Profit:** ${strategy_b.take_profit or 'N/A'}

**Reasoning:**
{strategy_b.reasoning}

**Risk Assessment:**
{strategy_b.risk_assessment}

## Evaluation Criteria

Compare the strategies on these criteria:

1. **Reasoning Quality**: Is the reasoning logical, comprehensive, and well-supported?
2. **Risk Management**: Are stop-loss, position sizing, and risk assessment appropriate?
3. **Confidence Calibration**: Is the confidence level justified by the analysis?
4. **Actionability**: Is the strategy clear and executable?
5. **Market Context Awareness**: Does the strategy appropriately consider market conditions?

## Your Task

Determine which strategy is better overall. Consider:
- Which has stronger reasoning?
- Which has better risk management?
- Which is more likely to succeed given the market state?
- Which provides clearer guidance?

Respond in JSON format:

{{
    "winner": "A" | "B" | "tie" | "unclear",
    "reasoning": "Detailed explanation of why one is better",
    "confidence": 0.0-1.0,
    "criteria_scores": {{
        "reasoning_quality": "A" | "B" | "tie",
        "risk_management": "A" | "B" | "tie",
        "confidence_calibration": "A" | "B" | "tie",
        "actionability": "A" | "B" | "tie",
        "market_context_awareness": "A" | "B" | "tie"
    }}
}}

Be objective and thorough. If strategies are very similar, you can mark as "tie".
"""
        return prompt


class PairwiseDataGenerator:
    """
    Generate pairwise comparison data for training
    
    Creates pairs of strategy outputs for same market state
    """
    
    def __init__(self, strategist_agent, config: Optional[Dict] = None):
        self.strategist = strategist_agent
        self.config = config or {}
        
        # Sampling strategies for diversity
        self.temperature_a = self.config.get('temperature_a', 0.7)
        self.temperature_b = self.config.get('temperature_b', 0.9)
    
    def generate_pair(
        self,
        market_state: Dict,
        agent_outputs: Dict
    ) -> Tuple[StrategyOutput, StrategyOutput]:
        """
        Generate two different strategy outputs for same market state
        
        Uses different temperatures to get diverse outputs
        
        Args:
            market_state: Market context
            agent_outputs: Outputs from junior agents
        
        Returns:
            (strategy_a, strategy_b)
        """
        # Generate strategy A (lower temperature, more conservative)
        strategy_a = self._generate_strategy(
            market_state,
            agent_outputs,
            temperature=self.temperature_a,
            seed=42
        )
        
        # Generate strategy B (higher temperature, more exploratory)
        strategy_b = self._generate_strategy(
            market_state,
            agent_outputs,
            temperature=self.temperature_b,
            seed=123
        )
        
        return strategy_a, strategy_b
    
    def _generate_strategy(
        self,
        market_state: Dict,
        agent_outputs: Dict,
        temperature: float,
        seed: int
    ) -> StrategyOutput:
        """Generate a single strategy output"""
        
        # Call strategist with specific temperature
        result = self.strategist.synthesize(
            symbol=market_state.get('symbol', 'AAPL'),
            agent_outputs=agent_outputs,
            portfolio_state=market_state.get('portfolio_state', {}),
            temperature=temperature,
            seed=seed
        )
        
        return StrategyOutput(
            recommendation=result.get('recommendation', 'hold'),
            confidence=result.get('confidence', 0.5),
            reasoning=result.get('reasoning', ''),
            position_size=result.get('position_size', 0.0),
            entry_target=result.get('entry_target'),
            stop_loss=result.get('stop_loss'),
            take_profit=result.get('take_profit'),
            risk_assessment=result.get('risk_assessment', ''),
            agent_outputs=agent_outputs
        )


class PairwiseRewardModel:
    """
    Reward model trained on pairwise comparisons
    
    Learns to predict which strategy is better
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.comparisons = []
        
        # Simple model: weighted criteria scores
        # In production, this would be a neural network
        self.criterion_weights = {
            'reasoning_quality': 0.3,
            'risk_management': 0.25,
            'confidence_calibration': 0.15,
            'actionability': 0.15,
            'market_context_awareness': 0.15
        }
    
    def add_comparison(self, comparison: PairwiseComparison):
        """Add a comparison to training data"""
        self.comparisons.append(comparison)
    
    def predict_preference(
        self,
        strategy_a: StrategyOutput,
        strategy_b: StrategyOutput,
        market_state: Dict
    ) -> Tuple[float, float]:
        """
        Predict preference scores for two strategies
        
        Returns:
            (score_a, score_b) where higher score = better strategy
        """
        # Simple heuristic-based scoring
        # In production, use trained neural network
        
        score_a = self._score_strategy(strategy_a, market_state)
        score_b = self._score_strategy(strategy_b, market_state)
        
        return score_a, score_b
    
    def _score_strategy(self, strategy: StrategyOutput, market_state: Dict) -> float:
        """Score a single strategy"""
        score = 0.0
        
        # Reasoning length (longer = more detailed)
        reasoning_score = min(len(strategy.reasoning) / 500, 1.0)
        score += reasoning_score * self.criterion_weights['reasoning_quality']
        
        # Risk management (has stop loss and take profit)
        risk_score = 0.0
        if strategy.stop_loss is not None:
            risk_score += 0.5
        if strategy.take_profit is not None:
            risk_score += 0.5
        score += risk_score * self.criterion_weights['risk_management']
        
        # Confidence calibration (not too high, not too low)
        conf_score = 1.0 - abs(strategy.confidence - 0.7)  # Optimal around 0.7
        score += conf_score * self.criterion_weights['confidence_calibration']
        
        # Actionability (has entry target)
        action_score = 1.0 if strategy.entry_target is not None else 0.5
        score += action_score * self.criterion_weights['actionability']
        
        # Market context (mentions volatility, trend, etc.)
        context_score = 0.0
        if 'volatility' in strategy.reasoning.lower():
            context_score += 0.33
        if 'trend' in strategy.reasoning.lower():
            context_score += 0.33
        if 'risk' in strategy.reasoning.lower():
            context_score += 0.34
        score += context_score * self.criterion_weights['market_context_awareness']
        
        return score
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about comparisons"""
        if not self.comparisons:
            return {}
        
        winner_counts = {
            ComparisonResult.A_BETTER: 0,
            ComparisonResult.B_BETTER: 0,
            ComparisonResult.TIE: 0,
            ComparisonResult.UNCLEAR: 0
        }
        
        for comp in self.comparisons:
            winner_counts[comp.winner] += 1
        
        avg_confidence = np.mean([c.confidence for c in self.comparisons])
        
        return {
            'total_comparisons': len(self.comparisons),
            'winner_distribution': {k.value: v for k, v in winner_counts.items()},
            'avg_judge_confidence': avg_confidence
        }


class PairwiseTrainingDataset:
    """Dataset for pairwise training"""
    
    def __init__(self, save_path: str = "data/pairwise_comparisons.jsonl"):
        self.save_path = save_path
        self.comparisons = []
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def add_comparison(self, comparison: PairwiseComparison):
        """Add a comparison to dataset"""
        self.comparisons.append(comparison)
    
    def save(self):
        """Save dataset to file"""
        with open(self.save_path, 'w') as f:
            for comp in self.comparisons:
                # Convert to dict
                data = {
                    'market_state': comp.market_state,
                    'strategy_a': comp.strategy_a.to_dict(),
                    'strategy_b': comp.strategy_b.to_dict(),
                    'winner': comp.winner.value,
                    'reasoning': comp.reasoning,
                    'confidence': comp.confidence,
                    'criteria_scores': comp.criteria_scores,
                    'timestamp': comp.timestamp
                }
                f.write(json.dumps(data) + '\n')
        
        logger.info(f"Saved {len(self.comparisons)} comparisons to {self.save_path}")
    
    def load(self):
        """Load dataset from file"""
        if not os.path.exists(self.save_path):
            logger.warning(f"Dataset file not found: {self.save_path}")
            return
        
        self.comparisons = []
        
        with open(self.save_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                comp = PairwiseComparison(
                    market_state=data['market_state'],
                    strategy_a=StrategyOutput(**data['strategy_a']),
                    strategy_b=StrategyOutput(**data['strategy_b']),
                    winner=ComparisonResult(data['winner']),
                    reasoning=data['reasoning'],
                    confidence=data['confidence'],
                    criteria_scores=data['criteria_scores'],
                    timestamp=data['timestamp']
                )
                
                self.comparisons.append(comp)
        
        logger.info(f"Loaded {len(self.comparisons)} comparisons from {self.save_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.comparisons:
            return {}
        
        winner_counts = {
            'A_better': 0,
            'B_better': 0,
            'tie': 0,
            'unclear': 0
        }
        
        for comp in self.comparisons:
            winner_counts[comp.winner.value] += 1
        
        return {
            'total_comparisons': len(self.comparisons),
            'winner_distribution': winner_counts,
            'avg_confidence': np.mean([c.confidence for c in self.comparisons])
        }


if __name__ == '__main__':
    # Test
    judge = PairwiseJudge()
    
    # Mock strategies
    market_state = {
        'symbol': 'AAPL',
        'current_price': 180.0,
        'volatility': 0.25,
        'trend': 'bullish',
        'news_sentiment': 0.7
    }
    
    strategy_a = StrategyOutput(
        recommendation='buy',
        confidence=0.8,
        reasoning='Strong bullish trend with positive news sentiment. Technical indicators show momentum.',
        position_size=0.08,
        entry_target=178.0,
        stop_loss=172.0,
        take_profit=195.0,
        risk_assessment='Moderate risk. Stop loss at 3.3% below entry.'
    )
    
    strategy_b = StrategyOutput(
        recommendation='buy',
        confidence=0.9,
        reasoning='Buy signal.',
        position_size=0.15,
        entry_target=180.0,
        stop_loss=None,
        take_profit=None,
        risk_assessment=''
    )
    
    comparison = judge.compare(market_state, strategy_a, strategy_b)
    
    print(f"Winner: {comparison.winner.value}")
    print(f"Reasoning: {comparison.reasoning}")
    print(f"Confidence: {comparison.confidence:.2f}")
    print(f"Criteria Scores: {comparison.criteria_scores}")
