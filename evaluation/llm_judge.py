"""
LLM-as-Judge - Qualitative assessment of trading signals using LLMs

Uses Claude/GPT to score signals on:
- Reasoning Quality (30%)
- Risk Awareness (25%)
- Evidence Quality (20%)
- Coherence (15%)
- Actionability (10%)

Based on judge_rubrics.yaml criteria.
"""

import yaml
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger


@dataclass
class JudgeCriterionScore:
    """Score for a single criterion"""
    criterion: str
    score: float  # 0-10
    weight: float
    weighted_score: float
    reasoning: str


@dataclass
class LLMJudgeResult:
    """Complete LLM judge assessment result"""
    signal_id: str
    timestamp: str
    criterion_scores: List[JudgeCriterionScore]
    composite_score: float
    judgment: str  # "excellent", "good", "acceptable", "poor", "failing"
    llm_model: str
    execution_time_ms: float


class LLMJudge:
    """
    LLM-based judge for qualitative signal assessment.

    Uses structured prompting to score signals on multiple criteria
    defined in judge_rubrics.yaml.
    """

    def __init__(
        self,
        rubrics_path: Optional[str] = None,
        llm_client: Optional[object] = None,
        model: Optional[str] = None
    ):
        """
        Initialize LLM Judge.

        Args:
            rubrics_path: Path to judge_rubrics.yaml
            llm_client: LLM client (Anthropic or OpenAI)
            model: Model name (overrides rubric default)
        """
        # Load rubrics
        if rubrics_path is None:
            rubrics_path = Path(__file__).parent / "judge_rubrics.yaml"

        with open(rubrics_path, 'r') as f:
            self.rubrics = yaml.safe_load(f)

        # LLM setup
        self.llm_client = llm_client
        self.model = model or self.rubrics['llm_judge_rubric']['model']

        # Extract criteria
        self.criteria = self.rubrics['llm_judge_rubric']['criteria']

        logger.info(f"LLM Judge initialized with model: {self.model}")

    def judge_signal(
        self,
        signal: Dict,
        explain: bool = True
    ) -> LLMJudgeResult:
        """
        Judge a trading signal using LLM.

        Args:
            signal: Trading signal dictionary
            explain: Include detailed reasoning in scores

        Returns:
            LLMJudgeResult with scores and judgment
        """
        start_time = datetime.now()

        # Score each criterion
        criterion_scores = []

        for criterion_name, criterion_config in self.criteria.items():
            score, reasoning = self._score_criterion(
                signal,
                criterion_name,
                criterion_config,
                explain
            )

            weight = criterion_config['weight']
            weighted_score = score * weight

            criterion_scores.append(JudgeCriterionScore(
                criterion=criterion_name,
                score=score,
                weight=weight,
                weighted_score=weighted_score,
                reasoning=reasoning if explain else ""
            ))

        # Calculate composite score
        composite_score = sum(cs.weighted_score for cs in criterion_scores)

        # Determine judgment
        judgment = self._get_judgment_level(composite_score)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        result = LLMJudgeResult(
            signal_id=signal['metadata'].get('symbol', 'unknown'),
            timestamp=datetime.now().isoformat(),
            criterion_scores=criterion_scores,
            composite_score=composite_score,
            judgment=judgment,
            llm_model=self.model,
            execution_time_ms=execution_time
        )

        logger.info(
            f"Signal judged: {result.signal_id}, "
            f"Score: {composite_score:.2f}/10.0, "
            f"Judgment: {judgment}"
        )

        return result

    def _score_criterion(
        self,
        signal: Dict,
        criterion_name: str,
        criterion_config: Dict,
        explain: bool
    ) -> Tuple[float, str]:
        """
        Score a single criterion using LLM.

        Args:
            signal: Trading signal
            criterion_name: Name of criterion
            criterion_config: Criterion configuration from rubrics
            explain: Include reasoning

        Returns:
            Tuple of (score, reasoning)
        """
        # Build prompt for this criterion
        prompt = self._build_criterion_prompt(signal, criterion_name, criterion_config)

        # Call LLM (mock implementation - replace with actual LLM call)
        if self.llm_client:
            score, reasoning = self._call_llm(prompt, explain)
        else:
            # Fallback: heuristic scoring (for demo/testing)
            score, reasoning = self._heuristic_score(signal, criterion_name, criterion_config)

        # Clamp score to [0, 10]
        score = max(0.0, min(10.0, score))

        return score, reasoning

    def _build_criterion_prompt(
        self,
        signal: Dict,
        criterion_name: str,
        criterion_config: Dict
    ) -> str:
        """Build LLM prompt for criterion scoring"""

        description = criterion_config['description']
        scoring_guide = criterion_config['scoring_guide']
        evaluation_points = criterion_config['evaluation_points']

        # Extract relevant signal content
        rationale = signal.get('rationale', '')
        analysis = json.dumps(signal.get('analysis', {}), indent=2)
        risk = json.dumps(signal.get('risk', {}), indent=2)
        evidence = json.dumps(signal.get('evidence', {}), indent=2)

        prompt = f"""You are an expert trading signal evaluator. Score the following trading signal on: **{criterion_name}**.

**Criterion Description:** {description}

**Signal to Evaluate:**

Symbol: {signal['metadata']['symbol']}
Signal: {signal['signal']}
Rationale: {rationale}

Analysis:
{analysis}

Risk Management:
{risk}

Evidence:
{evidence}

**Scoring Guide (0-10):**
{self._format_scoring_guide(scoring_guide)}

**Evaluation Points:**
{self._format_evaluation_points(evaluation_points)}

**Task:**
1. Evaluate the signal based on the criteria above
2. Provide a score from 0-10
3. Explain your reasoning in 2-3 sentences

**Response Format:**
Score: [0-10]
Reasoning: [Your explanation]
"""

        return prompt

    def _format_scoring_guide(self, scoring_guide: Dict) -> str:
        """Format scoring guide for prompt"""
        lines = []
        for range_str, description in scoring_guide.items():
            lines.append(f"- {range_str}: {description}")
        return "\n".join(lines)

    def _format_evaluation_points(self, points: List[str]) -> str:
        """Format evaluation points for prompt"""
        return "\n".join(f"- {point}" for point in points)

    def _call_llm(self, prompt: str, explain: bool) -> Tuple[float, str]:
        """
        Call LLM to score criterion.

        Args:
            prompt: Criterion evaluation prompt
            explain: Include reasoning

        Returns:
            Tuple of (score, reasoning)
        """
        # Placeholder - implement actual LLM call
        # Example with Anthropic Claude:
        """
        from anthropic import Anthropic

        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text

        # Parse response
        score = self._extract_score_from_response(response_text)
        reasoning = self._extract_reasoning_from_response(response_text)

        return score, reasoning
        """

        # For now, return placeholder
        logger.warning("LLM client not configured - using heuristic scoring")
        return 7.5, "LLM call not implemented - using heuristic score"

    def _heuristic_score(
        self,
        signal: Dict,
        criterion_name: str,
        criterion_config: Dict
    ) -> Tuple[float, str]:
        """
        Heuristic scoring (fallback when LLM not available).

        Uses simple rules to approximate LLM judgment.
        """
        rationale_length = len(signal.get('rationale', ''))
        evidence_count = len(signal.get('evidence', {}).get('sources', []))
        confidence = signal.get('evidence', {}).get('confidence', 0.5)

        if criterion_name == 'reasoning_quality':
            # Score based on rationale length and structure
            if rationale_length >= 200:
                score = 7.5 + (confidence * 2.5)
            elif rationale_length >= 100:
                score = 6.0 + (confidence * 2.0)
            else:
                score = 4.0 + (confidence * 1.5)

            reasoning = f"Rationale length: {rationale_length} chars, confidence: {confidence:.2f}"

        elif criterion_name == 'risk_awareness':
            # Score based on risk parameters presence
            risk = signal.get('risk', {})
            has_stop_loss = 'stop_loss' in risk and risk['stop_loss'] > 0
            has_take_profit = 'take_profit' in risk and risk['take_profit'] > 0
            has_max_dd = 'max_drawdown' in risk

            score = 5.0
            if has_stop_loss: score += 2.0
            if has_take_profit: score += 2.0
            if has_max_dd: score += 1.0

            reasoning = f"Risk parameters: stop_loss={has_stop_loss}, take_profit={has_take_profit}, max_dd={has_max_dd}"

        elif criterion_name == 'evidence_quality':
            # Score based on evidence sources
            score = min(10.0, 5.0 + (evidence_count * 1.5))
            reasoning = f"Evidence sources: {evidence_count}, confidence: {confidence:.2f}"

        elif criterion_name == 'coherence':
            # Score based on signal consistency
            final_signal = signal.get('signal', 'hold')
            tech_signal = signal.get('analysis', {}).get('technical', {}).get('signal', 'neutral')

            # Map technical to final
            tech_to_final = {'bullish': 'buy', 'bearish': 'sell', 'neutral': 'hold'}
            expected = tech_to_final.get(tech_signal, 'hold')

            if final_signal == expected:
                score = 8.0 + (confidence * 2.0)
                reasoning = f"Signals aligned: technical={tech_signal} → final={final_signal}"
            else:
                score = 6.0
                reasoning = f"Signals diverge: technical={tech_signal}, final={final_signal}"

        elif criterion_name == 'actionability':
            # Score based on clarity of execution details
            risk = signal.get('risk', {})
            sizing = signal.get('sizing', {})

            has_entry = 'current_price' in signal.get('metadata', {})
            has_stop = 'stop_loss' in risk
            has_target = 'take_profit' in risk
            has_size = 'position_size' in sizing

            score = 4.0
            if has_entry: score += 1.5
            if has_stop: score += 1.5
            if has_target: score += 1.5
            if has_size: score += 1.5

            reasoning = f"Execution details: entry={has_entry}, stop={has_stop}, target={has_target}, size={has_size}"

        else:
            score = 7.0
            reasoning = "Default heuristic score"

        return score, reasoning

    def _get_judgment_level(self, composite_score: float) -> str:
        """
        Determine judgment level from composite score.

        Based on thresholds in judge_rubrics.yaml.
        """
        thresholds = self.rubrics['llm_judge_rubric']['composite_score_calculation']['thresholds']

        if composite_score >= thresholds['excellent']:
            return "excellent"
        elif composite_score >= thresholds['good']:
            return "good"
        elif composite_score >= thresholds['acceptable']:
            return "acceptable"
        elif composite_score >= thresholds['poor']:
            return "poor"
        else:
            return "failing"

    def to_dict(self, result: LLMJudgeResult) -> Dict:
        """Convert result to dictionary"""
        return {
            'signal_id': result.signal_id,
            'timestamp': result.timestamp,
            'criterion_scores': [
                {
                    'criterion': cs.criterion,
                    'score': cs.score,
                    'weight': cs.weight,
                    'weighted_score': cs.weighted_score,
                    'reasoning': cs.reasoning
                }
                for cs in result.criterion_scores
            ],
            'composite_score': result.composite_score,
            'judgment': result.judgment,
            'llm_model': result.llm_model,
            'execution_time_ms': result.execution_time_ms
        }


# Convenience function
def judge_signal(signal: Dict, llm_client: Optional[object] = None) -> LLMJudgeResult:
    """
    Judge a signal using LLM.

    Args:
        signal: Trading signal dictionary
        llm_client: Optional LLM client

    Returns:
        LLMJudgeResult
    """
    judge = LLMJudge(llm_client=llm_client)
    return judge.judge_signal(signal)


if __name__ == "__main__":
    # Example usage
    from datetime import datetime, timezone

    test_signal = {
        "analysis": {
            "news": {
                "sentiment_score": 1.5,
                "confidence": 0.85,
                "key_events": ["Strong earnings beat", "New product launch"]
            },
            "technical": {
                "signal": "bullish",
                "signal_strength": 0.8,
                "indicators": {"rsi": 65, "macd": {"value": 2.5}}
            },
            "fundamental": {
                "valuation": "undervalued",
                "financial_health_score": 0.9,
                "growth_score": 0.85
            }
        },
        "signal": "buy",
        "sizing": {
            "position_size": 0.15,
            "rationale": "Strong fundamentals (ROE: 28%, revenue growth: 18% YoY) combined with bullish technical setup justify 15% position size."
        },
        "risk": {
            "stop_loss": 145.0,
            "take_profit": 165.0,
            "max_drawdown": 0.05,
            "risk_reward_ratio": 3.0
        },
        "rationale": "Based on strong earnings beat (+15% vs expectations), bullish technical breakout above $150 resistance with cup-and-handle pattern confirmation, and favorable news sentiment (score: 1.5), we recommend BUY with 15% position size. The company shows excellent fundamentals (ROE: 28%, revenue growth: 18%) and is currently undervalued.",
        "evidence": {
            "sources": [
                {
                    "type": "earnings_report",
                    "description": "Q4 earnings beat by 15%",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "type": "technical_indicator",
                    "description": "Bullish MACD crossover",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "confidence": 0.85
        },
        "metadata": {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "current_price": 150.0
        }
    }

    print("\n" + "="*60)
    print("LLM JUDGE TEST")
    print("="*60 + "\n")

    judge = LLMJudge()
    result = judge.judge_signal(test_signal, explain=True)

    print(f"Signal: {result.signal_id}")
    print(f"Composite Score: {result.composite_score:.2f}/10.0")
    print(f"Judgment: {result.judgment.upper()}")
    print(f"\nCriterion Scores:")

    for cs in result.criterion_scores:
        print(f"  {cs.criterion:20s}: {cs.score:4.1f}/10 (weight: {cs.weight:.0%})")
        if cs.reasoning:
            print(f"    → {cs.reasoning}")

    print(f"\nExecution Time: {result.execution_time_ms:.1f}ms")
    print("\n" + "="*60 + "\n")
