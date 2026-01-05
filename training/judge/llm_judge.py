#!/usr/bin/env python3
"""
LLM-as-a-Judge Integration - Task 3.1

Use LLMs to evaluate dataset quality with rubrics.

Features:
- LLMJudge class with OpenAI/Anthropic support
- Judge evaluation with rubrics
- Structured output (score + reasoning)
- Multi-criteria evaluation
- Batch evaluation support
- Judge result storage

Phase A1 Week 3-4: Task 3.1 COMPLETE
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from loguru import logger
import openai
from openai import OpenAI


class JudgeProvider(Enum):
    """LLM provider for judge"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class JudgeCriterion(Enum):
    """Evaluation criteria"""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    REASONING_QUALITY = "reasoning_quality"
    INSTRUCTION_FOLLOWING = "instruction_following"
    FACTUAL_CORRECTNESS = "factual_correctness"
    COHERENCE = "coherence"


@dataclass
class JudgeRubric:
    """
    Evaluation rubric for judge
    """
    criterion: str
    description: str
    min_score: float = 0.0
    max_score: float = 1.0
    weight: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class JudgeEvaluation:
    """
    Judge evaluation result
    """
    trajectory_id: str
    overall_score: float
    criterion_scores: Dict[str, float]
    reasoning: str
    pass_threshold: float
    passed: bool
    model: str
    provider: str
    rubrics: List[JudgeRubric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'trajectory_id': self.trajectory_id,
            'overall_score': self.overall_score,
            'criterion_scores': self.criterion_scores,
            'reasoning': self.reasoning,
            'pass_threshold': self.pass_threshold,
            'passed': self.passed,
            'model': self.model,
            'provider': self.provider,
            'rubrics': [r.to_dict() for r in self.rubrics],
            'metadata': self.metadata
        }


class LLMJudge:
    """
    LLM-as-a-Judge for dataset quality evaluation
    
    Supports OpenAI and Anthropic models
    """
    
    def __init__(
        self,
        provider: JudgeProvider = JudgeProvider.OPENAI,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        pass_threshold: float = 0.7
    ):
        """
        Initialize LLM Judge
        
        Args:
            provider: LLM provider (openai or anthropic)
            model: Model name (optional, uses default)
            api_key: API key (optional, uses env var)
            pass_threshold: Minimum score to pass (0.0 to 1.0)
        """
        self.provider = provider
        self.pass_threshold = pass_threshold
        
        # Set model
        if model:
            self.model = model
        else:
            if provider == JudgeProvider.OPENAI:
                self.model = "gpt-4o"
            elif provider == JudgeProvider.ANTHROPIC:
                self.model = "claude-3-5-sonnet-20241022"
        
        # Set API key
        if api_key:
            self.api_key = api_key
        elif provider == JudgeProvider.OPENAI:
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif provider == JudgeProvider.ANTHROPIC:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError(f"API key not provided for {provider.value}")
        
        # Initialize client
        if provider == JudgeProvider.OPENAI:
            self.client = OpenAI(api_key=self.api_key)
        elif provider == JudgeProvider.ANTHROPIC:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed")
        
        logger.info(
            f"LLMJudge initialized: provider={provider.value}, "
            f"model={self.model}, threshold={pass_threshold}"
        )
    
    def evaluate_trajectory(
        self,
        trajectory: Dict,
        rubrics: List[JudgeRubric],
        context: Optional[Dict] = None
    ) -> JudgeEvaluation:
        """
        Evaluate trajectory with rubrics
        
        Args:
            trajectory: Trajectory data
            rubrics: Evaluation rubrics
            context: Additional context (optional)
        
        Returns:
            JudgeEvaluation object
        """
        logger.info(f"Evaluating trajectory: {trajectory.get('id', 'unknown')}")
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(trajectory, rubrics, context)
        
        # Get LLM response
        if self.provider == JudgeProvider.OPENAI:
            response = self._evaluate_with_openai(prompt, rubrics)
        elif self.provider == JudgeProvider.ANTHROPIC:
            response = self._evaluate_with_anthropic(prompt, rubrics)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Parse response
        evaluation = self._parse_evaluation_response(
            response,
            trajectory.get('id', 'unknown'),
            rubrics
        )
        
        logger.info(
            f"Evaluation complete: score={evaluation.overall_score:.3f}, "
            f"passed={evaluation.passed}"
        )
        
        return evaluation
    
    def evaluate_batch(
        self,
        trajectories: List[Dict],
        rubrics: List[JudgeRubric],
        context: Optional[Dict] = None
    ) -> List[JudgeEvaluation]:
        """
        Evaluate multiple trajectories
        
        Args:
            trajectories: List of trajectory data
            rubrics: Evaluation rubrics
            context: Additional context (optional)
        
        Returns:
            List of JudgeEvaluation objects
        """
        logger.info(f"Evaluating batch of {len(trajectories)} trajectories")
        
        evaluations = []
        
        for trajectory in trajectories:
            try:
                evaluation = self.evaluate_trajectory(trajectory, rubrics, context)
                evaluations.append(evaluation)
            except Exception as e:
                logger.error(f"Failed to evaluate trajectory {trajectory.get('id')}: {e}")
                # Create failed evaluation
                evaluations.append(JudgeEvaluation(
                    trajectory_id=trajectory.get('id', 'unknown'),
                    overall_score=0.0,
                    criterion_scores={},
                    reasoning=f"Evaluation failed: {str(e)}",
                    pass_threshold=self.pass_threshold,
                    passed=False,
                    model=self.model,
                    provider=self.provider.value,
                    rubrics=rubrics,
                    metadata={'error': str(e)}
                ))
        
        passed_count = sum(1 for e in evaluations if e.passed)
        logger.info(
            f"Batch evaluation complete: {passed_count}/{len(evaluations)} passed"
        )
        
        return evaluations
    
    def _build_evaluation_prompt(
        self,
        trajectory: Dict,
        rubrics: List[JudgeRubric],
        context: Optional[Dict] = None
    ) -> str:
        """
        Build evaluation prompt
        
        Args:
            trajectory: Trajectory data
            rubrics: Evaluation rubrics
            context: Additional context
        
        Returns:
            Prompt string
        """
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("You are an expert evaluator assessing the quality of AI agent trajectories.")
        prompt_parts.append("")
        
        # Context
        if context:
            prompt_parts.append("Context:")
            prompt_parts.append(json.dumps(context, indent=2))
            prompt_parts.append("")
        
        # Trajectory
        prompt_parts.append("Trajectory to evaluate:")
        prompt_parts.append(json.dumps(trajectory, indent=2))
        prompt_parts.append("")
        
        # Rubrics
        prompt_parts.append("Evaluation Criteria:")
        for rubric in rubrics:
            prompt_parts.append(
                f"- {rubric.criterion} (weight: {rubric.weight}): {rubric.description}"
            )
            prompt_parts.append(f"  Score range: {rubric.min_score} to {rubric.max_score}")
        prompt_parts.append("")
        
        # Instructions
        prompt_parts.append("Instructions:")
        prompt_parts.append("1. Evaluate the trajectory against each criterion")
        prompt_parts.append("2. Provide a score for each criterion within the specified range")
        prompt_parts.append("3. Calculate the overall weighted score")
        prompt_parts.append("4. Provide detailed reasoning for your evaluation")
        prompt_parts.append("")
        
        # Output format
        prompt_parts.append("Respond with a JSON object in this exact format:")
        prompt_parts.append("{")
        prompt_parts.append('  "criterion_scores": {')
        for i, rubric in enumerate(rubrics):
            comma = "," if i < len(rubrics) - 1 else ""
            prompt_parts.append(f'    "{rubric.criterion}": <score>{comma}')
        prompt_parts.append('  },')
        prompt_parts.append('  "reasoning": "<detailed reasoning for each criterion>"')
        prompt_parts.append("}")
        
        return "\n".join(prompt_parts)
    
    def _evaluate_with_openai(
        self,
        prompt: str,
        rubrics: List[JudgeRubric]
    ) -> Dict:
        """
        Evaluate with OpenAI
        
        Args:
            prompt: Evaluation prompt
            rubrics: Evaluation rubrics
        
        Returns:
            Response dict
        """
        logger.debug("Calling OpenAI API")
        
        # Build JSON schema for structured output
        properties = {}
        for rubric in rubrics:
            properties[rubric.criterion] = {
                "type": "number",
                "minimum": rubric.min_score,
                "maximum": rubric.max_score,
                "description": f"Score for {rubric.criterion}"
            }
        
        schema = {
            "type": "object",
            "properties": {
                "criterion_scores": {
                    "type": "object",
                    "properties": properties,
                    "required": [r.criterion for r in rubrics],
                    "additionalProperties": False
                },
                "reasoning": {
                    "type": "string",
                    "description": "Detailed reasoning for the evaluation"
                }
            },
            "required": ["criterion_scores", "reasoning"],
            "additionalProperties": False
        }
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation",
                    "strict": True,
                    "schema": schema
                }
            },
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    
    def _evaluate_with_anthropic(
        self,
        prompt: str,
        rubrics: List[JudgeRubric]
    ) -> Dict:
        """
        Evaluate with Anthropic
        
        Args:
            prompt: Evaluation prompt
            rubrics: Evaluation rubrics
        
        Returns:
            Response dict
        """
        logger.debug("Calling Anthropic API")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.content[0].text
        
        # Parse JSON from response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                raise ValueError(f"Failed to parse JSON from response: {content}")
    
    def _parse_evaluation_response(
        self,
        response: Dict,
        trajectory_id: str,
        rubrics: List[JudgeRubric]
    ) -> JudgeEvaluation:
        """
        Parse evaluation response
        
        Args:
            response: LLM response dict
            trajectory_id: Trajectory ID
            rubrics: Evaluation rubrics
        
        Returns:
            JudgeEvaluation object
        """
        criterion_scores = response.get('criterion_scores', {})
        reasoning = response.get('reasoning', '')
        
        # Calculate weighted overall score
        total_weight = sum(r.weight for r in rubrics)
        weighted_sum = 0.0
        
        for rubric in rubrics:
            score = criterion_scores.get(rubric.criterion, 0.0)
            weighted_sum += score * rubric.weight
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Check if passed
        passed = overall_score >= self.pass_threshold
        
        return JudgeEvaluation(
            trajectory_id=trajectory_id,
            overall_score=overall_score,
            criterion_scores=criterion_scores,
            reasoning=reasoning,
            pass_threshold=self.pass_threshold,
            passed=passed,
            model=self.model,
            provider=self.provider.value,
            rubrics=rubrics
        )


# ============================================================================
# Helper Functions
# ============================================================================

def create_judge(
    provider: str = "openai",
    model: Optional[str] = None,
    pass_threshold: float = 0.7
) -> LLMJudge:
    """
    Create LLM judge (convenience function)
    
    Args:
        provider: Provider name (openai or anthropic)
        model: Model name (optional)
        pass_threshold: Pass threshold
    
    Returns:
        LLMJudge instance
    """
    provider_enum = JudgeProvider(provider)
    return LLMJudge(
        provider=provider_enum,
        model=model,
        pass_threshold=pass_threshold
    )


def create_rubric(
    criterion: str,
    description: str,
    weight: float = 1.0,
    min_score: float = 0.0,
    max_score: float = 1.0
) -> JudgeRubric:
    """
    Create evaluation rubric (convenience function)
    
    Args:
        criterion: Criterion name
        description: Criterion description
        weight: Weight for overall score
        min_score: Minimum score
        max_score: Maximum score
    
    Returns:
        JudgeRubric instance
    """
    return JudgeRubric(
        criterion=criterion,
        description=description,
        weight=weight,
        min_score=min_score,
        max_score=max_score
    )


if __name__ == "__main__":
    # Example usage
    print("=== LLM Judge Example ===\n")
    
    # Create rubrics
    rubrics = [
        JudgeRubric(
            criterion="accuracy",
            description="How accurate is the agent's analysis?",
            weight=2.0,
            min_score=0.0,
            max_score=1.0
        ),
        JudgeRubric(
            criterion="reasoning_quality",
            description="How clear and logical is the reasoning?",
            weight=1.5,
            min_score=0.0,
            max_score=1.0
        ),
        JudgeRubric(
            criterion="completeness",
            description="Does the response address all aspects?",
            weight=1.0,
            min_score=0.0,
            max_score=1.0
        )
    ]
    
    print("Evaluation Rubrics:")
    for rubric in rubrics:
        print(f"  - {rubric.criterion} (weight: {rubric.weight})")
        print(f"    {rubric.description}")
    print()
    
    # Mock trajectory
    trajectory = {
        'id': 'traj_001',
        'agent_type': 'technical',
        'query': 'Analyze AAPL stock',
        'response': 'AAPL shows strong fundamentals with good revenue growth.',
        'reasoning': 'Based on recent earnings and market position.',
        'final_reward': 0.85,
        'confidence_score': 0.9
    }
    
    print("Mock Trajectory:")
    print(f"  ID: {trajectory['id']}")
    print(f"  Query: {trajectory['query']}")
    print(f"  Response: {trajectory['response']}")
    print()
    
    print("✅ Example completed!")
    print("\nNote: Actual LLM evaluation requires API key and will be tested separately.")
