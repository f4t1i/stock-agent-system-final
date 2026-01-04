"""
LLM Judge - Automated Quality Evaluation using Claude
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path

from anthropic import Anthropic
import yaml

from utils.config_loader import load_config


class LLMJudge:
    """
    LLM-as-a-Judge System für automatisierte Qualitätsbewertung.

    Nutzt Claude (oder andere LLMs) um Agent-Outputs anhand von
    definierten Rubrics zu bewerten.
    """

    def __init__(
        self,
        rubrics_dir: str = "config/judge/rubrics",
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None
    ):
        """
        Args:
            rubrics_dir: Verzeichnis mit Rubric YAML-Dateien
            model: Claude Modell-Name
            api_key: Anthropic API-Key (oder aus Env)
        """
        self.rubrics_dir = Path(rubrics_dir)
        self.model = model

        # Initialize Anthropic client
        api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or parameters")

        self.client = Anthropic(api_key=api_key)

        # Load rubrics cache
        self.rubrics_cache = {}

    def load_rubric(self, rubric_name: str) -> Dict:
        """
        Lade Rubric aus YAML-Datei

        Args:
            rubric_name: Name der Rubric (z.B. 'news', 'technical', 'strategist')

        Returns:
            Rubric Dictionary
        """
        if rubric_name in self.rubrics_cache:
            return self.rubrics_cache[rubric_name]

        rubric_path = self.rubrics_dir / f"{rubric_name}_rubric.yaml"

        if not rubric_path.exists():
            raise FileNotFoundError(f"Rubric not found: {rubric_path}")

        with open(rubric_path, 'r') as f:
            rubric = yaml.safe_load(f)

        self.rubrics_cache[rubric_name] = rubric
        return rubric

    def evaluate(
        self,
        agent_output: Dict,
        rubric_name: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Bewerte Agent-Output anhand einer Rubric

        Args:
            agent_output: Output des Agenten
            rubric_name: Welche Rubric verwenden
            context: Zusätzlicher Kontext (z.B. Ground Truth, Marktdaten)

        Returns:
            Evaluation Result:
            {
                'overall_score': float,
                'dimension_scores': {...},
                'feedback': str,
                'strengths': [str],
                'weaknesses': [str],
                'suggestions': [str]
            }
        """
        # Load rubric
        rubric = self.load_rubric(rubric_name)

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(
            agent_output,
            rubric,
            context
        )

        # Call Claude
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.3,  # Low temperature for consistency
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Parse response
            result = self._parse_evaluation_response(
                response.content[0].text,
                rubric
            )

            result['metadata'] = {
                'rubric': rubric_name,
                'model': self.model,
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens
            }

            return result

        except Exception as e:
            return {
                'overall_score': 0.5,
                'dimension_scores': {},
                'feedback': f'Evaluation failed: {str(e)}',
                'strengths': [],
                'weaknesses': [],
                'suggestions': [],
                'error': str(e)
            }

    def _build_evaluation_prompt(
        self,
        agent_output: Dict,
        rubric: Dict,
        context: Optional[Dict]
    ) -> str:
        """Erstelle Evaluation Prompt"""

        # Rubric-Beschreibung
        rubric_description = f"""
## Evaluation Rubric: {rubric['name']}

Description: {rubric['description']}

## Evaluation Dimensions:

"""

        # Add each dimension
        for dimension in rubric['dimensions']:
            rubric_description += f"""
### {dimension['name']} (Weight: {dimension['weight']})
{dimension['description']}

Scoring Guide:
"""
            for level in dimension['levels']:
                rubric_description += f"- **{level['score']}**: {level['description']}\n"

            rubric_description += "\n"

        # Format agent output
        agent_output_str = json.dumps(agent_output, indent=2)

        # Format context if provided
        context_str = ""
        if context:
            context_str = f"""
## Additional Context:
{json.dumps(context, indent=2)}
"""

        # Build full prompt
        prompt = f"""You are an expert evaluator assessing the quality of financial analysis agent outputs.

{rubric_description}

## Agent Output to Evaluate:
```json
{agent_output_str}
```
{context_str}

## Your Task:

1. Evaluate the agent output against each dimension in the rubric
2. Assign a score (0.0 to 1.0) for each dimension based on the scoring guide
3. Calculate an overall weighted score
4. Provide constructive feedback

## Required Output Format (JSON):

```json
{{
    "overall_score": <float 0.0 to 1.0>,
    "dimension_scores": {{
        "<dimension_name>": {{
            "score": <float 0.0 to 1.0>,
            "justification": "<brief explanation>"
        }},
        ...
    }},
    "feedback": "<overall assessment>",
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
    "suggestions": ["<improvement suggestion 1>", ...]
}}
```

Provide your evaluation now:
"""

        return prompt

    def _parse_evaluation_response(
        self,
        response: str,
        rubric: Dict
    ) -> Dict:
        """Parse Judge-Response"""

        try:
            # Extract JSON
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            result = json.loads(response.strip())

            # Validate structure
            if 'overall_score' not in result:
                raise ValueError("Missing overall_score")

            if 'dimension_scores' not in result:
                result['dimension_scores'] = {}

            # Clamp scores
            result['overall_score'] = max(0.0, min(1.0, result['overall_score']))

            for dim_name, dim_data in result['dimension_scores'].items():
                if isinstance(dim_data, dict) and 'score' in dim_data:
                    dim_data['score'] = max(0.0, min(1.0, dim_data['score']))

            # Ensure required fields
            result.setdefault('feedback', 'No feedback provided')
            result.setdefault('strengths', [])
            result.setdefault('weaknesses', [])
            result.setdefault('suggestions', [])

            return result

        except Exception as e:
            # Fallback
            return {
                'overall_score': 0.5,
                'dimension_scores': {},
                'feedback': f'Failed to parse evaluation: {str(e)}',
                'strengths': [],
                'weaknesses': [],
                'suggestions': [],
                'parse_error': str(e)
            }

    def batch_evaluate(
        self,
        outputs: List[Dict],
        rubric_name: str,
        contexts: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Batch-Evaluation mehrerer Outputs"""

        results = []

        for i, output in enumerate(outputs):
            context = contexts[i] if contexts and i < len(contexts) else None
            result = self.evaluate(output, rubric_name, context)
            results.append(result)

        return results

    def calculate_reward(
        self,
        evaluation: Dict,
        reward_config: Optional[Dict] = None
    ) -> float:
        """
        Berechne RL-Reward aus Evaluation

        Args:
            evaluation: Evaluation result
            reward_config: Optional config für Reward-Berechnung

        Returns:
            Reward value (typically -1 to +1)
        """
        if reward_config is None:
            reward_config = {
                'score_weight': 1.0,
                'penalty_for_errors': -0.5,
                'bonus_for_excellence': 0.2
            }

        # Base reward from overall score
        # Map 0-1 score to -1 to +1 reward
        base_reward = (evaluation['overall_score'] * 2) - 1

        # Apply weights
        reward = base_reward * reward_config['score_weight']

        # Penalty for errors
        if 'error' in evaluation or 'parse_error' in evaluation:
            reward += reward_config['penalty_for_errors']

        # Bonus for excellent performance (score > 0.9)
        if evaluation['overall_score'] > 0.9:
            reward += reward_config['bonus_for_excellence']

        # Clamp to reasonable range
        return max(-2.0, min(2.0, reward))


if __name__ == "__main__":
    # Example usage
    judge = LLMJudge()

    # Example agent output
    agent_output = {
        'sentiment_score': 1.5,
        'confidence': 0.85,
        'reasoning': 'Strong positive sentiment from earnings beat and positive guidance',
        'price_impact': 'bullish',
        'time_horizon': 'short_term',
        'key_events': ['Earnings beat', 'Raised guidance'],
        'risk_factors': ['Market volatility']
    }

    # Evaluate
    evaluation = judge.evaluate(
        agent_output,
        rubric_name='news',
        context={'actual_price_change': 0.05}  # Stock went up 5%
    )

    print("Evaluation Result:")
    print(json.dumps(evaluation, indent=2))

    # Calculate reward
    reward = judge.calculate_reward(evaluation)
    print(f"\nRL Reward: {reward:.3f}")
