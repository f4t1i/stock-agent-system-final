"""
Multi-Judge System - Consensus-based Evaluation with Multiple LLMs
"""

import json
import os
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from anthropic import Anthropic
from openai import OpenAI

from .llm_judge import LLMJudge


class MultiJudge:
    """
    Multi-Judge System für robustere Evaluationen.

    Nutzt mehrere LLMs (Claude, GPT-4, optional DeepSeek) und
    bildet Konsens durch:
    - Median Scoring
    - Outlier Filtering
    - Inter-Rater Agreement Berechnung
    """

    def __init__(
        self,
        rubrics_dir: str = "config/judge/rubrics",
        judges: Optional[List[str]] = None,
        use_parallel: bool = True
    ):
        """
        Args:
            rubrics_dir: Verzeichnis mit Rubrics
            judges: Liste von Judge-IDs ['claude', 'gpt4', 'deepseek']
            use_parallel: Parallel execution für Judges
        """
        self.rubrics_dir = rubrics_dir
        self.use_parallel = use_parallel

        # Default judges
        if judges is None:
            judges = ['claude', 'gpt4']

        # Initialize judges
        self.judges = {}

        if 'claude' in judges:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.judges['claude'] = {
                    'client': Anthropic(api_key=api_key),
                    'model': 'claude-sonnet-4-20250514',
                    'type': 'anthropic'
                }

        if 'gpt4' in judges:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.judges['gpt4'] = {
                    'client': OpenAI(api_key=api_key),
                    'model': 'gpt-4o',
                    'type': 'openai'
                }

        if 'deepseek' in judges:
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if api_key:
                self.judges['deepseek'] = {
                    'client': OpenAI(
                        api_key=api_key,
                        base_url="https://api.deepseek.com"
                    ),
                    'model': 'deepseek-chat',
                    'type': 'openai'
                }

        if not self.judges:
            raise ValueError("No valid judge configurations found. Check API keys.")

        # Single judge instance for rubric loading
        self.judge_helper = LLMJudge(rubrics_dir=rubrics_dir)

    def evaluate(
        self,
        agent_output: Dict,
        rubric_name: str,
        context: Optional[Dict] = None,
        min_agreement: float = 0.7
    ) -> Dict:
        """
        Multi-Judge Evaluation mit Konsens

        Args:
            agent_output: Output des Agenten
            rubric_name: Rubric name
            context: Zusätzlicher Kontext
            min_agreement: Minimum Inter-Rater Agreement (0-1)

        Returns:
            Consensus Evaluation mit zusätzlichen Feldern:
            {
                'consensus_score': float,
                'judge_scores': {...},
                'agreement': float,
                'confidence': float,
                'outliers_removed': int,
                ...
            }
        """
        # Load rubric
        rubric = self.judge_helper.load_rubric(rubric_name)

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(agent_output, rubric, context)

        # Get evaluations from all judges
        if self.use_parallel:
            evaluations = self._evaluate_parallel(prompt)
        else:
            evaluations = self._evaluate_sequential(prompt)

        # Calculate consensus
        consensus = self._calculate_consensus(
            evaluations,
            min_agreement=min_agreement
        )

        # Add metadata
        consensus['metadata'] = {
            'rubric': rubric_name,
            'num_judges': len(self.judges),
            'judges_used': list(evaluations.keys())
        }

        return consensus

    def _build_evaluation_prompt(
        self,
        agent_output: Dict,
        rubric: Dict,
        context: Optional[Dict]
    ) -> str:
        """Build evaluation prompt (same as LLMJudge)"""

        rubric_description = f"""
## Evaluation Rubric: {rubric['name']}

{rubric['description']}

## Evaluation Dimensions:

"""

        for dimension in rubric['dimensions']:
            rubric_description += f"""
### {dimension['name']} (Weight: {dimension['weight']})
{dimension['description']}

Scoring Guide:
"""
            for level in dimension['levels']:
                rubric_description += f"- **{level['score']}**: {level['description']}\n"

            rubric_description += "\n"

        agent_output_str = json.dumps(agent_output, indent=2)

        context_str = ""
        if context:
            context_str = f"\n## Additional Context:\n{json.dumps(context, indent=2)}\n"

        prompt = f"""You are an expert evaluator assessing financial analysis agent outputs.

{rubric_description}

## Agent Output to Evaluate:
```json
{agent_output_str}
```
{context_str}

## Your Task:

1. Evaluate against each dimension
2. Assign scores (0.0 to 1.0) based on the rubric
3. Calculate weighted overall score
4. Provide constructive feedback

## Output Format (JSON):

```json
{{
    "overall_score": <float>,
    "dimension_scores": {{
        "<dimension_name>": {{
            "score": <float>,
            "justification": "<explanation>"
        }}
    }},
    "feedback": "<assessment>",
    "strengths": ["..."],
    "weaknesses": ["..."],
    "suggestions": ["..."]
}}
```

Provide your evaluation:
"""

        return prompt

    def _evaluate_parallel(self, prompt: str) -> Dict[str, Dict]:
        """Parallel evaluation mit allen Judges"""
        evaluations = {}

        with ThreadPoolExecutor(max_workers=len(self.judges)) as executor:
            future_to_judge = {
                executor.submit(self._call_judge, judge_id, config, prompt): judge_id
                for judge_id, config in self.judges.items()
            }

            for future in as_completed(future_to_judge):
                judge_id = future_to_judge[future]
                try:
                    result = future.result()
                    evaluations[judge_id] = result
                except Exception as e:
                    evaluations[judge_id] = {
                        'error': str(e),
                        'overall_score': 0.5
                    }

        return evaluations

    def _evaluate_sequential(self, prompt: str) -> Dict[str, Dict]:
        """Sequential evaluation"""
        evaluations = {}

        for judge_id, config in self.judges.items():
            try:
                result = self._call_judge(judge_id, config, prompt)
                evaluations[judge_id] = result
            except Exception as e:
                evaluations[judge_id] = {
                    'error': str(e),
                    'overall_score': 0.5
                }

        return evaluations

    def _call_judge(self, judge_id: str, config: Dict, prompt: str) -> Dict:
        """Call einzelnen Judge"""

        try:
            if config['type'] == 'anthropic':
                response = config['client'].messages.create(
                    model=config['model'],
                    max_tokens=2048,
                    temperature=0.3,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                response_text = response.content[0].text

            elif config['type'] == 'openai':
                response = config['client'].chat.completions.create(
                    model=config['model'],
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.3,
                    max_tokens=2048
                )
                response_text = response.choices[0].message.content

            else:
                raise ValueError(f"Unknown judge type: {config['type']}")

            # Parse response
            return self._parse_response(response_text, judge_id)

        except Exception as e:
            return {
                'error': str(e),
                'overall_score': 0.5,
                'dimension_scores': {},
                'feedback': f'Judge {judge_id} failed: {str(e)}'
            }

    def _parse_response(self, response: str, judge_id: str) -> Dict:
        """Parse Judge response"""

        try:
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            result = json.loads(response.strip())

            # Validate
            if 'overall_score' not in result:
                result['overall_score'] = 0.5

            result['overall_score'] = max(0.0, min(1.0, result['overall_score']))

            result.setdefault('dimension_scores', {})
            result.setdefault('feedback', 'No feedback')
            result.setdefault('strengths', [])
            result.setdefault('weaknesses', [])
            result.setdefault('suggestions', [])

            return result

        except Exception as e:
            return {
                'overall_score': 0.5,
                'dimension_scores': {},
                'feedback': f'Parse error for {judge_id}: {str(e)}',
                'parse_error': str(e)
            }

    def _calculate_consensus(
        self,
        evaluations: Dict[str, Dict],
        min_agreement: float = 0.7
    ) -> Dict:
        """Berechne Konsens aus mehreren Evaluations"""

        # Extract overall scores
        scores = []
        valid_judges = []

        for judge_id, eval_result in evaluations.items():
            if 'error' not in eval_result or 'overall_score' in eval_result:
                scores.append(eval_result['overall_score'])
                valid_judges.append(judge_id)

        if not scores:
            return {
                'consensus_score': 0.5,
                'judge_scores': evaluations,
                'agreement': 0.0,
                'confidence': 0.0,
                'feedback': 'All judges failed',
                'error': 'No valid evaluations'
            }

        scores = np.array(scores)

        # Remove outliers using IQR method
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_scores = scores[(scores >= lower_bound) & (scores <= upper_bound)]
        outliers_removed = len(scores) - len(filtered_scores)

        # Use median for consensus (robust to outliers)
        consensus_score = float(np.median(filtered_scores))

        # Calculate agreement (inverse of normalized std dev)
        if len(filtered_scores) > 1:
            std_dev = np.std(filtered_scores)
            # Normalize by score range (0-1)
            agreement = 1.0 - min(std_dev, 1.0)
        else:
            agreement = 1.0

        # Confidence based on agreement and number of judges
        confidence = agreement * min(len(valid_judges) / 3, 1.0)

        # Aggregate feedback
        all_strengths = []
        all_weaknesses = []
        all_suggestions = []
        feedback_texts = []

        for judge_id in valid_judges:
            eval_result = evaluations[judge_id]
            all_strengths.extend(eval_result.get('strengths', []))
            all_weaknesses.extend(eval_result.get('weaknesses', []))
            all_suggestions.extend(eval_result.get('suggestions', []))
            feedback_texts.append(eval_result.get('feedback', ''))

        # Deduplicate (simple approach)
        all_strengths = list(set(all_strengths))
        all_weaknesses = list(set(all_weaknesses))
        all_suggestions = list(set(all_suggestions))

        consensus_feedback = "\n\n".join([
            f"Judge {valid_judges[i]}: {feedback_texts[i]}"
            for i in range(len(valid_judges))
        ])

        return {
            'consensus_score': consensus_score,
            'judge_scores': {
                judge_id: evaluations[judge_id]['overall_score']
                for judge_id in valid_judges
            },
            'agreement': float(agreement),
            'confidence': float(confidence),
            'outliers_removed': outliers_removed,
            'num_valid_judges': len(valid_judges),
            'feedback': consensus_feedback,
            'strengths': all_strengths[:5],  # Top 5
            'weaknesses': all_weaknesses[:5],
            'suggestions': all_suggestions[:5],
            'detailed_evaluations': evaluations
        }

    def calculate_reward(
        self,
        consensus: Dict,
        reward_config: Optional[Dict] = None
    ) -> float:
        """Berechne RL-Reward aus Consensus"""

        if reward_config is None:
            reward_config = {
                'score_weight': 1.0,
                'confidence_weight': 0.3,
                'agreement_bonus': 0.2
            }

        # Base reward from consensus score
        base_reward = (consensus['consensus_score'] * 2) - 1

        # Confidence adjustment
        confidence_adjustment = (consensus['confidence'] - 0.5) * reward_config['confidence_weight']

        # Agreement bonus
        agreement_bonus = 0
        if consensus['agreement'] > 0.8:
            agreement_bonus = reward_config['agreement_bonus']

        reward = base_reward + confidence_adjustment + agreement_bonus

        return max(-2.0, min(2.0, reward))


if __name__ == "__main__":
    # Example usage
    multi_judge = MultiJudge()

    agent_output = {
        'sentiment_score': 1.5,
        'confidence': 0.85,
        'reasoning': 'Strong positive sentiment from earnings beat',
        'price_impact': 'bullish',
        'time_horizon': 'short_term'
    }

    consensus = multi_judge.evaluate(
        agent_output,
        rubric_name='news',
        context={'actual_price_change': 0.05}
    )

    print("Consensus Evaluation:")
    print(json.dumps({k: v for k, v in consensus.items() if k != 'detailed_evaluations'}, indent=2))

    reward = multi_judge.calculate_reward(consensus)
    print(f"\nRL Reward: {reward:.3f}")
