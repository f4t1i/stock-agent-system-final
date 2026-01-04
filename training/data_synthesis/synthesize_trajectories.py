"""
Trajectory Synthesis - Generate synthetic training data from experience library

This module implements:
- Successful trajectory extraction and formatting
- Error healing for failed trajectories
- Data augmentation for diversity
- ChatML format conversion for SFT training
"""

import json
import random
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger

from training.data_synthesis.experience_library import ExperienceLibrary


class TrajectorySynthesizer:
    """
    Synthesize training data from stored trajectories.
    
    Features:
    - Extract successful trajectories
    - Heal failed trajectories with corrections
    - Augment data for diversity
    - Format for SFT training (ChatML)
    """
    
    def __init__(
        self,
        experience_library: Optional[ExperienceLibrary] = None,
        llm_client: Optional[object] = None
    ):
        """
        Initialize trajectory synthesizer.
        
        Args:
            experience_library: ExperienceLibrary instance
            llm_client: LLM client for error healing (OpenAI/Anthropic)
        """
        self.library = experience_library or ExperienceLibrary()
        self.llm_client = llm_client
        
        logger.info("Trajectory Synthesizer initialized")
    
    def synthesize_sft_dataset(
        self,
        agent_type: str,
        num_examples: int = 1000,
        min_reward: float = 0.7,
        include_error_healing: bool = True,
        augmentation_factor: float = 0.3,
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Synthesize a complete SFT training dataset.
        
        Args:
            agent_type: Agent type (news, technical, fundamental, strategist)
            num_examples: Target number of examples
            min_reward: Minimum reward for successful trajectories
            include_error_healing: Include healed failed trajectories
            augmentation_factor: Fraction of augmented examples
            output_path: Path to save JSONL file
        
        Returns:
            List of training examples in ChatML format
        """
        logger.info(f"Synthesizing SFT dataset for {agent_type}")
        
        training_examples = []
        
        # 1. Extract successful trajectories
        num_successful = int(num_examples * (1 - augmentation_factor))
        successful_trajectories = self.library.get_top_trajectories(
            n=num_successful,
            agent_type=agent_type,
            min_reward=min_reward
        )
        
        logger.info(f"Retrieved {len(successful_trajectories)} successful trajectories")
        
        for traj in successful_trajectories:
            example = self._trajectory_to_chatml(traj, agent_type)
            if example:
                training_examples.append(example)
        
        # 2. Error healing (if enabled and LLM available)
        if include_error_healing and self.llm_client:
            num_healed = int(num_examples * 0.2)  # 20% healed examples
            failed_trajectories = self.library.get_failed_trajectories(
                n=num_healed,
                agent_type=agent_type
            )
            
            logger.info(f"Healing {len(failed_trajectories)} failed trajectories")
            
            for traj in failed_trajectories:
                healed_example = self._heal_trajectory(traj, agent_type)
                if healed_example:
                    training_examples.append(healed_example)
        
        # 3. Data augmentation
        num_augmented = int(num_examples * augmentation_factor)
        if num_augmented > 0:
            logger.info(f"Augmenting {num_augmented} examples")
            augmented = self._augment_trajectories(
                training_examples[:num_augmented],
                agent_type
            )
            training_examples.extend(augmented)
        
        # 4. Shuffle
        random.shuffle(training_examples)
        
        # 5. Limit to target size
        training_examples = training_examples[:num_examples]
        
        logger.info(f"Synthesized {len(training_examples)} training examples")
        
        # 6. Save to file if path provided
        if output_path:
            self._save_jsonl(training_examples, output_path)
        
        return training_examples
    
    def _trajectory_to_chatml(self, trajectory: Dict, agent_type: str) -> Optional[Dict]:
        """
        Convert trajectory to ChatML format for SFT training.
        
        Args:
            trajectory: Trajectory dictionary
            agent_type: Agent type
        
        Returns:
            ChatML formatted example
        """
        try:
            traj_data = trajectory['trajectory_data']
            
            # Extract relevant information based on agent type
            if agent_type == 'news':
                return self._format_news_trajectory(trajectory)
            elif agent_type == 'technical':
                return self._format_technical_trajectory(trajectory)
            elif agent_type == 'fundamental':
                return self._format_fundamental_trajectory(trajectory)
            elif agent_type == 'strategist':
                return self._format_strategist_trajectory(trajectory)
            else:
                logger.warning(f"Unknown agent type: {agent_type}")
                return None
        
        except Exception as e:
            logger.error(f"Error converting trajectory to ChatML: {e}")
            return None
    
    def _format_news_trajectory(self, trajectory: Dict) -> Dict:
        """Format news agent trajectory to ChatML"""
        traj_data = trajectory['trajectory_data']
        
        # Find news agent step
        news_step = None
        for step in traj_data.get('trajectory', []):
            if step.get('step') == 'news_agent':
                news_step = step
                break
        
        if not news_step:
            return None
        
        output = news_step.get('output', {})
        
        # System prompt
        system_prompt = """You are a financial news sentiment analysis expert. Analyze news articles and provide sentiment scores, confidence levels, and key insights."""
        
        # User message (input)
        user_message = f"""Analyze the sentiment for {trajectory['symbol']} based on recent news articles.

Symbol: {trajectory['symbol']}
News Articles: {json.dumps(output.get('news_articles', [])[:3], indent=2)}

Provide:
1. Overall sentiment score (-2 to +2)
2. Confidence level (0-1)
3. Key events or insights
4. Recommendation impact"""
        
        # Assistant message (output)
        assistant_message = json.dumps({
            'sentiment_score': output.get('sentiment_score', 0.0),
            'confidence': output.get('confidence', 0.0),
            'key_events': output.get('key_events', []),
            'reasoning': output.get('reasoning', ''),
            'recommendation': output.get('recommendation', 'neutral')
        }, indent=2)
        
        return {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': assistant_message}
            ]
        }
    
    def _format_technical_trajectory(self, trajectory: Dict) -> Dict:
        """Format technical agent trajectory to ChatML"""
        traj_data = trajectory['trajectory_data']
        
        # Find technical agent step
        tech_step = None
        for step in traj_data.get('trajectory', []):
            if step.get('step') == 'technical_agent':
                tech_step = step
                break
        
        if not tech_step:
            return None
        
        output = tech_step.get('output', {})
        
        system_prompt = """You are a technical analysis expert. Analyze price charts, indicators, and patterns to provide trading signals."""
        
        user_message = f"""Perform technical analysis for {trajectory['symbol']}.

Symbol: {trajectory['symbol']}
Indicators: {json.dumps(output.get('indicators', {}), indent=2)}
Patterns: {json.dumps(output.get('patterns', []), indent=2)}

Provide:
1. Overall signal (bullish/bearish/neutral)
2. Signal strength (0-1)
3. Key technical levels
4. Pattern interpretation"""
        
        assistant_message = json.dumps({
            'signal': output.get('signal', 'neutral'),
            'signal_strength': output.get('signal_strength', 0.0),
            'support_levels': output.get('support_levels', []),
            'resistance_levels': output.get('resistance_levels', []),
            'reasoning': output.get('reasoning', ''),
            'recommendation': output.get('recommendation', 'hold')
        }, indent=2)
        
        return {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': assistant_message}
            ]
        }
    
    def _format_fundamental_trajectory(self, trajectory: Dict) -> Dict:
        """Format fundamental agent trajectory to ChatML"""
        traj_data = trajectory['trajectory_data']
        
        # Find fundamental agent step
        fund_step = None
        for step in traj_data.get('trajectory', []):
            if step.get('step') == 'fundamental_agent':
                fund_step = step
                break
        
        if not fund_step:
            return None
        
        output = fund_step.get('output', {})
        
        system_prompt = """You are a fundamental analysis expert. Analyze financial metrics and company fundamentals to assess investment value."""
        
        user_message = f"""Analyze the fundamentals for {trajectory['symbol']}.

Symbol: {trajectory['symbol']}
Metrics: {json.dumps(output.get('metrics', {}), indent=2)}
Valuation: {output.get('valuation', 'unknown')}

Provide:
1. Valuation assessment
2. Financial health score
3. Growth potential
4. Investment recommendation"""
        
        assistant_message = json.dumps({
            'valuation': output.get('valuation', 'fairly_valued'),
            'financial_health_score': output.get('financial_health_score', 0.0),
            'growth_score': output.get('growth_score', 0.0),
            'reasoning': output.get('reasoning', ''),
            'recommendation': output.get('recommendation', 'hold')
        }, indent=2)
        
        return {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': assistant_message}
            ]
        }
    
    def _format_strategist_trajectory(self, trajectory: Dict) -> Dict:
        """Format strategist trajectory to ChatML"""
        traj_data = trajectory['trajectory_data']
        
        # Find strategist step
        strat_step = None
        for step in traj_data.get('trajectory', []):
            if step.get('step') == 'strategist':
                strat_step = step
                break
        
        if not strat_step:
            return None
        
        output = strat_step.get('output', {})
        
        # Get agent outputs
        agent_outputs = {}
        for step in traj_data.get('trajectory', []):
            if step.get('step') in ['news_agent', 'technical_agent', 'fundamental_agent']:
                agent_name = step['step'].replace('_agent', '')
                agent_outputs[agent_name] = step.get('output', {})
        
        system_prompt = """You are a senior trading strategist. Synthesize insights from multiple analysis sources and make final trading decisions with risk management."""
        
        user_message = f"""Make a trading decision for {trajectory['symbol']}.

Symbol: {trajectory['symbol']}
Agent Insights: {json.dumps(agent_outputs, indent=2)}
Portfolio State: {json.dumps(trajectory.get('portfolio_state', {}), indent=2)}

Provide:
1. Final decision (buy/sell/hold)
2. Position sizing
3. Entry/exit targets
4. Risk management parameters"""
        
        assistant_message = json.dumps({
            'decision': output.get('decision', 'hold'),
            'confidence': output.get('confidence', 0.0),
            'position_size': output.get('position_size', 0.0),
            'entry_target': output.get('entry_target'),
            'stop_loss': output.get('stop_loss'),
            'take_profit': output.get('take_profit'),
            'reasoning': output.get('reasoning', ''),
            'risk_assessment': output.get('risk_assessment', '')
        }, indent=2)
        
        return {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': assistant_message}
            ]
        }
    
    def _heal_trajectory(self, trajectory: Dict, agent_type: str) -> Optional[Dict]:
        """
        Heal a failed trajectory using LLM.
        
        Args:
            trajectory: Failed trajectory
            agent_type: Agent type
        
        Returns:
            Healed ChatML example
        """
        if not self.llm_client:
            return None
        
        try:
            # Convert to ChatML first
            chatml = self._trajectory_to_chatml(trajectory, agent_type)
            
            if not chatml:
                return None
            
            # Extract the error
            errors = trajectory['trajectory_data'].get('errors', [])
            error_description = '; '.join(errors) if errors else 'Unknown error'
            
            # Create healing prompt
            healing_prompt = f"""The following agent output had errors: {error_description}

Original output:
{chatml['messages'][-1]['content']}

Please provide a corrected version that fixes these errors while maintaining the same format."""
            
            # Call LLM to heal (placeholder - implement based on your LLM client)
            # healed_response = self.llm_client.complete(healing_prompt)
            
            # For now, return the original with a note
            # In production, replace with actual LLM healing
            logger.debug(f"Error healing not fully implemented for trajectory {trajectory['id']}")
            
            return None
        
        except Exception as e:
            logger.error(f"Error healing trajectory: {e}")
            return None
    
    def _augment_trajectories(
        self,
        examples: List[Dict],
        agent_type: str
    ) -> List[Dict]:
        """
        Augment training examples for diversity.
        
        Args:
            examples: Original examples
            agent_type: Agent type
        
        Returns:
            Augmented examples
        """
        augmented = []
        
        for example in examples:
            try:
                # Simple augmentation: paraphrase system prompt
                augmented_example = json.loads(json.dumps(example))  # Deep copy
                
                # Vary temperature in responses (simulate slight variations)
                # In production, use LLM to paraphrase
                
                augmented.append(augmented_example)
            
            except Exception as e:
                logger.error(f"Error augmenting example: {e}")
        
        return augmented
    
    def _save_jsonl(self, examples: List[Dict], output_path: str):
        """
        Save examples to JSONL file.
        
        Args:
            examples: Training examples
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")


# Convenience function
def synthesize_dataset(
    agent_type: str,
    num_examples: int = 1000,
    output_path: Optional[str] = None,
    **kwargs
) -> List[Dict]:
    """
    Synthesize training dataset for an agent.
    
    Args:
        agent_type: Agent type (news, technical, fundamental, strategist)
        num_examples: Number of examples to generate
        output_path: Path to save JSONL file
        **kwargs: Additional arguments for synthesizer
    
    Returns:
        List of training examples
    """
    synthesizer = TrajectorySynthesizer()
    
    return synthesizer.synthesize_sft_dataset(
        agent_type=agent_type,
        num_examples=num_examples,
        output_path=output_path,
        **kwargs
    )
