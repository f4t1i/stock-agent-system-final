"""
Data Synthesis Module

Automatically filters successful trajectories and prepares them as SFT training data.

Key Features:
1. Success filtering (correct predictions with high confidence)
2. Quality scoring
3. SFT format conversion (ChatML, Alpaca, etc.)
4. Batch processing
5. Data augmentation

Based on:
- TradingGroup data synthesis
- SIRIUS trajectory filtering
- PrimoAgent training data preparation
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from loguru import logger
import time

from data_pipeline.experience_library_postgres import (
    ExperienceLibraryPostgres,
    Trajectory
)


@dataclass
class SFTExample:
    """
    SFT training example in ChatML format
    """
    messages: List[Dict[str, str]]  # [{"role": "system/user/assistant", "content": "..."}]
    metadata: Dict  # trajectory_id, reward, etc.
    
    def to_chatml(self) -> str:
        """Convert to ChatML format string"""
        return json.dumps({"messages": self.messages})
    
    def to_alpaca(self) -> Dict:
        """Convert to Alpaca format"""
        # Extract instruction and response from messages
        instruction = ""
        input_text = ""
        output = ""
        
        for msg in self.messages:
            if msg['role'] == 'system':
                instruction = msg['content']
            elif msg['role'] == 'user':
                input_text = msg['content']
            elif msg['role'] == 'assistant':
                output = msg['content']
        
        return {
            'instruction': instruction,
            'input': input_text,
            'output': output
        }


class DataSynthesisModule:
    """
    Data Synthesis Module
    
    Filters successful trajectories and converts them to SFT training data.
    """
    
    def __init__(
        self,
        experience_library: ExperienceLibraryPostgres,
        min_reward: float = 0.5,
        min_confidence: float = 0.7,
        quality_threshold: float = 0.6
    ):
        """
        Initialize Data Synthesis Module
        
        Args:
            experience_library: Experience library instance
            min_reward: Minimum reward for success
            min_confidence: Minimum confidence for quality
            quality_threshold: Overall quality threshold
        """
        self.library = experience_library
        self.min_reward = min_reward
        self.min_confidence = min_confidence
        self.quality_threshold = quality_threshold
    
    def calculate_quality_score(self, trajectory: Trajectory) -> float:
        """
        Calculate quality score for a trajectory
        
        Quality = (reward * 0.5) + (confidence * 0.3) + (success * 0.2)
        
        Args:
            trajectory: Trajectory to score
        
        Returns:
            Quality score [0, 1]
        """
        reward_score = trajectory.reward if trajectory.reward else 0.0
        confidence_score = trajectory.confidence
        success_score = 1.0 if trajectory.success else 0.0
        
        quality = (
            reward_score * 0.5 +
            confidence_score * 0.3 +
            success_score * 0.2
        )
        
        return quality
    
    def filter_high_quality_trajectories(
        self,
        agent_type: Optional[str] = None,
        market_regime: Optional[str] = None,
        limit: int = 1000
    ) -> List[Tuple[Trajectory, float]]:
        """
        Filter high-quality trajectories for SFT training
        
        Args:
            agent_type: Filter by agent type
            market_regime: Filter by market regime
            limit: Maximum number of trajectories
        
        Returns:
            List of (trajectory, quality_score) tuples
        """
        # Retrieve successful trajectories
        trajectories = self.library.get_successful_trajectories(
            agent_type=agent_type,
            market_regime=market_regime,
            min_reward=self.min_reward,
            limit=limit * 2  # Get more, then filter
        )
        
        # Calculate quality scores
        scored_trajectories = []
        for traj in trajectories:
            # Skip low confidence
            if traj.confidence < self.min_confidence:
                continue
            
            quality = self.calculate_quality_score(traj)
            
            # Skip low quality
            if quality < self.quality_threshold:
                continue
            
            scored_trajectories.append((traj, quality))
        
        # Sort by quality (descending)
        scored_trajectories.sort(key=lambda x: x[1], reverse=True)
        
        # Limit
        scored_trajectories = scored_trajectories[:limit]
        
        logger.info(
            f"Filtered {len(scored_trajectories)} high-quality trajectories "
            f"from {len(trajectories)} successful ones"
        )
        
        return scored_trajectories
    
    def trajectory_to_sft_example(
        self,
        trajectory: Trajectory,
        agent_type: str
    ) -> SFTExample:
        """
        Convert trajectory to SFT training example
        
        Args:
            trajectory: Trajectory to convert
            agent_type: Agent type (news, technical, fundamental, strategist)
        
        Returns:
            SFT example in ChatML format
        """
        # System prompt (agent-specific)
        system_prompts = {
            'news': "You are a financial news sentiment analyst. Analyze news and provide sentiment scores.",
            'technical': "You are a technical analysis expert. Analyze charts and indicators to provide trading signals.",
            'fundamental': "You are a fundamental analysis expert. Analyze financial statements and provide valuation assessments.",
            'strategist': "You are a senior trading strategist. Synthesize multiple analyses and provide final trading recommendations."
        }
        
        system_prompt = system_prompts.get(agent_type, "You are a financial analyst.")
        
        # User prompt (input)
        user_prompt = self._format_user_prompt(trajectory, agent_type)
        
        # Assistant response (output)
        assistant_response = self._format_assistant_response(trajectory, agent_type)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
        
        metadata = {
            'trajectory_id': trajectory.trajectory_id,
            'symbol': trajectory.symbol,
            'agent_type': trajectory.agent_type,
            'reward': trajectory.reward,
            'confidence': trajectory.confidence,
            'quality_score': self.calculate_quality_score(trajectory)
        }
        
        return SFTExample(messages=messages, metadata=metadata)
    
    def _format_user_prompt(self, trajectory: Trajectory, agent_type: str) -> str:
        """Format user prompt from trajectory input"""
        if agent_type == 'news':
            return f"""Analyze the following market data and news for {trajectory.symbol}:

Market State: {json.dumps(trajectory.market_state, indent=2)}

News and Events: {json.dumps(trajectory.agent_inputs.get('news', []), indent=2)}

Provide sentiment score [-2, 2], confidence [0, 1], key events, and reasoning."""
        
        elif agent_type == 'technical':
            return f"""Analyze the following technical data for {trajectory.symbol}:

Market State: {json.dumps(trajectory.market_state, indent=2)}

Technical Indicators: {json.dumps(trajectory.agent_inputs.get('indicators', {}), indent=2)}

Provide signal (bullish/bearish/neutral), confidence [0, 1], and reasoning."""
        
        elif agent_type == 'fundamental':
            return f"""Analyze the following fundamental data for {trajectory.symbol}:

Market State: {json.dumps(trajectory.market_state, indent=2)}

Financial Metrics: {json.dumps(trajectory.agent_inputs.get('metrics', {}), indent=2)}

Provide valuation (undervalued/fairly_valued/overvalued), confidence [0, 1], and reasoning."""
        
        elif agent_type == 'strategist':
            return f"""Synthesize the following analyses for {trajectory.symbol}:

Market State: {json.dumps(trajectory.market_state, indent=2)}

Agent Analyses: {json.dumps(trajectory.agent_inputs.get('agent_outputs', {}), indent=2)}

Provide final recommendation (buy/sell/hold), confidence [0, 1], position size, stop loss, take profit, and reasoning."""
        
        else:
            return f"Analyze {trajectory.symbol}: {json.dumps(trajectory.market_state)}"
    
    def _format_assistant_response(self, trajectory: Trajectory, agent_type: str) -> str:
        """Format assistant response from trajectory output"""
        response = {
            'reasoning': trajectory.reasoning,
            'confidence': trajectory.confidence
        }
        
        if agent_type == 'news':
            response['sentiment_score'] = trajectory.agent_inputs.get('sentiment_score', 0.0)
            response['key_events'] = trajectory.agent_inputs.get('key_events', [])
        
        elif agent_type == 'technical':
            response['signal'] = trajectory.agent_inputs.get('signal', 'neutral')
            response['indicators'] = trajectory.agent_inputs.get('indicators', {})
        
        elif agent_type == 'fundamental':
            response['valuation'] = trajectory.agent_inputs.get('valuation', 'fairly_valued')
            response['key_metrics'] = trajectory.agent_inputs.get('key_metrics', {})
        
        elif agent_type == 'strategist':
            response['recommendation'] = trajectory.recommendation
            response['position_size'] = trajectory.position_size
            response['stop_loss'] = trajectory.stop_loss
            response['take_profit'] = trajectory.take_profit
        
        return json.dumps(response, indent=2)
    
    def synthesize_sft_dataset(
        self,
        agent_type: str,
        market_regime: Optional[str] = None,
        limit: int = 1000,
        output_format: str = 'chatml'
    ) -> List[Dict]:
        """
        Synthesize complete SFT dataset for an agent
        
        Args:
            agent_type: Agent type to synthesize for
            market_regime: Filter by market regime
            limit: Maximum number of examples
            output_format: 'chatml' or 'alpaca'
        
        Returns:
            List of SFT examples
        """
        logger.info(f"Synthesizing SFT dataset for {agent_type} agent...")
        
        # Filter high-quality trajectories
        scored_trajectories = self.filter_high_quality_trajectories(
            agent_type=agent_type,
            market_regime=market_regime,
            limit=limit
        )
        
        # Convert to SFT examples
        sft_examples = []
        for trajectory, quality in scored_trajectories:
            example = self.trajectory_to_sft_example(trajectory, agent_type)
            
            if output_format == 'chatml':
                sft_examples.append({
                    'messages': example.messages,
                    'metadata': example.metadata
                })
            elif output_format == 'alpaca':
                alpaca = example.to_alpaca()
                alpaca['metadata'] = example.metadata
                sft_examples.append(alpaca)
        
        logger.info(f"Synthesized {len(sft_examples)} SFT examples for {agent_type}")
        
        return sft_examples
    
    def save_sft_dataset(
        self,
        agent_type: str,
        output_path: str,
        market_regime: Optional[str] = None,
        limit: int = 1000,
        output_format: str = 'chatml'
    ):
        """
        Synthesize and save SFT dataset to file
        
        Args:
            agent_type: Agent type
            output_path: Output file path (.jsonl)
            market_regime: Filter by market regime
            limit: Maximum number of examples
            output_format: 'chatml' or 'alpaca'
        """
        # Synthesize dataset
        sft_examples = self.synthesize_sft_dataset(
            agent_type=agent_type,
            market_regime=market_regime,
            limit=limit,
            output_format=output_format
        )
        
        # Save to JSONL
        with open(output_path, 'w') as f:
            for example in sft_examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {len(sft_examples)} SFT examples to {output_path}")
    
    def get_synthesis_statistics(self) -> Dict:
        """
        Get statistics about synthesized data
        
        Returns:
            Dictionary with statistics
        """
        stats = self.library.get_statistics()
        
        # Add quality distribution
        all_trajectories = self.library.get_successful_trajectories(limit=10000)
        
        quality_scores = [
            self.calculate_quality_score(traj)
            for traj in all_trajectories
        ]
        
        if quality_scores:
            stats['quality'] = {
                'mean': sum(quality_scores) / len(quality_scores),
                'min': min(quality_scores),
                'max': max(quality_scores),
                'high_quality_count': sum(1 for q in quality_scores if q >= self.quality_threshold)
            }
        
        return stats


if __name__ == '__main__':
    # Test
    library = ExperienceLibraryPostgres(
        host='localhost',
        database='trading_experience',
        user='postgres',
        password='postgres'
    )
    
    synthesizer = DataSynthesisModule(
        experience_library=library,
        min_reward=0.5,
        min_confidence=0.7,
        quality_threshold=0.6
    )
    
    # Synthesize dataset for news agent
    synthesizer.save_sft_dataset(
        agent_type='news',
        output_path='sft_data_news.jsonl',
        limit=100,
        output_format='chatml'
    )
    
    # Get statistics
    stats = synthesizer.get_synthesis_statistics()
    print(f"Synthesis statistics: {json.dumps(stats, indent=2)}")
    
    library.close()
