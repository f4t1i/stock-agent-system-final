"""
ChatML/Alpaca Dataset Conversion - Task 1.4

Converts trading trajectories to SFT dataset formats:
1. ChatML format (OpenAI/Anthropic style)
2. Alpaca format (Stanford Alpaca style)

ChatML Format:
```
[
  {"role": "system", "content": "You are a stock trading agent..."},
  {"role": "user", "content": "Analyze AAPL with data: ..."},
  {"role": "assistant", "content": "Based on analysis... BUY recommendation"}
]
```

Alpaca Format:
```
{
  "instruction": "Analyze the stock and provide trading recommendation",
  "input": "Symbol: AAPL, Price: $150, Volume: ...",
  "output": "Based on technical analysis... BUY with confidence 0.85"
}
```

Features:
- Template-based conversion
- Customizable system prompts
- Market state formatting
- Reasoning inclusion
- Metadata preservation

Phase A1 Week 3-4: Task 1.4 COMPLETE
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from loguru import logger


@dataclass
class ChatMLMessage:
    """Single message in ChatML format"""
    role: str  # system, user, assistant
    content: str
    
    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content}


@dataclass
class AlpacaExample:
    """Single example in Alpaca format"""
    instruction: str
    input: str
    output: str
    
    def to_dict(self) -> Dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }


class DatasetFormatter:
    """
    Converts trading trajectories to SFT dataset formats
    
    Usage:
        formatter = DatasetFormatter()
        
        # ChatML format
        chatml = formatter.to_chatml(
            trajectory=trajectory_dict,
            system_prompt="You are a stock trading agent..."
        )
        
        # Alpaca format
        alpaca = formatter.to_alpaca(
            trajectory=trajectory_dict,
            instruction="Analyze the stock and provide recommendation"
        )
    """
    
    def __init__(self):
        """Initialize formatter"""
        self.default_system_prompt = (
            "You are an expert stock trading agent. "
            "Analyze market data and provide clear, actionable trading recommendations "
            "with detailed reasoning."
        )
        
        self.default_instruction = (
            "Analyze the given stock data and provide a trading recommendation "
            "(BUY, SELL, or HOLD) with confidence score and detailed reasoning."
        )
        
        logger.info("DatasetFormatter initialized")
    
    def to_chatml(
        self,
        trajectory: Dict,
        system_prompt: Optional[str] = None,
        include_reasoning: bool = True
    ) -> List[Dict]:
        """
        Convert trajectory to ChatML format
        
        Args:
            trajectory: Trajectory dictionary with keys:
                - symbol: str
                - agent_type: str
                - market_state: Dict
                - agent_inputs: Dict
                - reasoning: str
                - recommendation: str
                - confidence: float
            system_prompt: Custom system prompt (uses default if None)
            include_reasoning: Whether to include reasoning in assistant response
        
        Returns:
            List of message dictionaries in ChatML format
        """
        messages = []
        
        # System message
        system_content = system_prompt or self.default_system_prompt
        messages.append(ChatMLMessage(
            role="system",
            content=system_content
        ).to_dict())
        
        # User message (market data + question)
        user_content = self._format_user_message(trajectory)
        messages.append(ChatMLMessage(
            role="user",
            content=user_content
        ).to_dict())
        
        # Assistant message (recommendation + reasoning)
        assistant_content = self._format_assistant_message(
            trajectory,
            include_reasoning=include_reasoning
        )
        messages.append(ChatMLMessage(
            role="assistant",
            content=assistant_content
        ).to_dict())
        
        return messages
    
    def to_alpaca(
        self,
        trajectory: Dict,
        instruction: Optional[str] = None,
        include_reasoning: bool = True
    ) -> Dict:
        """
        Convert trajectory to Alpaca format
        
        Args:
            trajectory: Trajectory dictionary
            instruction: Custom instruction (uses default if None)
            include_reasoning: Whether to include reasoning in output
        
        Returns:
            Dictionary in Alpaca format
        """
        # Instruction
        instruction_text = instruction or self.default_instruction
        
        # Input (market data)
        input_text = self._format_input(trajectory)
        
        # Output (recommendation + reasoning)
        output_text = self._format_output(
            trajectory,
            include_reasoning=include_reasoning
        )
        
        return AlpacaExample(
            instruction=instruction_text,
            input=input_text,
            output=output_text
        ).to_dict()
    
    def _format_user_message(self, trajectory: Dict) -> str:
        """Format user message for ChatML"""
        symbol = trajectory.get('symbol', 'UNKNOWN')
        agent_type = trajectory.get('agent_type', 'unknown')
        market_state = trajectory.get('market_state', {})
        agent_inputs = trajectory.get('agent_inputs', {})
        
        # Build user message
        parts = [
            f"Analyze {symbol} stock as a {agent_type} agent.",
            "",
            "Market State:"
        ]
        
        # Add market state
        for key, value in market_state.items():
            if isinstance(value, (int, float)):
                parts.append(f"  {key}: {value:.2f}")
            else:
                parts.append(f"  {key}: {value}")
        
        # Add agent-specific inputs
        if agent_inputs:
            parts.append("")
            parts.append("Additional Data:")
            for key, value in agent_inputs.items():
                if isinstance(value, str) and len(value) > 100:
                    # Truncate long strings
                    parts.append(f"  {key}: {value[:100]}...")
                elif isinstance(value, (int, float)):
                    parts.append(f"  {key}: {value:.2f}")
                else:
                    parts.append(f"  {key}: {value}")
        
        parts.append("")
        parts.append("Provide your trading recommendation with confidence score and reasoning.")
        
        return "\n".join(parts)
    
    def _format_assistant_message(
        self,
        trajectory: Dict,
        include_reasoning: bool = True
    ) -> str:
        """Format assistant message for ChatML"""
        recommendation = trajectory.get('recommendation', 'HOLD')
        confidence = trajectory.get('confidence', 0.0)
        reasoning = trajectory.get('reasoning', '')
        
        parts = [
            f"Recommendation: {recommendation}",
            f"Confidence: {confidence:.2f}"
        ]
        
        if include_reasoning and reasoning:
            parts.append("")
            parts.append("Reasoning:")
            parts.append(reasoning)
        
        return "\n".join(parts)
    
    def _format_input(self, trajectory: Dict) -> str:
        """Format input for Alpaca"""
        symbol = trajectory.get('symbol', 'UNKNOWN')
        market_state = trajectory.get('market_state', {})
        agent_inputs = trajectory.get('agent_inputs', {})
        
        parts = [f"Symbol: {symbol}"]
        
        # Add market state
        for key, value in market_state.items():
            if isinstance(value, (int, float)):
                parts.append(f"{key}: {value:.2f}")
            else:
                parts.append(f"{key}: {value}")
        
        # Add agent inputs (truncated)
        for key, value in agent_inputs.items():
            if isinstance(value, str) and len(value) > 50:
                parts.append(f"{key}: {value[:50]}...")
            elif isinstance(value, (int, float)):
                parts.append(f"{key}: {value:.2f}")
            else:
                parts.append(f"{key}: {value}")
        
        return ", ".join(parts)
    
    def _format_output(
        self,
        trajectory: Dict,
        include_reasoning: bool = True
    ) -> str:
        """Format output for Alpaca"""
        recommendation = trajectory.get('recommendation', 'HOLD')
        confidence = trajectory.get('confidence', 0.0)
        reasoning = trajectory.get('reasoning', '')
        
        parts = [
            f"{recommendation} (confidence: {confidence:.2f})"
        ]
        
        if include_reasoning and reasoning:
            parts.append(f"Reasoning: {reasoning}")
        
        return ". ".join(parts)
    
    def batch_convert_chatml(
        self,
        trajectories: List[Dict],
        system_prompt: Optional[str] = None,
        include_reasoning: bool = True
    ) -> List[List[Dict]]:
        """
        Convert multiple trajectories to ChatML format
        
        Args:
            trajectories: List of trajectory dictionaries
            system_prompt: Custom system prompt
            include_reasoning: Whether to include reasoning
        
        Returns:
            List of ChatML conversations
        """
        conversations = []
        
        for traj in trajectories:
            conv = self.to_chatml(
                trajectory=traj,
                system_prompt=system_prompt,
                include_reasoning=include_reasoning
            )
            conversations.append(conv)
        
        logger.info(f"Converted {len(conversations)} trajectories to ChatML")
        
        return conversations
    
    def batch_convert_alpaca(
        self,
        trajectories: List[Dict],
        instruction: Optional[str] = None,
        include_reasoning: bool = True
    ) -> List[Dict]:
        """
        Convert multiple trajectories to Alpaca format
        
        Args:
            trajectories: List of trajectory dictionaries
            instruction: Custom instruction
            include_reasoning: Whether to include reasoning
        
        Returns:
            List of Alpaca examples
        """
        examples = []
        
        for traj in trajectories:
            example = self.to_alpaca(
                trajectory=traj,
                instruction=instruction,
                include_reasoning=include_reasoning
            )
            examples.append(example)
        
        logger.info(f"Converted {len(examples)} trajectories to Alpaca")
        
        return examples
    
    def save_chatml(
        self,
        conversations: List[List[Dict]],
        output_path: str
    ):
        """
        Save ChatML conversations to JSONL file
        
        Args:
            conversations: List of ChatML conversations
            output_path: Output file path (.jsonl)
        """
        with open(output_path, 'w') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
        
        logger.info(f"Saved {len(conversations)} ChatML conversations to {output_path}")
    
    def save_alpaca(
        self,
        examples: List[Dict],
        output_path: str
    ):
        """
        Save Alpaca examples to JSON file
        
        Args:
            examples: List of Alpaca examples
            output_path: Output file path (.json)
        """
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)
        
        logger.info(f"Saved {len(examples)} Alpaca examples to {output_path}")


if __name__ == "__main__":
    # Example usage
    formatter = DatasetFormatter()
    
    # Sample trajectory
    trajectory = {
        'symbol': 'AAPL',
        'agent_type': 'technical',
        'market_state': {
            'price': 150.25,
            'volume': 1000000,
            'rsi': 65.3,
            'macd': 1.2
        },
        'agent_inputs': {
            'trend': 'bullish',
            'support': 145.0,
            'resistance': 155.0
        },
        'reasoning': 'Strong bullish momentum with RSI at 65.3 indicating strength without overbought conditions. MACD crossover confirms uptrend. Price broke above 50-day MA with increasing volume.',
        'recommendation': 'BUY',
        'confidence': 0.85
    }
    
    # ChatML format
    print("ChatML Format:")
    print("=" * 50)
    chatml = formatter.to_chatml(trajectory)
    print(json.dumps(chatml, indent=2))
    
    print("\n\nAlpaca Format:")
    print("=" * 50)
    alpaca = formatter.to_alpaca(trajectory)
    print(json.dumps(alpaca, indent=2))
