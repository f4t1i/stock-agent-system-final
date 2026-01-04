#!/usr/bin/env python3
"""
Generate Synthetic Training Data

Uses GPT-4o/Claude to generate synthetic training examples for SFT training.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from openai import OpenAI
from anthropic import Anthropic
from loguru import logger
from tqdm import tqdm


class SyntheticDataGenerator:
    """
    Generate synthetic training data using LLMs.
    
    Supports:
    - News sentiment analysis examples
    - Technical analysis examples
    - Fundamental analysis examples
    - Strategist decision examples
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o"
    ):
        """
        Initialize generator.
        
        Args:
            provider: LLM provider (openai or anthropic)
            model: Model name
        """
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif provider == "anthropic":
            self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        logger.info(f"Initialized {provider} with model {model}")
    
    def generate_examples(
        self,
        agent_type: str,
        num_examples: int,
        output_path: str
    ) -> List[Dict]:
        """
        Generate synthetic training examples.
        
        Args:
            agent_type: Agent type (news, technical, fundamental, strategist)
            num_examples: Number of examples to generate
            output_path: Path to save JSONL file
        
        Returns:
            List of generated examples
        """
        logger.info(f"Generating {num_examples} examples for {agent_type} agent")
        
        # Get generation prompt based on agent type
        generation_prompt = self._get_generation_prompt(agent_type)
        
        examples = []
        
        # Generate in batches
        batch_size = 5
        for i in tqdm(range(0, num_examples, batch_size)):
            batch_count = min(batch_size, num_examples - i)
            
            try:
                batch_examples = self._generate_batch(
                    generation_prompt,
                    batch_count,
                    agent_type
                )
                examples.extend(batch_examples)
            
            except Exception as e:
                logger.error(f"Error generating batch {i}: {e}")
        
        # Save to file
        self._save_examples(examples, output_path)
        
        logger.info(f"Generated {len(examples)} examples, saved to {output_path}")
        
        return examples
    
    def _get_generation_prompt(self, agent_type: str) -> str:
        """Get generation prompt for agent type"""
        
        prompts = {
            'news': """Generate realistic training examples for a financial news sentiment analysis agent.

Each example should include:
1. A stock symbol (e.g., AAPL, MSFT, TSLA)
2. 2-3 recent news headlines about the company
3. Expected sentiment analysis output with:
   - sentiment_score (-2 to +2)
   - confidence (0-1)
   - key_events (list of important events)
   - reasoning (explanation)
   - recommendation (bullish/bearish/neutral)

Format as ChatML with system, user, and assistant messages.
Make the examples diverse and realistic.""",

            'technical': """Generate realistic training examples for a technical analysis agent.

Each example should include:
1. A stock symbol
2. Technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. Identified patterns (if any)
4. Expected technical analysis output with:
   - signal (bullish/bearish/neutral)
   - signal_strength (0-1)
   - support_levels (list)
   - resistance_levels (list)
   - reasoning (explanation)
   - recommendation (buy/sell/hold)

Format as ChatML with system, user, and assistant messages.
Make the indicators and patterns realistic.""",

            'fundamental': """Generate realistic training examples for a fundamental analysis agent.

Each example should include:
1. A stock symbol
2. Financial metrics (P/E, ROE, ROA, Debt/Equity, etc.)
3. Expected fundamental analysis output with:
   - valuation (undervalued/fairly_valued/overvalued)
   - financial_health_score (0-1)
   - growth_score (0-1)
   - reasoning (explanation)
   - recommendation (buy/hold/sell)

Format as ChatML with system, user, and assistant messages.
Make the metrics realistic for different company types.""",

            'strategist': """Generate realistic training examples for a senior trading strategist.

Each example should include:
1. A stock symbol
2. Insights from news, technical, and fundamental agents
3. Portfolio state (cash, positions, etc.)
4. Expected strategist output with:
   - decision (buy/sell/hold)
   - confidence (0-1)
   - position_size (0-1)
   - entry_target (price)
   - stop_loss (price)
   - take_profit (price)
   - reasoning (comprehensive explanation)
   - risk_assessment (description)

Format as ChatML with system, user, and assistant messages.
Make the decisions well-reasoned and risk-aware."""
        }
        
        return prompts.get(agent_type, prompts['news'])
    
    def _generate_batch(
        self,
        prompt: str,
        count: int,
        agent_type: str
    ) -> List[Dict]:
        """Generate a batch of examples"""
        
        full_prompt = f"""{prompt}

Generate {count} diverse examples. Return as a JSON array of objects, each with a "messages" field containing the ChatML conversation.

Example format:
[
  {{
    "messages": [
      {{"role": "system", "content": "..."}},
      {{"role": "user", "content": "..."}},
      {{"role": "assistant", "content": "..."}}
    ]
  }},
  ...
]

Generate realistic, diverse examples now:"""
        
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.8,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
        
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.8,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            content = response.content[0].text
        
        # Parse JSON response
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            examples = json.loads(content)
            
            if not isinstance(examples, list):
                examples = [examples]
            
            return examples
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {content}")
            return []
    
    def _save_examples(self, examples: List[Dict], output_path: str):
        """Save examples to JSONL file"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for agents"
    )
    
    parser.add_argument(
        '--agent-type',
        type=str,
        required=True,
        choices=['news', 'technical', 'fundamental', 'strategist'],
        help='Type of agent to generate data for'
    )
    
    parser.add_argument(
        '--num-examples',
        type=int,
        default=100,
        help='Number of examples to generate'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSONL file path'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        default='openai',
        choices=['openai', 'anthropic'],
        help='LLM provider'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model name'
    )
    
    args = parser.parse_args()
    
    # Generate
    generator = SyntheticDataGenerator(
        provider=args.provider,
        model=args.model
    )
    
    generator.generate_examples(
        agent_type=args.agent_type,
        num_examples=args.num_examples,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
