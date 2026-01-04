"""
Senior Strategist Agent - Final Decision Maker mit RL-Optimierung
"""

import json
from typing import Dict, List, Optional
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ..base_agent import BaseAgent


class StrategistAgent(BaseAgent):
    """
    Senior Strategist Agent - Trifft finale Trading-Entscheidungen.

    Kombiniert Outputs von allen Junior-Agenten und trifft finale
    Buy/Sell/Hold Entscheidungen unter Berücksichtigung von:
    - Portfolio-State
    - Risiko-Management
    - Marktbedingungen
    - Agent-Outputs (News, Technical, Fundamental)

    Trainiert via:
    1. Initial SFT auf synthetischen Daten
    2. GRPO/PPO Reinforcement Learning
    """

    def __init__(self, model_path: str, config: Dict):
        super().__init__(config)
        self.model_path = model_path
        self.config = config

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if config.get('fp16', True) else torch.float32,
            device_map="auto"
        )

        if config.get('lora_adapter_path'):
            self.model = PeftModel.from_pretrained(
                self.model,
                config['lora_adapter_path']
            )

        self.model.eval()

        # Risk Management Config
        self.max_position_size = config.get('max_position_size', 0.10)  # 10% max
        self.max_drawdown = config.get('max_drawdown', 0.15)  # 15% max
        self.stop_loss_pct = config.get('stop_loss', 0.05)  # 5% stop loss

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """System-Prompt für Senior Strategist"""
        return """You are a senior portfolio strategist and risk manager. Your task is to:

1. Synthesize insights from multiple analysts (news, technical, fundamental)
2. Make final trading decisions (buy/sell/hold)
3. Manage portfolio risk and position sizing
4. Set appropriate stop-loss and take-profit levels
5. Consider market conditions and portfolio constraints

You receive:
- News sentiment analysis
- Technical analysis signals
- Fundamental analysis metrics
- Current portfolio state
- Market conditions

You must provide:
- Final decision (buy/sell/hold)
- Position size (if buy)
- Entry price target
- Stop loss level
- Take profit target
- Confidence level (0 to 1)
- Detailed reasoning
- Risk assessment

Always consider:
- Portfolio diversification
- Risk-adjusted returns
- Maximum drawdown limits
- Position size limits
- Market regime (bull/bear/sideways)

Output must be valid JSON following this schema:
{
    "decision": "buy" | "sell" | "hold",
    "position_size": float,
    "entry_target": float | null,
    "stop_loss": float | null,
    "take_profit": float | null,
    "confidence": float,
    "reasoning": string,
    "risk_assessment": string,
    "time_horizon": "short_term" | "medium_term" | "long_term",
    "expected_return": float | null,
    "risk_reward_ratio": float | null,
    "key_factors": [string]
}
"""

    def analyze(
        self,
        symbol: str,
        agent_outputs: Dict,
        portfolio_state: Optional[Dict] = None,
        market_data: Optional[Dict] = None
    ) -> Dict:
        """
        Finale Trading-Entscheidung basierend auf allen Inputs

        Args:
            symbol: Stock symbol
            agent_outputs: Dict mit Outputs von Junior-Agenten
                {
                    'news': {...},
                    'technical': {...},
                    'fundamental': {...}
                }
            portfolio_state: Aktueller Portfolio-Status
                {
                    'cash': float,
                    'positions': {symbol: shares},
                    'total_value': float,
                    'current_drawdown': float
                }
            market_data: Aktuelle Marktdaten
                {
                    'price': float,
                    'volume': int,
                    'volatility': float
                }

        Returns:
            Dict mit finaler Trading-Entscheidung
        """
        # Default portfolio state
        if portfolio_state is None:
            portfolio_state = {
                'cash': 100000,
                'positions': {},
                'total_value': 100000,
                'current_drawdown': 0.0
            }

        # Format input für LLM
        strategy_input = self._format_strategy_input(
            symbol,
            agent_outputs,
            portfolio_state,
            market_data
        )

        # Generate decision
        decision = self._generate_decision(symbol, strategy_input)

        # Apply risk management constraints
        decision = self._apply_risk_management(
            decision,
            portfolio_state,
            market_data
        )

        # Add metadata
        decision['metadata'] = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'model': self.model_path,
            'portfolio_value': portfolio_state['total_value']
        }

        return decision

    def _format_strategy_input(
        self,
        symbol: str,
        agent_outputs: Dict,
        portfolio_state: Dict,
        market_data: Optional[Dict]
    ) -> str:
        """Formatiere alle Inputs für den Strategist"""

        # News Analysis Summary
        news_summary = "NOT AVAILABLE"
        if 'news' in agent_outputs and agent_outputs['news']:
            news = agent_outputs['news']
            news_summary = f"""
Sentiment Score: {news.get('sentiment_score', 0):.2f}/2.0
Confidence: {news.get('confidence', 0):.2%}
Price Impact: {news.get('price_impact', 'neutral')}
Time Horizon: {news.get('time_horizon', 'N/A')}
Reasoning: {news.get('reasoning', 'N/A')[:200]}
"""

        # Technical Analysis Summary
        technical_summary = "NOT AVAILABLE"
        if 'technical' in agent_outputs and agent_outputs['technical']:
            tech = agent_outputs['technical']
            tech_summary = f"""
Trend: {tech.get('trend', 'N/A')}
Momentum: {tech.get('momentum', 'N/A')}
Signal: {tech.get('signal', 'N/A')}
Signal Strength: {tech.get('signal_strength', 0):.2f}
Current Price: ${tech.get('current_price', 0):.2f}
Support Levels: {tech.get('support_levels', [])}
Resistance Levels: {tech.get('resistance_levels', [])}
Reasoning: {tech.get('reasoning', 'N/A')[:200]}
"""

        # Fundamental Analysis Summary
        fundamental_summary = "NOT AVAILABLE"
        if 'fundamental' in agent_outputs and agent_outputs['fundamental']:
            fund = agent_outputs['fundamental']
            fundamental_summary = f"""
Valuation: {fund.get('valuation', 'N/A')}
Financial Health: {fund.get('financial_health', 0):.2f}/1.0
Growth Quality: {fund.get('growth_quality', 0):.2f}/1.0
Recommendation: {fund.get('recommendation', 'N/A')}
Confidence: {fund.get('confidence', 0):.2%}
Price Target: ${fund.get('price_target', 'N/A')}
Reasoning: {fund.get('reasoning', 'N/A')[:200]}
"""

        # Portfolio State Summary
        position_in_symbol = portfolio_state.get('positions', {}).get(symbol, 0)
        portfolio_summary = f"""
Total Portfolio Value: ${portfolio_state['total_value']:,.2f}
Available Cash: ${portfolio_state['cash']:,.2f}
Current Position in {symbol}: {position_in_symbol} shares
Current Drawdown: {portfolio_state['current_drawdown']:.2%}
Max Drawdown Limit: {self.max_drawdown:.2%}
"""

        # Market Data Summary
        market_summary = "NOT AVAILABLE"
        if market_data:
            market_summary = f"""
Current Price: ${market_data.get('price', 0):.2f}
Volume: {market_data.get('volume', 0):,}
Volatility: {market_data.get('volatility', 0):.2%}
"""

        # Combine all summaries
        full_input = f"""Stock: {symbol}

=== NEWS SENTIMENT ANALYSIS ===
{news_summary}

=== TECHNICAL ANALYSIS ===
{technical_summary}

=== FUNDAMENTAL ANALYSIS ===
{fundamental_summary}

=== PORTFOLIO STATE ===
{portfolio_summary}

=== MARKET DATA ===
{market_summary}

=== RISK CONSTRAINTS ===
- Max Position Size: {self.max_position_size:.1%} of portfolio
- Max Drawdown: {self.max_drawdown:.1%}
- Stop Loss: {self.stop_loss_pct:.1%} from entry

Based on all available information, provide your strategic trading decision in JSON format.
"""

        return full_input

    def _generate_decision(self, symbol: str, strategy_input: str) -> Dict:
        """Generate trading decision using LLM"""

        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': strategy_input}
        ]

        # Tokenize
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.config.get('max_new_tokens', 768),
                temperature=self.config.get('temperature', 0.7),
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Parse JSON
        try:
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]

            result = json.loads(response.strip())

            # Validate required fields
            required_fields = [
                'decision', 'position_size', 'confidence',
                'reasoning', 'risk_assessment'
            ]

            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Validate decision value
            if result['decision'] not in ['buy', 'sell', 'hold']:
                result['decision'] = 'hold'

            # Clamp values
            result['confidence'] = max(0.0, min(1.0, result['confidence']))
            result['position_size'] = max(0.0, min(1.0, result['position_size']))

            return result

        except Exception as e:
            # Fallback to conservative hold decision
            return {
                'decision': 'hold',
                'position_size': 0.0,
                'entry_target': None,
                'stop_loss': None,
                'take_profit': None,
                'confidence': 0.3,
                'reasoning': f'Error parsing response: {str(e)}. Defaulting to hold.',
                'risk_assessment': 'Unable to assess due to parsing error',
                'time_horizon': 'medium_term',
                'expected_return': None,
                'risk_reward_ratio': None,
                'key_factors': [],
                'error': str(e)
            }

    def _apply_risk_management(
        self,
        decision: Dict,
        portfolio_state: Dict,
        market_data: Optional[Dict]
    ) -> Dict:
        """Apply risk management constraints to decision"""

        # Check drawdown limit
        if portfolio_state['current_drawdown'] >= self.max_drawdown:
            decision['decision'] = 'hold'
            decision['reasoning'] += f"\n\nRISK OVERRIDE: Max drawdown limit ({self.max_drawdown:.1%}) reached."
            decision['position_size'] = 0.0
            return decision

        # Enforce position size limits
        if decision['decision'] == 'buy':
            # Limit position size to max_position_size of portfolio
            max_allowed_size = self.max_position_size
            decision['position_size'] = min(decision['position_size'], max_allowed_size)

            # Ensure we have enough cash
            if market_data and 'price' in market_data:
                price = market_data['price']
                max_affordable_pct = portfolio_state['cash'] / (portfolio_state['total_value'] * price)
                decision['position_size'] = min(decision['position_size'], max_affordable_pct)

            # Set stop loss if not provided
            if decision['stop_loss'] is None and market_data and 'price' in market_data:
                decision['stop_loss'] = market_data['price'] * (1 - self.stop_loss_pct)

        # Enforce minimum confidence threshold
        if decision['confidence'] < 0.5 and decision['decision'] in ['buy', 'sell']:
            decision['decision'] = 'hold'
            decision['reasoning'] += f"\n\nCONFIDENCE OVERRIDE: Confidence ({decision['confidence']:.1%}) below threshold (50%)."

        return decision

    def batch_analyze(
        self,
        symbols: List[str],
        agent_outputs_dict: Dict[str, Dict],
        portfolio_state: Optional[Dict] = None,
        market_data_dict: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Dict]:
        """Batch-Analyse für mehrere Symbole"""
        results = {}

        for symbol in symbols:
            try:
                agent_outputs = agent_outputs_dict.get(symbol, {})
                market_data = market_data_dict.get(symbol) if market_data_dict else None

                results[symbol] = self.analyze(
                    symbol,
                    agent_outputs,
                    portfolio_state,
                    market_data
                )
            except Exception as e:
                results[symbol] = {
                    'decision': 'hold',
                    'error': str(e),
                    'reasoning': f'Error during analysis: {str(e)}'
                }

        return results


if __name__ == "__main__":
    config = {
        'fp16': True,
        'max_new_tokens': 768,
        'temperature': 0.7,
        'max_position_size': 0.10,
        'max_drawdown': 0.15,
        'stop_loss': 0.05
    }

    agent = StrategistAgent(
        model_path="models/strategist_v1",
        config=config
    )

    # Example usage
    agent_outputs = {
        'news': {
            'sentiment_score': 1.5,
            'confidence': 0.85,
            'price_impact': 'bullish',
            'time_horizon': 'short_term',
            'reasoning': 'Strong positive sentiment from earnings beat'
        },
        'technical': {
            'trend': 'uptrend',
            'momentum': 'bullish',
            'signal': 'buy',
            'signal_strength': 0.78,
            'current_price': 175.50,
            'support_levels': [170.00, 165.00],
            'resistance_levels': [180.00, 185.00],
            'reasoning': 'Golden cross signal with strong momentum'
        },
        'fundamental': {
            'valuation': 'fairly_valued',
            'financial_health': 0.85,
            'growth_quality': 0.75,
            'recommendation': 'buy',
            'confidence': 0.80,
            'price_target': 190.00,
            'reasoning': 'Strong fundamentals with solid growth'
        }
    }

    portfolio_state = {
        'cash': 50000,
        'positions': {'AAPL': 100},
        'total_value': 100000,
        'current_drawdown': 0.05
    }

    market_data = {
        'price': 175.50,
        'volume': 50000000,
        'volatility': 0.25
    }

    result = agent.analyze("AAPL", agent_outputs, portfolio_state, market_data)
    print(json.dumps(result, indent=2))
