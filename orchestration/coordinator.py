"""
System Coordinator - Orchestrates all agents and manages workflow
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from loguru import logger

from agents.junior.news_agent import NewsAgent
from agents.junior.technical_agent import TechnicalAgent
from agents.junior.fundamental_agent import FundamentalAgent
from agents.supervisor.supervisor_agent import SupervisorAgent
from agents.senior.strategist_agent import StrategistAgent

from utils.news_fetcher import NewsFetcher
from utils.market_data import MarketDataFetcher
from utils.config_loader import load_config


class SystemCoordinator:
    """
    System Coordinator - Hauptorchestrator für das Multi-Agenten-System.

    Verantwortlich für:
    - Agent-Initialisierung
    - Workflow-Koordination
    - State-Management
    - Error-Handling
    - Logging & Monitoring
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to system config YAML
        """
        # Load config
        if config_path:
            self.config = load_config(config_path)
        else:
            # Default config
            self.config = self._get_default_config()

        logger.info("Initializing System Coordinator")

        # Initialize utilities
        self.news_fetcher = NewsFetcher()
        self.market_data_fetcher = MarketDataFetcher()

        # Initialize agents
        self._initialize_agents()

        # Portfolio state
        self.portfolio_state = {
            'cash': 100000,  # Starting capital
            'positions': {},
            'total_value': 100000,
            'current_drawdown': 0.0
        }

        logger.info("System Coordinator initialized successfully")

    def _get_default_config(self) -> Dict:
        """Get default configuration if no config file provided"""
        return {
            'agents': {
                'news': {
                    'enabled': True,
                    'model_path': 'models/news_agent_v1',
                    'weight': 0.4,
                    'fp16': True,
                    'max_new_tokens': 512,
                    'temperature': 0.7
                },
                'technical': {
                    'enabled': True,
                    'model_path': 'models/technical_agent_v1',
                    'weight': 0.35,
                    'fp16': True,
                    'max_new_tokens': 512,
                    'temperature': 0.5
                },
                'fundamental': {
                    'enabled': True,
                    'model_path': 'models/fundamental_agent_v1',
                    'weight': 0.25,
                    'fp16': True,
                    'max_new_tokens': 512,
                    'temperature': 0.6
                }
            },
            'supervisor': {
                'enabled': True,
                'model_path': 'models/supervisor_v1',
                'exploration_factor': 0.5
            },
            'strategist': {
                'model_path': 'models/strategist_v1',
                'fp16': True,
                'max_new_tokens': 768,
                'temperature': 0.7,
                'max_position_size': 0.10,
                'max_drawdown': 0.15,
                'stop_loss': 0.05
            }
        }

    def _initialize_agents(self):
        """Initialize all agents"""
        logger.info("Initializing agents...")

        # Junior Agents
        self.agents = {}

        if self.config['agents']['news']['enabled']:
            try:
                self.agents['news'] = NewsAgent(
                    model_path=self.config['agents']['news']['model_path'],
                    config=self.config['agents']['news']
                )
                logger.info("News Agent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize News Agent: {e}")
                self.agents['news'] = None

        if self.config['agents']['technical']['enabled']:
            try:
                self.agents['technical'] = TechnicalAgent(
                    model_path=self.config['agents']['technical']['model_path'],
                    config=self.config['agents']['technical']
                )
                logger.info("Technical Agent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Technical Agent: {e}")
                self.agents['technical'] = None

        if self.config['agents']['fundamental']['enabled']:
            try:
                self.agents['fundamental'] = FundamentalAgent(
                    model_path=self.config['agents']['fundamental']['model_path'],
                    config=self.config['agents']['fundamental']
                )
                logger.info("Fundamental Agent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Fundamental Agent: {e}")
                self.agents['fundamental'] = None

        # Supervisor Agent (optional)
        if self.config.get('supervisor', {}).get('enabled', False):
            try:
                self.supervisor = SupervisorAgent(
                    model_path=self.config['supervisor']['model_path'],
                    config=self.config['supervisor']
                )
                logger.info("Supervisor Agent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Supervisor: {e}")
                self.supervisor = None
        else:
            self.supervisor = None

        # Senior Strategist
        try:
            self.strategist = StrategistAgent(
                model_path=self.config['strategist']['model_path'],
                config=self.config['strategist']
            )
            logger.info("Strategist Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Strategist: {e}")
            raise

    def analyze_symbol(
        self,
        symbol: str,
        use_supervisor: bool = False,
        lookback_days: int = 7
    ) -> Dict:
        """
        Analyse ein Symbol und treffe Trading-Entscheidung

        Args:
            symbol: Stock symbol (z.B. 'AAPL')
            use_supervisor: Nutze Supervisor für intelligentes Routing
            lookback_days: Tage für News-Recherche

        Returns:
            Dict mit finaler Analyse und Entscheidung
        """
        logger.info(f"Analyzing symbol: {symbol}")

        try:
            # Fetch market data
            market_data = self.market_data_fetcher.get_realtime(symbol)

            if not market_data:
                return {
                    'symbol': symbol,
                    'error': 'Failed to fetch market data',
                    'recommendation': 'hold'
                }

            # Determine which agents to activate
            if use_supervisor and self.supervisor:
                active_agents = self._supervisor_routing(symbol, market_data)
            else:
                # Use all enabled agents
                active_agents = [
                    name for name, agent in self.agents.items()
                    if agent is not None
                ]

            # Run junior agents
            agent_outputs = {}

            for agent_name in active_agents:
                logger.info(f"Running {agent_name} agent")

                try:
                    if agent_name == 'news' and self.agents['news']:
                        output = self.agents['news'].analyze(
                            symbol,
                            lookback_days=lookback_days
                        )
                        agent_outputs['news'] = output

                    elif agent_name == 'technical' and self.agents['technical']:
                        output = self.agents['technical'].analyze(
                            symbol,
                            period="3mo"
                        )
                        agent_outputs['technical'] = output

                    elif agent_name == 'fundamental' and self.agents['fundamental']:
                        output = self.agents['fundamental'].analyze(
                            symbol,
                            period="Q"
                        )
                        agent_outputs['fundamental'] = output

                except Exception as e:
                    logger.error(f"Error in {agent_name} agent: {e}")
                    agent_outputs[agent_name] = {'error': str(e)}

            # Run Senior Strategist
            logger.info("Running Senior Strategist")

            try:
                strategist_output = self.strategist.analyze(
                    symbol,
                    agent_outputs,
                    self.portfolio_state,
                    market_data
                )
            except Exception as e:
                logger.error(f"Error in Strategist: {e}")
                strategist_output = {
                    'decision': 'hold',
                    'error': str(e),
                    'reasoning': f'Strategist failed: {str(e)}'
                }

            # Compile final result
            result = {
                'symbol': symbol,
                'recommendation': strategist_output.get('decision', 'hold'),
                'confidence': strategist_output.get('confidence', 0.0),
                'reasoning': strategist_output.get('reasoning', ''),
                'position_size': strategist_output.get('position_size', 0.0),
                'entry_target': strategist_output.get('entry_target'),
                'stop_loss': strategist_output.get('stop_loss'),
                'take_profit': strategist_output.get('take_profit'),
                'risk_assessment': strategist_output.get('risk_assessment', ''),
                'agent_outputs': agent_outputs,
                'strategist_output': strategist_output,
                'market_data': market_data,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Analysis complete: {result['recommendation']}")

            return result

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'recommendation': 'hold',
                'confidence': 0.0
            }

    def _supervisor_routing(self, symbol: str, market_data: Dict) -> List[str]:
        """
        Use supervisor to determine which agents to activate

        Args:
            symbol: Stock symbol
            market_data: Current market data

        Returns:
            List of agent names to activate
        """
        if not self.supervisor:
            # Fallback to all agents
            return list(self.agents.keys())

        try:
            # Build context for supervisor
            context = {
                'market_regime': 'bull',  # TODO: Detect from market data
                'volatility': 20.0,  # TODO: Calculate from market data
                'query_type': 'mixed',
                'time_horizon': 'short',
                'information_density': 10,
                'trading_session': 'market_hours'
            }

            # Supervisor makes routing decision
            strategy_name, strategy_config, confidence = self.supervisor.select_routing_strategy(
                context,
                explore=True
            )

            logger.info(f"Supervisor selected: {strategy_name} (confidence: {confidence:.2f})")

            # Convert strategy config to agent list
            agents_to_activate = [
                agent_name for agent_name, enabled in strategy_config.items()
                if enabled and agent_name in self.agents
            ]

            return agents_to_activate

        except Exception as e:
            logger.error(f"Supervisor routing failed: {e}")
            # Fallback to all agents
            return list(self.agents.keys())

    def batch_analyze(
        self,
        symbols: List[str],
        use_supervisor: bool = False
    ) -> Dict[str, Dict]:
        """
        Batch-Analyse für mehrere Symbole

        Args:
            symbols: Liste von Symbolen
            use_supervisor: Nutze Supervisor

        Returns:
            Dict mapping symbol zu Analyse-Ergebnis
        """
        results = {}

        for symbol in symbols:
            logger.info(f"Batch analyzing: {symbol}")
            results[symbol] = self.analyze_symbol(symbol, use_supervisor)

        return results

    def execute_decision(
        self,
        symbol: str,
        decision: Dict,
        dry_run: bool = True
    ) -> Dict:
        """
        Execute trading decision (optional, for backtesting/paper trading)

        Args:
            symbol: Stock symbol
            decision: Decision dict from analyze_symbol
            dry_run: If True, don't execute real trades

        Returns:
            Execution result
        """
        if dry_run:
            logger.info(f"DRY RUN: {decision['recommendation']} {symbol}")

            return {
                'executed': False,
                'dry_run': True,
                'symbol': symbol,
                'action': decision['recommendation'],
                'message': 'Dry run - no real execution'
            }

        # TODO: Implement actual trade execution via broker API
        logger.warning("Real trade execution not implemented yet")

        return {
            'executed': False,
            'error': 'Trade execution not implemented'
        }

    def update_portfolio(self, trades: List[Dict]):
        """
        Update portfolio state based on executed trades

        Args:
            trades: List of executed trades
        """
        for trade in trades:
            symbol = trade['symbol']
            action = trade['action']
            quantity = trade['quantity']
            price = trade['price']

            if action == 'buy':
                # Add to positions
                self.portfolio_state['positions'][symbol] = (
                    self.portfolio_state['positions'].get(symbol, 0) + quantity
                )
                self.portfolio_state['cash'] -= quantity * price

            elif action == 'sell':
                # Remove from positions
                self.portfolio_state['positions'][symbol] = (
                    self.portfolio_state['positions'].get(symbol, 0) - quantity
                )
                self.portfolio_state['cash'] += quantity * price

        # Update total value
        self._update_portfolio_value()

    def _update_portfolio_value(self):
        """Update total portfolio value"""
        total_value = self.portfolio_state['cash']

        for symbol, quantity in self.portfolio_state['positions'].items():
            quote = self.market_data_fetcher.get_quote(symbol)
            if quote:
                total_value += quantity * quote['price']

        self.portfolio_state['total_value'] = total_value

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        self._update_portfolio_value()

        return {
            'cash': self.portfolio_state['cash'],
            'positions': self.portfolio_state['positions'],
            'total_value': self.portfolio_state['total_value'],
            'current_drawdown': self.portfolio_state['current_drawdown'],
            'num_positions': len(self.portfolio_state['positions'])
        }


if __name__ == "__main__":
    # Example usage
    coordinator = SystemCoordinator()

    # Analyze single symbol
    result = coordinator.analyze_symbol("AAPL")

    print("\n" + "="*80)
    print(f"Analysis for AAPL")
    print("="*80)
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nReasoning:")
    print(result['reasoning'])
    print("="*80 + "\n")

    # Portfolio summary
    portfolio = coordinator.get_portfolio_summary()
    print("Portfolio Summary:")
    print(json.dumps(portfolio, indent=2))
