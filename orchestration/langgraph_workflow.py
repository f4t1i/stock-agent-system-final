"""
LangGraph Workflow - State-based orchestration for multi-agent stock analysis

This module implements a LangGraph-based workflow that orchestrates the multi-agent
stock analysis system with:
- State management across agent executions
- Conditional routing based on supervisor decisions
- Error handling and retry logic
- Trajectory logging for training
"""

from typing import Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import json
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from agents.junior.news_agent import NewsAgent
from agents.junior.technical_agent import TechnicalAgent
from agents.junior.fundamental_agent import FundamentalAgent
from agents.supervisor.supervisor_agent import SupervisorAgent
from agents.senior.strategist_agent import StrategistAgent

from utils.news_fetcher import NewsFetcher
from utils.market_data import MarketDataFetcher


# State Schema Definition
class AnalysisState(TypedDict):
    """
    State schema for the analysis workflow.
    
    This state is passed between nodes and accumulates information
    as the workflow progresses.
    """
    # Input
    symbol: str
    lookback_days: int
    use_supervisor: bool
    
    # Market Data
    market_data: Optional[Dict]
    
    # Supervisor Decision
    active_agents: Optional[List[str]]
    supervisor_reasoning: Optional[str]
    
    # Agent Outputs
    news_output: Optional[Dict]
    technical_output: Optional[Dict]
    fundamental_output: Optional[Dict]
    
    # Final Decision
    strategist_output: Optional[Dict]
    final_recommendation: Optional[str]
    
    # Portfolio State
    portfolio_state: Dict
    
    # Metadata
    timestamp: str
    errors: List[str]
    retry_count: int
    
    # Trajectory for training
    trajectory: List[Dict]


class StockAnalysisWorkflow:
    """
    LangGraph-based workflow for stock analysis.
    
    Implements a state-based orchestration with:
    - Conditional routing via supervisor
    - Parallel agent execution (optional)
    - Error handling and retries
    - Trajectory logging
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the workflow with configuration.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        logger.info("Initializing LangGraph Workflow")
        
        # Initialize utilities
        self.news_fetcher = NewsFetcher()
        self.market_data_fetcher = MarketDataFetcher()
        
        # Initialize agents
        self._initialize_agents()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        # Compile with checkpointing
        self.app = self.workflow.compile(
            checkpointer=MemorySaver()
        )
        
        logger.info("LangGraph Workflow initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all agents from configuration"""
        logger.info("Initializing agents for workflow...")
        
        self.agents = {}
        
        # News Agent
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
        
        # Technical Agent
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
        
        # Fundamental Agent
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
        
        # Supervisor Agent
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
        
        # Strategist Agent
        try:
            self.strategist = StrategistAgent(
                model_path=self.config['strategist']['model_path'],
                config=self.config['strategist']
            )
            logger.info("Strategist Agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Strategist: {e}")
            raise
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Workflow structure:
        1. fetch_market_data -> Fetch market data
        2. supervisor_routing -> Determine which agents to run (conditional)
        3. run_news_agent -> Run news analysis (conditional)
        4. run_technical_agent -> Run technical analysis (conditional)
        5. run_fundamental_agent -> Run fundamental analysis (conditional)
        6. run_strategist -> Make final decision
        7. log_trajectory -> Save trajectory for training
        """
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("fetch_market_data", self._fetch_market_data_node)
        workflow.add_node("supervisor_routing", self._supervisor_routing_node)
        workflow.add_node("run_news_agent", self._run_news_agent_node)
        workflow.add_node("run_technical_agent", self._run_technical_agent_node)
        workflow.add_node("run_fundamental_agent", self._run_fundamental_agent_node)
        workflow.add_node("run_strategist", self._run_strategist_node)
        workflow.add_node("log_trajectory", self._log_trajectory_node)
        
        # Define edges
        workflow.set_entry_point("fetch_market_data")
        
        # After fetching market data, decide routing
        workflow.add_conditional_edges(
            "fetch_market_data",
            self._should_use_supervisor,
            {
                "supervisor": "supervisor_routing",
                "all_agents": "run_news_agent"
            }
        )
        
        # After supervisor routing, go to first active agent
        workflow.add_conditional_edges(
            "supervisor_routing",
            self._route_to_first_agent,
            {
                "news": "run_news_agent",
                "technical": "run_technical_agent",
                "fundamental": "run_fundamental_agent",
                "strategist": "run_strategist"
            }
        )
        
        # After news agent, check if technical should run
        workflow.add_conditional_edges(
            "run_news_agent",
            self._should_run_technical,
            {
                "yes": "run_technical_agent",
                "no": "run_strategist"
            }
        )
        
        # After technical agent, check if fundamental should run
        workflow.add_conditional_edges(
            "run_technical_agent",
            self._should_run_fundamental,
            {
                "yes": "run_fundamental_agent",
                "no": "run_strategist"
            }
        )
        
        # After fundamental agent, go to strategist
        workflow.add_edge("run_fundamental_agent", "run_strategist")
        
        # After strategist, log trajectory
        workflow.add_edge("run_strategist", "log_trajectory")
        
        # After logging, end
        workflow.add_edge("log_trajectory", END)
        
        return workflow
    
    # ========== Node Functions ==========
    
    def _fetch_market_data_node(self, state: AnalysisState) -> AnalysisState:
        """Fetch market data for the symbol"""
        logger.info(f"Fetching market data for {state['symbol']}")
        
        try:
            market_data = self.market_data_fetcher.get_realtime(state['symbol'])
            
            if not market_data:
                state['errors'].append("Failed to fetch market data")
                market_data = {'error': 'No data available'}
            
            state['market_data'] = market_data
            state['trajectory'].append({
                'step': 'fetch_market_data',
                'timestamp': datetime.now().isoformat(),
                'success': 'error' not in market_data
            })
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            state['errors'].append(f"Market data error: {str(e)}")
            state['market_data'] = {'error': str(e)}
        
        return state
    
    def _supervisor_routing_node(self, state: AnalysisState) -> AnalysisState:
        """Use supervisor to determine which agents to activate"""
        logger.info("Running supervisor routing")
        
        try:
            if self.supervisor:
                # Create context for supervisor
                context = {
                    'symbol': state['symbol'],
                    'market_data': state['market_data'],
                    'portfolio_state': state['portfolio_state']
                }
                
                # Get routing decision
                routing_decision = self.supervisor.route(context)
                
                state['active_agents'] = routing_decision.get('active_agents', ['news', 'technical', 'fundamental'])
                state['supervisor_reasoning'] = routing_decision.get('reasoning', '')
                
                logger.info(f"Supervisor activated agents: {state['active_agents']}")
            else:
                # Default: all agents
                state['active_agents'] = ['news', 'technical', 'fundamental']
                state['supervisor_reasoning'] = 'No supervisor available, using all agents'
            
            state['trajectory'].append({
                'step': 'supervisor_routing',
                'timestamp': datetime.now().isoformat(),
                'active_agents': state['active_agents'],
                'reasoning': state['supervisor_reasoning']
            })
            
        except Exception as e:
            logger.error(f"Error in supervisor routing: {e}")
            state['errors'].append(f"Supervisor error: {str(e)}")
            state['active_agents'] = ['news', 'technical', 'fundamental']
        
        return state
    
    def _run_news_agent_node(self, state: AnalysisState) -> AnalysisState:
        """Run news sentiment analysis"""
        logger.info("Running News Agent")
        
        try:
            if self.agents.get('news'):
                output = self.agents['news'].analyze(
                    state['symbol'],
                    lookback_days=state['lookback_days']
                )
                state['news_output'] = output
                
                state['trajectory'].append({
                    'step': 'news_agent',
                    'timestamp': datetime.now().isoformat(),
                    'output': output
                })
            else:
                state['news_output'] = {'error': 'News agent not available'}
                state['errors'].append('News agent not available')
        
        except Exception as e:
            logger.error(f"Error in News Agent: {e}")
            state['errors'].append(f"News agent error: {str(e)}")
            state['news_output'] = {'error': str(e)}
        
        return state
    
    def _run_technical_agent_node(self, state: AnalysisState) -> AnalysisState:
        """Run technical analysis"""
        logger.info("Running Technical Agent")
        
        try:
            if self.agents.get('technical'):
                output = self.agents['technical'].analyze(
                    state['symbol'],
                    period="3mo"
                )
                state['technical_output'] = output
                
                state['trajectory'].append({
                    'step': 'technical_agent',
                    'timestamp': datetime.now().isoformat(),
                    'output': output
                })
            else:
                state['technical_output'] = {'error': 'Technical agent not available'}
                state['errors'].append('Technical agent not available')
        
        except Exception as e:
            logger.error(f"Error in Technical Agent: {e}")
            state['errors'].append(f"Technical agent error: {str(e)}")
            state['technical_output'] = {'error': str(e)}
        
        return state
    
    def _run_fundamental_agent_node(self, state: AnalysisState) -> AnalysisState:
        """Run fundamental analysis"""
        logger.info("Running Fundamental Agent")
        
        try:
            if self.agents.get('fundamental'):
                output = self.agents['fundamental'].analyze(
                    state['symbol'],
                    period="Q"
                )
                state['fundamental_output'] = output
                
                state['trajectory'].append({
                    'step': 'fundamental_agent',
                    'timestamp': datetime.now().isoformat(),
                    'output': output
                })
            else:
                state['fundamental_output'] = {'error': 'Fundamental agent not available'}
                state['errors'].append('Fundamental agent not available')
        
        except Exception as e:
            logger.error(f"Error in Fundamental Agent: {e}")
            state['errors'].append(f"Fundamental agent error: {str(e)}")
            state['fundamental_output'] = {'error': str(e)}
        
        return state
    
    def _run_strategist_node(self, state: AnalysisState) -> AnalysisState:
        """Run senior strategist for final decision"""
        logger.info("Running Senior Strategist")
        
        try:
            # Compile agent outputs
            agent_outputs = {}
            if state.get('news_output'):
                agent_outputs['news'] = state['news_output']
            if state.get('technical_output'):
                agent_outputs['technical'] = state['technical_output']
            if state.get('fundamental_output'):
                agent_outputs['fundamental'] = state['fundamental_output']
            
            # Run strategist
            strategist_output = self.strategist.analyze(
                state['symbol'],
                agent_outputs,
                state['portfolio_state'],
                state['market_data']
            )
            
            state['strategist_output'] = strategist_output
            state['final_recommendation'] = strategist_output.get('decision', 'hold')
            
            state['trajectory'].append({
                'step': 'strategist',
                'timestamp': datetime.now().isoformat(),
                'output': strategist_output
            })
            
        except Exception as e:
            logger.error(f"Error in Strategist: {e}")
            state['errors'].append(f"Strategist error: {str(e)}")
            state['strategist_output'] = {
                'decision': 'hold',
                'error': str(e)
            }
            state['final_recommendation'] = 'hold'
        
        return state
    
    def _log_trajectory_node(self, state: AnalysisState) -> AnalysisState:
        """Log the complete trajectory for training"""
        logger.info("Logging trajectory")
        
        try:
            # Create trajectory directory if needed
            trajectory_dir = Path("data/trajectories")
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            
            # Save trajectory
            trajectory_file = trajectory_dir / f"{state['symbol']}_{state['timestamp']}.json"
            
            trajectory_data = {
                'symbol': state['symbol'],
                'timestamp': state['timestamp'],
                'final_recommendation': state['final_recommendation'],
                'trajectory': state['trajectory'],
                'errors': state['errors']
            }
            
            with open(trajectory_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            
            logger.info(f"Trajectory saved to {trajectory_file}")
            
        except Exception as e:
            logger.error(f"Error logging trajectory: {e}")
            state['errors'].append(f"Trajectory logging error: {str(e)}")
        
        return state
    
    # ========== Conditional Edge Functions ==========
    
    def _should_use_supervisor(self, state: AnalysisState) -> str:
        """Determine if supervisor should be used for routing"""
        if state['use_supervisor'] and self.supervisor:
            return "supervisor"
        else:
            return "all_agents"
    
    def _route_to_first_agent(self, state: AnalysisState) -> str:
        """Route to the first active agent after supervisor"""
        active_agents = state.get('active_agents', [])
        
        if 'news' in active_agents:
            return "news"
        elif 'technical' in active_agents:
            return "technical"
        elif 'fundamental' in active_agents:
            return "fundamental"
        else:
            return "strategist"
    
    def _should_run_technical(self, state: AnalysisState) -> str:
        """Check if technical agent should run"""
        active_agents = state.get('active_agents', ['news', 'technical', 'fundamental'])
        
        if 'technical' in active_agents:
            return "yes"
        else:
            return "no"
    
    def _should_run_fundamental(self, state: AnalysisState) -> str:
        """Check if fundamental agent should run"""
        active_agents = state.get('active_agents', ['news', 'technical', 'fundamental'])
        
        if 'fundamental' in active_agents:
            return "yes"
        else:
            return "no"
    
    # ========== Public API ==========
    
    def analyze(
        self,
        symbol: str,
        portfolio_state: Optional[Dict] = None,
        use_supervisor: bool = False,
        lookback_days: int = 7
    ) -> Dict:
        """
        Run the complete analysis workflow for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            portfolio_state: Current portfolio state
            use_supervisor: Whether to use supervisor for routing
            lookback_days: Days to look back for news
        
        Returns:
            Dict with complete analysis results
        """
        logger.info(f"Starting workflow analysis for {symbol}")
        
        # Initialize state
        initial_state: AnalysisState = {
            'symbol': symbol,
            'lookback_days': lookback_days,
            'use_supervisor': use_supervisor,
            'market_data': None,
            'active_agents': None,
            'supervisor_reasoning': None,
            'news_output': None,
            'technical_output': None,
            'fundamental_output': None,
            'strategist_output': None,
            'final_recommendation': None,
            'portfolio_state': portfolio_state or {
                'cash': 100000,
                'positions': {},
                'total_value': 100000,
                'current_drawdown': 0.0
            },
            'timestamp': datetime.now().isoformat(),
            'errors': [],
            'retry_count': 0,
            'trajectory': []
        }
        
        # Run workflow
        try:
            # Execute the workflow
            result = self.app.invoke(
                initial_state,
                config={"configurable": {"thread_id": f"{symbol}_{datetime.now().timestamp()}"}}
            )
            
            # Compile final result
            final_result = {
                'symbol': result['symbol'],
                'recommendation': result['final_recommendation'],
                'confidence': result.get('strategist_output', {}).get('confidence', 0.0),
                'reasoning': result.get('strategist_output', {}).get('reasoning', ''),
                'position_size': result.get('strategist_output', {}).get('position_size', 0.0),
                'entry_target': result.get('strategist_output', {}).get('entry_target'),
                'stop_loss': result.get('strategist_output', {}).get('stop_loss'),
                'take_profit': result.get('strategist_output', {}).get('take_profit'),
                'risk_assessment': result.get('strategist_output', {}).get('risk_assessment', ''),
                'agent_outputs': {
                    'news': result.get('news_output'),
                    'technical': result.get('technical_output'),
                    'fundamental': result.get('fundamental_output')
                },
                'strategist_output': result.get('strategist_output'),
                'market_data': result.get('market_data'),
                'supervisor_routing': {
                    'active_agents': result.get('active_agents'),
                    'reasoning': result.get('supervisor_reasoning')
                },
                'timestamp': result['timestamp'],
                'errors': result['errors']
            }
            
            logger.info(f"Workflow complete: {final_result['recommendation']}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                'symbol': symbol,
                'recommendation': 'hold',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_analyze(
        self,
        symbols: List[str],
        portfolio_state: Optional[Dict] = None,
        use_supervisor: bool = False
    ) -> List[Dict]:
        """
        Analyze multiple symbols sequentially.
        
        Args:
            symbols: List of stock symbols
            portfolio_state: Current portfolio state
            use_supervisor: Whether to use supervisor routing
        
        Returns:
            List of analysis results
        """
        results = []
        
        for symbol in symbols:
            try:
                result = self.analyze(
                    symbol,
                    portfolio_state=portfolio_state,
                    use_supervisor=use_supervisor
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'error': str(e),
                    'recommendation': 'hold'
                })
        
        return results


# Factory function for easy instantiation
def create_workflow(config: Optional[Dict] = None) -> StockAnalysisWorkflow:
    """
    Create a StockAnalysisWorkflow instance with configuration.
    
    Args:
        config: System configuration. If None, uses default config.
    
    Returns:
        StockAnalysisWorkflow instance
    """
    if config is None:
        # Default configuration
        config = {
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
    
    return StockAnalysisWorkflow(config)
