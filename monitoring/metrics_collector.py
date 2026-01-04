"""
Metrics Collector

Collects and exposes metrics for Prometheus monitoring.

Key Metrics:
1. Agent performance (accuracy, confidence, latency)
2. Training metrics (loss, reward, iterations)
3. System metrics (throughput, errors, uptime)
4. Business metrics (trades, returns, Sharpe ratio)
"""

from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
from typing import Dict, Optional
from loguru import logger
import time
import threading


class MetricsCollector:
    """
    Metrics Collector for Prometheus
    
    Collects and exposes metrics via HTTP endpoint.
    """
    
    def __init__(self, port: int = 9090):
        """
        Initialize metrics collector
        
        Args:
            port: Port to expose metrics on
        """
        self.port = port
        self.running = False
        
        # Define metrics
        
        # Agent metrics
        self.agent_requests = Counter(
            'agent_requests_total',
            'Total number of agent requests',
            ['agent_type']
        )
        
        self.agent_errors = Counter(
            'agent_errors_total',
            'Total number of agent errors',
            ['agent_type', 'error_type']
        )
        
        self.agent_latency = Histogram(
            'agent_latency_seconds',
            'Agent request latency',
            ['agent_type']
        )
        
        self.agent_confidence = Gauge(
            'agent_confidence',
            'Agent confidence score',
            ['agent_type']
        )
        
        # Judge metrics
        self.judge_evaluations = Counter(
            'judge_evaluations_total',
            'Total number of judge evaluations',
            ['agent_type']
        )
        
        self.judge_scores = Histogram(
            'judge_scores',
            'Judge evaluation scores',
            ['agent_type']
        )
        
        self.judge_disagreements = Counter(
            'judge_disagreements_total',
            'Total number of judge disagreements'
        )
        
        # Training metrics
        self.training_runs = Counter(
            'training_runs_total',
            'Total number of training runs',
            ['agent_type', 'training_type']  # training_type: sft, rl
        )
        
        self.training_loss = Gauge(
            'training_loss',
            'Current training loss',
            ['agent_type', 'training_type']
        )
        
        self.training_reward = Gauge(
            'training_reward',
            'Current training reward',
            ['agent_type']
        )
        
        # Experience Library metrics
        self.trajectories_stored = Counter(
            'trajectories_stored_total',
            'Total number of trajectories stored',
            ['agent_type']
        )
        
        self.trajectories_successful = Counter(
            'trajectories_successful_total',
            'Total number of successful trajectories',
            ['agent_type']
        )
        
        self.library_size = Gauge(
            'library_size',
            'Current size of experience library'
        )
        
        # Trading metrics
        self.trades_executed = Counter(
            'trades_executed_total',
            'Total number of trades executed',
            ['symbol', 'action']  # action: buy, sell, hold
        )
        
        self.trade_returns = Histogram(
            'trade_returns',
            'Trade returns',
            ['symbol']
        )
        
        self.portfolio_value = Gauge(
            'portfolio_value',
            'Current portfolio value'
        )
        
        self.sharpe_ratio = Gauge(
            'sharpe_ratio',
            'Current Sharpe ratio'
        )
        
        # System metrics
        self.system_uptime = Gauge(
            'system_uptime_seconds',
            'System uptime in seconds'
        )
        
        self.active_components = Gauge(
            'active_components',
            'Number of active components'
        )
        
        logger.info("Metrics collector initialized")
    
    def start(self):
        """Start metrics HTTP server"""
        if self.running:
            logger.warning("Metrics collector already running")
            return
        
        try:
            start_http_server(self.port)
            self.running = True
            logger.info(f"✅ Metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def stop(self):
        """Stop metrics collector"""
        self.running = False
        logger.info("Metrics collector stopped")
    
    # Agent metrics
    
    def record_agent_request(self, agent_type: str, latency: float, confidence: float):
        """Record agent request"""
        self.agent_requests.labels(agent_type=agent_type).inc()
        self.agent_latency.labels(agent_type=agent_type).observe(latency)
        self.agent_confidence.labels(agent_type=agent_type).set(confidence)
    
    def record_agent_error(self, agent_type: str, error_type: str):
        """Record agent error"""
        self.agent_errors.labels(agent_type=agent_type, error_type=error_type).inc()
    
    # Judge metrics
    
    def record_judge_evaluation(self, agent_type: str, score: float):
        """Record judge evaluation"""
        self.judge_evaluations.labels(agent_type=agent_type).inc()
        self.judge_scores.labels(agent_type=agent_type).observe(score)
    
    def record_judge_disagreement(self):
        """Record judge disagreement"""
        self.judge_disagreements.inc()
    
    # Training metrics
    
    def record_training_run(self, agent_type: str, training_type: str):
        """Record training run"""
        self.training_runs.labels(agent_type=agent_type, training_type=training_type).inc()
    
    def update_training_loss(self, agent_type: str, training_type: str, loss: float):
        """Update training loss"""
        self.training_loss.labels(agent_type=agent_type, training_type=training_type).set(loss)
    
    def update_training_reward(self, agent_type: str, reward: float):
        """Update training reward"""
        self.training_reward.labels(agent_type=agent_type).set(reward)
    
    # Experience Library metrics
    
    def record_trajectory_stored(self, agent_type: str, successful: bool):
        """Record trajectory stored"""
        self.trajectories_stored.labels(agent_type=agent_type).inc()
        if successful:
            self.trajectories_successful.labels(agent_type=agent_type).inc()
    
    def update_library_size(self, size: int):
        """Update library size"""
        self.library_size.set(size)
    
    # Trading metrics
    
    def record_trade_executed(self, symbol: str, action: str, trade_return: float):
        """Record trade executed"""
        self.trades_executed.labels(symbol=symbol, action=action).inc()
        self.trade_returns.labels(symbol=symbol).observe(trade_return)
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value"""
        self.portfolio_value.set(value)
    
    def update_sharpe_ratio(self, sharpe: float):
        """Update Sharpe ratio"""
        self.sharpe_ratio.set(sharpe)
    
    # System metrics
    
    def update_system_uptime(self, uptime: float):
        """Update system uptime"""
        self.system_uptime.set(uptime)
    
    def update_active_components(self, count: int):
        """Update active components count"""
        self.active_components.set(count)
    
    # Batch updates
    
    def update_from_status(self, status: Dict):
        """
        Update metrics from status dictionary
        
        Args:
            status: Status dictionary from orchestrator
        """
        # System uptime
        if 'uptime_seconds' in status:
            self.update_system_uptime(status['uptime_seconds'])
        
        # Active components
        if 'components' in status:
            active_count = sum(
                1 for v in status['components'].values()
                if v == 'running'
            )
            self.update_active_components(active_count)
        
        # Library size
        if 'experience_library' in status:
            lib_stats = status['experience_library']
            if 'overall' in lib_stats:
                self.update_library_size(lib_stats['overall'].get('count', 0))
        
        # Training metrics
        if 'learning_loop' in status:
            learning_status = status['learning_loop']
            if 'recent_trainings' in learning_status:
                for training in learning_status['recent_trainings']:
                    agent_type = training.get('agent_type', 'unknown')
                    training_type = training.get('type', 'unknown')
                    
                    if training.get('success'):
                        self.record_training_run(agent_type, training_type)
    
    def get_current_metrics(self) -> Dict:
        """
        Get current metric values
        
        Returns:
            Dictionary of current metrics
        """
        # This is a simplified version
        # In production, you'd query Prometheus or maintain internal state
        return {
            'running': self.running,
            'port': self.port
        }


if __name__ == '__main__':
    # Test
    collector = MetricsCollector(port=9090)
    collector.start()
    
    logger.info("Metrics collector started. Generating test metrics...")
    
    # Generate test metrics
    for i in range(10):
        collector.record_agent_request('news', latency=0.5, confidence=0.85)
        collector.record_agent_request('technical', latency=0.3, confidence=0.78)
        collector.record_judge_evaluation('news', score=85.0)
        collector.record_trade_executed('AAPL', 'buy', trade_return=0.05)
        
        time.sleep(1)
    
    logger.info(f"Metrics available at http://localhost:{collector.port}/metrics")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        collector.stop()
