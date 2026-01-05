"""
Backtester Completion Detection - Task 1.1

Monitors backtest execution and triggers data synthesis pipeline upon completion.

Key Features:
- Detects when Backtester v2 completes execution
- Extracts backtest_id from results
- Triggers callback functions on completion
- Handles errors and incomplete backtests
- Logs completion events

Architecture:
1. BacktestCompletionMonitor wraps Backtester v2
2. Intercepts run() method completion
3. Validates results (success/failure)
4. Calls registered callbacks with backtest_id
5. Logs all events for audit trail

Integration:
- Wraps existing Backtester v2 (no modifications needed)
- Callback-based architecture (decoupled)
- Can register multiple callbacks
- Thread-safe for concurrent backtests

Phase A1 Week 3-4: Task 1.1 COMPLETE
"""

import time
from typing import Dict, List, Callable, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from loguru import logger


@dataclass
class BacktestCompletionEvent:
    """Event fired when backtest completes"""
    backtest_id: str
    status: str  # 'success' or 'failure'
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    metrics: Dict
    error: Optional[str] = None


class BacktestCompletionMonitor:
    """
    Monitors backtest execution and triggers callbacks on completion.
    
    Usage:
        monitor = BacktestCompletionMonitor()
        monitor.register_callback(my_callback_function)
        
        # Wrap backtester
        backtester = Backtester(config)
        results = monitor.run_with_monitoring(backtester, backtest_id="test_001")
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize monitor
        
        Args:
            log_dir: Directory to save completion logs (optional)
        """
        self.callbacks: List[Callable[[BacktestCompletionEvent], None]] = []
        self.log_dir = Path(log_dir) if log_dir else None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("BacktestCompletionMonitor initialized")
    
    def register_callback(self, callback: Callable[[BacktestCompletionEvent], None]):
        """
        Register a callback function to be called on backtest completion
        
        Args:
            callback: Function that takes BacktestCompletionEvent as argument
        """
        self.callbacks.append(callback)
        logger.info(f"Registered callback: {callback.__name__}")
    
    def run_with_monitoring(
        self,
        backtester,
        backtest_id: str
    ) -> Dict:
        """
        Run backtest with completion monitoring
        
        Args:
            backtester: Backtester v2 instance
            backtest_id: Unique identifier for this backtest
        
        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting monitored backtest: {backtest_id}")
        
        start_time = datetime.now()
        status = 'failure'
        metrics = {}
        error = None
        
        try:
            # Run backtest
            metrics = backtester.run()
            
            # Check if backtest succeeded
            if 'error' in metrics:
                status = 'failure'
                error = metrics['error']
                logger.error(f"Backtest {backtest_id} failed: {error}")
            else:
                status = 'success'
                logger.info(f"Backtest {backtest_id} completed successfully")
        
        except Exception as e:
            status = 'failure'
            error = str(e)
            logger.error(f"Backtest {backtest_id} crashed: {e}")
            metrics = {'error': error}
        
        finally:
            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create completion event
            event = BacktestCompletionEvent(
                backtest_id=backtest_id,
                status=status,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                metrics=metrics,
                error=error
            )
            
            # Log event
            self._log_event(event)
            
            # Trigger callbacks
            self._trigger_callbacks(event)
        
        return metrics
    
    def _log_event(self, event: BacktestCompletionEvent):
        """Log completion event to file"""
        if not self.log_dir:
            return
        
        log_file = self.log_dir / f"backtest_completion_{event.backtest_id}.json"
        
        event_dict = asdict(event)
        event_dict['start_time'] = event.start_time.isoformat()
        event_dict['end_time'] = event.end_time.isoformat()
        
        with open(log_file, 'w') as f:
            json.dump(event_dict, f, indent=2)
        
        logger.debug(f"Logged completion event to {log_file}")
    
    def _trigger_callbacks(self, event: BacktestCompletionEvent):
        """Trigger all registered callbacks"""
        if not self.callbacks:
            logger.debug("No callbacks registered")
            return
        
        logger.info(f"Triggering {len(self.callbacks)} callbacks for backtest {event.backtest_id}")
        
        for callback in self.callbacks:
            try:
                callback(event)
                logger.debug(f"Callback {callback.__name__} completed successfully")
            except Exception as e:
                logger.error(f"Callback {callback.__name__} failed: {e}")
    
    def get_completion_history(self) -> List[BacktestCompletionEvent]:
        """
        Get history of all backtest completions
        
        Returns:
            List of BacktestCompletionEvent
        """
        if not self.log_dir:
            return []
        
        events = []
        
        for log_file in self.log_dir.glob("backtest_completion_*.json"):
            try:
                with open(log_file, 'r') as f:
                    event_dict = json.load(f)
                
                # Convert ISO strings back to datetime
                event_dict['start_time'] = datetime.fromisoformat(event_dict['start_time'])
                event_dict['end_time'] = datetime.fromisoformat(event_dict['end_time'])
                
                events.append(BacktestCompletionEvent(**event_dict))
            except Exception as e:
                logger.warning(f"Failed to load event from {log_file}: {e}")
        
        # Sort by end time
        events.sort(key=lambda e: e.end_time, reverse=True)
        
        return events


# Example callback function
def example_callback(event: BacktestCompletionEvent):
    """Example callback that prints completion info"""
    print(f"\n{'='*60}")
    print(f"BACKTEST COMPLETED: {event.backtest_id}")
    print(f"{'='*60}")
    print(f"Status: {event.status}")
    print(f"Duration: {event.duration_seconds:.1f}s")
    
    if event.status == 'success':
        print(f"Total Return: {event.metrics.get('total_return', 'N/A')}")
        print(f"Sharpe Ratio: {event.metrics.get('sharpe_ratio', 'N/A')}")
    else:
        print(f"Error: {event.error}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    from training.rl.backtester_v2 import Backtester, BacktestConfig
    
    # Create monitor
    monitor = BacktestCompletionMonitor(log_dir="logs/backtest_completions")
    
    # Register callback
    monitor.register_callback(example_callback)
    
    # Create backtester
    config = BacktestConfig(
        symbols=['AAPL'],
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    backtester = Backtester(config, agent=None)
    
    # Run with monitoring
    results = monitor.run_with_monitoring(backtester, backtest_id="example_001")
    
    print(f"\nBacktest results: {results}")
