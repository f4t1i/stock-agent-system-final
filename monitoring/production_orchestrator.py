"""
Production Orchestrator

Coordinates all components of the self-improving multi-agent system:
1. Data Pipeline (Experience Library, Real-time Streaming)
2. Learning Loop (SFT, RL, Online Learning)
3. Judge & Reward System
4. Agent Inference
5. Monitoring & Observability

This is the "control center" that brings everything together in production.
"""

import time
import threading
from typing import Dict, List, Optional
from loguru import logger
from dataclasses import dataclass, asdict
import json

from data_pipeline.experience_library_postgres import ExperienceLibraryPostgres
from data_pipeline.realtime_streaming import RealtimeDataPipeline
from training.pipelines.online_learning_loop import OnlineLearningLoop, OnlineLearningConfig
from monitoring.metrics_collector import MetricsCollector
from monitoring.health_checker import HealthChecker


@dataclass
class OrchestratorConfig:
    """Configuration for production orchestrator"""
    
    # Components to enable
    enable_streaming: bool = True
    enable_learning_loop: bool = True
    enable_monitoring: bool = True
    
    # Streaming
    finnhub_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    streaming_symbols: List[str] = None
    
    # Learning loop
    learning_check_interval_hours: int = 6
    
    # Monitoring
    metrics_port: int = 9090
    health_check_interval_seconds: int = 60
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "trading_experience"
    db_user: str = "postgres"
    db_password: str = "postgres"


class ProductionOrchestrator:
    """
    Production Orchestrator
    
    Coordinates all components of the self-improving system.
    """
    
    def __init__(self, config: OrchestratorConfig):
        """
        Initialize orchestrator
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config
        
        # State
        self.running = False
        self.components = {}
        self.status = {
            'started_at': None,
            'uptime_seconds': 0,
            'components': {}
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        logger.info("Initializing components...")
        
        # 1. Experience Library (always needed)
        try:
            self.components['experience_library'] = ExperienceLibraryPostgres(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
            self.status['components']['experience_library'] = 'initialized'
            logger.info("✅ Experience Library initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Experience Library: {e}")
            self.status['components']['experience_library'] = f'failed: {e}'
        
        # 2. Real-time Streaming (optional)
        if self.config.enable_streaming:
            try:
                self.components['streaming'] = RealtimeDataPipeline(
                    finnhub_api_key=self.config.finnhub_api_key,
                    polygon_api_key=self.config.polygon_api_key
                )
                self.status['components']['streaming'] = 'initialized'
                logger.info("✅ Real-time Streaming initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Streaming: {e}")
                self.status['components']['streaming'] = f'failed: {e}'
        
        # 3. Online Learning Loop (optional)
        if self.config.enable_learning_loop:
            try:
                learning_config = OnlineLearningConfig(
                    check_interval_hours=self.config.learning_check_interval_hours,
                    db_host=self.config.db_host,
                    db_port=self.config.db_port,
                    db_name=self.config.db_name,
                    db_user=self.config.db_user,
                    db_password=self.config.db_password
                )
                self.components['learning_loop'] = OnlineLearningLoop(learning_config)
                self.status['components']['learning_loop'] = 'initialized'
                logger.info("✅ Online Learning Loop initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Learning Loop: {e}")
                self.status['components']['learning_loop'] = f'failed: {e}'
        
        # 4. Metrics Collector (optional)
        if self.config.enable_monitoring:
            try:
                self.components['metrics'] = MetricsCollector(
                    port=self.config.metrics_port
                )
                self.status['components']['metrics'] = 'initialized'
                logger.info("✅ Metrics Collector initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Metrics: {e}")
                self.status['components']['metrics'] = f'failed: {e}'
        
        # 5. Health Checker (optional)
        if self.config.enable_monitoring:
            try:
                self.components['health_checker'] = HealthChecker(
                    check_interval=self.config.health_check_interval_seconds
                )
                self.status['components']['health_checker'] = 'initialized'
                logger.info("✅ Health Checker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Health Checker: {e}")
                self.status['components']['health_checker'] = f'failed: {e}'
    
    def start(self):
        """Start all components"""
        if self.running:
            logger.warning("Orchestrator already running")
            return
        
        logger.info("🚀 Starting Production Orchestrator...")
        
        self.running = True
        self.status['started_at'] = time.time()
        
        # Start components
        
        # 1. Start streaming
        if 'streaming' in self.components and self.config.streaming_symbols:
            try:
                import asyncio
                
                def start_streaming():
                    asyncio.run(
                        self.components['streaming'].start(
                            self.config.streaming_symbols
                        )
                    )
                
                streaming_thread = threading.Thread(
                    target=start_streaming,
                    daemon=True
                )
                streaming_thread.start()
                
                self.status['components']['streaming'] = 'running'
                logger.info("✅ Streaming started")
            except Exception as e:
                logger.error(f"Failed to start streaming: {e}")
                self.status['components']['streaming'] = f'failed: {e}'
        
        # 2. Start learning loop
        if 'learning_loop' in self.components:
            try:
                self.components['learning_loop'].start()
                self.status['components']['learning_loop'] = 'running'
                logger.info("✅ Learning Loop started")
            except Exception as e:
                logger.error(f"Failed to start learning loop: {e}")
                self.status['components']['learning_loop'] = f'failed: {e}'
        
        # 3. Start metrics collector
        if 'metrics' in self.components:
            try:
                self.components['metrics'].start()
                self.status['components']['metrics'] = 'running'
                logger.info("✅ Metrics Collector started")
            except Exception as e:
                logger.error(f"Failed to start metrics: {e}")
                self.status['components']['metrics'] = f'failed: {e}'
        
        # 4. Start health checker
        if 'health_checker' in self.components:
            try:
                self.components['health_checker'].start(self.components)
                self.status['components']['health_checker'] = 'running'
                logger.info("✅ Health Checker started")
            except Exception as e:
                logger.error(f"Failed to start health checker: {e}")
                self.status['components']['health_checker'] = f'failed: {e}'
        
        logger.info("✅ Production Orchestrator started successfully!")
    
    def stop(self):
        """Stop all components"""
        if not self.running:
            logger.warning("Orchestrator not running")
            return
        
        logger.info("Stopping Production Orchestrator...")
        
        self.running = False
        
        # Stop components in reverse order
        
        # 1. Stop health checker
        if 'health_checker' in self.components:
            try:
                self.components['health_checker'].stop()
                self.status['components']['health_checker'] = 'stopped'
                logger.info("Health Checker stopped")
            except Exception as e:
                logger.error(f"Failed to stop health checker: {e}")
        
        # 2. Stop metrics collector
        if 'metrics' in self.components:
            try:
                self.components['metrics'].stop()
                self.status['components']['metrics'] = 'stopped'
                logger.info("Metrics Collector stopped")
            except Exception as e:
                logger.error(f"Failed to stop metrics: {e}")
        
        # 3. Stop learning loop
        if 'learning_loop' in self.components:
            try:
                self.components['learning_loop'].stop()
                self.status['components']['learning_loop'] = 'stopped'
                logger.info("Learning Loop stopped")
            except Exception as e:
                logger.error(f"Failed to stop learning loop: {e}")
        
        # 4. Stop streaming
        if 'streaming' in self.components:
            try:
                import asyncio
                asyncio.run(self.components['streaming'].stop())
                self.status['components']['streaming'] = 'stopped'
                logger.info("Streaming stopped")
            except Exception as e:
                logger.error(f"Failed to stop streaming: {e}")
        
        # 5. Close experience library
        if 'experience_library' in self.components:
            try:
                self.components['experience_library'].close()
                self.status['components']['experience_library'] = 'closed'
                logger.info("Experience Library closed")
            except Exception as e:
                logger.error(f"Failed to close experience library: {e}")
        
        logger.info("✅ Production Orchestrator stopped")
    
    def get_status(self) -> Dict:
        """
        Get orchestrator status
        
        Returns:
            Status dictionary
        """
        if self.status['started_at']:
            self.status['uptime_seconds'] = int(time.time() - self.status['started_at'])
        
        # Get component-specific status
        if 'learning_loop' in self.components:
            try:
                learning_status = self.components['learning_loop'].get_status()
                self.status['learning_loop'] = learning_status
            except:
                pass
        
        if 'experience_library' in self.components:
            try:
                library_stats = self.components['experience_library'].get_statistics()
                self.status['experience_library'] = library_stats
            except:
                pass
        
        if 'metrics' in self.components:
            try:
                metrics = self.components['metrics'].get_current_metrics()
                self.status['metrics'] = metrics
            except:
                pass
        
        if 'health_checker' in self.components:
            try:
                health = self.components['health_checker'].get_health_status()
                self.status['health'] = health
            except:
                pass
        
        return self.status
    
    def get_component(self, name: str):
        """
        Get component by name
        
        Args:
            name: Component name
        
        Returns:
            Component instance or None
        """
        return self.components.get(name)


def start_production_orchestrator(
    config: Optional[OrchestratorConfig] = None
) -> ProductionOrchestrator:
    """
    Start production orchestrator
    
    Args:
        config: Orchestrator configuration
    
    Returns:
        ProductionOrchestrator instance
    """
    if config is None:
        config = OrchestratorConfig()
    
    orchestrator = ProductionOrchestrator(config)
    orchestrator.start()
    
    return orchestrator


if __name__ == '__main__':
    # Test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_streaming', action='store_true')
    parser.add_argument('--enable_learning_loop', action='store_true')
    parser.add_argument('--enable_monitoring', action='store_true')
    parser.add_argument('--streaming_symbols', type=str, nargs='+', default=['AAPL', 'MSFT', 'GOOGL'])
    
    args = parser.parse_args()
    
    config = OrchestratorConfig(
        enable_streaming=args.enable_streaming,
        enable_learning_loop=args.enable_learning_loop,
        enable_monitoring=args.enable_monitoring,
        streaming_symbols=args.streaming_symbols
    )
    
    orchestrator = start_production_orchestrator(config)
    
    logger.info("Production Orchestrator started. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(60)
            
            # Print status every 10 minutes
            if int(time.time()) % 600 == 0:
                status = orchestrator.get_status()
                logger.info(f"Status: {json.dumps(status, indent=2, default=str)}")
    
    except KeyboardInterrupt:
        logger.info("Stopping...")
        orchestrator.stop()
