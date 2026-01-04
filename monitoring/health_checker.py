"""
Health Checker

Monitors health of all system components and triggers alerts.

Health Checks:
1. Component availability
2. Database connectivity
3. API responsiveness
4. Memory/CPU usage
5. Error rates
"""

import time
import threading
from typing import Dict, List, Optional, Callable
from loguru import logger
from dataclasses import dataclass, asdict
import psutil


@dataclass
class HealthStatus:
    """Health status of a component"""
    component: str
    healthy: bool
    message: str
    last_check: float
    response_time: Optional[float] = None
    error_count: int = 0


class HealthChecker:
    """
    Health Checker
    
    Monitors system health and triggers alerts.
    """
    
    def __init__(
        self,
        check_interval: int = 60,  # seconds
        alert_threshold: int = 3  # consecutive failures
    ):
        """
        Initialize health checker
        
        Args:
            check_interval: Interval between health checks (seconds)
            alert_threshold: Number of consecutive failures before alert
        """
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        
        self.running = False
        self.thread = None
        
        self.health_status = {}
        self.failure_counts = {}
        self.alert_callbacks = []
    
    def start(self, components: Dict):
        """
        Start health checker
        
        Args:
            components: Dictionary of components to monitor
        """
        if self.running:
            logger.warning("Health checker already running")
            return
        
        self.running = True
        self.components = components
        
        # Start monitoring thread
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        
        logger.info("✅ Health checker started")
    
    def stop(self):
        """Stop health checker"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info("Health checker stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_all_components()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    def _check_all_components(self):
        """Check health of all components"""
        logger.debug("Running health checks...")
        
        for name, component in self.components.items():
            try:
                status = self._check_component(name, component)
                self.health_status[name] = status
                
                # Track failures
                if not status.healthy:
                    self.failure_counts[name] = self.failure_counts.get(name, 0) + 1
                    
                    # Trigger alert if threshold reached
                    if self.failure_counts[name] >= self.alert_threshold:
                        self._trigger_alert(status)
                else:
                    # Reset failure count on success
                    self.failure_counts[name] = 0
            
            except Exception as e:
                logger.error(f"Failed to check {name}: {e}")
                self.health_status[name] = HealthStatus(
                    component=name,
                    healthy=False,
                    message=f"Health check failed: {e}",
                    last_check=time.time()
                )
    
    def _check_component(self, name: str, component) -> HealthStatus:
        """
        Check health of a single component
        
        Args:
            name: Component name
            component: Component instance
        
        Returns:
            HealthStatus
        """
        start_time = time.time()
        
        try:
            # Component-specific health checks
            
            if name == 'experience_library':
                # Check database connectivity
                stats = component.get_statistics()
                healthy = stats is not None
                message = "Database connected" if healthy else "Database unreachable"
            
            elif name == 'streaming':
                # Check if streaming is running
                healthy = component.running if hasattr(component, 'running') else False
                message = "Streaming active" if healthy else "Streaming inactive"
            
            elif name == 'learning_loop':
                # Check if learning loop is running
                healthy = component.running if hasattr(component, 'running') else False
                message = "Learning loop active" if healthy else "Learning loop inactive"
            
            elif name == 'metrics':
                # Check if metrics server is running
                healthy = component.running if hasattr(component, 'running') else False
                message = "Metrics server active" if healthy else "Metrics server inactive"
            
            else:
                # Generic check
                healthy = component is not None
                message = "Component available" if healthy else "Component unavailable"
            
            response_time = time.time() - start_time
            
            return HealthStatus(
                component=name,
                healthy=healthy,
                message=message,
                last_check=time.time(),
                response_time=response_time
            )
        
        except Exception as e:
            return HealthStatus(
                component=name,
                healthy=False,
                message=f"Health check error: {e}",
                last_check=time.time(),
                error_count=1
            )
    
    def _trigger_alert(self, status: HealthStatus):
        """
        Trigger alert for unhealthy component
        
        Args:
            status: Health status
        """
        logger.error(
            f"🚨 ALERT: Component {status.component} unhealthy for "
            f"{self.failure_counts[status.component]} consecutive checks. "
            f"Message: {status.message}"
        )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[HealthStatus], None]):
        """
        Add alert callback
        
        Args:
            callback: Callback function
        """
        self.alert_callbacks.append(callback)
    
    def get_health_status(self) -> Dict:
        """
        Get current health status
        
        Returns:
            Dictionary of health statuses
        """
        return {
            name: asdict(status)
            for name, status in self.health_status.items()
        }
    
    def get_system_health(self) -> Dict:
        """
        Get overall system health
        
        Returns:
            System health dictionary
        """
        all_healthy = all(
            status.healthy
            for status in self.health_status.values()
        )
        
        unhealthy_components = [
            name for name, status in self.health_status.items()
            if not status.healthy
        ]
        
        # System resource usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'overall_healthy': all_healthy,
            'unhealthy_components': unhealthy_components,
            'component_count': len(self.health_status),
            'healthy_count': sum(1 for s in self.health_status.values() if s.healthy),
            'system_resources': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        }


if __name__ == '__main__':
    # Test
    
    # Mock components
    class MockComponent:
        def __init__(self, name):
            self.name = name
            self.running = True
        
        def get_statistics(self):
            return {'count': 100}
    
    components = {
        'experience_library': MockComponent('library'),
        'streaming': MockComponent('streaming'),
        'learning_loop': MockComponent('learning')
    }
    
    # Alert callback
    def alert_handler(status: HealthStatus):
        logger.warning(f"Alert received: {status.component} - {status.message}")
    
    checker = HealthChecker(check_interval=5, alert_threshold=2)
    checker.add_alert_callback(alert_handler)
    checker.start(components)
    
    logger.info("Health checker started. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(10)
            
            # Print health status
            health = checker.get_system_health()
            logger.info(f"System Health: {health}")
    
    except KeyboardInterrupt:
        logger.info("Stopping...")
        checker.stop()
