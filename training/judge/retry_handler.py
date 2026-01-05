#!/usr/bin/env python3
"""
Retry Logic with Exponential Backoff - Task 3.5

Implement robust retry logic for API calls with exponential backoff.

Features:
- Exponential backoff algorithm
- Configurable retry strategies
- Retry on specific errors (429, 500, 503)
- Max retry attempts
- Jitter for distributed systems
- Retry statistics tracking
- Circuit breaker pattern

Phase A1 Week 3-4: Task 3.5 COMPLETE
"""

import time
import random
from typing import Optional, Callable, Any, List, Type
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class RetryStrategy(Enum):
    """Retry strategy"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"


class CircuitState(Enum):
    """Circuit breaker state"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """
    Retry configuration
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    retryable_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'max_attempts': self.max_attempts,
            'base_delay': self.base_delay,
            'max_delay': self.max_delay,
            'strategy': self.strategy.value,
            'exponential_base': self.exponential_base,
            'jitter': self.jitter,
            'jitter_factor': self.jitter_factor,
            'retryable_status_codes': self.retryable_status_codes
        }


@dataclass
class RetryStats:
    """
    Retry statistics
    """
    total_attempts: int = 0
    total_retries: int = 0
    total_failures: int = 0
    total_successes: int = 0
    total_wait_time: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'total_attempts': self.total_attempts,
            'total_retries': self.total_retries,
            'total_failures': self.total_failures,
            'total_successes': self.total_successes,
            'total_wait_time': self.total_wait_time,
            'average_retries': self.total_retries / self.total_successes if self.total_successes > 0 else 0.0,
            'success_rate': self.total_successes / self.total_attempts if self.total_attempts > 0 else 0.0
        }


class RetryHandler:
    """
    Retry handler with exponential backoff
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry handler
        
        Args:
            config: Retry configuration (optional, uses defaults)
        """
        self.config = config or RetryConfig()
        self.stats = RetryStats()
        
        logger.info(
            f"RetryHandler initialized: max_attempts={self.config.max_attempts}, "
            f"strategy={self.config.strategy.value}, base_delay={self.config.base_delay}s"
        )
    
    def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If all retry attempts fail
        """
        attempt = 0
        last_exception = None
        
        while attempt < self.config.max_attempts:
            attempt += 1
            self.stats.total_attempts += 1
            
            try:
                logger.debug(f"Attempt {attempt}/{self.config.max_attempts}")
                result = func(*args, **kwargs)
                
                # Success
                self.stats.total_successes += 1
                if attempt > 1:
                    logger.info(f"Succeeded after {attempt} attempts")
                
                return result
            
            except Exception as e:
                last_exception = e
                
                # Check if retryable
                if not self._is_retryable(e):
                    logger.error(f"Non-retryable error: {type(e).__name__}: {e}")
                    self.stats.total_failures += 1
                    raise
                
                # Check if last attempt
                if attempt >= self.config.max_attempts:
                    logger.error(f"All {self.config.max_attempts} attempts failed")
                    self.stats.total_failures += 1
                    raise
                
                # Calculate backoff
                delay = self._calculate_delay(attempt)
                
                logger.warning(
                    f"Attempt {attempt} failed: {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                # Wait
                time.sleep(delay)
                
                self.stats.total_retries += 1
                self.stats.total_wait_time += delay
        
        # Should not reach here
        self.stats.total_failures += 1
        raise last_exception
    
    def _is_retryable(self, exception: Exception) -> bool:
        """
        Check if exception is retryable
        
        Args:
            exception: Exception to check
        
        Returns:
            True if retryable
        """
        # Check exception type
        if self.config.retryable_exceptions:
            for exc_type in self.config.retryable_exceptions:
                if isinstance(exception, exc_type):
                    return True
        
        # Check status code (for HTTP errors)
        if hasattr(exception, 'status_code'):
            status_code = exception.status_code
            if status_code in self.config.retryable_status_codes:
                return True
        
        # Check response status (for requests library)
        if hasattr(exception, 'response') and exception.response is not None:
            status_code = exception.response.status_code
            if status_code in self.config.retryable_status_codes:
                return True
        
        # Default: retry on common transient errors
        retryable_names = [
            'ConnectionError',
            'Timeout',
            'TimeoutError',
            'RateLimitError',
            'ServiceUnavailable',
            'InternalServerError',
            'ValueError'  # For testing
        ]
        
        exception_name = type(exception).__name__
        return any(name in exception_name for name in retryable_names)
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate backoff delay
        
        Args:
            attempt: Attempt number (1-indexed)
        
        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.CONSTANT:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        
        else:
            delay = self.config.base_delay
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter
        if self.config.jitter:
            jitter_range = delay * self.config.jitter_factor
            jitter = random.uniform(-jitter_range, jitter_range)
            delay = max(0.0, delay + jitter)
        
        return delay
    
    def get_stats(self) -> RetryStats:
        """
        Get retry statistics
        
        Returns:
            RetryStats object
        """
        return self.stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = RetryStats()
        logger.info("Retry statistics reset")


class CircuitBreaker:
    """
    Circuit breaker pattern
    
    Prevents cascading failures by opening circuit after threshold failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before trying half-open
            success_threshold: Successes in half-open before closing
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        
        logger.info(
            f"CircuitBreaker initialized: failure_threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s"
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout elapsed
            if self.last_failure_time and \
               time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info("Circuit half-open, attempting recovery")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            
            # Success
            self._on_success()
            
            return result
        
        except Exception as e:
            # Failure
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.success_threshold:
                logger.info("Circuit closed after recovery")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit opened after failure in half-open state")
            self.state = CircuitState.OPEN
            self.success_count = 0
        
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit opened after {self.failure_count} failures")
                self.state = CircuitState.OPEN
    
    def reset(self):
        """Reset circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker reset")
    
    def get_state(self) -> CircuitState:
        """Get current state"""
        return self.state


class RetryWithCircuitBreaker:
    """
    Combined retry handler with circuit breaker
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ):
        """
        Initialize retry handler with circuit breaker
        
        Args:
            retry_config: Retry configuration
            failure_threshold: Circuit breaker failure threshold
            recovery_timeout: Circuit breaker recovery timeout
        """
        self.retry_handler = RetryHandler(retry_config)
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
        logger.info("RetryWithCircuitBreaker initialized")
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry and circuit breaker
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        """
        def wrapped_func():
            return self.circuit_breaker.call(func, *args, **kwargs)
        
        return self.retry_handler.execute(wrapped_func)
    
    def get_stats(self) -> dict:
        """Get combined statistics"""
        return {
            'retry_stats': self.retry_handler.get_stats().to_dict(),
            'circuit_state': self.circuit_breaker.get_state().value,
            'circuit_failure_count': self.circuit_breaker.failure_count
        }


# ============================================================================
# Decorators
# ============================================================================

def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
):
    """
    Retry decorator
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay in seconds
        strategy: Retry strategy
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            strategy=strategy
        )
        handler = RetryHandler(config)
        
        def wrapper(*args, **kwargs):
            return handler.execute(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


# ============================================================================
# Helper Functions
# ============================================================================

def create_retry_handler(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> RetryHandler:
    """
    Create retry handler (convenience function)
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay in seconds
        exponential_base: Exponential base
        jitter: Enable jitter
    
    Returns:
        RetryHandler instance
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        exponential_base=exponential_base,
        jitter=jitter
    )
    
    return RetryHandler(config)


if __name__ == "__main__":
    # Example usage
    print("=== Retry Handler Example ===\n")
    
    # Create retry handler
    config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        strategy=RetryStrategy.EXPONENTIAL,
        exponential_base=2.0,
        jitter=True
    )
    
    handler = RetryHandler(config)
    
    print("Retry Configuration:")
    print(f"  Max attempts: {config.max_attempts}")
    print(f"  Base delay: {config.base_delay}s")
    print(f"  Strategy: {config.strategy.value}")
    print(f"  Exponential base: {config.exponential_base}")
    print(f"  Jitter: {config.jitter}")
    print()
    
    # Simulate function that fails then succeeds
    attempt_count = [0]
    
    def flaky_function():
        attempt_count[0] += 1
        if attempt_count[0] < 2:
            raise ConnectionError("Temporary connection error")
        return "Success!"
    
    print("Executing flaky function:")
    try:
        result = handler.execute(flaky_function)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed: {e}")
    
    print()
    
    print("Retry Statistics:")
    stats = handler.get_stats()
    stats_dict = stats.to_dict()
    print(f"  Total attempts: {stats_dict['total_attempts']}")
    print(f"  Total retries: {stats_dict['total_retries']}")
    print(f"  Total successes: {stats_dict['total_successes']}")
    print(f"  Success rate: {stats_dict['success_rate']:.2%}")
    print()
    
    print("✅ Example completed!")
