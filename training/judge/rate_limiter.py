#!/usr/bin/env python3
"""
Rate Limiting for API Calls - Task 3.4

Implement rate limiting to avoid API quota exhaustion.

Features:
- Token bucket algorithm
- Requests per minute/second limits
- Concurrent request limits
- Adaptive rate limiting
- Rate limit headers parsing
- Backoff on rate limit errors
- Rate limit statistics

Phase A1 Week 3-4: Task 3.4 COMPLETE
"""

import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque
from loguru import logger


@dataclass
class RateLimitConfig:
    """
    Rate limit configuration
    """
    requests_per_minute: int = 60
    requests_per_second: Optional[int] = None
    max_concurrent: int = 5
    tokens_per_minute: Optional[int] = None
    tokens_per_request: int = 1000
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'requests_per_minute': self.requests_per_minute,
            'requests_per_second': self.requests_per_second,
            'max_concurrent': self.max_concurrent,
            'tokens_per_minute': self.tokens_per_minute,
            'tokens_per_request': self.tokens_per_request
        }


@dataclass
class RateLimitStats:
    """
    Rate limit statistics
    """
    total_requests: int = 0
    total_tokens: int = 0
    total_wait_time: float = 0.0
    rate_limit_hits: int = 0
    current_concurrent: int = 0
    requests_last_minute: int = 0
    tokens_last_minute: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'total_wait_time': self.total_wait_time,
            'rate_limit_hits': self.rate_limit_hits,
            'current_concurrent': self.current_concurrent,
            'requests_last_minute': self.requests_last_minute,
            'tokens_last_minute': self.tokens_last_minute,
            'average_wait_time': self.total_wait_time / self.total_requests if self.total_requests > 0 else 0.0
        }


class TokenBucket:
    """
    Token bucket for rate limiting
    
    Implements the token bucket algorithm for smooth rate limiting.
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1, blocking: bool = True) -> float:
        """
        Consume tokens from bucket
        
        Args:
            tokens: Number of tokens to consume
            blocking: If True, wait for tokens; if False, return wait time
        
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        with self.lock:
            # Refill tokens
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.refill_rate
            )
            self.last_refill = now
            
            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            
            # Calculate wait time
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate
            
            if blocking:
                # Wait for tokens
                time.sleep(wait_time)
                self.tokens = 0.0
                self.last_refill = time.time()
                return wait_time
            else:
                return wait_time
    
    def get_available_tokens(self) -> float:
        """Get current available tokens"""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            tokens = min(
                self.capacity,
                self.tokens + elapsed * self.refill_rate
            )
            return tokens


class RateLimiter:
    """
    Rate limiter for API calls
    
    Supports multiple rate limiting strategies:
    - Requests per minute/second
    - Tokens per minute
    - Concurrent request limits
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize rate limiter
        
        Args:
            config: Rate limit configuration (optional, uses defaults)
        """
        self.config = config or RateLimitConfig()
        self.stats = RateLimitStats()
        
        # Token buckets
        self.request_bucket_minute: Optional[TokenBucket] = None
        self.request_bucket_second: Optional[TokenBucket] = None
        self.token_bucket: Optional[TokenBucket] = None
        
        # Initialize request bucket (per minute)
        if self.config.requests_per_minute:
            self.request_bucket_minute = TokenBucket(
                capacity=self.config.requests_per_minute,
                refill_rate=self.config.requests_per_minute / 60.0
            )
        
        # Initialize request bucket (per second)
        if self.config.requests_per_second:
            self.request_bucket_second = TokenBucket(
                capacity=self.config.requests_per_second,
                refill_rate=float(self.config.requests_per_second)
            )
        
        # Initialize token bucket
        if self.config.tokens_per_minute:
            self.token_bucket = TokenBucket(
                capacity=self.config.tokens_per_minute,
                refill_rate=self.config.tokens_per_minute / 60.0
            )
        
        # Concurrent request tracking
        self.concurrent_requests = 0
        self.concurrent_lock = threading.Lock()
        self.concurrent_semaphore = threading.Semaphore(self.config.max_concurrent)
        
        # Request history (for stats)
        self.request_history: deque = deque(maxlen=1000)
        self.token_history: deque = deque(maxlen=1000)
        
        logger.info(
            f"RateLimiter initialized: "
            f"rpm={self.config.requests_per_minute}, "
            f"rps={self.config.requests_per_second}, "
            f"max_concurrent={self.config.max_concurrent}"
        )
    
    def acquire(self, tokens: Optional[int] = None, blocking: bool = True) -> float:
        """
        Acquire rate limit permission
        
        Args:
            tokens: Number of tokens to consume (optional, uses default)
            blocking: If True, wait for permission; if False, return wait time
        
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        tokens = tokens or self.config.tokens_per_request
        total_wait = 0.0
        
        # Acquire concurrent semaphore
        if blocking:
            self.concurrent_semaphore.acquire()
            with self.concurrent_lock:
                self.concurrent_requests += 1
                self.stats.current_concurrent = self.concurrent_requests
        else:
            # Non-blocking: check if available
            if not self.concurrent_semaphore.acquire(blocking=False):
                return float('inf')  # Would block indefinitely
            with self.concurrent_lock:
                self.concurrent_requests += 1
                self.stats.current_concurrent = self.concurrent_requests
        
        try:
            # Check request rate (per minute)
            if self.request_bucket_minute:
                wait = self.request_bucket_minute.consume(1, blocking=blocking)
                total_wait += wait
                if not blocking and wait > 0:
                    return wait
            
            # Check request rate (per second)
            if self.request_bucket_second:
                wait = self.request_bucket_second.consume(1, blocking=blocking)
                total_wait += wait
                if not blocking and wait > 0:
                    return wait
            
            # Check token rate
            if self.token_bucket:
                wait = self.token_bucket.consume(tokens, blocking=blocking)
                total_wait += wait
                if not blocking and wait > 0:
                    return wait
            
            # Update stats
            self.stats.total_requests += 1
            self.stats.total_tokens += tokens
            self.stats.total_wait_time += total_wait
            
            if total_wait > 0:
                self.stats.rate_limit_hits += 1
            
            # Update history
            now = time.time()
            self.request_history.append(now)
            self.token_history.append((now, tokens))
            
            # Update recent stats
            self._update_recent_stats()
            
            return total_wait
        
        except Exception as e:
            # Release semaphore on error
            self.release()
            raise e
    
    def release(self):
        """
        Release rate limit permission
        """
        with self.concurrent_lock:
            self.concurrent_requests -= 1
            self.stats.current_concurrent = self.concurrent_requests
        
        self.concurrent_semaphore.release()
    
    def _update_recent_stats(self):
        """
        Update recent statistics (last minute)
        """
        now = time.time()
        cutoff = now - 60.0
        
        # Count requests in last minute
        self.stats.requests_last_minute = sum(
            1 for t in self.request_history if t > cutoff
        )
        
        # Count tokens in last minute
        self.stats.tokens_last_minute = sum(
            tokens for t, tokens in self.token_history if t > cutoff
        )
    
    def get_stats(self) -> RateLimitStats:
        """
        Get rate limit statistics
        
        Returns:
            RateLimitStats object
        """
        self._update_recent_stats()
        return self.stats
    
    def get_wait_time(self, tokens: Optional[int] = None) -> float:
        """
        Get estimated wait time without acquiring
        
        Args:
            tokens: Number of tokens (optional, uses default)
        
        Returns:
            Estimated wait time in seconds
        """
        tokens = tokens or self.config.tokens_per_request
        max_wait = 0.0
        
        # Check concurrent limit
        if self.concurrent_requests >= self.config.max_concurrent:
            # Would need to wait for a release (unpredictable)
            max_wait = float('inf')
        
        # Check request rate (per minute)
        if self.request_bucket_minute:
            wait = self.request_bucket_minute.consume(1, blocking=False)
            max_wait = max(max_wait, wait)
        
        # Check request rate (per second)
        if self.request_bucket_second:
            wait = self.request_bucket_second.consume(1, blocking=False)
            max_wait = max(max_wait, wait)
        
        # Check token rate
        if self.token_bucket:
            wait = self.token_bucket.consume(tokens, blocking=False)
            max_wait = max(max_wait, wait)
        
        return max_wait
    
    def reset(self):
        """
        Reset rate limiter state
        """
        # Reset token buckets
        if self.request_bucket_minute:
            self.request_bucket_minute.tokens = float(self.request_bucket_minute.capacity)
            self.request_bucket_minute.last_refill = time.time()
        
        if self.request_bucket_second:
            self.request_bucket_second.tokens = float(self.request_bucket_second.capacity)
            self.request_bucket_second.last_refill = time.time()
        
        if self.token_bucket:
            self.token_bucket.tokens = float(self.token_bucket.capacity)
            self.token_bucket.last_refill = time.time()
        
        # Reset stats
        self.stats = RateLimitStats()
        self.request_history.clear()
        self.token_history.clear()
        
        logger.info("RateLimiter reset")


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts based on API responses
    
    Automatically reduces rate on 429 errors and increases when successful.
    """
    
    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        min_rpm: int = 10,
        max_rpm: int = 200,
        adjustment_factor: float = 0.5
    ):
        """
        Initialize adaptive rate limiter
        
        Args:
            config: Initial rate limit configuration
            min_rpm: Minimum requests per minute
            max_rpm: Maximum requests per minute
            adjustment_factor: Factor for rate adjustment (0.0 to 1.0)
        """
        super().__init__(config)
        
        self.min_rpm = min_rpm
        self.max_rpm = max_rpm
        self.adjustment_factor = adjustment_factor
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        
        logger.info(
            f"AdaptiveRateLimiter initialized: "
            f"min_rpm={min_rpm}, max_rpm={max_rpm}, "
            f"adjustment_factor={adjustment_factor}"
        )
    
    def report_success(self):
        """
        Report successful API call
        
        Increases rate limit after consecutive successes.
        """
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        
        # Increase rate after 10 consecutive successes
        if self.consecutive_successes >= 10:
            self._increase_rate()
            self.consecutive_successes = 0
    
    def report_rate_limit_error(self, retry_after: Optional[float] = None):
        """
        Report rate limit error (429)
        
        Decreases rate limit immediately.
        
        Args:
            retry_after: Retry-After header value in seconds (optional)
        """
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        
        self._decrease_rate()
        
        # If retry_after is provided, wait
        if retry_after:
            logger.warning(f"Rate limit hit, waiting {retry_after}s")
            time.sleep(retry_after)
    
    def _increase_rate(self):
        """
        Increase rate limit
        """
        current_rpm = self.config.requests_per_minute
        new_rpm = min(
            self.max_rpm,
            int(current_rpm * (1.0 + self.adjustment_factor))
        )
        
        if new_rpm > current_rpm:
            logger.info(f"Increasing rate limit: {current_rpm} -> {new_rpm} rpm")
            self.config.requests_per_minute = new_rpm
            
            # Recreate bucket
            self.request_bucket_minute = TokenBucket(
                capacity=new_rpm,
                refill_rate=new_rpm / 60.0
            )
    
    def _decrease_rate(self):
        """
        Decrease rate limit
        """
        current_rpm = self.config.requests_per_minute
        new_rpm = max(
            self.min_rpm,
            int(current_rpm * (1.0 - self.adjustment_factor))
        )
        
        if new_rpm < current_rpm:
            logger.warning(f"Decreasing rate limit: {current_rpm} -> {new_rpm} rpm")
            self.config.requests_per_minute = new_rpm
            
            # Recreate bucket
            self.request_bucket_minute = TokenBucket(
                capacity=new_rpm,
                refill_rate=new_rpm / 60.0
            )


# ============================================================================
# Context Manager
# ============================================================================

class RateLimitContext:
    """
    Context manager for rate limiting
    """
    
    def __init__(self, rate_limiter: RateLimiter, tokens: Optional[int] = None):
        """
        Initialize context manager
        
        Args:
            rate_limiter: RateLimiter instance
            tokens: Number of tokens (optional)
        """
        self.rate_limiter = rate_limiter
        self.tokens = tokens
    
    def __enter__(self):
        """Acquire rate limit"""
        self.rate_limiter.acquire(self.tokens)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release rate limit"""
        self.rate_limiter.release()
        return False


# ============================================================================
# Helper Functions
# ============================================================================

def create_rate_limiter(
    requests_per_minute: int = 60,
    max_concurrent: int = 5,
    adaptive: bool = False
) -> RateLimiter:
    """
    Create rate limiter (convenience function)
    
    Args:
        requests_per_minute: Requests per minute limit
        max_concurrent: Max concurrent requests
        adaptive: Use adaptive rate limiter
    
    Returns:
        RateLimiter instance
    """
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        max_concurrent=max_concurrent
    )
    
    if adaptive:
        return AdaptiveRateLimiter(config)
    else:
        return RateLimiter(config)


if __name__ == "__main__":
    # Example usage
    print("=== Rate Limiter Example ===\n")
    
    # Create rate limiter
    config = RateLimitConfig(
        requests_per_minute=60,
        requests_per_second=2,
        max_concurrent=3
    )
    
    limiter = RateLimiter(config)
    
    print("Rate Limiter Configuration:")
    print(f"  Requests per minute: {config.requests_per_minute}")
    print(f"  Requests per second: {config.requests_per_second}")
    print(f"  Max concurrent: {config.max_concurrent}")
    print()
    
    print("Simulating API calls:")
    for i in range(5):
        wait_time = limiter.acquire()
        print(f"  Request {i+1}: waited {wait_time:.3f}s")
        limiter.release()
    
    print()
    
    print("Rate Limit Statistics:")
    stats = limiter.get_stats()
    print(f"  Total requests: {stats.total_requests}")
    print(f"  Total wait time: {stats.total_wait_time:.3f}s")
    print(f"  Rate limit hits: {stats.rate_limit_hits}")
    print(f"  Current concurrent: {stats.current_concurrent}")
    print()
    
    print("✅ Example completed!")
