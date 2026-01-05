#!/usr/bin/env python3
"""
Alert Evaluator - Evaluate Alert Conditions

Evaluates alert conditions against current market data and triggers
notifications when conditions are met.

Supported Conditions:
- above: Value is above threshold
- below: Value is below threshold
- crosses_above: Value crosses above threshold (was below, now above)
- crosses_below: Value crosses below threshold (was above, now below)
- equals: Value equals threshold (within tolerance)

Usage:
    evaluator = AlertEvaluator()
    triggered = evaluator.evaluate_condition(current_value, condition, threshold)
"""

from typing import Dict, Optional, Literal
from loguru import logger


ConditionType = Literal["above", "below", "crosses_above", "crosses_below", "equals"]


class AlertEvaluator:
    """Evaluate alert conditions"""
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize alert evaluator
        
        Args:
            tolerance: Tolerance for 'equals' condition (default: 1%)
        """
        self.tolerance = tolerance
        self.previous_values: Dict[str, float] = {}
    
    def evaluate_condition(
        self,
        current_value: float,
        condition: ConditionType,
        threshold: float,
        alert_id: Optional[str] = None
    ) -> bool:
        """
        Evaluate if condition is met
        
        Args:
            current_value: Current value to evaluate
            condition: Condition type
            threshold: Threshold value
            alert_id: Alert identifier (for tracking previous values)
            
        Returns:
            True if condition is met, False otherwise
        """
        try:
            if condition == "above":
                return current_value > threshold
            
            elif condition == "below":
                return current_value < threshold
            
            elif condition == "equals":
                # Within tolerance
                return abs(current_value - threshold) <= abs(threshold * self.tolerance)
            
            elif condition == "crosses_above":
                # Need previous value
                if alert_id and alert_id in self.previous_values:
                    prev_value = self.previous_values[alert_id]
                    result = prev_value <= threshold and current_value > threshold
                    self.previous_values[alert_id] = current_value
                    return result
                else:
                    # First evaluation - store current value
                    if alert_id:
                        self.previous_values[alert_id] = current_value
                    return False
            
            elif condition == "crosses_below":
                # Need previous value
                if alert_id and alert_id in self.previous_values:
                    prev_value = self.previous_values[alert_id]
                    result = prev_value >= threshold and current_value < threshold
                    self.previous_values[alert_id] = current_value
                    return result
                else:
                    # First evaluation - store current value
                    if alert_id:
                        self.previous_values[alert_id] = current_value
                    return False
            
            else:
                logger.warning(f"Unknown condition type: {condition}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    def clear_history(self, alert_id: str):
        """Clear previous value for an alert"""
        if alert_id in self.previous_values:
            del self.previous_values[alert_id]


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    evaluator = AlertEvaluator()
    
    print("=== Alert Evaluator Tests ===\n")
    
    # Test 1: Above
    print("Test 1: Above condition")
    print(f"  100 > 90: {evaluator.evaluate_condition(100, 'above', 90)}")  # True
    print(f"  80 > 90: {evaluator.evaluate_condition(80, 'above', 90)}")   # False
    
    # Test 2: Below
    print("\nTest 2: Below condition")
    print(f"  80 < 90: {evaluator.evaluate_condition(80, 'below', 90)}")   # True
    print(f"  100 < 90: {evaluator.evaluate_condition(100, 'below', 90)}") # False
    
    # Test 3: Equals
    print("\nTest 3: Equals condition (tolerance=1%)")
    print(f"  100 ≈ 100: {evaluator.evaluate_condition(100, 'equals', 100)}")    # True
    print(f"  100.5 ≈ 100: {evaluator.evaluate_condition(100.5, 'equals', 100)}") # True
    print(f"  102 ≈ 100: {evaluator.evaluate_condition(102, 'equals', 100)}")    # False
    
    # Test 4: Crosses above
    print("\nTest 4: Crosses above condition")
    alert_id = "test_alert_1"
    print(f"  First: 80 crosses 90: {evaluator.evaluate_condition(80, 'crosses_above', 90, alert_id)}")  # False (first)
    print(f"  Second: 85 crosses 90: {evaluator.evaluate_condition(85, 'crosses_above', 90, alert_id)}") # False (still below)
    print(f"  Third: 95 crosses 90: {evaluator.evaluate_condition(95, 'crosses_above', 90, alert_id)}")  # True (crossed!)
    print(f"  Fourth: 100 crosses 90: {evaluator.evaluate_condition(100, 'crosses_above', 90, alert_id)}") # False (already above)
    
    # Test 5: Crosses below
    print("\nTest 5: Crosses below condition")
    alert_id = "test_alert_2"
    print(f"  First: 100 crosses 90: {evaluator.evaluate_condition(100, 'crosses_below', 90, alert_id)}")  # False (first)
    print(f"  Second: 95 crosses 90: {evaluator.evaluate_condition(95, 'crosses_below', 90, alert_id)}")   # False (still above)
    print(f"  Third: 85 crosses 90: {evaluator.evaluate_condition(85, 'crosses_below', 90, alert_id)}")    # True (crossed!)
    print(f"  Fourth: 80 crosses 90: {evaluator.evaluate_condition(80, 'crosses_below', 90, alert_id)}")   # False (already below)
    
    print("\n✅ All tests complete")


# ============================================================================
# Specific Alert Type Evaluators
# ============================================================================

class PriceThresholdEvaluator:
    """Evaluate price threshold alerts"""
    
    def __init__(self):
        self.evaluator = AlertEvaluator()
    
    def evaluate(self, current_price: float, alert_config: dict, alert_id: str) -> bool:
        """
        Evaluate price threshold alert
        
        Args:
            current_price: Current stock price
            alert_config: Alert configuration (condition, threshold)
            alert_id: Alert identifier
            
        Returns:
            True if alert should trigger
        """
        condition = alert_config.get("condition", "above")
        threshold = alert_config.get("threshold", 0)
        
        return self.evaluator.evaluate_condition(
            current_price, condition, threshold, alert_id
        )


class ConfidenceChangeEvaluator:
    """Check confidence changes"""
    
    def __init__(self):
        self.previous_confidences: Dict[str, float] = {}
    
    def evaluate(self, current_confidence: float, alert_config: dict, alert_id: str) -> bool:
        """
        Evaluate confidence change alert
        
        Args:
            current_confidence: Current confidence score (0-1)
            alert_config: Alert configuration (min_change_threshold)
            alert_id: Alert identifier
            
        Returns:
            True if confidence changed significantly
        """
        min_change = alert_config.get("min_change_threshold", 0.1)  # 10% default
        
        if alert_id in self.previous_confidences:
            prev_conf = self.previous_confidences[alert_id]
            change = abs(current_confidence - prev_conf)
            
            self.previous_confidences[alert_id] = current_confidence
            return change >= min_change
        else:
            # First evaluation
            self.previous_confidences[alert_id] = current_confidence
            return False


class RecommendationChangeEvaluator:
    """Monitor recommendation changes"""
    
    def __init__(self):
        self.previous_recommendations: Dict[str, str] = {}
    
    def evaluate(self, current_recommendation: str, alert_config: dict, alert_id: str) -> bool:
        """
        Evaluate recommendation change alert
        
        Args:
            current_recommendation: Current recommendation (BUY/SELL/HOLD)
            alert_config: Alert configuration (optional target_recommendation)
            alert_id: Alert identifier
            
        Returns:
            True if recommendation changed
        """
        target_rec = alert_config.get("target_recommendation")  # Optional filter
        
        if alert_id in self.previous_recommendations:
            prev_rec = self.previous_recommendations[alert_id]
            changed = prev_rec != current_recommendation
            
            self.previous_recommendations[alert_id] = current_recommendation
            
            # If target specified, only trigger if changed TO target
            if target_rec:
                return changed and current_recommendation == target_rec
            return changed
        else:
            # First evaluation
            self.previous_recommendations[alert_id] = current_recommendation
            return False


class TechnicalSignalEvaluator:
    """Detect technical signals (RSI, MACD crossovers)"""
    
    def __init__(self):
        self.previous_indicators: Dict[str, dict] = {}
    
    def evaluate_rsi(self, rsi: float, alert_config: dict, alert_id: str) -> bool:
        """
        Evaluate RSI alert
        
        Args:
            rsi: Current RSI value (0-100)
            alert_config: Alert configuration (overbought_threshold, oversold_threshold)
            alert_id: Alert identifier
            
        Returns:
            True if RSI crosses overbought/oversold threshold
        """
        overbought = alert_config.get("overbought_threshold", 70)
        oversold = alert_config.get("oversold_threshold", 30)
        
        # Check overbought
        if rsi >= overbought:
            return True
        
        # Check oversold
        if rsi <= oversold:
            return True
        
        return False
    
    def evaluate_macd_crossover(self, macd: float, signal: float, alert_config: dict, alert_id: str) -> bool:
        """
        Evaluate MACD crossover alert
        
        Args:
            macd: Current MACD value
            signal: Current signal line value
            alert_config: Alert configuration
            alert_id: Alert identifier
            
        Returns:
            True if MACD crosses signal line
        """
        key = f"{alert_id}_macd"
        
        if key in self.previous_indicators:
            prev_macd = self.previous_indicators[key]["macd"]
            prev_signal = self.previous_indicators[key]["signal"]
            
            # Bullish crossover: MACD crosses above signal
            bullish = prev_macd <= prev_signal and macd > signal
            
            # Bearish crossover: MACD crosses below signal
            bearish = prev_macd >= prev_signal and macd < signal
            
            self.previous_indicators[key] = {"macd": macd, "signal": signal}
            
            return bullish or bearish
        else:
            # First evaluation
            self.previous_indicators[key] = {"macd": macd, "signal": signal}
            return False


# ============================================================================
# Background Task Scheduler
# ============================================================================

import asyncio
from typing import Callable, List

class AlertScheduler:
    """Background task scheduler for alert evaluation"""
    
    def __init__(self, evaluation_interval: int = 300):  # 5 minutes default
        """
        Initialize alert scheduler
        
        Args:
            evaluation_interval: Interval between evaluations in seconds
        """
        self.evaluation_interval = evaluation_interval
        self.is_running = False
        self.tasks: List[Callable] = []
    
    def add_task(self, task: Callable):
        """Add evaluation task"""
        self.tasks.append(task)
    
    async def run(self):
        """Run scheduler loop"""
        self.is_running = True
        logger.info(f"Alert scheduler started (interval: {self.evaluation_interval}s)")
        
        while self.is_running:
            try:
                logger.info("Evaluating alerts...")
                
                # Run all tasks
                for task in self.tasks:
                    try:
                        if asyncio.iscoroutinefunction(task):
                            await task()
                        else:
                            task()
                    except Exception as e:
                        logger.error(f"Error in task: {e}")
                
                logger.info(f"Alert evaluation complete. Next run in {self.evaluation_interval}s")
                
                # Wait for next interval
                await asyncio.sleep(self.evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    def stop(self):
        """Stop scheduler"""
        self.is_running = False
        logger.info("Alert scheduler stopped")
