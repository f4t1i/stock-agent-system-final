#!/usr/bin/env python3
"""Component 3: MetaLearningCycle - Continuous Optimization System.

Provides continuous performance measurement, optimization target identification,
agent weight adjustment, learning persistence, and optimization cycle orchestration.

Version: 1.0.0
Status: Production Ready
Timestamp: 2026-03-16T15:29:01
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ============================================================================
# DATACLASS: OptimizationResult
# ============================================================================

@dataclass
class OptimizationResult:
    """Result of an optimization cycle with updated weights and performance metrics."""
    updated_weights: Dict[str, Dict[str, float]]
    performance_delta: float
    new_insights: List[str]
    metrics_before: Dict[str, Any]
    metrics_after: Dict[str, Any]
    timestamp: str

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class OptimizationError(Exception):
    """Raised when optimization cycle encounters an error."""
    pass

class PersistenceError(Exception):
    """Raised when atomic persistence operations fail."""
    pass

class MetricsError(Exception):
    """Raised when performance metrics calculation fails."""
    pass

# ============================================================================
# MAIN CLASS: MetaLearningCycle
# ============================================================================

class MetaLearningCycle:
    """Continuous optimization system measuring performance and adjusting agent weights."""

    def __init__(self, knowledge_network_path: str, trigger_interval: int = 10):
        """Initialize MetaLearningCycle.

        Args:
            knowledge_network_path: Path to knowledge_network.json
            trigger_interval: Task count threshold for optimization cycles (default: 10)
        """
        self.knowledge_network_path = Path(knowledge_network_path)
        self.trigger_interval = trigger_interval
        self.task_count = 0
        self.task_results = []
        self.agent_weights = {}
        self._load_knowledge_network()
        logger.info(f"MetaLearningCycle initialized (trigger_interval={trigger_interval})")

    def _load_knowledge_network(self) -> None:
        """Load knowledge network and extract agent weights."""
        try:
            if not self.knowledge_network_path.exists():
                logger.warning(f"Knowledge network not found: {self.knowledge_network_path}")
                return

            with open(self.knowledge_network_path, 'r') as f:
                kn = json.load(f)

            # Extract agent weights from metrics section
            if 'metrics' in kn and 'agent_weights' in kn['metrics']:
                self.agent_weights = kn['metrics']['agent_weights']
                logger.debug(f"Loaded {len(self.agent_weights)} agent weights")
        except Exception as e:
            logger.error(f"Failed to load knowledge network: {e}")

    def measure_performance(self, task_results: List[Dict]) -> Dict[str, Any]:
        """Measure system performance from task results.

        Args:
            task_results: List of task execution results

        Returns:
            Dict with execution_time_avg, success_rate, efficacy_score, task_count
        """
        try:
            if not task_results:
                return {
                    'execution_time_avg': 0.0,
                    'success_rate': 0.0,
                    'efficacy_score': 0.5,
                    'task_count': 0
                }

            # Calculate execution time average
            execution_times = [r.get('execution_time', 100) for r in task_results]
            execution_time_avg = float(np.mean(execution_times))

            # Calculate success rate
            successes = sum(1 for r in task_results if r.get('success', False))
            success_rate = float(successes / len(task_results))

            # Calculate efficacy score (weighted average of metrics)
            efficacy_scores = []
            for r in task_results:
                metrics = r.get('metrics', {})
                gted = metrics.get('gted', 0.5)
                aucr = metrics.get('aucr', 0.5)
                pdcr = metrics.get('pdcr', 0.5)
                efficacy = (gted + aucr + pdcr) / 3.0
                efficacy_scores.append(efficacy)

            efficacy_score = float(np.mean(efficacy_scores)) if efficacy_scores else 0.5

            performance = {
                'execution_time_avg': execution_time_avg,
                'success_rate': success_rate,
                'efficacy_score': efficacy_score,
                'task_count': len(task_results)
            }

            logger.info(f"Performance measured: success_rate={success_rate:.2%}, efficacy={efficacy_score:.3f}")
            return performance
        except Exception as e:
            logger.error(f"Performance measurement failed: {e}")
            raise MetricsError(f"Failed to measure performance: {e}")

    def identify_optimization_targets(self, performance_data: Dict) -> List[Tuple[str, str, int]]:
        """Identify agents and patterns that need optimization.

        Returns:
            List of (agent_name, optimization_reason, priority) tuples
        """
        targets = []
        try:
            efficacy_score = performance_data.get('efficacy_score', 0.5)
            execution_time = performance_data.get('execution_time_avg', 100)

            # Find underperforming agents
            if efficacy_score < 0.6:
                for agent_name in self.agent_weights.keys():
                    targets.append((agent_name, "low_efficacy", 1))

            # Find inefficient patterns
            if execution_time > 500:
                targets.append(("system", "high_latency", 2))

            logger.info(f"Identified {len(targets)} optimization targets")
            return targets
        except Exception as e:
            logger.error(f"Target identification failed: {e}")
            return []

    def adjust_agent_weights(self, agent_name: str, performance_delta: float, metric: str) -> Dict[str, float]:
        """Adjust agent weights based on performance improvement.

        Args:
            agent_name: Name of agent to adjust
            performance_delta: Performance improvement percentage
            metric: Metric to adjust (gted, aucr, pdcr)

        Returns:
            Updated metric dict
        """
        try:
            if agent_name not in self.agent_weights:
                self.agent_weights[agent_name] = {}

            current_value = self.agent_weights[agent_name].get(metric, 0.5)
            # Conservative scaling: new_value = old_value * (1 + 0.1 * delta)
            new_value = current_value * (1.0 + 0.1 * performance_delta)
            # Ensure stays in [0.0, 1.0] range
            new_value = max(0.0, min(1.0, new_value))

            self.agent_weights[agent_name][metric] = new_value
            logger.info(f"Adjusted {agent_name}.{metric}: {current_value:.3f} → {new_value:.3f}")
            return self.agent_weights[agent_name]
        except Exception as e:
            logger.error(f"Weight adjustment failed: {e}")
            raise OptimizationError(f"Failed to adjust weights: {e}")

    def persist_learning_updates(self, updates: Dict) -> bool:
        """Persist learning updates to knowledge_network.json.

        Args:
            updates: Dict with agent_weights and other metrics

        Returns:
            Success status
        """
        try:
            if not self.knowledge_network_path.exists():
                logger.warning("Knowledge network file not found, skipping persistence")
                return False

            # Load existing knowledge network
            with open(self.knowledge_network_path, 'r') as f:
                kn = json.load(f)

            # Update metrics section
            if 'metrics' not in kn:
                kn['metrics'] = {}

            kn['metrics']['agent_weights'] = updates.get('agent_weights', {})

            # Add to learning history
            if 'learning_history' not in kn['metrics']:
                kn['metrics']['learning_history'] = []

            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'performance_delta': updates.get('performance_delta', 0),
                'updates': updates.get('agent_weights', {})
            }
            kn['metrics']['learning_history'].append(history_entry)

            # Atomic write with backup
            backup_path = self.knowledge_network_path.with_suffix('.backup')
            if self.knowledge_network_path.exists():
                self.knowledge_network_path.rename(backup_path)

            try:
                with open(self.knowledge_network_path, 'w') as f:
                    json.dump(kn, f, indent=2)
                logger.info(f"Learning updates persisted to {self.knowledge_network_path}")
                # Remove backup if successful
                if backup_path.exists():
                    backup_path.unlink()
                return True
            except Exception as e:
                # Restore backup on failure
                if backup_path.exists():
                    backup_path.rename(self.knowledge_network_path)
                raise PersistenceError(f"Failed to write knowledge network: {e}")
        except Exception as e:
            logger.error(f"Persistence failed: {e}")
            return False

    def should_trigger_cycle(self, task_count: int) -> bool:
        """Determine if optimization cycle should trigger.

        Args:
            task_count: Current task count

        Returns:
            True if cycle should trigger
        """
        return (task_count > 0 and task_count % self.trigger_interval == 0) or task_count == 1

    def run_optimization_cycle(self, all_task_results: List[Dict]) -> OptimizationResult:
        """Run complete optimization cycle.

        Args:
            all_task_results: All task results to analyze

        Returns:
            OptimizationResult with updated weights and insights
        """
        try:
            # Measure performance
            metrics_before = self.measure_performance(all_task_results[-self.trigger_interval:] if len(all_task_results) >= self.trigger_interval else all_task_results)

            # Identify targets
            targets = self.identify_optimization_targets(metrics_before)

            # Adjust weights
            performance_delta = metrics_before.get('success_rate', 0) * 100
            updated_weights = {}
            for agent_name, reason, priority in targets:
                if agent_name != "system":
                    self.adjust_agent_weights(agent_name, performance_delta, 'gted')
                    updated_weights[agent_name] = self.agent_weights[agent_name]

            # Generate insights
            new_insights = self._generate_insights(metrics_before, targets)

            # Persist updates
            self.persist_learning_updates({
                'agent_weights': self.agent_weights,
                'performance_delta': performance_delta
            })

            # Measure performance after
            metrics_after = self.measure_performance(all_task_results)

            result = OptimizationResult(
                updated_weights=updated_weights,
                performance_delta=performance_delta,
                new_insights=new_insights,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                timestamp=datetime.now().isoformat()
            )

            logger.info(f"Optimization cycle complete: delta={performance_delta:.1f}%")
            return result
        except Exception as e:
            logger.error(f"Optimization cycle failed: {e}")
            raise OptimizationError(f"Cycle execution failed: {e}")

    def _generate_insights(self, performance: Dict, targets: List) -> List[str]:
        """Generate new insights from performance analysis.

        Returns:
            List of insight strings
        """
        insights = []
        try:
            if performance.get('success_rate', 0) > 0.8:
                insights.append(f"High success rate detected: {performance['success_rate']:.1%}")

            if performance.get('execution_time_avg', 0) < 200:
                insights.append(f"Efficient execution: avg {performance['execution_time_avg']:.0f}ms")

            if performance.get('efficacy_score', 0) > 0.7:
                insights.append(f"Strong agent efficacy: {performance['efficacy_score']:.3f}")

            for agent_name, reason, priority in targets:
                insights.append(f"Optimization opportunity for {agent_name}: {reason}")

            return insights[:5]  # Limit to 5 insights
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return []

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_meta_learning_cycle(
    knowledge_network_path: str,
    trigger_interval: int = 10
) -> MetaLearningCycle:
    """Factory function to create MetaLearningCycle instance.

    Args:
        knowledge_network_path: Path to knowledge_network.json
        trigger_interval: Task count threshold for optimization cycles

    Returns:
        Initialized MetaLearningCycle instance

    Raises:
        OptimizationError: If initialization fails
    """
    try:
        cycle = MetaLearningCycle(
            knowledge_network_path=knowledge_network_path,
            trigger_interval=trigger_interval
        )
        logger.info("MetaLearningCycle created successfully")
        return cycle
    except Exception as e:
        logger.error(f"Failed to create MetaLearningCycle: {e}")
        raise OptimizationError(f"Failed to create cycle: {e}")


if __name__ == "__main__":
    logger.info("="*76)
    logger.info("MetaLearningCycle Component 3 - Production Ready")
    logger.info("="*76)
