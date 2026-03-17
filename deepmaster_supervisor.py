"""DeepMaster Meta-Orchestration Core Logic - Component 1: DeepMasterSupervisor.

This module implements intelligent task routing and metric computation for DeepALL's
multi-agent orchestration system. It uses knowledge network insights to compute
agent-specific metrics and route tasks based on causal potential rather than speed.

Author: DeepALL Meta-Orchestration System
Version: 1.0.0
Last Updated: 2026-03-16
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pathlib import Path

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# ============================================================================
# Custom Exceptions
# ============================================================================

class RoutingError(Exception):
    """Raised when task routing computation fails."""
    pass


class MetricsError(Exception):
    """Raised when metrics computation fails."""
    pass


class KnowledgeNetworkError(Exception):
    """Raised when knowledge network loading or parsing fails."""
    pass


# ============================================================================
# Enumerations
# ============================================================================

class RoutingAlgorithm(Enum):
    """Available routing algorithms for task assignment."""
    NEURAL_UCB = "neural_ucb"
    BAYESIAN = "bayesian"
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON_SAMPLING = "thompson_sampling"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RoutingDecision:
    """Represents a task routing decision.

    Attributes:
        agent_name: Name of the agent assigned to handle the task.
        confidence: Confidence score of the routing decision (0.0-1.0).
        algorithm: Routing algorithm used for decision (RoutingAlgorithm enum).
        context: Context information used in routing decision.
        timestamp: ISO format timestamp of decision creation.
        metrics: Dictionary of computed metrics influencing the decision.
    """
    agent_name: str
    confidence: float  # 0.0-1.0
    algorithm: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: "2026-03-16T15:29:01")
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate routing decision values."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
        if not self.agent_name:
            raise ValueError("agent_name cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert routing decision to dictionary.

        Returns:
            Dictionary representation of the routing decision.
        """
        return asdict(self)


# ============================================================================
# DeepMasterSupervisor Class
# ============================================================================

class DeepMasterSupervisor:
    """DeepMaster supervisor for intelligent task routing and metric computation.

    This class implements the core orchestration logic for routing tasks to
    appropriate agents based on computed metrics derived from the knowledge
    network and historical performance data.

    Attributes:
        knowledge_network_path: Path to knowledge_network.json file.
        knowledge_network: Loaded knowledge network data.
        phase_c_insights: Extracted phase C insights for calibration.
        regime_affinity_matrix: Agent performance across market regimes.
    """

    def __init__(self, knowledge_network_path: str = 
                 '/home/deepall/deepall_implementation/knowledge_network.json') -> None:
        """Initialize DeepMasterSupervisor with knowledge network.

        Args:
            knowledge_network_path: Path to knowledge_network.json.

        Raises:
            KnowledgeNetworkError: If knowledge network loading fails.
        """
        self.knowledge_network_path = knowledge_network_path
        self.knowledge_network: Dict[str, Any] = {}
        self.phase_c_insights: Dict[str, List[Dict[str, Any]]] = {}
        self.regime_affinity_matrix: Dict[str, Any] = {}
        self._agent_history: Dict[str, List[Dict[str, Any]]] = {}
        self._agent_cache: Dict[str, Dict[str, Any]] = {}  # Cache for agent data

        self._load_knowledge_network()
        self._build_agent_cache()
        logger.info("DeepMasterSupervisor initialized successfully")

    def _load_knowledge_network(self) -> None:
        """Load and parse knowledge network JSON file.

        Raises:
            KnowledgeNetworkError: If file loading or parsing fails.
        """
        try:
            path = Path(self.knowledge_network_path)
            if not path.exists():
                raise FileNotFoundError(f"Knowledge network file not found: {self.knowledge_network_path}")

            with open(path, 'r', encoding='utf-8') as f:
                self.knowledge_network = json.load(f)

            logger.debug(f"Loaded knowledge_network.json ({len(self.knowledge_network)} keys)")

            # Extract phase_c_insights
            if 'phase_c_insights' in self.knowledge_network:
                self.phase_c_insights = self.knowledge_network['phase_c_insights']
                meta_refl_count = len(self.phase_c_insights.get('meta_reflection', []))
                synth_count = len(self.phase_c_insights.get('synthesis', []))
                logger.info(f"Extracted phase_c_insights: {meta_refl_count} meta_reflection + {synth_count} synthesis")
            else:
                logger.warning("phase_c_insights not found in knowledge_network")

            # Extract regime_affinity_matrix
            if 'regime_affinity_matrix' in self.knowledge_network:
                self.regime_affinity_matrix = self.knowledge_network['regime_affinity_matrix']
                agent_count = len(self.regime_affinity_matrix.get('agents', []))
                logger.info(f"Extracted regime_affinity_matrix with {agent_count} agents")
            else:
                logger.warning("regime_affinity_matrix not found in knowledge_network")

        except FileNotFoundError as e:
            raise KnowledgeNetworkError(f"File not found: {e}") from e
        except json.JSONDecodeError as e:
            raise KnowledgeNetworkError(f"Invalid JSON format: {e}") from e
        except Exception as e:
            raise KnowledgeNetworkError(f"Unexpected error loading knowledge network: {e}") from e

    def _build_agent_cache(self) -> None:
        """Build cache mapping agent names to agent data for fast lookup."""
        agents_list = self.regime_affinity_matrix.get('agents', [])
        for agent_data in agents_list:
            agent_id = agent_data.get('agent_id')
            if agent_id:
                self._agent_cache[agent_id] = agent_data
        logger.debug(f"Built agent cache with {len(self._agent_cache)} agents")

    def _get_agent_data(self, agent_name: str) -> Dict[str, Any]:
        """Get agent data from cache.

        Args:
            agent_name: Name/ID of the agent.

        Returns:
            Agent data dictionary or empty dict if not found.
        """
        return self._agent_cache.get(agent_name, {})

    def compute_gted(self, agent_name: str, task_domain: str) -> float:
        """Compute Goal Task Execution Distance (GTED) metric.

        GTED measures the semantic distance between agent specialization and
        the target task domain. Lower values indicate better domain alignment.
        Returns normalized value: 0.0 (perfect alignment) to 1.0 (no alignment).

        Args:
            agent_name: Name of the agent.
            task_domain: Target task domain.

        Returns:
            GTED score as float in range [0.0, 1.0].

        Raises:
            MetricsError: If GTED computation fails.
        """
        try:
            agent_data = self._get_agent_data(agent_name)

            if not agent_data:
                logger.warning(f"No agent data found for {agent_name}, returning neutral GTED=0.5")
                return 0.5

            # Extract regime performance scores
            regime_scores = {
                'HIGH_VOLATILITY': agent_data.get('HIGH_VOLATILITY', 0.5),
                'BULL_TREND': agent_data.get('BULL_TREND', 0.5),
                'BEAR_TREND': agent_data.get('BEAR_TREND', 0.5),
                'SIDEWAYS': agent_data.get('SIDEWAYS', 0.5),
            }

            # Map domain to relevant regimes
            domain_regimes = self._map_domain_to_regimes(task_domain)

            if not domain_regimes:
                return 0.5

            # Average performance across relevant regimes
            relevant_scores = [regime_scores.get(regime, 0.5) for regime in domain_regimes]
            avg_performance = np.mean(relevant_scores) if relevant_scores else 0.5

            # Convert performance to distance (lower distance = better match)
            gted = max(0.0, 1.0 - avg_performance)
            gted = float(np.clip(gted, 0.0, 1.0))

            logger.debug(f"GTED({agent_name}, {task_domain}) = {gted:.4f}")
            return gted

        except Exception as e:
            raise MetricsError(f"Failed to compute GTED: {e}") from e

    def compute_aucr(self, agent_name: str, recent_tasks: Optional[List[Dict[str, Any]]] = None) -> float:
        """Compute Agent-specific Usage Confidence Ratio (AUCR) metric.

        AUCR reflects the consistency and reliability of agent performance
        based on recent task history. Ranges from 0.0 (unreliable) to 1.0 (highly reliable).

        Args:
            agent_name: Name of the agent.
            recent_tasks: Optional list of recent task results for the agent.

        Returns:
            AUCR score as float in range [0.0, 1.0].

        Raises:
            MetricsError: If AUCR computation fails.
        """
        try:
            if recent_tasks is None:
                recent_tasks = self._agent_history.get(agent_name, [])

            if not recent_tasks or len(recent_tasks) == 0:
                logger.debug(f"No recent tasks for {agent_name}, returning neutral AUCR=0.5")
                return 0.5

            # Extract success rates and quality scores
            success_indicator = np.array(
                [task.get('success', 0.5) for task in recent_tasks],
                dtype=np.float32
            )
            quality_scores = np.array(
                [task.get('quality_score', 0.5) for task in recent_tasks],
                dtype=np.float32
            )

            # Compute consistency (inverse of variance)
            consistency = max(0.0, 1.0 - np.std(success_indicator))
            reliability = float(np.mean(success_indicator))
            quality = float(np.mean(quality_scores))

            # Weighted combination: reliability (50%) + quality (30%) + consistency (20%)
            aucr = 0.5 * reliability + 0.3 * quality + 0.2 * consistency
            aucr = float(np.clip(aucr, 0.0, 1.0))

            logger.debug(f"AUCR({agent_name}) = {aucr:.4f} (reliability={reliability:.3f}, quality={quality:.3f})")
            return aucr

        except Exception as e:
            raise MetricsError(f"Failed to compute AUCR: {e}") from e

    def compute_pdcr(self, agent_name: str, decision: Optional[Dict[str, Any]] = None) -> float:
        """Compute Precision Distribution Confidence Ratio (PDCR) metric.

        PDCR measures decision precision - the probability that an agent's
        decision falls within acceptable error bounds. Target: < 0.05.
        Returns error rate in range [0.0, 1.0].

        Args:
            agent_name: Name of the agent.
            decision: Optional decision data for error computation.

        Returns:
            PDCR error rate as float (lower is better, target < 0.05).

        Raises:
            MetricsError: If PDCR computation fails.
        """
        try:
            # Base error rate - derived from regime performance variance
            agent_data = self._get_agent_data(agent_name)
            regime_scores = [
                agent_data.get('HIGH_VOLATILITY', 0.5),
                agent_data.get('BULL_TREND', 0.5),
                agent_data.get('BEAR_TREND', 0.5),
                agent_data.get('SIDEWAYS', 0.5),
            ]
            regime_variance = float(np.std(regime_scores))
            base_error = min(0.1, regime_variance)  # Cap at 0.1

            if decision is None:
                logger.debug(f"PDCR({agent_name}) = {base_error:.4f} (no decision data)")
                return float(np.clip(base_error, 0.0, 1.0))

            # Compute decision-specific error
            expected_value = decision.get('expected_value', 0.0)
            actual_value = decision.get('actual_value', 0.0)
            variance = decision.get('variance', 0.1)

            if variance == 0:
                variance = 0.1

            # Normalized error as z-score contribution
            deviation = abs(actual_value - expected_value)
            z_score = deviation / variance if variance > 0 else deviation
            decision_error = min(1.0, float(2.0 / (1.0 + np.exp(-z_score)))) - 1.0
            decision_error = max(0.0, decision_error)

            # Combine base and decision-specific error
            pdcr = 0.7 * base_error + 0.3 * decision_error
            pdcr = float(np.clip(pdcr, 0.0, 1.0))

            logger.debug(f"PDCR({agent_name}) = {pdcr:.4f} (target < 0.05)")
            return pdcr

        except Exception as e:
            raise MetricsError(f"Failed to compute PDCR: {e}") from e

    def analyze_cognitive_friction(self, agent_name: str, task_complexity: float) -> float:
        """Analyze cognitive friction metric.

        Cognitive friction measures the effort required for an agent to handle
        a task of given complexity. Lower values indicate better cognitive fit.
        Range: 0.0 (no friction) to 1.0 (maximum friction).

        Args:
            agent_name: Name of the agent.
            task_complexity: Task complexity score in range [0.0, 1.0].

        Returns:
            Cognitive friction score as float in range [0.0, 1.0].

        Raises:
            MetricsError: If computation fails.
        """
        try:
            if not 0.0 <= task_complexity <= 1.0:
                raise ValueError(f"task_complexity must be in [0.0, 1.0], got {task_complexity}")

            agent_data = self._get_agent_data(agent_name)

            # Derive capability from average regime performance
            regime_scores = [
                agent_data.get('HIGH_VOLATILITY', 0.5),
                agent_data.get('BULL_TREND', 0.5),
                agent_data.get('BEAR_TREND', 0.5),
                agent_data.get('SIDEWAYS', 0.5),
            ]
            capability = float(np.mean(regime_scores))

            # Friction = how far task complexity exceeds agent capability
            friction = max(0.0, task_complexity - capability)
            cognitive_friction = float(np.clip(friction / 1.0, 0.0, 1.0))

            logger.debug(f"Cognitive_friction({agent_name}, complexity={task_complexity:.2f}) = {cognitive_friction:.4f}")
            return cognitive_friction

        except Exception as e:
            raise MetricsError(f"Failed to compute cognitive friction: {e}") from e

    def compute_domain_affinity(self, agent_name: str, task_domain: str) -> float:
        """Compute domain affinity score for agent-task pairing.

        Domain affinity measures how well an agent's expertise aligns with
        the required task domain. Range: 0.0 (no affinity) to 1.0 (perfect match).

        Args:
            agent_name: Name of the agent.
            task_domain: Target task domain.

        Returns:
            Domain affinity score as float in range [0.0, 1.0].

        Raises:
            MetricsError: If computation fails.
        """
        try:
            agent_data = self._get_agent_data(agent_name)

            if not agent_data:
                logger.debug(f"No domain data for {agent_name}, returning neutral affinity=0.5")
                return 0.5

            # Domain affinity based on regime specialization
            # Agents with higher performance in regimes relevant to domain have better affinity
            domain_regimes = self._map_domain_to_regimes(task_domain)

            if not domain_regimes:
                return 0.5

            # Extract performance on relevant regimes
            relevant_scores = [
                agent_data.get(regime, 0.5) for regime in domain_regimes
            ]
            affinity = float(np.mean(relevant_scores)) if relevant_scores else 0.5
            domain_affinity = float(np.clip(affinity, 0.0, 1.0))

            logger.debug(f"Domain_affinity({agent_name}, {task_domain}) = {domain_affinity:.4f}")
            return domain_affinity

        except Exception as e:
            raise MetricsError(f"Failed to compute domain affinity: {e}") from e

    def route_task(
        self,
        task_domain: str,
        task_complexity: float,
        available_agents: List[str],
        algorithm: RoutingAlgorithm = RoutingAlgorithm.NEURAL_UCB
    ) -> RoutingDecision:
        """Route task to optimal agent based on computed metrics.

        This method implements intelligent task routing using a weighted
        combination of agent capability metrics. Routing is based on causal
        potential and agent-task fit, not execution speed.

        Args:
            task_domain: Domain/category of the task.
            task_complexity: Complexity score of the task (0.0-1.0).
            available_agents: List of available agent names.
            algorithm: Routing algorithm to use (default: NEURAL_UCB).

        Returns:
            RoutingDecision with assigned agent and metrics.

        Raises:
            RoutingError: If routing computation fails.
        """
        try:
            if not available_agents:
                raise RoutingError("No available agents for routing")

            if not 0.0 <= task_complexity <= 1.0:
                raise ValueError(f"task_complexity must be in [0.0, 1.0], got {task_complexity}")

            # Compute scores for each agent
            agent_scores: Dict[str, Dict[str, float]] = {}

            for agent_name in available_agents:
                try:
                    gted = self.compute_gted(agent_name, task_domain)
                    aucr = self.compute_aucr(agent_name)
                    pdcr = self.compute_pdcr(agent_name)
                    cognitive_friction = self.analyze_cognitive_friction(agent_name, task_complexity)
                    domain_affinity = self.compute_domain_affinity(agent_name, task_domain)

                    # Composite score: favor good fit (low GTED), reliability (high AUCR),
                    # precision (low PDCR), low cognitive friction, and domain match
                    composite_score = (
                        0.25 * (1.0 - gted) +
                        0.25 * aucr +
                        0.15 * (1.0 - pdcr) +
                        0.20 * (1.0 - cognitive_friction) +
                        0.15 * domain_affinity
                    )

                    agent_scores[agent_name] = {
                        'composite': float(np.clip(composite_score, 0.0, 1.0)),
                        'gted': gted,
                        'aucr': aucr,
                        'pdcr': pdcr,
                        'cognitive_friction': cognitive_friction,
                        'domain_affinity': domain_affinity,
                    }

                except MetricsError as e:
                    logger.warning(f"Error computing metrics for {agent_name}: {e}")
                    agent_scores[agent_name] = {'composite': 0.5}

            # Select best agent (highest composite score)
            best_agent = max(agent_scores.items(), key=lambda x: x[1]['composite'])
            agent_name = best_agent[0]
            metrics = best_agent[1]
            confidence = float(np.clip(metrics['composite'], 0.0, 1.0))

            routing_decision = RoutingDecision(
                agent_name=agent_name,
                confidence=confidence,
                algorithm=algorithm.value,
                context={
                    'task_domain': task_domain,
                    'task_complexity': task_complexity,
                    'available_agents': available_agents,
                    'all_scores': agent_scores,
                },
                timestamp="2026-03-16T15:29:01",
                metrics=metrics,
            )

            logger.info(
                f"Routed task (domain={task_domain}, complexity={task_complexity:.2f}) "
                f"to agent '{agent_name}' (confidence={confidence:.4f})"
            )

            return routing_decision

        except RoutingError:
            raise
        except Exception as e:
            raise RoutingError(f"Task routing failed: {e}") from e

    def _map_domain_to_regimes(self, task_domain: str) -> List[str]:
        """Map task domain to relevant market regimes.

        Args:
            task_domain: Task domain name.

        Returns:
            List of relevant regime names.
        """
        domain_lower = task_domain.lower()

        regime_mapping = {
            'technology': ['HIGH_VOLATILITY', 'BULL_TREND'],
            'tech': ['HIGH_VOLATILITY', 'BULL_TREND'],
            'finance': ['BEAR_TREND', 'BULL_TREND', 'SIDEWAYS'],
            'trading': ['HIGH_VOLATILITY', 'BULL_TREND', 'BEAR_TREND'],
            'science': ['SIDEWAYS'],
            'business': ['BULL_TREND', 'SIDEWAYS'],
            'mathematics': ['HIGH_VOLATILITY', 'BULL_TREND', 'BEAR_TREND', 'SIDEWAYS'],
        }

        for key, regimes in regime_mapping.items():
            if key in domain_lower:
                return regimes

        return ['HIGH_VOLATILITY', 'BULL_TREND']  # Default

    def add_agent_history(self, agent_name: str, task_result: Dict[str, Any]) -> None:
        """Add task result to agent history for metric computation.

        Args:
            agent_name: Name of the agent.
            task_result: Dictionary with task result data (success, quality_score, etc.).
        """
        if agent_name not in self._agent_history:
            self._agent_history[agent_name] = []

        self._agent_history[agent_name].append(task_result)
        self._agent_history[agent_name] = self._agent_history[agent_name][-100:]
        logger.debug(f"Added task result for {agent_name}")


# ============================================================================
# Module-level convenience functions
# ============================================================================

def create_supervisor(
    knowledge_network_path: str = '/home/deepall/deepall_implementation/knowledge_network.json'
) -> DeepMasterSupervisor:
    """Create and initialize a DeepMasterSupervisor instance.

    Args:
        knowledge_network_path: Path to knowledge_network.json.

    Returns:
        Initialized DeepMasterSupervisor instance.

    Raises:
        KnowledgeNetworkError: If initialization fails.
    """
    return DeepMasterSupervisor(knowledge_network_path)


if __name__ == "__main__":
    """Example usage of DeepMasterSupervisor."""
    try:
        supervisor = create_supervisor()
        available_agents = ["TrendFollowingAgent", "MomentumAgent", "MeanReversionAgent"]

        decision = supervisor.route_task(
            task_domain="technology",
            task_complexity=0.7,
            available_agents=available_agents,
        )

        print(f"\n✓ Routing decision: {decision.agent_name}")
        print(f"  Confidence: {decision.confidence:.4f}")
        print(f"  Metrics: {decision.metrics}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
