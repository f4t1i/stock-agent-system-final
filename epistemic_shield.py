"""EpistemicShield - Predictive Conflict Prevention (PCP) System.

This module detects and prevents conflicts in multi-agent decision systems by:
- Detecting semantic contradictions between agent outputs
- Identifying causal paradoxes (Vektor Gamma corruption patterns)
- Computing cascade failure risks across agent networks
- Recommending veto decisions to prevent system corruption

Author: DeepALL Epistemic Safety System
Version: 1.0.0
Last Updated: 2026-03-16
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from pathlib import Path
from collections import defaultdict

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class ConflictError(Exception):
    """Raised when conflict detection computation fails."""
    pass


class ParadoxError(Exception):
    """Raised when paradox detection fails."""
    pass


class CascadeAnalysisError(Exception):
    """Raised when cascade risk computation fails."""
    pass


class EpistemicShieldError(Exception):
    """Raised for general EpistemicShield operation failures."""
    pass


class ParadoxType(Enum):
    """Types of detected paradoxes."""
    CAUSAL = "causal"
    SEMANTIC = "semantic"
    CIRCULAR_LOGIC = "circular_logic"
    SELF_DEFEATING = "self_defeating"
    NONE = "none"


@dataclass
class ConflictAnalysis:
    """Represents comprehensive conflict analysis result.

    Attributes:
        contradiction_score: Semantic contradiction score [0.0-1.0].
        paradox_detected: Whether a paradox was detected.
        paradox_type: Type of paradox ('causal', 'semantic', 'circular_logic', 'self_defeating', 'none').
        cascade_risk: Cascade failure risk (PDCR value, target < 0.05).
        veto_recommended: Whether output should be vetoed.
        reasoning: Detailed explanation of analysis.
        timestamp: ISO format timestamp of analysis.
    """
    contradiction_score: float
    paradox_detected: bool
    paradox_type: str
    cascade_risk: float
    veto_recommended: bool
    reasoning: str
    timestamp: str = field(default_factory=lambda: "2026-03-16T15:29:01")

    def __post_init__(self) -> None:
        """Validate ConflictAnalysis values."""
        if not 0.0 <= self.contradiction_score <= 1.0:
            raise ValueError(f"contradiction_score must be in [0.0, 1.0], got {self.contradiction_score}")
        if not 0.0 <= self.cascade_risk <= 1.0:
            raise ValueError(f"cascade_risk must be in [0.0, 1.0], got {self.cascade_risk}")
        valid_types = ['causal', 'semantic', 'circular_logic', 'self_defeating', 'none']
        if self.paradox_type not in valid_types:
            raise ValueError(f"Invalid paradox_type: {self.paradox_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class EpistemicShield:
    """Predictive Conflict Prevention (PCP) system for multi-agent outputs.

    Detects semantic contradictions, causal paradoxes, and cascade risks
    to prevent system corruption and maintain epistemic consistency.
    """

    def __init__(self, knowledge_network_path: str = 
                 '/home/deepall/deepall_implementation/knowledge_network.json') -> None:
        """Initialize EpistemicShield with knowledge network.

        Args:
            knowledge_network_path: Path to knowledge_network.json.

        Raises:
            EpistemicShieldError: If initialization fails.
        """
        self.knowledge_network_path = knowledge_network_path
        self.knowledge_network: Dict[str, Any] = {}
        self.phase_c_insights: Dict[str, List[Dict[str, Any]]] = {}
        self._known_paradox_patterns: Set[str] = set()
        self._contradiction_cache: Dict[str, float] = {}
        self._conflict_log: List[Dict[str, Any]] = []

        self._load_knowledge_network()
        self._build_paradox_patterns()
        logger.info("EpistemicShield initialized successfully")

    def _load_knowledge_network(self) -> None:
        """Load knowledge network for causal patterns."""
        try:
            path = Path(self.knowledge_network_path)
            if not path.exists():
                raise FileNotFoundError(f"Knowledge network not found: {self.knowledge_network_path}")

            with open(path, 'r', encoding='utf-8') as f:
                self.knowledge_network = json.load(f)

            if 'phase_c_insights' in self.knowledge_network:
                self.phase_c_insights = self.knowledge_network['phase_c_insights']
                logger.info(f"Loaded phase_c_insights")

        except Exception as e:
            raise EpistemicShieldError(f"Failed to load knowledge network: {e}") from e

    def _build_paradox_patterns(self) -> None:
        """Build known paradox patterns from phase C insights."""
        try:
            meta_reflection = self.phase_c_insights.get('meta_reflection', [])
            for insight in meta_reflection:
                if isinstance(insight, dict):
                    name = insight.get('name', '')
                    if name:
                        self._known_paradox_patterns.add(name.lower())
            logger.debug(f"Built {len(self._known_paradox_patterns)} paradox patterns")
        except Exception as e:
            logger.warning(f"Failed to build paradox patterns: {e}")

    def detect_semantic_contradiction(self, output1: Dict[str, Any], 
                                     output2: Dict[str, Any]) -> float:
        """Detect semantic contradiction between two outputs [0.0-1.0].

        Args:
            output1: First agent output.
            output2: Second agent output.

        Returns:
            Contradiction score [0.0-1.0].
        """
        try:
            key1 = json.dumps(output1, sort_keys=True, default=str)[:100]
            key2 = json.dumps(output2, sort_keys=True, default=str)[:100]
            cache_key = f"{key1}|{key2}"

            if cache_key in self._contradiction_cache:
                return self._contradiction_cache[cache_key]

            contradiction_score = 0.0
            vec1 = self._extract_semantic_vector(output1)
            vec2 = self._extract_semantic_vector(output2)

            if vec1 is not None and vec2 is not None and len(vec1) > 0 and len(vec2) > 0:
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
                contradiction_score = max(0.0, 1.0 - similarity)

            value_conflict = self._check_value_conflicts(output1, output2)
            contradiction_score = max(contradiction_score, value_conflict)

            objective_conflict = self._check_objective_conflicts(output1, output2)
            contradiction_score = max(contradiction_score, objective_conflict)

            contradiction_score = float(np.clip(contradiction_score, 0.0, 1.0))
            self._contradiction_cache[cache_key] = contradiction_score

            logger.debug(f"Semantic contradiction score: {contradiction_score:.4f}")
            return contradiction_score

        except Exception as e:
            raise ConflictError(f"Failed to detect semantic contradiction: {e}") from e

    def detect_causal_paradox(self, output: Dict[str, Any], 
                             context: Dict[str, Any]) -> Tuple[bool, str]:
        """Detect causal paradoxes [bool, str].

        Args:
            output: Agent output to analyze.
            context: Decision context.

        Returns:
            Tuple of (paradox_detected, paradox_type).
        """
        try:
            paradox_detected = False
            paradox_type = 'none'

            if self._detect_circular_logic(output):
                return (True, 'circular_logic')

            if self._detect_self_defeating(output):
                return (True, 'self_defeating')

            if self._detect_causal_loop(output, context):
                return (True, 'causal')

            output_text = json.dumps(output, default=str).lower()
            for pattern in self._known_paradox_patterns:
                if pattern in output_text:
                    return (True, 'semantic')

            logger.debug(f"No causal paradox detected")
            return (paradox_detected, paradox_type)

        except Exception as e:
            raise ParadoxError(f"Failed to detect causal paradox: {e}") from e

    def compute_cascade_risk(self, decision: Dict[str, Any], 
                            agent_outputs: List[Dict[str, Any]]) -> float:
        """Compute cascade failure risk (PDCR metric) [0.0-1.0].

        Target: < 0.05 (5% failure propagation).

        Args:
            decision: The decision to evaluate.
            agent_outputs: List of agent outputs.

        Returns:
            Cascade risk score [0.0-1.0].
        """
        try:
            if not agent_outputs:
                return 0.05

            num_agents = len(agent_outputs)
            base_risk = 0.01 * num_agents

            disagreement_vectors = [self._extract_semantic_vector(output) 
                                   for output in agent_outputs]
            disagreement_vectors = [v for v in disagreement_vectors if v is not None]

            if disagreement_vectors and len(disagreement_vectors) > 1:
                variance = np.std([np.linalg.norm(v) for v in disagreement_vectors])
                disagreement_risk = float(np.clip(variance / 2.0, 0.0, 0.3))
            else:
                disagreement_risk = 0.0

            interconnectedness = self._compute_interconnectedness(agent_outputs)
            interconnection_risk = interconnectedness * 0.3

            cascade_risk = base_risk + disagreement_risk + interconnection_risk
            cascade_risk = float(np.clip(cascade_risk, 0.0, 1.0))

            logger.debug(f"Cascade risk: {cascade_risk:.4f} (target < 0.05)")
            return cascade_risk

        except Exception as e:
            raise CascadeAnalysisError(f"Failed to compute cascade risk: {e}") from e

    def trigger_cross_reflection(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Trigger cross-reflection when conflicts detected.

        Args:
            outputs: List of agent outputs in conflict.

        Returns:
            Structured reflection results.
        """
        try:
            reflection_results = {
                'reflection_triggered': True,
                'num_outputs': len(outputs),
                'conflict_areas': [],
                'reflection_prompts': [],
                'recommended_actions': [],
            }

            if len(outputs) >= 2:
                conflict_areas = self._identify_conflict_areas(outputs[0], outputs[1])
                reflection_results['conflict_areas'] = conflict_areas

                for i, area in enumerate(conflict_areas[:3]):
                    prompt = f"Reflect on disagreement in area: {area}. Explain your reasoning."
                    reflection_results['reflection_prompts'].append(prompt)

            reflection_results['recommended_actions'] = [
                "Wait for reflection responses before proceeding",
                "Escalate to human reviewer",
                "Compute alternative consensus path",
            ]

            logger.info(f"Cross-reflection triggered for {len(outputs)} outputs")
            return reflection_results

        except Exception as e:
            logger.error(f"Failed to trigger cross-reflection: {e}")
            return {'reflection_triggered': False, 'error': str(e)}

    def should_veto(self, analysis: ConflictAnalysis) -> bool:
        """Determine if output should be vetoed.

        Criteria: high contradiction OR paradox OR high cascade risk.

        Args:
            analysis: ConflictAnalysis result.

        Returns:
            True if veto recommended.
        """
        try:
            veto_contradiction = analysis.contradiction_score > 0.75
            veto_paradox = analysis.paradox_detected
            veto_cascade = analysis.cascade_risk > 0.10

            should_veto = veto_contradiction or veto_paradox or veto_cascade

            logger.debug(
                f"Veto decision: {should_veto} "
                f"(contradiction={analysis.contradiction_score:.2f}, "
                f"paradox={analysis.paradox_detected}, "
                f"cascade={analysis.cascade_risk:.4f})"
            )

            return should_veto

        except Exception as e:
            logger.error(f"Failed to determine veto decision: {e}")
            return True

    def analyze_conflicts(self, agent_outputs: List[Dict[str, Any]], 
                         decision_context: Dict[str, Any]) -> ConflictAnalysis:
        """Main entry point for comprehensive conflict analysis.

        Args:
            agent_outputs: List of agent outputs to analyze.
            decision_context: Context information.

        Returns:
            ConflictAnalysis with comprehensive findings.
        """
        try:
            if not agent_outputs:
                raise ValueError("No agent outputs provided")

            contradiction_score = 0.0
            paradox_detected = False
            paradox_type = 'none'
            cascade_risk = 0.05
            reasoning_parts = []

            if len(agent_outputs) >= 2:
                contradictions = []
                for i in range(len(agent_outputs) - 1):
                    for j in range(i + 1, len(agent_outputs)):
                        contradiction = self.detect_semantic_contradiction(
                            agent_outputs[i], agent_outputs[j]
                        )
                        contradictions.append(contradiction)

                if contradictions:
                    contradiction_score = float(np.mean(contradictions))
                    reasoning_parts.append(
                        f"Semantic contradictions detected (score={contradiction_score:.4f})"
                    )

            for i, output in enumerate(agent_outputs):
                pd, pt = self.detect_causal_paradox(output, decision_context)
                if pd:
                    paradox_detected = True
                    paradox_type = pt
                    reasoning_parts.append(f"Causal paradox in output {i}: {pt}")
                    break

            cascade_risk = self.compute_cascade_risk(decision_context, agent_outputs)
            if cascade_risk > 0.05:
                reasoning_parts.append(
                    f"High cascade risk (score={cascade_risk:.4f}, target < 0.05)"
                )

            veto_recommended = (
                contradiction_score > 0.75 or 
                paradox_detected or 
                cascade_risk > 0.10
            )

            reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No conflicts detected"

            analysis = ConflictAnalysis(
                contradiction_score=float(np.clip(contradiction_score, 0.0, 1.0)),
                paradox_detected=paradox_detected,
                paradox_type=paradox_type,
                cascade_risk=float(np.clip(cascade_risk, 0.0, 1.0)),
                veto_recommended=veto_recommended,
                reasoning=reasoning,
                timestamp="2026-03-16T15:29:01",
            )

            self._conflict_log.append(analysis.to_dict())
            logger.info(
                f"Conflict analysis: contradiction={contradiction_score:.4f}, "
                f"paradox={paradox_detected}, cascade={cascade_risk:.4f}, "
                f"veto={veto_recommended}"
            )

            return analysis

        except Exception as e:
            raise EpistemicShieldError(f"Failed to analyze conflicts: {e}") from e

    def _extract_semantic_vector(self, output: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract semantic vector using TF-IDF style approach."""
        try:
            text = json.dumps(output, default=str).lower()
            words = text.split()

            if not words:
                return None

            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1

            vector = np.array(list(word_freq.values()), dtype=np.float32)
            if len(vector) > 0:
                vector = vector / (np.linalg.norm(vector) + 1e-10)
            return vector
        except Exception:
            return None

    def _check_value_conflicts(self, output1: Dict[str, Any], 
                              output2: Dict[str, Any]) -> float:
        """Check for explicit value conflicts."""
        try:
            conflicts = 0
            total_checks = 0

            for key in output1:
                if key in output2:
                    val1 = output1[key]
                    val2 = output2[key]
                    total_checks += 1

                    if isinstance(val1, bool) and isinstance(val2, bool):
                        if val1 != val2:
                            conflicts += 1
                    elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        if abs(val1 - val2) > 0.8 * max(abs(val1), abs(val2)):
                            conflicts += 1

            if total_checks == 0:
                return 0.0

            return float(conflicts / total_checks)
        except Exception:
            return 0.0

    def _check_objective_conflicts(self, output1: Dict[str, Any], 
                                  output2: Dict[str, Any]) -> float:
        """Check for conflicting objectives/goals."""
        try:
            objective_conflict = 0.0
            goals1 = set()
            goals2 = set()

            for key in ['goal', 'objective', 'target', 'action', 'recommendation']:
                if key in output1:
                    goals1.add(str(output1[key]).lower())
                if key in output2:
                    goals2.add(str(output2[key]).lower())

            if goals1 and goals2:
                opposing_words = [('buy', 'sell'), ('increase', 'decrease'), 
                                ('accept', 'reject'), ('proceed', 'halt')]
                for word1, word2 in opposing_words:
                    if any(word1 in g for g in goals1) and any(word2 in g for g in goals2):
                        objective_conflict = 1.0
                        break
                    if any(word2 in g for g in goals1) and any(word1 in g for g in goals2):
                        objective_conflict = 1.0
                        break

            return float(objective_conflict)
        except Exception:
            return 0.0

    def _detect_circular_logic(self, output: Dict[str, Any]) -> bool:
        """Detect circular reasoning."""
        try:
            text = json.dumps(output, default=str).lower()
            circular_indicators = [
                'because i said so',
                'because we decided',
                'as stated',
                'as mentioned',
                'recursive',
                'loop',
            ]
            return any(indicator in text for indicator in circular_indicators)
        except Exception:
            return False

    def _detect_self_defeating(self, output: Dict[str, Any]) -> bool:
        """Detect self-defeating recommendations."""
        try:
            text = json.dumps(output, default=str).lower()
            defeating_pairs = [
                ('do', "don't"),
                ('proceed', 'stop'),
                ('recommend', 'advise against'),
                ('accept', 'reject'),
            ]
            for word1, word2 in defeating_pairs:
                if word1 in text and word2 in text:
                    return True
            return False
        except Exception:
            return False

    def _detect_causal_loop(self, output: Dict[str, Any], 
                           context: Dict[str, Any]) -> bool:
        """Detect causal loops."""
        try:
            output_text = json.dumps(output, default=str).lower()
            context_text = json.dumps(context, default=str).lower()

            output_words = set(output_text.split())
            context_words = set(context_text.split())

            overlap = output_words & context_words
            if len(overlap) > len(output_words) * 0.5:
                return True
            return False
        except Exception:
            return False

    def _compute_interconnectedness(self, agent_outputs: List[Dict[str, Any]]) -> float:
        """Compute interconnectedness between agents [0.0-1.0]."""
        try:
            if len(agent_outputs) < 2:
                return 0.0

            similarities = []
            for i in range(len(agent_outputs) - 1):
                for j in range(i + 1, len(agent_outputs)):
                    vec1 = self._extract_semantic_vector(agent_outputs[i])
                    vec2 = self._extract_semantic_vector(agent_outputs[j])
                    if vec1 is not None and vec2 is not None:
                        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
                        similarities.append(sim)

            if similarities:
                return float(np.mean(similarities))
            return 0.5
        except Exception:
            return 0.5

    def _identify_conflict_areas(self, output1: Dict[str, Any], 
                                output2: Dict[str, Any]) -> List[str]:
        """Identify specific areas of conflict."""
        conflict_areas = []
        try:
            for key in set(list(output1.keys()) + list(output2.keys())):
                if key in output1 and key in output2:
                    if output1[key] != output2[key]:
                        conflict_areas.append(key)
        except Exception:
            pass
        return conflict_areas[:5]

    def get_conflict_log(self) -> List[Dict[str, Any]]:
        """Get all conflict analyses from this session."""
        return self._conflict_log


def create_shield(knowledge_network_path: str = 
                 '/home/deepall/deepall_implementation/knowledge_network.json') -> EpistemicShield:
    """Create and initialize EpistemicShield instance."""
    return EpistemicShield(knowledge_network_path)


if __name__ == "__main__":
    try:
        shield = create_shield()
        print("✓ EpistemicShield initialized successfully")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
