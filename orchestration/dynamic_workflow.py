"""
Dynamic Workflow Manager with Graph Rewriting

Enables runtime workflow reconfiguration based on:
1. Conflicting signals (positive news + negative technicals)
2. Confidence thresholds (low confidence → add validator)
3. Latency constraints (skip agents to save time)
4. Agent agreement (high agreement → skip redundant agents)

Key Concepts:
- Graph Rewriting: Modify workflow graph at runtime
- Adaptive Routing: Choose paths based on context
- Conflict Resolution: Handle contradictory agent outputs
- Validator Injection: Add validation when confidence is low
"""

import time
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import numpy as np


class WorkflowAction(Enum):
    """Possible workflow modifications"""
    SKIP_AGENT = "skip_agent"
    ADD_VALIDATOR = "add_validator"
    PARALLEL_EXECUTION = "parallel_execution"
    SEQUENTIAL_EXECUTION = "sequential_execution"
    EARLY_TERMINATION = "early_termination"
    RETRY_AGENT = "retry_agent"


@dataclass
class AgentOutput:
    """Standardized agent output"""
    agent_name: str
    recommendation: str  # buy, sell, hold, etc.
    confidence: float  # 0-1
    reasoning: str
    signals: Dict[str, float]  # e.g., {'sentiment': 0.8, 'technical_score': -0.5}
    execution_time: float
    timestamp: float


@dataclass
class WorkflowState:
    """Current state of workflow execution"""
    executed_agents: List[str] = field(default_factory=list)
    pending_agents: List[str] = field(default_factory=list)
    agent_outputs: Dict[str, AgentOutput] = field(default_factory=dict)
    conflicts_detected: List[Dict] = field(default_factory=list)
    total_latency: float = 0.0
    rewrite_history: List[Dict] = field(default_factory=list)


@dataclass
class RewriteRule:
    """Graph rewriting rule"""
    name: str
    condition: callable  # Function that checks if rule applies
    action: WorkflowAction
    target_agents: List[str]
    priority: int = 0  # Higher priority rules applied first
    description: str = ""


class ConflictDetector:
    """Detect conflicts between agent outputs"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.conflict_threshold = self.config.get('conflict_threshold', 0.7)
    
    def detect_conflicts(
        self,
        agent_outputs: Dict[str, AgentOutput]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between agent outputs
        
        Returns list of conflicts with:
        - type: Type of conflict
        - agents: Involved agents
        - severity: Conflict severity (0-1)
        - description: Human-readable description
        """
        conflicts = []
        
        if len(agent_outputs) < 2:
            return conflicts
        
        # Check recommendation conflicts
        recommendations = {
            name: output.recommendation 
            for name, output in agent_outputs.items()
        }
        
        # Conflict: Opposite recommendations (buy vs sell)
        if 'buy' in recommendations.values() and 'sell' in recommendations.values():
            buy_agents = [name for name, rec in recommendations.items() if 'buy' in rec.lower()]
            sell_agents = [name for name, rec in recommendations.items() if 'sell' in rec.lower()]
            
            conflicts.append({
                'type': 'opposite_recommendations',
                'agents': buy_agents + sell_agents,
                'severity': 1.0,
                'description': f"Buy recommendation from {buy_agents} conflicts with sell from {sell_agents}",
                'details': {
                    'buy_agents': buy_agents,
                    'sell_agents': sell_agents
                }
            })
        
        # Check signal conflicts (e.g., positive news + negative technical)
        signal_conflicts = self._detect_signal_conflicts(agent_outputs)
        conflicts.extend(signal_conflicts)
        
        # Check confidence disagreement
        confidence_conflicts = self._detect_confidence_conflicts(agent_outputs)
        conflicts.extend(confidence_conflicts)
        
        return conflicts
    
    def _detect_signal_conflicts(
        self,
        agent_outputs: Dict[str, AgentOutput]
    ) -> List[Dict]:
        """Detect conflicts in underlying signals"""
        conflicts = []
        
        # Extract signals
        all_signals = {}
        for name, output in agent_outputs.items():
            for signal_name, signal_value in output.signals.items():
                if signal_name not in all_signals:
                    all_signals[signal_name] = {}
                all_signals[signal_name][name] = signal_value
        
        # Check for conflicting signals
        for signal_name, agent_values in all_signals.items():
            if len(agent_values) < 2:
                continue
            
            values = list(agent_values.values())
            
            # Check if signs differ significantly
            positive_agents = [a for a, v in agent_values.items() if v > 0.3]
            negative_agents = [a for a, v in agent_values.items() if v < -0.3]
            
            if positive_agents and negative_agents:
                # Calculate severity based on magnitude
                pos_magnitude = np.mean([agent_values[a] for a in positive_agents])
                neg_magnitude = np.mean([agent_values[a] for a in negative_agents])
                severity = min(abs(pos_magnitude) + abs(neg_magnitude), 1.0)
                
                conflicts.append({
                    'type': 'signal_conflict',
                    'signal': signal_name,
                    'agents': positive_agents + negative_agents,
                    'severity': severity,
                    'description': f"Signal '{signal_name}' conflict: positive from {positive_agents}, negative from {negative_agents}",
                    'details': {
                        'positive_agents': positive_agents,
                        'negative_agents': negative_agents,
                        'pos_magnitude': pos_magnitude,
                        'neg_magnitude': neg_magnitude
                    }
                })
        
        return conflicts
    
    def _detect_confidence_conflicts(
        self,
        agent_outputs: Dict[str, AgentOutput]
    ) -> List[Dict]:
        """Detect large confidence disagreements"""
        conflicts = []
        
        confidences = {name: output.confidence for name, output in agent_outputs.items()}
        
        if len(confidences) < 2:
            return conflicts
        
        # Check for high variance in confidence
        conf_values = list(confidences.values())
        conf_std = np.std(conf_values)
        
        if conf_std > 0.3:  # High disagreement
            high_conf_agents = [name for name, conf in confidences.items() if conf > 0.7]
            low_conf_agents = [name for name, conf in confidences.items() if conf < 0.4]
            
            if high_conf_agents and low_conf_agents:
                conflicts.append({
                    'type': 'confidence_disagreement',
                    'agents': high_conf_agents + low_conf_agents,
                    'severity': conf_std,
                    'description': f"High confidence disagreement: {high_conf_agents} confident, {low_conf_agents} uncertain",
                    'details': {
                        'high_conf_agents': high_conf_agents,
                        'low_conf_agents': low_conf_agents,
                        'std': conf_std
                    }
                })
        
        return conflicts


class DynamicWorkflowManager:
    """
    Manages dynamic workflow execution with graph rewriting
    
    Features:
    - Runtime workflow modification
    - Conflict detection and resolution
    - Adaptive agent routing
    - Latency optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Components
        self.conflict_detector = ConflictDetector(config)
        
        # Rewrite rules
        self.rewrite_rules = self._initialize_rewrite_rules()
        
        # Thresholds
        self.low_confidence_threshold = self.config.get('low_confidence_threshold', 0.5)
        self.high_agreement_threshold = self.config.get('high_agreement_threshold', 0.9)
        self.max_latency_ms = self.config.get('max_latency_ms', 5000)
        
        # Agents
        self.available_agents = ['news', 'technical', 'fundamental']
        self.validator_agent = 'validator'  # Special agent for conflict resolution
    
    def _initialize_rewrite_rules(self) -> List[RewriteRule]:
        """Initialize graph rewriting rules"""
        rules = []
        
        # Rule 1: Add validator when confidence is low
        rules.append(RewriteRule(
            name="add_validator_low_confidence",
            condition=lambda state: self._check_low_confidence(state),
            action=WorkflowAction.ADD_VALIDATOR,
            target_agents=[self.validator_agent],
            priority=10,
            description="Add validator agent when strategist confidence < threshold"
        ))
        
        # Rule 2: Skip redundant agents when high agreement
        rules.append(RewriteRule(
            name="skip_redundant_high_agreement",
            condition=lambda state: self._check_high_agreement(state),
            action=WorkflowAction.SKIP_AGENT,
            target_agents=['fundamental'],  # Skip if news + technical agree
            priority=5,
            description="Skip fundamental agent when news and technical strongly agree"
        ))
        
        # Rule 3: Early termination on extreme confidence
        rules.append(RewriteRule(
            name="early_termination_extreme_confidence",
            condition=lambda state: self._check_extreme_confidence(state),
            action=WorkflowAction.EARLY_TERMINATION,
            target_agents=[],
            priority=15,
            description="Terminate early when all agents have extreme confidence and agree"
        ))
        
        # Rule 4: Parallel execution when latency is critical
        rules.append(RewriteRule(
            name="parallel_execution_latency",
            condition=lambda state: self._check_latency_critical(state),
            action=WorkflowAction.PARALLEL_EXECUTION,
            target_agents=self.available_agents,
            priority=20,
            description="Execute remaining agents in parallel to reduce latency"
        ))
        
        # Rule 5: Add validator on detected conflicts
        rules.append(RewriteRule(
            name="add_validator_conflict",
            condition=lambda state: len(state.conflicts_detected) > 0,
            action=WorkflowAction.ADD_VALIDATOR,
            target_agents=[self.validator_agent],
            priority=12,
            description="Add validator agent when conflicts detected between agents"
        ))
        
        # Rule 6: Retry agent on very low confidence
        rules.append(RewriteRule(
            name="retry_agent_very_low_confidence",
            condition=lambda state: self._check_very_low_confidence(state),
            action=WorkflowAction.RETRY_AGENT,
            target_agents=[],  # Determined dynamically
            priority=8,
            description="Retry agent when confidence is extremely low (< 0.3)"
        ))
        
        return sorted(rules, key=lambda r: r.priority, reverse=True)
    
    def execute_workflow(
        self,
        initial_agents: List[str],
        context: Dict,
        agents: Dict[str, Any]  # Agent instances
    ) -> Tuple[Dict[str, AgentOutput], WorkflowState]:
        """
        Execute workflow with dynamic rewriting
        
        Args:
            initial_agents: Initial list of agents to execute
            context: Market context
            agents: Dictionary of agent instances
        
        Returns:
            (agent_outputs, workflow_state)
        """
        state = WorkflowState(
            pending_agents=initial_agents.copy(),
            executed_agents=[],
            agent_outputs={},
            conflicts_detected=[],
            total_latency=0.0,
            rewrite_history=[]
        )
        
        start_time = time.time()
        
        while state.pending_agents:
            # Check if we should rewrite the workflow
            rewrite_applied = self._apply_rewrite_rules(state, agents, context)
            
            if rewrite_applied and rewrite_applied['action'] == WorkflowAction.EARLY_TERMINATION:
                logger.info("Early termination triggered")
                break
            
            # Execute next agent
            agent_name = state.pending_agents.pop(0)
            
            if agent_name not in agents:
                logger.warning(f"Agent {agent_name} not available, skipping")
                continue
            
            # Execute agent
            agent_start = time.time()
            try:
                output = self._execute_agent(agent_name, agents[agent_name], context, state)
                agent_time = time.time() - agent_start
                
                output.execution_time = agent_time
                output.timestamp = time.time()
                
                state.agent_outputs[agent_name] = output
                state.executed_agents.append(agent_name)
                state.total_latency += agent_time * 1000  # Convert to ms
                
                logger.info(f"Executed {agent_name}: {output.recommendation} (conf={output.confidence:.2f}, time={agent_time:.3f}s)")
            
            except Exception as e:
                logger.error(f"Error executing {agent_name}: {e}")
                continue
            
            # Detect conflicts after each agent
            if len(state.agent_outputs) >= 2:
                conflicts = self.conflict_detector.detect_conflicts(state.agent_outputs)
                state.conflicts_detected.extend(conflicts)
                
                if conflicts:
                    logger.warning(f"Detected {len(conflicts)} conflicts")
                    for conflict in conflicts:
                        logger.warning(f"  - {conflict['description']}")
        
        total_time = time.time() - start_time
        logger.info(f"Workflow completed in {total_time:.3f}s with {len(state.executed_agents)} agents")
        
        return state.agent_outputs, state
    
    def _apply_rewrite_rules(
        self,
        state: WorkflowState,
        agents: Dict[str, Any],
        context: Dict
    ) -> Optional[Dict]:
        """Apply applicable rewrite rules"""
        
        for rule in self.rewrite_rules:
            if rule.condition(state):
                logger.info(f"Applying rewrite rule: {rule.name}")
                
                # Apply action
                if rule.action == WorkflowAction.SKIP_AGENT:
                    for agent in rule.target_agents:
                        if agent in state.pending_agents:
                            state.pending_agents.remove(agent)
                            logger.info(f"  Skipped agent: {agent}")
                
                elif rule.action == WorkflowAction.ADD_VALIDATOR:
                    if self.validator_agent not in state.pending_agents and \
                       self.validator_agent not in state.executed_agents:
                        state.pending_agents.append(self.validator_agent)
                        logger.info(f"  Added validator agent")
                
                elif rule.action == WorkflowAction.EARLY_TERMINATION:
                    state.pending_agents.clear()
                    logger.info(f"  Early termination triggered")
                
                elif rule.action == WorkflowAction.PARALLEL_EXECUTION:
                    # Note: Actual parallel execution would require async
                    logger.info(f"  Parallel execution mode activated")
                
                elif rule.action == WorkflowAction.RETRY_AGENT:
                    # Find agent with lowest confidence
                    if state.agent_outputs:
                        lowest_conf_agent = min(
                            state.agent_outputs.items(),
                            key=lambda x: x[1].confidence
                        )[0]
                        if state.agent_outputs[lowest_conf_agent].confidence < 0.3:
                            state.pending_agents.insert(0, lowest_conf_agent)
                            logger.info(f"  Retrying agent: {lowest_conf_agent}")
                
                # Record rewrite
                state.rewrite_history.append({
                    'rule': rule.name,
                    'action': rule.action.value,
                    'targets': rule.target_agents,
                    'timestamp': time.time()
                })
                
                return {
                    'rule': rule.name,
                    'action': rule.action,
                    'targets': rule.target_agents
                }
        
        return None
    
    def _execute_agent(
        self,
        agent_name: str,
        agent: Any,
        context: Dict,
        state: WorkflowState
    ) -> AgentOutput:
        """Execute a single agent"""
        
        # Special handling for validator agent
        if agent_name == self.validator_agent:
            return self._execute_validator(context, state)
        
        # Execute regular agent
        result = agent.analyze(
            symbol=context.get('symbol', 'AAPL'),
            market_data=context
        )
        
        # Convert to AgentOutput
        return AgentOutput(
            agent_name=agent_name,
            recommendation=result.get('recommendation', 'hold'),
            confidence=result.get('confidence', 0.5),
            reasoning=result.get('reasoning', ''),
            signals=result.get('signals', {}),
            execution_time=0.0,  # Will be set by caller
            timestamp=time.time()
        )
    
    def _execute_validator(
        self,
        context: Dict,
        state: WorkflowState
    ) -> AgentOutput:
        """
        Execute validator agent to resolve conflicts
        
        Validator analyzes existing agent outputs and conflicts,
        then provides a tie-breaking recommendation
        """
        logger.info("Executing validator agent for conflict resolution")
        
        # Analyze conflicts
        if not state.conflicts_detected:
            # No conflicts, just aggregate
            recommendations = [o.recommendation for o in state.agent_outputs.values()]
            confidences = [o.confidence for o in state.agent_outputs.values()]
            
            # Simple majority vote
            from collections import Counter
            vote_counts = Counter(recommendations)
            final_rec = vote_counts.most_common(1)[0][0]
            final_conf = np.mean(confidences)
            
            reasoning = f"Validator: Aggregated {len(recommendations)} agent outputs via majority vote"
        
        else:
            # Resolve conflicts
            conflict = state.conflicts_detected[0]  # Handle first conflict
            
            if conflict['type'] == 'opposite_recommendations':
                # Weight by confidence
                buy_agents = conflict['details']['buy_agents']
                sell_agents = conflict['details']['sell_agents']
                
                buy_conf = np.mean([
                    state.agent_outputs[a].confidence 
                    for a in buy_agents
                ])
                sell_conf = np.mean([
                    state.agent_outputs[a].confidence 
                    for a in sell_agents
                ])
                
                if buy_conf > sell_conf:
                    final_rec = 'buy'
                    final_conf = buy_conf
                    reasoning = f"Validator: Buy signals ({buy_agents}) have higher confidence ({buy_conf:.2f}) than sell ({sell_conf:.2f})"
                else:
                    final_rec = 'sell'
                    final_conf = sell_conf
                    reasoning = f"Validator: Sell signals ({sell_agents}) have higher confidence ({sell_conf:.2f}) than buy ({buy_conf:.2f})"
            
            elif conflict['type'] == 'signal_conflict':
                # Favor agent with higher confidence
                involved_agents = conflict['agents']
                best_agent = max(
                    involved_agents,
                    key=lambda a: state.agent_outputs[a].confidence
                )
                
                final_rec = state.agent_outputs[best_agent].recommendation
                final_conf = state.agent_outputs[best_agent].confidence
                reasoning = f"Validator: Following {best_agent} due to highest confidence in signal conflict"
            
            else:
                # Default: majority vote
                recommendations = [o.recommendation for o in state.agent_outputs.values()]
                from collections import Counter
                vote_counts = Counter(recommendations)
                final_rec = vote_counts.most_common(1)[0][0]
                final_conf = 0.6
                reasoning = f"Validator: Majority vote resolution"
        
        return AgentOutput(
            agent_name=self.validator_agent,
            recommendation=final_rec,
            confidence=final_conf,
            reasoning=reasoning,
            signals={'validation_score': final_conf},
            execution_time=0.0,
            timestamp=time.time()
        )
    
    # Condition checkers for rewrite rules
    
    def _check_low_confidence(self, state: WorkflowState) -> bool:
        """Check if any agent has low confidence"""
        if not state.agent_outputs:
            return False
        
        min_conf = min(o.confidence for o in state.agent_outputs.values())
        return min_conf < self.low_confidence_threshold
    
    def _check_very_low_confidence(self, state: WorkflowState) -> bool:
        """Check if any agent has very low confidence"""
        if not state.agent_outputs:
            return False
        
        min_conf = min(o.confidence for o in state.agent_outputs.values())
        return min_conf < 0.3
    
    def _check_high_agreement(self, state: WorkflowState) -> bool:
        """Check if agents strongly agree"""
        if len(state.agent_outputs) < 2:
            return False
        
        # Check if news and technical agree
        if 'news' in state.agent_outputs and 'technical' in state.agent_outputs:
            news_rec = state.agent_outputs['news'].recommendation
            tech_rec = state.agent_outputs['technical'].recommendation
            
            news_conf = state.agent_outputs['news'].confidence
            tech_conf = state.agent_outputs['technical'].confidence
            
            # Strong agreement if same recommendation and both confident
            if news_rec == tech_rec and news_conf > 0.7 and tech_conf > 0.7:
                return True
        
        return False
    
    def _check_extreme_confidence(self, state: WorkflowState) -> bool:
        """Check if all agents have extreme confidence and agree"""
        if len(state.agent_outputs) < 2:
            return False
        
        # All agents must have confidence > 0.9
        all_high_conf = all(o.confidence > 0.9 for o in state.agent_outputs.values())
        
        if not all_high_conf:
            return False
        
        # All must agree on recommendation
        recommendations = [o.recommendation for o in state.agent_outputs.values()]
        all_agree = len(set(recommendations)) == 1
        
        return all_agree
    
    def _check_latency_critical(self, state: WorkflowState) -> bool:
        """Check if latency is approaching limit"""
        return state.total_latency > (self.max_latency_ms * 0.7)


if __name__ == '__main__':
    # Test
    manager = DynamicWorkflowManager()
    
    # Mock agents
    class MockAgent:
        def __init__(self, name, rec, conf):
            self.name = name
            self.rec = rec
            self.conf = conf
        
        def analyze(self, symbol, market_data):
            return {
                'recommendation': self.rec,
                'confidence': self.conf,
                'reasoning': f'{self.name} analysis',
                'signals': {}
            }
    
    agents = {
        'news': MockAgent('news', 'buy', 0.9),
        'technical': MockAgent('technical', 'sell', 0.8),
        'fundamental': MockAgent('fundamental', 'hold', 0.6)
    }
    
    context = {'symbol': 'AAPL'}
    
    outputs, state = manager.execute_workflow(
        initial_agents=['news', 'technical', 'fundamental'],
        context=context,
        agents=agents
    )
    
    print(f"Executed agents: {state.executed_agents}")
    print(f"Conflicts: {len(state.conflicts_detected)}")
    print(f"Rewrites: {len(state.rewrite_history)}")
