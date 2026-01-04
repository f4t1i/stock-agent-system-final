"""
Advanced Conflict Resolution Strategies

Handles contradictory signals between agents:
1. Positive news + negative technical indicators
2. High confidence disagreements
3. Signal magnitude conflicts
4. Time-sensitive conflicts

Resolution Strategies:
- Confidence-weighted voting
- Signal strength analysis
- Temporal priority (recent signals weighted higher)
- Domain expertise weighting (technical for trends, news for events)
- Ensemble methods
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class ConflictType(Enum):
    """Types of conflicts"""
    OPPOSITE_RECOMMENDATIONS = "opposite_recommendations"
    SIGNAL_CONFLICT = "signal_conflict"
    CONFIDENCE_DISAGREEMENT = "confidence_disagreement"
    MAGNITUDE_CONFLICT = "magnitude_conflict"


class ResolutionStrategy(Enum):
    """Conflict resolution strategies"""
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    SIGNAL_STRENGTH = "signal_strength"
    DOMAIN_EXPERTISE = "domain_expertise"
    TEMPORAL_PRIORITY = "temporal_priority"
    ENSEMBLE_VOTING = "ensemble_voting"
    VALIDATOR_AGENT = "validator_agent"


@dataclass
class ConflictResolution:
    """Result of conflict resolution"""
    strategy_used: ResolutionStrategy
    final_recommendation: str
    final_confidence: float
    reasoning: str
    contributing_agents: List[str]
    weights: Dict[str, float]


class AdvancedConflictResolver:
    """
    Advanced conflict resolution with multiple strategies
    
    Selects appropriate strategy based on conflict type and context
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Domain expertise weights
        self.domain_weights = {
            'news': {
                'event_driven': 0.9,
                'sentiment': 0.9,
                'trend': 0.3,
                'technical': 0.2
            },
            'technical': {
                'event_driven': 0.2,
                'sentiment': 0.3,
                'trend': 0.9,
                'technical': 0.9
            },
            'fundamental': {
                'event_driven': 0.6,
                'sentiment': 0.5,
                'trend': 0.7,
                'technical': 0.5
            }
        }
        
        # Temporal decay factor (how much to discount older signals)
        self.temporal_decay = self.config.get('temporal_decay', 0.95)
    
    def resolve_conflict(
        self,
        conflict: Dict[str, Any],
        agent_outputs: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ConflictResolution:
        """
        Resolve a conflict using appropriate strategy
        
        Args:
            conflict: Conflict information
            agent_outputs: All agent outputs
            context: Market context
        
        Returns:
            ConflictResolution
        """
        conflict_type = ConflictType(conflict['type'])
        
        # Select strategy based on conflict type and context
        strategy = self._select_strategy(conflict_type, conflict, context)
        
        logger.info(f"Resolving {conflict_type.value} using {strategy.value}")
        
        # Apply strategy
        if strategy == ResolutionStrategy.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_resolution(conflict, agent_outputs)
        
        elif strategy == ResolutionStrategy.SIGNAL_STRENGTH:
            return self._signal_strength_resolution(conflict, agent_outputs)
        
        elif strategy == ResolutionStrategy.DOMAIN_EXPERTISE:
            return self._domain_expertise_resolution(conflict, agent_outputs, context)
        
        elif strategy == ResolutionStrategy.TEMPORAL_PRIORITY:
            return self._temporal_priority_resolution(conflict, agent_outputs)
        
        elif strategy == ResolutionStrategy.ENSEMBLE_VOTING:
            return self._ensemble_voting_resolution(conflict, agent_outputs)
        
        else:
            # Default to confidence-weighted
            return self._confidence_weighted_resolution(conflict, agent_outputs)
    
    def _select_strategy(
        self,
        conflict_type: ConflictType,
        conflict: Dict,
        context: Dict
    ) -> ResolutionStrategy:
        """Select appropriate resolution strategy"""
        
        # Opposite recommendations → Use confidence weighting
        if conflict_type == ConflictType.OPPOSITE_RECOMMENDATIONS:
            # If high volatility, prefer news (events drive market)
            if context.get('volatility', 0.5) > 0.7:
                return ResolutionStrategy.DOMAIN_EXPERTISE
            else:
                return ResolutionStrategy.CONFIDENCE_WEIGHTED
        
        # Signal conflicts → Analyze signal strength
        elif conflict_type == ConflictType.SIGNAL_CONFLICT:
            return ResolutionStrategy.SIGNAL_STRENGTH
        
        # Confidence disagreement → Use ensemble
        elif conflict_type == ConflictType.CONFIDENCE_DISAGREEMENT:
            return ResolutionStrategy.ENSEMBLE_VOTING
        
        else:
            return ResolutionStrategy.CONFIDENCE_WEIGHTED
    
    def _confidence_weighted_resolution(
        self,
        conflict: Dict,
        agent_outputs: Dict[str, Any]
    ) -> ConflictResolution:
        """
        Resolve by weighting recommendations by confidence
        
        Higher confidence agents have more influence
        """
        involved_agents = conflict['agents']
        
        # Calculate weighted votes
        recommendation_scores = {}
        total_confidence = 0.0
        
        for agent in involved_agents:
            if agent not in agent_outputs:
                continue
            
            output = agent_outputs[agent]
            rec = output.recommendation
            conf = output.confidence
            
            if rec not in recommendation_scores:
                recommendation_scores[rec] = 0.0
            
            recommendation_scores[rec] += conf
            total_confidence += conf
        
        # Normalize
        if total_confidence > 0:
            recommendation_scores = {
                rec: score / total_confidence
                for rec, score in recommendation_scores.items()
            }
        
        # Select recommendation with highest weighted score
        final_rec = max(recommendation_scores.items(), key=lambda x: x[1])[0]
        final_conf = recommendation_scores[final_rec]
        
        # Calculate agent weights
        weights = {
            agent: agent_outputs[agent].confidence / total_confidence
            for agent in involved_agents
            if agent in agent_outputs
        }
        
        reasoning = f"Confidence-weighted resolution: {final_rec} selected with {final_conf:.2%} weighted confidence"
        
        return ConflictResolution(
            strategy_used=ResolutionStrategy.CONFIDENCE_WEIGHTED,
            final_recommendation=final_rec,
            final_confidence=final_conf,
            reasoning=reasoning,
            contributing_agents=involved_agents,
            weights=weights
        )
    
    def _signal_strength_resolution(
        self,
        conflict: Dict,
        agent_outputs: Dict[str, Any]
    ) -> ConflictResolution:
        """
        Resolve by analyzing underlying signal strengths
        
        Stronger signals (higher magnitude) have more influence
        """
        signal_name = conflict.get('signal', 'unknown')
        involved_agents = conflict['agents']
        
        # Get signal values
        signal_values = {}
        for agent in involved_agents:
            if agent not in agent_outputs:
                continue
            
            output = agent_outputs[agent]
            if signal_name in output.signals:
                signal_values[agent] = output.signals[signal_name]
        
        # Calculate weighted average based on absolute magnitude
        total_magnitude = sum(abs(v) for v in signal_values.values())
        
        if total_magnitude == 0:
            # Fallback to confidence weighting
            return self._confidence_weighted_resolution(conflict, agent_outputs)
        
        weighted_signal = sum(
            v * (abs(v) / total_magnitude)
            for v in signal_values.values()
        )
        
        # Determine recommendation based on signal direction
        if weighted_signal > 0.3:
            final_rec = 'buy'
        elif weighted_signal < -0.3:
            final_rec = 'sell'
        else:
            final_rec = 'hold'
        
        final_conf = min(abs(weighted_signal), 1.0)
        
        weights = {
            agent: abs(signal_values[agent]) / total_magnitude
            for agent in signal_values
        }
        
        reasoning = f"Signal strength resolution: {signal_name} weighted value = {weighted_signal:.2f} → {final_rec}"
        
        return ConflictResolution(
            strategy_used=ResolutionStrategy.SIGNAL_STRENGTH,
            final_recommendation=final_rec,
            final_confidence=final_conf,
            reasoning=reasoning,
            contributing_agents=list(signal_values.keys()),
            weights=weights
        )
    
    def _domain_expertise_resolution(
        self,
        conflict: Dict,
        agent_outputs: Dict[str, Any],
        context: Dict
    ) -> ConflictResolution:
        """
        Resolve by weighting agents based on domain expertise
        
        Example: In high volatility, news agent has more weight
        """
        involved_agents = conflict['agents']
        
        # Determine market condition
        volatility = context.get('volatility', 0.5)
        trend_strength = abs(context.get('trend_strength', 0.0))
        news_impact = context.get('news_impact', 0.5)
        
        # Determine dominant factor
        if news_impact > 0.7 or volatility > 0.7:
            condition = 'event_driven'
        elif trend_strength > 0.6:
            condition = 'trend'
        else:
            condition = 'technical'
        
        # Calculate domain-weighted scores
        recommendation_scores = {}
        total_weight = 0.0
        
        for agent in involved_agents:
            if agent not in agent_outputs:
                continue
            
            output = agent_outputs[agent]
            rec = output.recommendation
            
            # Get domain weight
            domain_weight = self.domain_weights.get(agent, {}).get(condition, 0.5)
            
            # Combine with confidence
            weight = domain_weight * output.confidence
            
            if rec not in recommendation_scores:
                recommendation_scores[rec] = 0.0
            
            recommendation_scores[rec] += weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            recommendation_scores = {
                rec: score / total_weight
                for rec, score in recommendation_scores.items()
            }
        
        final_rec = max(recommendation_scores.items(), key=lambda x: x[1])[0]
        final_conf = recommendation_scores[final_rec]
        
        # Calculate agent weights
        weights = {}
        for agent in involved_agents:
            if agent in agent_outputs:
                domain_weight = self.domain_weights.get(agent, {}).get(condition, 0.5)
                weights[agent] = domain_weight * agent_outputs[agent].confidence / total_weight
        
        reasoning = f"Domain expertise resolution: {condition} condition → {final_rec} (news weight={self.domain_weights['news'][condition]:.2f}, technical weight={self.domain_weights['technical'][condition]:.2f})"
        
        return ConflictResolution(
            strategy_used=ResolutionStrategy.DOMAIN_EXPERTISE,
            final_recommendation=final_rec,
            final_confidence=final_conf,
            reasoning=reasoning,
            contributing_agents=involved_agents,
            weights=weights
        )
    
    def _temporal_priority_resolution(
        self,
        conflict: Dict,
        agent_outputs: Dict[str, Any]
    ) -> ConflictResolution:
        """
        Resolve by giving priority to more recent signals
        
        Applies temporal decay to older signals
        """
        involved_agents = conflict['agents']
        
        # Get timestamps
        current_time = max(
            agent_outputs[a].timestamp 
            for a in involved_agents 
            if a in agent_outputs
        )
        
        # Calculate time-weighted scores
        recommendation_scores = {}
        total_weight = 0.0
        
        for agent in involved_agents:
            if agent not in agent_outputs:
                continue
            
            output = agent_outputs[agent]
            rec = output.recommendation
            
            # Calculate temporal weight
            time_diff = current_time - output.timestamp
            temporal_weight = self.temporal_decay ** time_diff
            
            # Combine with confidence
            weight = temporal_weight * output.confidence
            
            if rec not in recommendation_scores:
                recommendation_scores[rec] = 0.0
            
            recommendation_scores[rec] += weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            recommendation_scores = {
                rec: score / total_weight
                for rec, score in recommendation_scores.items()
            }
        
        final_rec = max(recommendation_scores.items(), key=lambda x: x[1])[0]
        final_conf = recommendation_scores[final_rec]
        
        reasoning = f"Temporal priority resolution: Recent signals weighted higher → {final_rec}"
        
        return ConflictResolution(
            strategy_used=ResolutionStrategy.TEMPORAL_PRIORITY,
            final_recommendation=final_rec,
            final_confidence=final_conf,
            reasoning=reasoning,
            contributing_agents=involved_agents,
            weights={}  # Complex to calculate
        )
    
    def _ensemble_voting_resolution(
        self,
        conflict: Dict,
        agent_outputs: Dict[str, Any]
    ) -> ConflictResolution:
        """
        Resolve using ensemble voting methods
        
        Combines multiple strategies
        """
        involved_agents = conflict['agents']
        
        # Strategy 1: Majority vote
        recommendations = [
            agent_outputs[a].recommendation 
            for a in involved_agents 
            if a in agent_outputs
        ]
        
        from collections import Counter
        vote_counts = Counter(recommendations)
        majority_rec = vote_counts.most_common(1)[0][0]
        
        # Strategy 2: Confidence-weighted
        conf_weighted = self._confidence_weighted_resolution(conflict, agent_outputs)
        
        # Combine strategies
        if majority_rec == conf_weighted.final_recommendation:
            # Agreement → High confidence
            final_rec = majority_rec
            final_conf = min(conf_weighted.final_confidence * 1.2, 1.0)
            reasoning = f"Ensemble: Majority vote and confidence weighting agree on {final_rec}"
        else:
            # Disagreement → Use confidence weighting but lower confidence
            final_rec = conf_weighted.final_recommendation
            final_conf = conf_weighted.final_confidence * 0.8
            reasoning = f"Ensemble: Confidence weighting selected {final_rec} despite different majority vote"
        
        return ConflictResolution(
            strategy_used=ResolutionStrategy.ENSEMBLE_VOTING,
            final_recommendation=final_rec,
            final_confidence=final_conf,
            reasoning=reasoning,
            contributing_agents=involved_agents,
            weights=conf_weighted.weights
        )


class ConflictAnalyzer:
    """Analyze patterns in conflicts for system improvement"""
    
    def __init__(self):
        self.conflict_history = []
    
    def add_conflict(self, conflict: Dict, resolution: ConflictResolution):
        """Record a conflict and its resolution"""
        self.conflict_history.append({
            'conflict': conflict,
            'resolution': resolution,
            'timestamp': np.datetime64('now')
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get conflict statistics"""
        if not self.conflict_history:
            return {}
        
        # Count by type
        type_counts = {}
        for record in self.conflict_history:
            ctype = record['conflict']['type']
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
        
        # Count by resolution strategy
        strategy_counts = {}
        for record in self.conflict_history:
            strategy = record['resolution'].strategy_used.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Average confidence after resolution
        avg_conf = np.mean([
            record['resolution'].final_confidence
            for record in self.conflict_history
        ])
        
        return {
            'total_conflicts': len(self.conflict_history),
            'conflicts_by_type': type_counts,
            'resolutions_by_strategy': strategy_counts,
            'avg_resolution_confidence': avg_conf
        }


if __name__ == '__main__':
    # Test
    from orchestration.dynamic_workflow import AgentOutput
    import time
    
    resolver = AdvancedConflictResolver()
    
    # Create mock conflict: Positive news + negative technical
    agent_outputs = {
        'news': AgentOutput(
            agent_name='news',
            recommendation='buy',
            confidence=0.85,
            reasoning='Positive earnings',
            signals={'sentiment': 0.8},
            execution_time=0.5,
            timestamp=time.time()
        ),
        'technical': AgentOutput(
            agent_name='technical',
            recommendation='sell',
            confidence=0.75,
            reasoning='Bearish technical pattern',
            signals={'sentiment': -0.6},
            execution_time=0.3,
            timestamp=time.time()
        )
    }
    
    conflict = {
        'type': 'opposite_recommendations',
        'agents': ['news', 'technical'],
        'severity': 1.0,
        'description': 'Buy vs Sell conflict',
        'details': {
            'buy_agents': ['news'],
            'sell_agents': ['technical']
        }
    }
    
    context = {
        'volatility': 0.8,
        'trend_strength': -0.3,
        'news_impact': 0.9
    }
    
    resolution = resolver.resolve_conflict(conflict, agent_outputs, context)
    
    print(f"Strategy: {resolution.strategy_used.value}")
    print(f"Final Recommendation: {resolution.final_recommendation}")
    print(f"Confidence: {resolution.final_confidence:.2f}")
    print(f"Reasoning: {resolution.reasoning}")
    print(f"Weights: {resolution.weights}")
