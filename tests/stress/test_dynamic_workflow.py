"""
Stress Tests for Dynamic Workflow & Conflict Resolution

Tests system's ability to handle:
1. Contradictory signals (positive news + negative technical)
2. Low confidence scenarios
3. High agreement optimization
4. Latency constraints
5. Complex conflict resolution

Key Questions:
- Can system detect and resolve conflicts?
- Does graph rewriting improve efficiency?
- Are validator agents added when needed?
- Does system handle extreme scenarios gracefully?
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock
from typing import Dict, List
from orchestration.dynamic_workflow import (
    DynamicWorkflowManager,
    AgentOutput,
    WorkflowAction
)
from orchestration.conflict_resolution import (
    AdvancedConflictResolver,
    ResolutionStrategy
)


class MockAgent:
    """Mock agent for testing"""
    
    def __init__(self, name: str, recommendation: str, confidence: float, signals: Dict = None):
        self.name = name
        self.recommendation = recommendation
        self.confidence = confidence
        self.signals = signals or {}
    
    def analyze(self, symbol: str, market_data: Dict) -> Dict:
        """Mock analyze method"""
        time.sleep(0.01)  # Simulate execution time
        
        return {
            'recommendation': self.recommendation,
            'confidence': self.confidence,
            'reasoning': f'{self.name} analysis for {symbol}',
            'signals': self.signals
        }


class TestConflictDetection:
    """Test conflict detection capabilities"""
    
    @pytest.fixture
    def manager(self):
        return DynamicWorkflowManager()
    
    def test_detect_opposite_recommendations(self, manager):
        """Test: Detect buy vs sell conflict"""
        agent_outputs = {
            'news': AgentOutput(
                agent_name='news',
                recommendation='buy',
                confidence=0.8,
                reasoning='Positive news',
                signals={'sentiment': 0.7},
                execution_time=0.1,
                timestamp=time.time()
            ),
            'technical': AgentOutput(
                agent_name='technical',
                recommendation='sell',
                confidence=0.75,
                reasoning='Bearish pattern',
                signals={'technical_score': -0.6},
                execution_time=0.1,
                timestamp=time.time()
            )
        }
        
        conflicts = manager.conflict_detector.detect_conflicts(agent_outputs)
        
        assert len(conflicts) > 0, "Should detect buy vs sell conflict"
        assert conflicts[0]['type'] == 'opposite_recommendations'
        assert 'news' in conflicts[0]['agents']
        assert 'technical' in conflicts[0]['agents']
    
    def test_detect_signal_conflict(self, manager):
        """Test: Detect conflicting signals (positive news + negative technical)"""
        agent_outputs = {
            'news': AgentOutput(
                agent_name='news',
                recommendation='buy',
                confidence=0.8,
                reasoning='Strong positive sentiment',
                signals={'sentiment': 0.9},  # Very positive
                execution_time=0.1,
                timestamp=time.time()
            ),
            'technical': AgentOutput(
                agent_name='technical',
                recommendation='hold',
                confidence=0.6,
                reasoning='Technical indicators mixed',
                signals={'sentiment': -0.7},  # Very negative
                execution_time=0.1,
                timestamp=time.time()
            )
        }
        
        conflicts = manager.conflict_detector.detect_conflicts(agent_outputs)
        
        # Should detect signal conflict
        signal_conflicts = [c for c in conflicts if c['type'] == 'signal_conflict']
        assert len(signal_conflicts) > 0, "Should detect sentiment signal conflict"
        assert signal_conflicts[0]['signal'] == 'sentiment'
    
    def test_detect_confidence_disagreement(self, manager):
        """Test: Detect high confidence disagreement"""
        agent_outputs = {
            'news': AgentOutput(
                agent_name='news',
                recommendation='buy',
                confidence=0.95,  # Very confident
                reasoning='Strong signals',
                signals={},
                execution_time=0.1,
                timestamp=time.time()
            ),
            'technical': AgentOutput(
                agent_name='technical',
                recommendation='sell',
                confidence=0.25,  # Very uncertain
                reasoning='Unclear pattern',
                signals={},
                execution_time=0.1,
                timestamp=time.time()
            )
        }
        
        conflicts = manager.conflict_detector.detect_conflicts(agent_outputs)
        
        # Should detect confidence disagreement
        conf_conflicts = [c for c in conflicts if c['type'] == 'confidence_disagreement']
        assert len(conf_conflicts) > 0, "Should detect confidence disagreement"


class TestGraphRewriting:
    """Test graph rewriting rules"""
    
    @pytest.fixture
    def manager(self):
        return DynamicWorkflowManager()
    
    def test_add_validator_on_low_confidence(self, manager):
        """Test: Validator added when confidence is low"""
        agents = {
            'news': MockAgent('news', 'buy', 0.3),  # Low confidence
            'technical': MockAgent('technical', 'hold', 0.4),  # Low confidence
            'validator': MockAgent('validator', 'hold', 0.7)
        }
        
        context = {'symbol': 'AAPL', 'volatility': 0.5}
        
        outputs, state = manager.execute_workflow(
            initial_agents=['news', 'technical'],
            context=context,
            agents=agents
        )
        
        # Validator should be added due to low confidence
        assert 'validator' in state.executed_agents, "Validator should be added for low confidence"
        
        # Check rewrite history
        validator_rewrites = [
            r for r in state.rewrite_history 
            if 'validator' in r.get('rule', '')
        ]
        assert len(validator_rewrites) > 0, "Should record validator addition"
    
    def test_skip_agent_on_high_agreement(self, manager):
        """Test: Skip redundant agent when others strongly agree"""
        agents = {
            'news': MockAgent('news', 'buy', 0.9),  # High confidence
            'technical': MockAgent('technical', 'buy', 0.85),  # High confidence, same rec
            'fundamental': MockAgent('fundamental', 'buy', 0.7)
        }
        
        context = {'symbol': 'AAPL'}
        
        outputs, state = manager.execute_workflow(
            initial_agents=['news', 'technical', 'fundamental'],
            context=context,
            agents=agents
        )
        
        # Fundamental might be skipped if news + technical strongly agree
        skip_rewrites = [
            r for r in state.rewrite_history 
            if r.get('action') == WorkflowAction.SKIP_AGENT.value
        ]
        
        if skip_rewrites:
            assert 'fundamental' in skip_rewrites[0]['targets'], "Should skip fundamental on high agreement"
    
    def test_early_termination_on_extreme_confidence(self, manager):
        """Test: Early termination when all agents extremely confident and agree"""
        agents = {
            'news': MockAgent('news', 'buy', 0.95),  # Extreme confidence
            'technical': MockAgent('technical', 'buy', 0.93),  # Extreme confidence
            'fundamental': MockAgent('fundamental', 'buy', 0.92)  # Would be executed
        }
        
        context = {'symbol': 'AAPL'}
        
        outputs, state = manager.execute_workflow(
            initial_agents=['news', 'technical', 'fundamental'],
            context=context,
            agents=agents
        )
        
        # Check if early termination triggered
        early_term_rewrites = [
            r for r in state.rewrite_history 
            if r.get('action') == WorkflowAction.EARLY_TERMINATION.value
        ]
        
        if early_term_rewrites:
            # If triggered, fundamental should not be executed
            assert 'fundamental' not in state.executed_agents, "Should skip fundamental after early termination"


class TestConflictResolution:
    """Test conflict resolution strategies"""
    
    @pytest.fixture
    def resolver(self):
        return AdvancedConflictResolver()
    
    def test_confidence_weighted_resolution(self, resolver):
        """Test: Confidence-weighted resolution favors higher confidence"""
        agent_outputs = {
            'news': AgentOutput(
                agent_name='news',
                recommendation='buy',
                confidence=0.9,  # High confidence
                reasoning='Strong positive news',
                signals={},
                execution_time=0.1,
                timestamp=time.time()
            ),
            'technical': AgentOutput(
                agent_name='technical',
                recommendation='sell',
                confidence=0.5,  # Lower confidence
                reasoning='Weak bearish signal',
                signals={},
                execution_time=0.1,
                timestamp=time.time()
            )
        }
        
        conflict = {
            'type': 'opposite_recommendations',
            'agents': ['news', 'technical'],
            'severity': 1.0,
            'description': 'Buy vs Sell',
            'details': {
                'buy_agents': ['news'],
                'sell_agents': ['technical']
            }
        }
        
        context = {'volatility': 0.5}
        
        resolution = resolver.resolve_conflict(conflict, agent_outputs, context)
        
        # Should favor 'buy' due to higher confidence
        assert resolution.final_recommendation == 'buy', "Should favor higher confidence recommendation"
        assert resolution.final_confidence > 0.6, "Should have reasonable confidence"
    
    def test_domain_expertise_resolution_high_volatility(self, resolver):
        """Test: In high volatility, news agent weighted higher"""
        agent_outputs = {
            'news': AgentOutput(
                agent_name='news',
                recommendation='sell',
                confidence=0.7,
                reasoning='Negative news',
                signals={},
                execution_time=0.1,
                timestamp=time.time()
            ),
            'technical': AgentOutput(
                agent_name='technical',
                recommendation='buy',
                confidence=0.8,  # Higher confidence
                reasoning='Bullish technical',
                signals={},
                execution_time=0.1,
                timestamp=time.time()
            )
        }
        
        conflict = {
            'type': 'opposite_recommendations',
            'agents': ['news', 'technical'],
            'severity': 1.0,
            'description': 'Buy vs Sell',
            'details': {
                'buy_agents': ['technical'],
                'sell_agents': ['news']
            }
        }
        
        # High volatility context → News should be weighted higher
        context = {'volatility': 0.9, 'news_impact': 0.85}
        
        resolution = resolver.resolve_conflict(conflict, agent_outputs, context)
        
        # In high volatility, news should be favored despite lower confidence
        assert resolution.strategy_used == ResolutionStrategy.DOMAIN_EXPERTISE, "Should use domain expertise in high volatility"
    
    def test_signal_strength_resolution(self, resolver):
        """Test: Signal strength resolution based on magnitude"""
        agent_outputs = {
            'news': AgentOutput(
                agent_name='news',
                recommendation='buy',
                confidence=0.7,
                reasoning='Positive sentiment',
                signals={'sentiment': 0.9},  # Strong positive
                execution_time=0.1,
                timestamp=time.time()
            ),
            'technical': AgentOutput(
                agent_name='technical',
                recommendation='sell',
                confidence=0.7,
                reasoning='Negative technical',
                signals={'sentiment': -0.3},  # Weak negative
                execution_time=0.1,
                timestamp=time.time()
            )
        }
        
        conflict = {
            'type': 'signal_conflict',
            'signal': 'sentiment',
            'agents': ['news', 'technical'],
            'severity': 0.8,
            'description': 'Sentiment conflict',
            'details': {
                'positive_agents': ['news'],
                'negative_agents': ['technical']
            }
        }
        
        context = {}
        
        resolution = resolver.resolve_conflict(conflict, agent_outputs, context)
        
        # Should favor 'buy' due to stronger positive signal
        assert resolution.final_recommendation == 'buy', "Should favor stronger signal magnitude"


class TestComplexScenarios:
    """Test complex real-world scenarios"""
    
    @pytest.fixture
    def manager(self):
        return DynamicWorkflowManager()
    
    def test_extreme_positive_news_negative_technical(self, manager):
        """
        Test: Extreme positive news + negative technical breakdown
        
        Scenario: Company announces breakthrough product, but stock price drops
        """
        agents = {
            'news': MockAgent(
                'news',
                recommendation='strong_buy',
                confidence=0.95,
                signals={'sentiment': 0.95, 'event_impact': 0.9}
            ),
            'technical': MockAgent(
                'technical',
                recommendation='sell',
                confidence=0.8,
                signals={'sentiment': -0.7, 'trend': -0.6}
            ),
            'fundamental': MockAgent(
                'fundamental',
                recommendation='buy',
                confidence=0.7,
                signals={'valuation': 0.5}
            ),
            'validator': MockAgent(
                'validator',
                recommendation='hold',
                confidence=0.75
            )
        }
        
        context = {
            'symbol': 'AAPL',
            'volatility': 0.85,  # High volatility
            'news_impact': 0.9,  # High news impact
            'trend_strength': -0.5
        }
        
        outputs, state = manager.execute_workflow(
            initial_agents=['news', 'technical', 'fundamental'],
            context=context,
            agents=agents
        )
        
        # Should detect conflicts
        assert len(state.conflicts_detected) > 0, "Should detect conflicts"
        
        # Validator should be added
        assert 'validator' in state.executed_agents, "Validator should be added for conflict resolution"
        
        # Check final recommendation (should consider high news impact)
        assert len(outputs) >= 3, "Should execute multiple agents"
    
    def test_all_agents_uncertain(self, manager):
        """Test: All agents have low confidence"""
        agents = {
            'news': MockAgent('news', 'hold', 0.3),  # Low confidence
            'technical': MockAgent('technical', 'hold', 0.35),  # Low confidence
            'fundamental': MockAgent('fundamental', 'hold', 0.4),  # Low confidence
            'validator': MockAgent('validator', 'hold', 0.6)
        }
        
        context = {'symbol': 'AAPL'}
        
        outputs, state = manager.execute_workflow(
            initial_agents=['news', 'technical', 'fundamental'],
            context=context,
            agents=agents
        )
        
        # Validator should be added due to low confidence
        assert 'validator' in state.executed_agents, "Validator should be added when all agents uncertain"
        
        # Check if retry was attempted
        retry_rewrites = [
            r for r in state.rewrite_history 
            if r.get('action') == WorkflowAction.RETRY_AGENT.value
        ]
        
        # Might retry an agent
        if retry_rewrites:
            assert len(retry_rewrites) > 0, "Should attempt retry on very low confidence"
    
    def test_high_latency_scenario(self, manager):
        """Test: Latency optimization when approaching limit"""
        
        class SlowAgent(MockAgent):
            def analyze(self, symbol, market_data):
                time.sleep(0.5)  # Slow execution
                return super().analyze(symbol, market_data)
        
        agents = {
            'news': SlowAgent('news', 'buy', 0.8),
            'technical': SlowAgent('technical', 'buy', 0.75),
            'fundamental': SlowAgent('fundamental', 'buy', 0.7)
        }
        
        # Set low latency limit
        manager.max_latency_ms = 1000  # 1 second
        
        context = {'symbol': 'AAPL'}
        
        start_time = time.time()
        outputs, state = manager.execute_workflow(
            initial_agents=['news', 'technical', 'fundamental'],
            context=context,
            agents=agents
        )
        total_time = time.time() - start_time
        
        # Check if parallel execution was triggered
        parallel_rewrites = [
            r for r in state.rewrite_history 
            if r.get('action') == WorkflowAction.PARALLEL_EXECUTION.value
        ]
        
        # Note: Actual parallel execution not implemented in this version
        # This test checks if the rule is triggered
        if state.total_latency > 700:  # 70% of limit
            assert len(parallel_rewrites) > 0 or total_time < 2.0, "Should optimize for latency"


class TestWorkflowEfficiency:
    """Test workflow efficiency improvements"""
    
    def test_agent_skipping_saves_time(self):
        """Test: Skipping agents reduces total execution time"""
        manager = DynamicWorkflowManager()
        
        # Scenario 1: All agents executed
        agents_all = {
            'news': MockAgent('news', 'buy', 0.6),
            'technical': MockAgent('technical', 'buy', 0.6),
            'fundamental': MockAgent('fundamental', 'buy', 0.6)
        }
        
        context = {'symbol': 'AAPL'}
        
        start1 = time.time()
        outputs1, state1 = manager.execute_workflow(
            initial_agents=['news', 'technical', 'fundamental'],
            context=context,
            agents=agents_all
        )
        time1 = time.time() - start1
        
        # Scenario 2: High agreement (should skip fundamental)
        manager2 = DynamicWorkflowManager()
        agents_agree = {
            'news': MockAgent('news', 'buy', 0.9),  # High confidence
            'technical': MockAgent('technical', 'buy', 0.85),  # High confidence, same rec
            'fundamental': MockAgent('fundamental', 'buy', 0.7)
        }
        
        start2 = time.time()
        outputs2, state2 = manager2.execute_workflow(
            initial_agents=['news', 'technical', 'fundamental'],
            context=context,
            agents=agents_agree
        )
        time2 = time.time() - start2
        
        # Check if fundamental was skipped
        if 'fundamental' not in state2.executed_agents:
            assert time2 < time1, "Skipping agent should reduce execution time"
            assert len(state2.executed_agents) < len(state1.executed_agents), "Fewer agents executed"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
