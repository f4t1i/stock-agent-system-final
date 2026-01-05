#!/usr/bin/env python3
"""
Reasoning Extractor - Extract and Parse Agent Reasoning

Extracts reasoning, confidence factors, and generates alternative scenarios
from agent outputs for explainability.

Usage:
    extractor = ReasoningExtractor()
    reasoning = extractor.extract_reasoning(agent_output, agent_name)
    factors = extractor.extract_factors(agent_output, agent_name)
    alternatives = extractor.generate_alternatives(agent_output, agent_name)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
from loguru import logger


@dataclass
class FactorImportance:
    """Factor contributing to decision"""
    name: str
    importance: float  # 0-1
    value: Any
    description: str


@dataclass
class AlternativeScenario:
    """Alternative decision scenario"""
    scenario: str
    recommendation: str
    confidence: float
    reasoning: str


class ReasoningExtractor:
    """Extract reasoning and factors from agent outputs"""
    
    def __init__(self):
        """Initialize reasoning extractor"""
        self.agent_parsers = {
            "news_agent": self._parse_news_agent,
            "technical_agent": self._parse_technical_agent,
            "fundamental_agent": self._parse_fundamental_agent,
            "strategist": self._parse_strategist
        }
    
    def extract_reasoning(self, agent_output: Dict[str, Any], agent_name: str) -> str:
        """
        Extract human-readable reasoning from agent output
        
        Args:
            agent_output: Agent output dictionary
            agent_name: Name of the agent
            
        Returns:
            Formatted reasoning text
        """
        try:
            # Get reasoning field
            reasoning = agent_output.get("reasoning", "")
            
            if not reasoning:
                # Try to construct reasoning from other fields
                reasoning = self._construct_reasoning(agent_output, agent_name)
            
            # Format reasoning
            reasoning = self._format_reasoning(reasoning)
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error extracting reasoning: {e}")
            return "Unable to extract reasoning from agent output."
    
    def extract_factors(self, agent_output: Dict[str, Any], agent_name: str) -> List[Dict[str, Any]]:
        """
        Extract key decision factors with importance weights
        
        Args:
            agent_output: Agent output dictionary
            agent_name: Name of the agent
            
        Returns:
            List of factors with importance, value, and description
        """
        try:
            # Use agent-specific parser
            parser = self.agent_parsers.get(agent_name, self._parse_generic)
            factors = parser(agent_output)
            
            # Sort by importance
            factors.sort(key=lambda x: x["importance"], reverse=True)
            
            return factors
            
        except Exception as e:
            logger.error(f"Error extracting factors: {e}")
            return []
    
    def generate_alternatives(
        self,
        agent_output: Dict[str, Any],
        agent_name: str
    ) -> List[Dict[str, Any]]:
        """
        Generate alternative decision scenarios
        
        Args:
            agent_output: Agent output dictionary
            agent_name: Name of the agent
            
        Returns:
            List of alternative scenarios
        """
        try:
            alternatives = []
            
            recommendation = agent_output.get("recommendation", "HOLD")
            confidence = agent_output.get("confidence", 0.5)
            
            # Generate alternative based on confidence
            if confidence < 0.7:
                # Low confidence - suggest opposite
                alt_recommendation = self._opposite_recommendation(recommendation)
                alternatives.append({
                    "scenario": "Low confidence scenario",
                    "recommendation": alt_recommendation,
                    "confidence": 1.0 - confidence,
                    "reasoning": f"Given the low confidence ({confidence:.2f}), the opposite action might be considered."
                })
            
            # Generate conservative alternative
            if recommendation != "HOLD":
                alternatives.append({
                    "scenario": "Conservative approach",
                    "recommendation": "HOLD",
                    "confidence": 0.6,
                    "reasoning": "A wait-and-see approach to gather more information before acting."
                })
            
            return alternatives
            
        except Exception as e:
            logger.error(f"Error generating alternatives: {e}")
            return []
    
    # ========================================================================
    # Agent-Specific Parsers
    # ========================================================================
    
    def _parse_news_agent(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse news agent output"""
        factors = []
        
        # Sentiment score
        if "sentiment_score" in output:
            factors.append({
                "name": "Sentiment Score",
                "importance": 0.9,
                "value": output["sentiment_score"],
                "description": f"News sentiment: {output['sentiment_score']:.2f} (-2 to 2)"
            })
        
        # News count
        if "news_count" in output:
            factors.append({
                "name": "News Volume",
                "importance": 0.6,
                "value": output["news_count"],
                "description": f"Number of articles analyzed: {output['news_count']}"
            })
        
        # Key events
        if "key_events" in output and output["key_events"]:
            factors.append({
                "name": "Key Events",
                "importance": 0.8,
                "value": len(output["key_events"]),
                "description": f"Significant events: {', '.join(output['key_events'][:3])}"
            })
        
        return factors
    
    def _parse_technical_agent(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse technical agent output"""
        factors = []
        
        # Signal strength
        if "signal_strength" in output:
            factors.append({
                "name": "Signal Strength",
                "importance": 0.95,
                "value": output["signal_strength"],
                "description": f"Technical signal strength: {output['signal_strength']:.2f}"
            })
        
        # Support/Resistance levels
        if "support_levels" in output:
            factors.append({
                "name": "Support Levels",
                "importance": 0.7,
                "value": output["support_levels"],
                "description": f"Key support: {output['support_levels']}"
            })
        
        if "resistance_levels" in output:
            factors.append({
                "name": "Resistance Levels",
                "importance": 0.7,
                "value": output["resistance_levels"],
                "description": f"Key resistance: {output['resistance_levels']}"
            })
        
        # Indicators
        if "indicators" in output:
            for name, value in output["indicators"].items():
                factors.append({
                    "name": name.upper(),
                    "importance": 0.6,
                    "value": value,
                    "description": f"{name}: {value}"
                })
        
        return factors
    
    def _parse_fundamental_agent(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse fundamental agent output"""
        factors = []
        
        # Valuation
        if "valuation" in output:
            factors.append({
                "name": "Valuation",
                "importance": 0.9,
                "value": output["valuation"],
                "description": f"Valuation assessment: {output['valuation']}"
            })
        
        # Financial health
        if "financial_health_score" in output:
            factors.append({
                "name": "Financial Health",
                "importance": 0.85,
                "value": output["financial_health_score"],
                "description": f"Financial health score: {output['financial_health_score']:.2f}"
            })
        
        # Growth score
        if "growth_score" in output:
            factors.append({
                "name": "Growth Potential",
                "importance": 0.8,
                "value": output["growth_score"],
                "description": f"Growth score: {output['growth_score']:.2f}"
            })
        
        # Metrics
        if "metrics" in output:
            for name, value in output["metrics"].items():
                factors.append({
                    "name": name,
                    "importance": 0.6,
                    "value": value,
                    "description": f"{name}: {value}"
                })
        
        return factors
    
    def _parse_strategist(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse strategist output"""
        factors = []
        
        # Position size
        if "position_size" in output:
            factors.append({
                "name": "Position Size",
                "importance": 0.9,
                "value": output["position_size"],
                "description": f"Recommended position: {output['position_size']:.1%}"
            })
        
        # Risk assessment
        if "risk_assessment" in output:
            factors.append({
                "name": "Risk Assessment",
                "importance": 0.85,
                "value": output["risk_assessment"],
                "description": output["risk_assessment"]
            })
        
        # Entry/Exit targets
        if "entry_target" in output:
            factors.append({
                "name": "Entry Target",
                "importance": 0.7,
                "value": output["entry_target"],
                "description": f"Entry price: ${output['entry_target']:.2f}"
            })
        
        if "stop_loss" in output:
            factors.append({
                "name": "Stop Loss",
                "importance": 0.8,
                "value": output["stop_loss"],
                "description": f"Stop loss: ${output['stop_loss']:.2f}"
            })
        
        if "take_profit" in output:
            factors.append({
                "name": "Take Profit",
                "importance": 0.75,
                "value": output["take_profit"],
                "description": f"Take profit: ${output['take_profit']:.2f}"
            })
        
        return factors
    
    def _parse_generic(self, output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generic parser for unknown agents"""
        factors = []
        
        # Extract confidence
        if "confidence" in output:
            factors.append({
                "name": "Confidence",
                "importance": 0.9,
                "value": output["confidence"],
                "description": f"Decision confidence: {output['confidence']:.2f}"
            })
        
        return factors
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _construct_reasoning(self, output: Dict[str, Any], agent_name: str) -> str:
        """Construct reasoning from output fields"""
        parts = []
        
        recommendation = output.get("recommendation", "HOLD")
        confidence = output.get("confidence", 0.5)
        
        parts.append(f"Recommendation: {recommendation} (confidence: {confidence:.2f})")
        
        # Add agent-specific details
        if agent_name == "news_agent":
            if "sentiment_score" in output:
                parts.append(f"News sentiment: {output['sentiment_score']:.2f}")
        elif agent_name == "technical_agent":
            if "signal" in output:
                parts.append(f"Technical signal: {output['signal']}")
        elif agent_name == "fundamental_agent":
            if "valuation" in output:
                parts.append(f"Valuation: {output['valuation']}")
        
        return " | ".join(parts)
    
    def _format_reasoning(self, reasoning: str) -> str:
        """Format reasoning text for display"""
        # Remove extra whitespace
        reasoning = re.sub(r'\s+', ' ', reasoning).strip()
        
        # Capitalize first letter
        if reasoning:
            reasoning = reasoning[0].upper() + reasoning[1:]
        
        # Ensure ends with period
        if reasoning and not reasoning.endswith('.'):
            reasoning += '.'
        
        return reasoning
    
    def _opposite_recommendation(self, recommendation: str) -> str:
        """Get opposite recommendation"""
        opposites = {
            "BUY": "SELL",
            "SELL": "BUY",
            "HOLD": "HOLD"
        }
        return opposites.get(recommendation.upper(), "HOLD")


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    extractor = ReasoningExtractor()
    
    # Test with news agent output
    news_output = {
        "recommendation": "BUY",
        "confidence": 0.75,
        "reasoning": "Positive earnings report and strong market sentiment",
        "sentiment_score": 1.5,
        "news_count": 15,
        "key_events": ["Earnings beat", "New product launch", "CEO interview"]
    }
    
    print("=== News Agent Test ===")
    reasoning = extractor.extract_reasoning(news_output, "news_agent")
    print(f"Reasoning: {reasoning}")
    
    factors = extractor.extract_factors(news_output, "news_agent")
    print(f"\nFactors ({len(factors)}):")
    for f in factors:
        print(f"  - {f['name']}: {f['value']} (importance: {f['importance']:.2f})")
    
    alternatives = extractor.generate_alternatives(news_output, "news_agent")
    print(f"\nAlternatives ({len(alternatives)}):")
    for alt in alternatives:
        print(f"  - {alt['scenario']}: {alt['recommendation']} ({alt['confidence']:.2f})")
    
    print("\n✅ Reasoning Extractor test complete")
