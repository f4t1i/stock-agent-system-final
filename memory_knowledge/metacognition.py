"""
Metacognition Module for DeepALL system.

This module implements enhanced metacognitive capabilities for the DeepALL system,
including multi-perspective reflection, pattern-based synthesis, and more.
"""

import logging
import time
import random
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# Import the TransdisciplinaryKnowledgeNetwork from our implementation
from knowledge_network import TransdisciplinaryKnowledgeNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedMetacognition:
    """Enhanced metacognition capabilities for DeepALL system."""
    
    def __init__(self, system):
        """Initialize enhanced metacognition.
        
        Args:
            system: The DeepALL system instance
        """
        self.system = system
        self.perspectives = {
            "logical": "Focus on formal logic, consistency, and rational analysis",
            "creative": "Focus on novel approaches, innovation, and thinking outside the box",
            "critical": "Focus on identifying weaknesses, risks, and potential issues",
            "practical": "Focus on feasibility, resource efficiency, and implementation",
            "strategic": "Focus on long-term implications, alignment with goals, and strategic fit",
            "ethical": "Focus on ethical implications, fairness, and potential biases"
        }
        self.reflection_history = []
        self.initialized = True
        logging.info("Enhanced metacognition initialized")
    
    # ================ ENHANCEMENT 1: MULTI-PERSPECTIVE REFLECTION ================
    
    def multi_perspective_reflect(self, decision, context=None, perspectives=None):
        """Reflect on a decision from multiple perspectives.
        
        Args:
            decision: The decision to reflect on (dict or object)
            context: Additional context for the reflection (optional)
            perspectives: Specific perspectives to use, or None for all
            
        Returns:
            dict: A structured reflection result with insights from multiple perspectives
        """
        logging.info(f"Multi-perspective reflection on decision: {decision}")
        
        if context is None:
            context = {}
        
        # If no perspectives are specified, use all available perspectives
        if perspectives is None:
            perspectives = list(self.perspectives.keys())
        
        # Limit to 3-5 perspectives for efficiency if too many are provided
        if len(perspectives) > 5:
            perspectives = random.sample(perspectives, 5)
        
        reflection_result = {
            "decision": decision,
            "context": context,
            "perspectives": {},
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Generate reflection from each perspective
        for perspective in perspectives:
            if perspective in self.perspectives:
                perspective_reflection = self._reflect_from_perspective(decision, context, perspective)
                reflection_result["perspectives"][perspective] = perspective_reflection
        
        # Save reflection for future analysis
        self._save_reflection_result(reflection_result)
        
        return reflection_result
    
    def _reflect_from_perspective(self, decision, context, perspective):
        """Generate reflection insights from a specific perspective.
        
        This is a simplified implementation. In a full implementation, this would use
        more sophisticated analysis tailored to each perspective.
        """
        perspective_description = self.perspectives[perspective]
        
        # For demonstration, we'll simulate perspective-specific reflection
        reflection = {
            "perspective": perspective,
            "description": perspective_description,
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        # Generate strengths, weaknesses, and suggestions based on perspective
        # This is a simple example - a real implementation would have more sophisticated logic
        if perspective == "logical":
            if isinstance(decision, dict) and "priority" in decision and decision["priority"] == "high":
                reflection["strengths"].append("High priority allocation is logical given the importance")
            else:
                reflection["weaknesses"].append("Lack of clear prioritization may lead to inefficient resource allocation")
                reflection["suggestions"].append("Implement a formal priority ranking system")
                
        elif perspective == "creative":
            reflection["strengths"].append("The approach allows for flexible adaptation")
            reflection["suggestions"].append("Consider alternative implementation paths for redundancy")
            
        elif perspective == "critical":
            reflection["weaknesses"].append("Potential bottlenecks in implementation are not addressed")
            reflection["weaknesses"].append("Failure scenarios are not fully explored")
            reflection["suggestions"].append("Conduct a comprehensive risk analysis")
            
        elif perspective == "practical":
            if isinstance(decision, dict) and "resource_allocation" in decision and decision["resource_allocation"] == "increased":
                if isinstance(context, dict) and "available_resources" in context and context["available_resources"] == "limited":
                    reflection["weaknesses"].append("Increased resource allocation may not be practical with limited resources")
                    reflection["suggestions"].append("Consider a phased approach to resource allocation")
                else:
                    reflection["strengths"].append("Appropriate resource allocation for the task")
            
        elif perspective == "strategic":
            reflection["strengths"].append("Aligns with long-term system evolution goals")
            reflection["suggestions"].append("Establish clear success metrics for future evaluation")
            
        elif perspective == "ethical":
            reflection["strengths"].append("Transparent decision-making process")
            reflection["suggestions"].append("Monitor for unintended consequences or biases in outcomes")
        
        return reflection
    
    def _save_reflection_result(self, reflection_result):
        """Save reflection result for future analysis."""
        self.reflection_history.append(reflection_result)
        
        # Limit history size to prevent memory issues
        if len(self.reflection_history) > 100:
            self.reflection_history = self.reflection_history[-100:]
    
    # ================ ENHANCEMENT 2: PATTERN-BASED FEEDBACK SYNTHESIS ================
    
    def pattern_based_synthesis(self, decision, context=None):
        """Pattern-based synthesis that identifies patterns across multiple perspectives."""
        logging.info(f"Pattern-based synthesis for decision: {decision}")
        
        # Get multi-perspective reflection
        reflection = self.multi_perspective_reflect(decision, context)
        
        # Extract patterns
        patterns = self._extract_patterns(reflection)
        
        # Create synthesis
        synthesis = self._create_synthesis(reflection, patterns)
        
        # Combine reflection and synthesis
        result = {
            "reflection": reflection,
            "patterns": patterns,
            "synthesis": synthesis
        }
        
        return result
    
    def _extract_patterns(self, reflection):
        """Extract patterns from multi-perspective reflection."""
        patterns = {
            "common_strengths": [],
            "common_weaknesses": [],
            "common_suggestions": [],
            "unique_insights": {}
        }
        
        # Track all strengths, weaknesses, and suggestions from all perspectives
        all_strengths = {}
        all_weaknesses = {}
        all_suggestions = {}
        
        # Count occurrences of each strength/weakness/suggestion
        for perspective, insights in reflection["perspectives"].items():
            # Track strengths
            for strength in insights.get("strengths", []):
                if strength not in all_strengths:
                    all_strengths[strength] = []
                all_strengths[strength].append(perspective)
            
            # Track weaknesses
            for weakness in insights.get("weaknesses", []):
                if weakness not in all_weaknesses:
                    all_weaknesses[weakness] = []
                all_weaknesses[weakness].append(perspective)
            
            # Track suggestions
            for suggestion in insights.get("suggestions", []):
                if suggestion not in all_suggestions:
                    all_suggestions[suggestion] = []
                all_suggestions[suggestion].append(perspective)
        
        # Find common strengths (mentioned by multiple perspectives)
        for strength, perspectives in all_strengths.items():
            if len(perspectives) > 1:  # Common if mentioned by at least 2 perspectives
                patterns["common_strengths"].append({
                    "text": strength,
                    "perspectives": perspectives,
                    "count": len(perspectives)
                })
        
        # Find common weaknesses (mentioned by multiple perspectives)
        for weakness, perspectives in all_weaknesses.items():
            if len(perspectives) > 1:  # Common if mentioned by at least 2 perspectives
                patterns["common_weaknesses"].append({
                    "text": weakness,
                    "perspectives": perspectives,
                    "count": len(perspectives)
                })
        
        # Find common suggestions (mentioned by multiple perspectives)
        for suggestion, perspectives in all_suggestions.items():
            if len(perspectives) > 1:  # Common if mentioned by at least 2 perspectives
                patterns["common_suggestions"].append({
                    "text": suggestion,
                    "perspectives": perspectives,
                    "count": len(perspectives)
                })
        
        # Find unique insights (mentioned by only one perspective)
        for perspective, insights in reflection["perspectives"].items():
            unique_for_perspective = []
            
            # Check strengths
            for strength in insights.get("strengths", []):
                if len(all_strengths[strength]) == 1:  # Unique if mentioned by exactly 1 perspective
                    unique_for_perspective.append({"type": "strength", "text": strength})
            
            # Check weaknesses
            for weakness in insights.get("weaknesses", []):
                if len(all_weaknesses[weakness]) == 1:  # Unique if mentioned by exactly 1 perspective
                    unique_for_perspective.append({"type": "weakness", "text": weakness})
            
            # Check suggestions
            for suggestion in insights.get("suggestions", []):
                if len(all_suggestions[suggestion]) == 1:  # Unique if mentioned by exactly 1 perspective
                    unique_for_perspective.append({"type": "suggestion", "text": suggestion})
            
            if unique_for_perspective:
                patterns["unique_insights"][perspective] = unique_for_perspective
        
        return patterns
    
    def _create_synthesis(self, reflection, patterns):
        """Create a synthesis from the extracted patterns."""
        synthesis = {
            "key_strengths": [],
            "key_weaknesses": [],
            "key_suggestions": [],
            "perspective_highlights": {},
            "executive_summary": ""
        }
        
        # Extract key strengths (sort by count - most common first)
        sorted_strengths = sorted(patterns["common_strengths"], key=lambda x: x["count"], reverse=True)
        for strength in sorted_strengths:
            synthesis["key_strengths"].append(strength["text"])
        
        # If no common strengths, find the most important individual strengths
        if not synthesis["key_strengths"]:
            for perspective, insights in reflection["perspectives"].items():
                if insights.get("strengths") and len(insights["strengths"]) > 0:
                    synthesis["key_strengths"].append(insights["strengths"][0])
                    if len(synthesis["key_strengths"]) >= 3:  # Limit to top 3
                        break
        
        # Extract key weaknesses (sort by count - most common first)
        sorted_weaknesses = sorted(patterns["common_weaknesses"], key=lambda x: x["count"], reverse=True)
        for weakness in sorted_weaknesses:
            synthesis["key_weaknesses"].append(weakness["text"])
        
        # If no common weaknesses, find the most important individual weaknesses
        if not synthesis["key_weaknesses"]:
            for perspective, insights in reflection["perspectives"].items():
                if insights.get("weaknesses") and len(insights["weaknesses"]) > 0:
                    synthesis["key_weaknesses"].append(insights["weaknesses"][0])
                    if len(synthesis["key_weaknesses"]) >= 3:  # Limit to top 3
                        break
        
        # Extract key suggestions (sort by count - most common first)
        sorted_suggestions = sorted(patterns["common_suggestions"], key=lambda x: x["count"], reverse=True)
        for suggestion in sorted_suggestions:
            synthesis["key_suggestions"].append(suggestion["text"])
        
        # If no common suggestions, find the most important individual suggestions
        if not synthesis["key_suggestions"]:
            for perspective, insights in reflection["perspectives"].items():
                if insights.get("suggestions") and len(insights["suggestions"]) > 0:
                    synthesis["key_suggestions"].append(insights["suggestions"][0])
                    if len(synthesis["key_suggestions"]) >= 3:  # Limit to top 3
                        break
        
        # Extract perspective highlights (unique insights from each perspective)
        for perspective, insights in patterns["unique_insights"].items():
            if insights:
                synthesis["perspective_highlights"][perspective] = insights
        
        # Create executive summary
        executive_summary = ["Executive Summary:"]
        
        if synthesis["key_strengths"]:
            executive_summary.append("Key strengths: " + "; ".join(synthesis["key_strengths"][:3]))
        
        if synthesis["key_weaknesses"]:
            executive_summary.append("Key weaknesses: " + "; ".join(synthesis["key_weaknesses"][:3]))
        
        if synthesis["key_suggestions"]:
            executive_summary.append("Key suggestions: " + "; ".join(synthesis["key_suggestions"][:3]))
        
        unique_perspective_count = len(patterns["unique_insights"])
        if unique_perspective_count > 0:
            executive_summary.append(f"Unique insights from {unique_perspective_count} different perspectives.")
        
        synthesis["executive_summary"] = " ".join(executive_summary)
        
        return synthesis
    
    # ================ ENHANCEMENT 4: TRANSDISCIPLINARY KNOWLEDGE NETWORK ================
    
    def initialize_knowledge_network(self):
        """Initialize the transdisciplinary knowledge network."""
        logging.info("Initializing transdisciplinary knowledge network")
        self.knowledge_network = TransdisciplinaryKnowledgeNetwork(self.system)
        return self.knowledge_network
    
    def add_knowledge_concept(self, domain, concept, attributes=None):
        """Add a concept to the knowledge network."""
        if not hasattr(self, 'knowledge_network') or not self.knowledge_network:
            self.initialize_knowledge_network()
        
        concept_id = self.knowledge_network.add_concept(domain, concept, attributes)
        return concept_id
    
    def create_knowledge_link(self, source_domain, source_concept_id,
                            target_domain, target_concept_id,
                            link_type, strength=1.0):
        """Create a link between concepts in the knowledge network."""
        if not hasattr(self, 'knowledge_network') or not self.knowledge_network:
            self.initialize_knowledge_network()
        
        # Check if it's within the same domain or across domains
        if source_domain == target_domain:
            return self.knowledge_network.relate_concepts_within_domain(
                source_domain, source_concept_id, target_concept_id, link_type, strength)
        else:
            return self.knowledge_network.create_cross_domain_link(
                source_domain, source_concept_id, target_domain, target_concept_id, link_type, strength)
    
    def analyze_knowledge_gaps(self):
        """Analyze knowledge gaps in the network."""
        if not hasattr(self, 'knowledge_network') or not self.knowledge_network:
            self.initialize_knowledge_network()
        
        return self.knowledge_network.identify_knowledge_gaps()
    
    def discover_concept_relationships(self, concept_text, context=None):
        """Discover relationships between a new piece of knowledge and existing concepts."""
        if not hasattr(self, 'knowledge_network') or not self.knowledge_network:
            self.initialize_knowledge_network()
        
        # Identify domain from context or infer it
        domain = None
        if context and 'domain' in context:
            domain = context['domain']
        else:
            # Simple domain inference based on keywords (in real implementation, this would use ML)
            domain_keywords = {
                'technology': ['software', 'hardware', 'algorithm', 'code', 'programming', 'computer'],
                'science': ['experiment', 'hypothesis', 'theory', 'research', 'data', 'laboratory'],
                'business': ['strategy', 'market', 'customer', 'profit', 'revenue', 'product'],
                'humanities': ['culture', 'ethics', 'philosophy', 'history', 'literature', 'art'],
                'mathematics': ['equation', 'formula', 'proof', 'theorem', 'calculation', 'number']
            }
            
            # Count domain keyword occurrences in concept_text
            domain_counts = {}
            concept_lower = concept_text.lower()
            for domain_name, keywords in domain_keywords.items():
                count = sum(keyword in concept_lower for keyword in keywords)
                domain_counts[domain_name] = count
            
            # Select domain with highest keyword count, if any
            if max(domain_counts.values()) > 0:
                domain = max(domain_counts, key=domain_counts.get)
            else:
                # Default to technology if no clear domain is found
                domain = 'technology'
        
        # Extract concept name (in a real implementation, this would use NLP)
        # Here using a simple approach: use first sentence or first N words
        concept_name = concept_text.split('.')[0].strip()
        if len(concept_name) > 50:  # If too long, take first 5 words
            concept_name = ' '.join(concept_text.split()[:5])
        
        # Add concept to knowledge network
        concept_id = self.add_knowledge_concept(domain, concept_name)
        
        # Find potential connections to existing concepts
        connections = []
        
        # For each domain
        for domain_name, domain_obj in self.knowledge_network.domains.items():
            for existing_id, existing_concept in domain_obj.concepts.items():
                # Skip the concept we just added
                if domain_name == domain and existing_id == concept_id:
                    continue
                
                existing_name = existing_concept['name']
                
                # Simple relevance check (in a real implementation, this would use semantic similarity)
                relevance = self._calculate_concept_relevance(concept_text, existing_name)
                
                if relevance > 0.2:  # Arbitrary threshold
                    connections.append({
                        'domain': domain_name,
                        'concept_id': existing_id,
                        'concept_name': existing_name,
                        'relevance': relevance,
                        'suggested_link': 'related_to' if domain_name != domain else 'similar_to'
                    })
        
        # Sort connections by relevance
        connections.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Return top 5 connections
        return connections[:5]
    
    def _calculate_concept_relevance(self, concept_text, existing_concept):
        """Calculate relevance between a concept text and existing concept name.
        
        In a real implementation, this would use semantic similarity or embeddings.
        Here using a simple keyword-based approach for demonstration.
        """
        # Convert to lowercase for comparison
        concept_lower = concept_text.lower()
        existing_lower = existing_concept.lower()
        
        # Check if existing concept appears in the concept text
        if existing_lower in concept_lower:
            return 0.8
        
        # Check if any words from existing concept appear in concept text
        existing_words = existing_lower.split()
        matched_words = sum(word in concept_lower for word in existing_words)
        
        if matched_words > 0:
            return 0.3 + (0.1 * matched_words)
        
        return 0.1  # Default low relevance
    
    def apply_insights_from_reflection(self, reflection_result, context=None):
        """Apply insights from reflection to enrich the knowledge network."""
        if not hasattr(self, 'knowledge_network') or not self.knowledge_network:
            self.initialize_knowledge_network()
        
        insights = {
            "concepts_added": [],
            "relationships_created": [],
            "domains_enriched": set()
        }
        
        # Extract information from the reflection
        if "patterns" in reflection_result and "synthesis" in reflection_result:
            # Add key insights as concepts
            synthesis = reflection_result["synthesis"]
            
            # Process strengths
            for strength in synthesis.get("key_strengths", []):
                concept_id = self.add_knowledge_concept("business", f"Strength: {strength}", 
                                                      {"type": "strength", "source": "reflection"})
                insights["concepts_added"].append({
                    "domain": "business", 
                    "concept": f"Strength: {strength}",
                    "id": concept_id
                })
                insights["domains_enriched"].add("business")
            
            # Process weaknesses
            for weakness in synthesis.get("key_weaknesses", []):
                concept_id = self.add_knowledge_concept("business", f"Weakness: {weakness}", 
                                                      {"type": "weakness", "source": "reflection"})
                insights["concepts_added"].append({
                    "domain": "business", 
                    "concept": f"Weakness: {weakness}",
                    "id": concept_id
                })
                insights["domains_enriched"].add("business")
            
            # Process suggestions
            for suggestion in synthesis.get("key_suggestions", []):
                concept_id = self.add_knowledge_concept("business", f"Action: {suggestion}", 
                                                      {"type": "action", "source": "reflection"})
                insights["concepts_added"].append({
                    "domain": "business", 
                    "concept": f"Action: {suggestion}",
                    "id": concept_id
                })
                insights["domains_enriched"].add("business")
            
            # Create relationships between concepts where appropriate
            for i in range(len(insights["concepts_added"])):
                for j in range(i+1, len(insights["concepts_added"])):
                    # Connect strengths to actions
                    if ("Strength" in insights["concepts_added"][i]["concept"] and 
                        "Action" in insights["concepts_added"][j]["concept"]):
                        relation = self.create_knowledge_link(
                            "business", insights["concepts_added"][i]["id"],
                            "business", insights["concepts_added"][j]["id"],
                            "enables", 0.7)
                        if relation:
                            insights["relationships_created"].append({
                                "source": insights["concepts_added"][i]["concept"],
                                "target": insights["concepts_added"][j]["concept"],
                                "type": "enables"
                            })
                    
                    # Connect weaknesses to actions
                    elif ("Weakness" in insights["concepts_added"][i]["concept"] and 
                          "Action" in insights["concepts_added"][j]["concept"]):
                        relation = self.create_knowledge_link(
                            "business", insights["concepts_added"][i]["id"],
                            "business", insights["concepts_added"][j]["id"],
                            "addressed_by", 0.8)
                        if relation:
                            insights["relationships_created"].append({
                                "source": insights["concepts_added"][i]["concept"],
                                "target": insights["concepts_added"][j]["concept"],
                                "type": "addressed_by"
                            })
        
        insights["domains_enriched"] = list(insights["domains_enriched"])
        return insights

# Function to register metacognition with system
def register_metacognition(system):
    """Register enhanced metacognition with the DeepALL system."""
    metacognition = EnhancedMetacognition(system)
    return metacognition
