"""
Self-Improvement Loop for Agent Zero

This module implements a self-improvement mechanism for Agent Zero based on
the DeepALL system's metacognition and knowledge network capabilities.
It reflects on performance, extracts insights, stores them in a knowledge network,
and suggests improvements.
"""

import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Import the EnhancedMetacognition and TransdisciplinaryKnowledgeNetwork from our implementation
from metacognition import EnhancedMetacognition
from knowledge_network import TransdisciplinaryKnowledgeNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SelfImprovementLoop:
    """Self-improvement loop for Agent Zero."""
    
    def __init__(self, agent_zero_instance=None):
        """Initialize self-improvement loop.
        
        Args:
            agent_zero_instance: The Agent Zero instance (optional)
        """
        self.agent_zero = agent_zero_instance
        self.metacognition = EnhancedMetacognition(agent_zero_instance) if agent_zero_instance else EnhancedMetacognition(None)
        self.knowledge_network = TransdisciplinaryKnowledgeNetwork(agent_zero_instance) if agent_zero_instance else TransdisciplinaryKnowledgeNetwork(None)
        self.improvement_history = []
        self.last_improvement_check = datetime.now()
        self.improvement_interval = timedelta(hours=1)  # Check for improvements every hour
        logger.info("Self-improvement loop initialized")
    
    def reflect_on_task(self, task_description: str, task_result: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Reflect on a completed task to extract insights for self-improvement.
        
        Args:
            task_description: Description of the task that was performed
            task_result: Result of the task (success, metrics, etc.)
            context: Additional context about the task (tools used, time taken, etc.)
            
        Returns:
            dict: Reflection results including insights and improvement suggestions
        """
        logger.info(f"Reflecting on task: {task_description}")
        
        if context is None:
            context = {}
        
        # Create a decision object representing the task performance
        decision = {
            "task_description": task_description,
            "task_result": task_result,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use metacognition to perform multi-perspective reflection
        reflection = self.metacognition.multi_perspective_reflect(decision, context)
        
        # Perform pattern-based synthesis to extract insights
        synthesis_result = self.metacognition.pattern_based_synthesis(decision, context)
        
        # Extract actionable insights for self-improvement
        insights = self._extract_improvement_insights(reflection, synthesis_result, task_description, task_result, context)
        
        # Store insights in the knowledge network
        storage_result = self._store_insights_in_knowledge_network(insights, task_description)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(insights, storage_result)
        
        # Record in improvement history
        improvement_record = {
            "timestamp": datetime.now().isoformat(),
            "task_description": task_description,
            "reflection": reflection,
            "synthesis": synthesis_result,
            "insights": insights,
            "storage_result": storage_result,
            "improvement_suggestions": improvement_suggestions
        }
        self.improvement_history.append(improvement_record)
        
        # Limit history size
        if len(self.improvement_history) > 50:
            self.improvement_history = self.improvement_history[-50:]
        
        result = {
            "reflection": reflection,
            "synthesis": synthesis_result,
            "insights": insights,
            "storage_result": storage_result,
            "improvement_suggestions": improvement_suggestions
        }
        
        logger.info(f"Completed reflection on task: {task_description}. Generated {len(improvement_suggestions)} improvement suggestions.")
        return result
    
    def _extract_improvement_insights(self, reflection: Dict[str, Any], synthesis_result: Dict[str, Any], 
                                    task_description: str, task_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actionable insights for self-improvement from reflection results."""
        insights = {
            "strengths_to_preserve": [],
            "weaknesses_to_address": [],
            "skills_to_develop": [],
            "workflow_adjustments": [],
            "knowledge_gaps": []
        }
        
        # Extract from synthesis
        synthesis = synthesis_result.get("synthesis", {})
        
        # Process key strengths
        for strength in synthesis.get("key_strengths", []):
            insights["strengths_to_preserve"].append({
                "description": strength,
                "source": "synthesis",
                "type": "strength"
            })
        
        # Process key weaknesses
        for weakness in synthesis.get("key_weaknesses", []):
            insights["weaknesses_to_address"].append({
                "description": weakness,
                "source": "synthesis",
                "type": "weakness"
            })
        
        # Process key suggestions as potential improvements
        for suggestion in synthesis.get("key_suggestions", []):
            # Categorize suggestions
            suggestion_lower = suggestion.lower()
            if any(keyword in suggestion_lower for keyword in ["skill", "learn", "training", "study"]):
                insights["skills_to_develop"].append({
                    "description": suggestion,
                    "source": "synthesis_suggestion",
                    "type": "skill_development"
                })
            elif any(keyword in suggestion_lower for keyword in ["workflow", "process", "method", "approach"]):
                insights["workflow_adjustments"].append({
                    "description": suggestion,
                    "source": "synthesis_suggestion",
                    "type": "workflow_adjustment"
                })
            elif any(keyword in suggestion_lower for keyword in ["knowledge", "learn", "understand", "study"]):
                insights["knowledge_gaps"].append({
                    "description": suggestion,
                    "source": "synthesis_suggestion",
                    "type": "knowledge_gap"
                })
            else:
                # Default to workflow adjustment
                insights["workflow_adjustments"].append({
                    "description": suggestion,
                    "source": "synthesis_suggestion",
                    "type": "general_improvement"
                })
        
        # Extract from task result if available
        if isinstance(task_result, dict):
            # Check for performance metrics
            if "performance" in task_result and task_result["performance"] < 0.7:  # Arbitrary threshold
                insights["weaknesses_to_address"].append({
                    "description": f"Low performance score: {task_result['performance']}",
                    "source": "task_result",
                    "type": "performance_issue"
                })
            
            if "error_rate" in task_result and task_result["error_rate"] > 0.3:  # Arbitrary threshold
                insights["weaknesses_to_address"].append({
                    "description": f"High error rate: {task_result['error_rate']}",
                    "source": "task_result",
                    "type": "quality_issue"
                })
            
            # Check for tool usage patterns
            if "tools_used" in task_result:
                tools_used = task_result["tools_used"]
                if isinstance(tools_used, list) and len(tools_used) > 5:
                    insights["workflow_adjustments"].append({
                        "description": f"High tool usage count ({len(tools_used)}). Consider consolidating or planning better.",
                        "source": "task_result",
                        "type": "tool_usage"
                    })
        
        # Extract from context
        if "duration" in context and context["duration"] > 300:  # More than 5 minutes
            insights["workflow_adjustments"].append({
                "description": f"Task took a long time ({context['duration']}s). Look for optimization opportunities.",
                "source": "context",
                "type": "duration_issue"
            })
        
        if "interventions" in context and context["interventions"] > 2:  # More than 2 interventions
            insights["weaknesses_to_address"].append({
                "description": f"Required multiple interventions ({context['interventions']}). May indicate unclear instructions or complexity.",
                "source": "context",
                "type": "intervention_frequency"
            })
        
        return insights
    
    def _store_insights_in_knowledge_network(self, insights: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Store insights in the transdisciplinary knowledge network."""
        logger.info("Storing insights in knowledge network")
        
        storage_result = {
            "concepts_added": [],
            "relationships_created": [],
            "domains_enriched": []
        }
        
        # Add strengths as concepts in the 'performance' domain
        performance_domain = self.knowledge_network.get_or_create_domain(
            "performance", "Performance metrics and strengths"
        )
        for strength in insights.get("strengths_to_preserve", []):
            concept_id = performance_domain.add_concept(
                f"Strength: {strength['description']}",
                {
                    "type": "strength",
                    "source": "self_improvement",
                    "related_task": task_description,
                    "timestamp": datetime.now().isoformat()
                }
            )
            storage_result["concepts_added"].append({
                "domain": "performance",
                "concept": f"Strength: {strength['description']}",
                "id": concept_id,
                "type": strength["type"]
            })
        
        # Add weaknesses as concepts in the 'improvement' domain
        improvement_domain = self.knowledge_network.get_or_create_domain(
            "improvement", "Areas for improvement and weaknesses"
        )
        for weakness in insights.get("weaknesses_to_address", []):
            concept_id = improvement_domain.add_concept(
                f"Weakness: {weakness['description']}",
                {
                    "type": "weakness",
                    "source": "self_improvement",
                    "related_task": task_description,
                    "timestamp": datetime.now().isoformat()
                }
            )
            storage_result["concepts_added"].append({
                "domain": "improvement",
                "concept": f"Weakness: {weakness['description']}",
                "id": concept_id,
                "type": weakness["type"]
            })
        
        # Add skill development needs as concepts in the 'learning' domain
        learning_domain = self.knowledge_network.get_or_create_domain(
            "learning", "Skills to develop and knowledge gaps"
        )
        for skill in insights.get("skills_to_develop", []):
            concept_id = learning_domain.add_concept(
                f"Skill to Develop: {skill['description']}",
                {
                    "type": "skill_development",
                    "source": "self_improvement",
                    "related_task": task_description,
                    "timestamp": datetime.now().isoformat()
                }
            )
            storage_result["concepts_added"].append({
                "domain": "learning",
                "concept": f"Skill to Develop: {skill['description']}",
                "id": concept_id,
                "type": skill["type"]
            })
        
        # Record domains enriched
        domains_enriched = set()
        for concept in storage_result["concepts_added"]:
            domains_enriched.add(concept["domain"])
        storage_result["domains_enriched"] = list(domains_enriched)
        
        logger.info(f"Stored {len(storage_result['concepts_added'])} concepts in knowledge network across {len(storage_result['domains_enriched'])} domains.")
        return storage_result
    def _generate_improvement_suggestions(self, insights: Dict[str, Any], storage_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable improvement suggestions from insights."""
        suggestions = []
        
        # Suggest preserving strengths
        for strength in insights.get("strengths_to_preserve", []):
            suggestions.append({
                "type": "preserve_strength",
                "description": f"Continue to leverage strength: {strength['description']}",
                "priority": "medium",
                "action": "maintain_current_approach"
            })
        
        # Suggest addressing weaknesses
        for weakness in insights.get("weaknesses_to_address", []):
            suggestions.append({
                "type": "address_weakness",
                "description": f"Address weakness: {weakness['description']}",
                "priority": "high" if weakness['type'] in ["performance_issue", "quality_issue"] else "medium",
                "action": "develop_improvement_plan"
            })
        
        # Suggest developing skills
        for skill in insights.get("skills_to_develop", []):
            suggestions.append({
                "type": "develop_skill",
                "description": f"Develop skill: {skill['description']}",
                "priority": "medium",
                "action": "create_learning_plan"
            })
        
        # Suggest workflow adjustments
        for workflow in insights.get("workflow_adjustments", []):
            suggestions.append({
                "type": "adjust_workflow",
                "description": f"Adjust workflow: {workflow['description']}",
                "priority": "medium",
                "action": "refine_process"
            })
        
        # Suggest filling knowledge gaps
        for gap in insights.get("knowledge_gaps", []):
            suggestions.append({
                "type": "fill_knowledge_gap",
                "description": f"Fill knowledge gap: {gap['description']}",
                "priority": "low",
                "action": "research_and_learn"
            })
        
        # Sort by priority (high -> medium -> low)
        priority_order = {"high": 3, "medium": 2, "low": 1}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 0), reverse=True)
        
        return suggestions
    
    def get_improvement_suggestions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent improvement suggestions."""
        if not self.improvement_history:
            return []
        
        # Get suggestions from the most recent reflection
        latest = self.improvement_history[-1]
        suggestions = latest.get("improvement_suggestions", [])
        return suggestions[:limit]
    
    def apply_improvements(self, max_suggestions: int = 3) -> Dict[str, Any]:
        """Apply the top improvement suggestions.
        
        In a full implementation, this would automatically adjust the agent's behavior,
        create new skills, modify workflows, etc. For now, we'll return what actions
        would be taken.
        
        Args:
            max_suggestions: Maximum number of suggestions to apply
            
        Returns:
            dict: Summary of improvements applied
        """
        suggestions = self.get_improvement_suggestions(limit=max_suggestions)
        
        applied = []
        for suggestion in suggestions:
            applied_action = {
                "suggestion": suggestion,
                "action_taken": "logged_for_manual_review",  # In full implementation, this would be automated
                "timestamp": datetime.now().isoformat()
            }
            applied.append(applied_action)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "suggestions_considered": len(suggestions),
            "improvements_applied": len(applied),
            "details": applied
        }
        
        logger.info(f"Applied {len(applied)} improvements based on self-reflection.")
        return result
    
    def get_knowledge_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge network."""
        if not hasattr(self.knowledge_network, 'domains'):
            return {"error": "Knowledge network not initialized"}
        
        stats = {
            "domains": {},
            "total_concepts": 0,
            "total_relationships": 0
        }
        
        for domain_name, domain_obj in self.knowledge_network.domains.items():
            concept_count = len(domain_obj.concepts)
            relationship_count = sum(
                len(concept.get("related_concepts", {}))
                for concept in domain_obj.concepts.values()
            )
            stats["domains"][domain_name] = {
                "concepts": concept_count,
                "relationships": relationship_count
            }
            stats["total_concepts"] += concept_count
            stats["total_relationships"] += relationship_count
        
        # Since each relationship is counted twice (once in each concept), divide by 2
        stats["total_relationships"] = stats["total_relationships"] // 2
        
        return stats


# Function to register self-improvement loop with agent zero
def register_self_improvement_loop(agent_zero_instance):
    """Register self-improvement loop with Agent Zero instance."""
    loop = SelfImprovementLoop(agent_zero_instance)
    return loop


if __name__ == "__main__":
    # Demo / test script
    print("=== Self-Improvement Loop Demo ===")
    
    # Create a self-improvement loop instance (without agent zero for demo)
    loop = SelfImprovementLoop()
    
    # Simulate a task reflection
    task_description = "Analyze customer feedback data to identify product improvement opportunities"
    task_result = {
        "success": True,
        "performance": 0.75,
        "error_rate": 0.2,
        "tools_used": ["document_query", "data_analysis", "visualization"],
        "duration": 420,  # 7 minutes
        "interventions": 1
    }
    context = {
        "user_requirements": "Identify top 3 customer pain points",
        "data_quality": "good",
        "team_size": 1
    }
    
    print(f"\nReflecting on task: {task_description}")
    reflection_result = loop.reflect_on_task(task_description, task_result, context)
    
    print(f"\n--- Improvement Suggestions ---")
    suggestions = loop.get_improvement_suggestions()
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. [{suggestion['priority'].upper()}] {suggestion['description']}")
    
    print(f"\n--- Knowledge Network Stats ---")
    stats = loop.get_knowledge_network_stats()
    print(f"Domains: {list(stats['domains'].keys())}")
    for domain, domain_stats in stats['domains'].items():
        print(f"  {domain}: {domain_stats['concepts']} concepts, {domain_stats['relationships']} relationships")
    print(f"Total: {stats['total_concepts']} concepts, {stats['total_relationships']} relationships")
    
    print(f"\n--- Applying Improvements ---")
    application_result = loop.apply_improvements(max_suggestions=3)
    print(f"Applied {application_result['improvements_applied']} improvements.")
    for i, improvement in enumerate(application_result['details'], 1):
        print(f"{i}. {improvement['suggestion']['description']} -> {improvement['action_taken']}")
    
    print("\n=== Demo Complete ===")
