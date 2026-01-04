"""
Taxonomy-Guided Judge

Enhanced judge that uses error taxonomy for precise feedback.

Instead of simple scoring, provides:
1. Detailed error categorization
2. Severity classification
3. Specific fault localization
4. Actionable suggested fixes
5. High-quality feedback for SFT training

This enables junior agents to learn from precise, structured feedback
rather than vague scores.
"""

from typing import Dict, List, Optional, Any
import json
from loguru import logger

from evaluation.error_taxonomy import (
    ErrorInstance,
    ErrorReport,
    ErrorSeverity,
    ErrorTaxonomyManager
)
from evaluation.fault_localization import FaultLocalizationEngine


class TaxonomyGuidedJudge:
    """
    Enhanced judge using error taxonomy
    
    Combines:
    1. Automated fault detection (FaultLocalizationEngine)
    2. LLM-based evaluation (for complex errors)
    3. Error taxonomy classification
    4. Structured feedback generation
    """
    
    def __init__(self, llm_judge=None):
        """
        Args:
            llm_judge: Optional LLM judge for complex evaluation
        """
        self.fault_engine = FaultLocalizationEngine()
        self.llm_judge = llm_judge
        self.taxonomy_manager = ErrorTaxonomyManager()
    
    def evaluate(
        self,
        agent_type: str,
        agent_output: Dict,
        ground_truth: Optional[Dict] = None,
        use_llm: bool = True
    ) -> ErrorReport:
        """
        Evaluate agent output using taxonomy-guided approach
        
        Args:
            agent_type: Type of agent
            agent_output: Agent's output
            ground_truth: Optional ground truth
            use_llm: Whether to use LLM for additional evaluation
        
        Returns:
            ErrorReport with detailed feedback
        """
        # Step 1: Automated fault detection
        report = self.fault_engine.detect_faults(agent_type, agent_output, ground_truth)
        
        # Step 2: LLM-based evaluation (if enabled and available)
        if use_llm and self.llm_judge:
            llm_errors = self._llm_evaluation(agent_type, agent_output, ground_truth)
            report.errors.extend(llm_errors)
            
            # Re-calculate overall severity
            from evaluation.error_taxonomy import ErrorSeverityClassifier
            classifier = ErrorSeverityClassifier()
            report.overall_severity = classifier.determine_overall_severity(report.errors)
            report.is_acceptable = classifier.is_acceptable(report.overall_severity)
        
        return report
    
    def _llm_evaluation(
        self,
        agent_type: str,
        agent_output: Dict,
        ground_truth: Optional[Dict]
    ) -> List[ErrorInstance]:
        """
        Use LLM to detect complex errors
        
        LLM is good at:
        - Detecting logical fallacies
        - Identifying contradictions
        - Assessing reasoning quality
        - Finding subtle hallucinations
        """
        errors = []
        
        # Build prompt for LLM
        prompt = self._build_llm_prompt(agent_type, agent_output, ground_truth)
        
        # Get LLM response
        try:
            response = self.llm_judge.evaluate(prompt)
            
            # Parse LLM response to extract errors
            llm_errors = self._parse_llm_response(response, agent_type)
            errors.extend(llm_errors)
        
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
        
        return errors
    
    def _build_llm_prompt(
        self,
        agent_type: str,
        agent_output: Dict,
        ground_truth: Optional[Dict]
    ) -> str:
        """Build prompt for LLM evaluation"""
        
        # Get possible error categories
        categories = self.taxonomy_manager.get_all_error_categories(agent_type)
        
        prompt = f"""You are an expert evaluator for a {agent_type} agent in a stock analysis system.

Your task is to identify errors in the agent's output using a structured error taxonomy.

Agent Output:
{json.dumps(agent_output, indent=2)}

"""
        
        if ground_truth:
            prompt += f"""Ground Truth (for comparison):
{json.dumps(ground_truth, indent=2)}

"""
        
        prompt += f"""Error Categories to Check:
{', '.join(categories)}

For each error you find, provide:
1. category: Error category from the list above
2. description: Clear description of the error
3. location: Where in the output (field name)
4. suggested_fix: How to fix it
5. evidence: Evidence of the error (quote from output)

Focus on:
- Logical fallacies and contradictions
- Unsupported claims
- Reasoning quality
- Hallucinations (fabricated data)
- Inappropriate confidence

Output your findings as a JSON array of errors:
[
  {{
    "category": "logical_fallacy",
    "description": "...",
    "location": "reasoning",
    "suggested_fix": "...",
    "evidence": "..."
  }},
  ...
]

If no errors found, return empty array: []
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, agent_type: str) -> List[ErrorInstance]:
        """Parse LLM response to extract errors"""
        errors = []
        
        try:
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                return errors
            
            json_str = response[json_start:json_end]
            error_dicts = json.loads(json_str)
            
            # Convert to ErrorInstance objects
            from evaluation.error_taxonomy import ErrorSeverityClassifier
            classifier = ErrorSeverityClassifier()
            
            for error_dict in error_dicts:
                category = error_dict.get('category', 'unknown')
                severity = classifier.classify(category, agent_type)
                
                error = ErrorInstance(
                    category=category,
                    severity=severity,
                    description=error_dict.get('description', ''),
                    location=error_dict.get('location', 'unknown'),
                    suggested_fix=error_dict.get('suggested_fix', ''),
                    evidence=error_dict.get('evidence')
                )
                
                errors.append(error)
        
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
        
        return errors
    
    def generate_sft_feedback(self, report: ErrorReport) -> str:
        """
        Generate high-quality feedback for SFT training
        
        Format:
        - Clear error descriptions
        - Severity levels
        - Specific locations
        - Actionable fixes
        - Examples of correct behavior
        """
        if not report.errors:
            return "Excellent output! No errors detected."
        
        feedback = f"Found {len(report.errors)} errors in your {report.agent_type} analysis:\n\n"
        
        # Group errors by severity
        fatal_errors = report.get_errors_by_severity(ErrorSeverity.FATAL)
        high_errors = report.get_errors_by_severity(ErrorSeverity.HIGH)
        medium_errors = report.get_errors_by_severity(ErrorSeverity.MEDIUM)
        low_errors = report.get_errors_by_severity(ErrorSeverity.LOW)
        
        # FATAL errors first
        if fatal_errors:
            feedback += "🔴 FATAL ERRORS (Must Fix):\n"
            for i, error in enumerate(fatal_errors, 1):
                feedback += f"{i}. [{error.category}] in '{error.location}'\n"
                feedback += f"   Problem: {error.description}\n"
                feedback += f"   Fix: {error.suggested_fix}\n"
                if error.evidence:
                    feedback += f"   Evidence: {error.evidence}\n"
                feedback += "\n"
        
        # HIGH errors
        if high_errors:
            feedback += "🟠 HIGH PRIORITY ERRORS:\n"
            for i, error in enumerate(high_errors, 1):
                feedback += f"{i}. [{error.category}] in '{error.location}'\n"
                feedback += f"   Problem: {error.description}\n"
                feedback += f"   Fix: {error.suggested_fix}\n"
                if error.evidence:
                    feedback += f"   Evidence: {error.evidence}\n"
                feedback += "\n"
        
        # MEDIUM errors
        if medium_errors:
            feedback += "🟡 MEDIUM PRIORITY ERRORS:\n"
            for i, error in enumerate(medium_errors, 1):
                feedback += f"{i}. [{error.category}] in '{error.location}'\n"
                feedback += f"   Problem: {error.description}\n"
                feedback += f"   Fix: {error.suggested_fix}\n"
                feedback += "\n"
        
        # LOW errors (brief)
        if low_errors:
            feedback += f"🟢 LOW PRIORITY: {len(low_errors)} minor issues\n\n"
        
        # Overall assessment
        feedback += f"\nOverall Severity: {report.overall_severity.value.upper()}\n"
        feedback += f"Output Acceptable: {'Yes' if report.is_acceptable else 'No'}\n"
        
        return feedback
    
    def generate_training_example(
        self,
        agent_type: str,
        agent_output: Dict,
        report: ErrorReport
    ) -> Dict:
        """
        Generate training example for SFT
        
        Format:
        {
            "input": <agent input>,
            "output": <agent output>,
            "feedback": <structured feedback>,
            "corrected_output": <how it should be>,
            "errors": <list of errors>
        }
        """
        feedback = self.generate_sft_feedback(report)
        
        # Generate corrected output (simplified)
        corrected_output = self._generate_corrected_output(agent_output, report)
        
        training_example = {
            "agent_type": agent_type,
            "output": agent_output,
            "feedback": feedback,
            "corrected_output": corrected_output,
            "errors": [e.to_dict() for e in report.errors],
            "overall_severity": report.overall_severity.value,
            "is_acceptable": report.is_acceptable
        }
        
        return training_example
    
    def _generate_corrected_output(
        self,
        agent_output: Dict,
        report: ErrorReport
    ) -> Dict:
        """
        Generate corrected version of output
        
        This is simplified - in practice, would use LLM or human feedback
        """
        corrected = agent_output.copy()
        
        # Apply suggested fixes
        for error in report.errors:
            location = error.location
            
            # Simple corrections
            if error.category == 'missing_field':
                field_name = location
                corrected[field_name] = f"<{field_name} should be provided>"
            
            elif error.category == 'invalid_value':
                # Mark as needs correction
                if '.' in location:
                    parts = location.split('.')
                    if len(parts) == 2 and parts[0] in corrected:
                        corrected[parts[0]][parts[1]] = f"<correct {parts[1]}>"
                else:
                    corrected[location] = f"<correct {location}>"
        
        return corrected


if __name__ == '__main__':
    # Test
    judge = TaxonomyGuidedJudge()
    
    # Test News Agent
    news_output = {
        'sentiment_score': 1.5,
        'confidence': 0.8,
        'key_events': [
            'Strong earnings beat',
            'Positive analyst upgrade'
        ],
        'reasoning': 'Very positive news. Stock looks good.'  # Too short
    }
    
    report = judge.evaluate('news', news_output, use_llm=False)
    
    print("=== Error Report ===")
    print(f"Agent: {report.agent_type}")
    print(f"Errors: {len(report.errors)}")
    print(f"Overall Severity: {report.overall_severity.value}")
    print(f"Acceptable: {report.is_acceptable}")
    print(f"\nSummary: {report.summary}")
    
    print("\n=== SFT Feedback ===")
    feedback = judge.generate_sft_feedback(report)
    print(feedback)
    
    print("\n=== Training Example ===")
    training_example = judge.generate_training_example('news', news_output, report)
    print(json.dumps(training_example, indent=2))
