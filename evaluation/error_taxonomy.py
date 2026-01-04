"""
Error Taxonomy Framework for Junior Agents

Instead of binary "correct/incorrect", provides detailed error classification
with severity levels and specific fault localization.

Key Benefits:
1. Precise feedback for SFT training
2. Severity-based prioritization
3. Actionable error descriptions
4. Agent-specific error categories
5. Enables targeted improvements

Error Severity Levels:
- NEGLIGIBLE: Minor issues (formatting, style)
- LOW: Small inaccuracies (minor calculation errors)
- MEDIUM: Significant issues (missing analysis)
- HIGH: Major problems (wrong recommendation)
- FATAL: Critical errors (hallucinated data, dangerous advice)

Based on:
- Software fault localization research
- Medical error taxonomy (severity classification)
- LLM hallucination detection
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time


class ErrorSeverity(Enum):
    """Error severity levels"""
    NEGLIGIBLE = "negligible"  # Minor formatting, style issues
    LOW = "low"                # Small inaccuracies, non-critical
    MEDIUM = "medium"          # Significant issues affecting quality
    HIGH = "high"              # Major problems affecting correctness
    FATAL = "fatal"            # Critical errors, dangerous advice


class ErrorCategory(Enum):
    """General error categories"""
    # Data errors
    HALLUCINATION = "hallucination"              # Fabricated data
    DATA_INACCURACY = "data_inaccuracy"          # Wrong numbers
    MISSING_DATA = "missing_data"                # Required data absent
    
    # Reasoning errors
    LOGICAL_FALLACY = "logical_fallacy"          # Flawed logic
    INCOMPLETE_ANALYSIS = "incomplete_analysis"  # Missing key factors
    CONTRADICTORY = "contradictory"              # Self-contradicting
    UNSUPPORTED_CLAIM = "unsupported_claim"      # No evidence
    
    # Recommendation errors
    WRONG_RECOMMENDATION = "wrong_recommendation"  # Incorrect action
    INAPPROPRIATE_CONFIDENCE = "inappropriate_confidence"  # Overconfident/underconfident
    MISSING_RISK_ASSESSMENT = "missing_risk_assessment"  # No risk analysis
    
    # Technical errors
    CALCULATION_ERROR = "calculation_error"      # Math mistakes
    INDICATOR_MISUSE = "indicator_misuse"        # Wrong technical indicator usage
    TIMEFRAME_MISMATCH = "timeframe_mismatch"    # Wrong time period
    
    # Formatting errors
    FORMATTING_ERROR = "formatting_error"        # Structure issues
    MISSING_FIELD = "missing_field"              # Required field absent
    INVALID_VALUE = "invalid_value"              # Out of range


# Agent-specific error categories

class NewsAgentError(Enum):
    """News Agent specific errors"""
    SENTIMENT_MISCALCULATION = "sentiment_miscalculation"
    OUTDATED_NEWS = "outdated_news"
    NEWS_HALLUCINATION = "news_hallucination"
    MISSING_SOURCE = "missing_source"
    SENTIMENT_JUSTIFICATION_MISSING = "sentiment_justification_missing"
    WRONG_EVENT_IMPACT = "wrong_event_impact"
    CONFLATING_COMPANIES = "conflating_companies"  # Mixing up different companies


class TechnicalAgentError(Enum):
    """Technical Agent specific errors"""
    INDICATOR_CALCULATION_ERROR = "indicator_calculation_error"
    WRONG_SIGNAL_INTERPRETATION = "wrong_signal_interpretation"
    TIMEFRAME_CONFUSION = "timeframe_confusion"
    SUPPORT_RESISTANCE_HALLUCINATION = "support_resistance_hallucination"
    PATTERN_MISIDENTIFICATION = "pattern_misidentification"
    VOLUME_ANALYSIS_ERROR = "volume_analysis_error"
    TREND_MISCLASSIFICATION = "trend_misclassification"


class FundamentalAgentError(Enum):
    """Fundamental Agent specific errors"""
    VALUATION_CALCULATION_ERROR = "valuation_calculation_error"
    FINANCIAL_RATIO_MISINTERPRETATION = "financial_ratio_misinterpretation"
    EARNINGS_HALLUCINATION = "earnings_hallucination"
    DEBT_ANALYSIS_ERROR = "debt_analysis_error"
    GROWTH_PROJECTION_UNREALISTIC = "growth_projection_unrealistic"
    SECTOR_COMPARISON_INVALID = "sector_comparison_invalid"
    MISSING_KEY_METRIC = "missing_key_metric"


class StrategistAgentError(Enum):
    """Strategist Agent specific errors"""
    CONFLICTING_SIGNALS_UNRESOLVED = "conflicting_signals_unresolved"
    POSITION_SIZE_INAPPROPRIATE = "position_size_inappropriate"
    STOP_LOSS_MISSING = "stop_loss_missing"
    TAKE_PROFIT_UNREALISTIC = "take_profit_unrealistic"
    RISK_REWARD_MISCALCULATION = "risk_reward_miscalculation"
    PORTFOLIO_CONTEXT_IGNORED = "portfolio_context_ignored"
    TIMING_INAPPROPRIATE = "timing_inappropriate"


@dataclass
class ErrorInstance:
    """A specific error instance"""
    category: str  # ErrorCategory or agent-specific
    severity: ErrorSeverity
    description: str
    location: str  # Where in the output (e.g., "reasoning", "sentiment_score")
    suggested_fix: str
    evidence: Optional[str] = None  # Evidence of the error
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            'category': self.category,
            'severity': self.severity.value,
            'description': self.description,
            'location': self.location,
            'suggested_fix': self.suggested_fix,
            'evidence': self.evidence,
            'timestamp': self.timestamp
        }


@dataclass
class ErrorReport:
    """Complete error report for an agent output"""
    agent_type: str  # news, technical, fundamental, strategist
    errors: List[ErrorInstance]
    overall_severity: ErrorSeverity
    is_acceptable: bool  # Whether output is acceptable despite errors
    summary: str
    timestamp: float = field(default_factory=time.time)
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorInstance]:
        """Get all errors of a specific severity"""
        return [e for e in self.errors if e.severity == severity]
    
    def get_fatal_errors(self) -> List[ErrorInstance]:
        """Get all fatal errors"""
        return self.get_errors_by_severity(ErrorSeverity.FATAL)
    
    def has_fatal_errors(self) -> bool:
        """Check if report has any fatal errors"""
        return len(self.get_fatal_errors()) > 0
    
    def get_error_count_by_category(self) -> Dict[str, int]:
        """Count errors by category"""
        counts = {}
        for error in self.errors:
            counts[error.category] = counts.get(error.category, 0) + 1
        return counts
    
    def to_dict(self) -> Dict:
        return {
            'agent_type': self.agent_type,
            'errors': [e.to_dict() for e in self.errors],
            'overall_severity': self.overall_severity.value,
            'is_acceptable': self.is_acceptable,
            'summary': self.summary,
            'timestamp': self.timestamp,
            'error_counts': {
                'total': len(self.errors),
                'fatal': len(self.get_errors_by_severity(ErrorSeverity.FATAL)),
                'high': len(self.get_errors_by_severity(ErrorSeverity.HIGH)),
                'medium': len(self.get_errors_by_severity(ErrorSeverity.MEDIUM)),
                'low': len(self.get_errors_by_severity(ErrorSeverity.LOW)),
                'negligible': len(self.get_errors_by_severity(ErrorSeverity.NEGLIGIBLE))
            }
        }


class ErrorSeverityClassifier:
    """
    Classifies error severity based on category and context
    
    Rules:
    - Hallucinations: FATAL
    - Wrong recommendations: HIGH
    - Missing risk assessment: MEDIUM-HIGH
    - Calculation errors: LOW-MEDIUM
    - Formatting errors: NEGLIGIBLE
    """
    
    # Severity mapping for general categories
    CATEGORY_SEVERITY_MAP = {
        ErrorCategory.HALLUCINATION: ErrorSeverity.FATAL,
        ErrorCategory.DATA_INACCURACY: ErrorSeverity.MEDIUM,
        ErrorCategory.MISSING_DATA: ErrorSeverity.MEDIUM,
        ErrorCategory.LOGICAL_FALLACY: ErrorSeverity.HIGH,
        ErrorCategory.INCOMPLETE_ANALYSIS: ErrorSeverity.MEDIUM,
        ErrorCategory.CONTRADICTORY: ErrorSeverity.HIGH,
        ErrorCategory.UNSUPPORTED_CLAIM: ErrorSeverity.MEDIUM,
        ErrorCategory.WRONG_RECOMMENDATION: ErrorSeverity.HIGH,
        ErrorCategory.INAPPROPRIATE_CONFIDENCE: ErrorSeverity.MEDIUM,
        ErrorCategory.MISSING_RISK_ASSESSMENT: ErrorSeverity.HIGH,
        ErrorCategory.CALCULATION_ERROR: ErrorSeverity.MEDIUM,
        ErrorCategory.INDICATOR_MISUSE: ErrorSeverity.MEDIUM,
        ErrorCategory.TIMEFRAME_MISMATCH: ErrorSeverity.LOW,
        ErrorCategory.FORMATTING_ERROR: ErrorSeverity.NEGLIGIBLE,
        ErrorCategory.MISSING_FIELD: ErrorSeverity.LOW,
        ErrorCategory.INVALID_VALUE: ErrorSeverity.MEDIUM
    }
    
    # Agent-specific severity mappings
    NEWS_AGENT_SEVERITY = {
        NewsAgentError.NEWS_HALLUCINATION: ErrorSeverity.FATAL,
        NewsAgentError.SENTIMENT_MISCALCULATION: ErrorSeverity.MEDIUM,
        NewsAgentError.OUTDATED_NEWS: ErrorSeverity.LOW,
        NewsAgentError.MISSING_SOURCE: ErrorSeverity.LOW,
        NewsAgentError.SENTIMENT_JUSTIFICATION_MISSING: ErrorSeverity.MEDIUM,
        NewsAgentError.WRONG_EVENT_IMPACT: ErrorSeverity.HIGH,
        NewsAgentError.CONFLATING_COMPANIES: ErrorSeverity.FATAL
    }
    
    TECHNICAL_AGENT_SEVERITY = {
        TechnicalAgentError.INDICATOR_CALCULATION_ERROR: ErrorSeverity.HIGH,
        TechnicalAgentError.WRONG_SIGNAL_INTERPRETATION: ErrorSeverity.HIGH,
        TechnicalAgentError.TIMEFRAME_CONFUSION: ErrorSeverity.MEDIUM,
        TechnicalAgentError.SUPPORT_RESISTANCE_HALLUCINATION: ErrorSeverity.FATAL,
        TechnicalAgentError.PATTERN_MISIDENTIFICATION: ErrorSeverity.MEDIUM,
        TechnicalAgentError.VOLUME_ANALYSIS_ERROR: ErrorSeverity.LOW,
        TechnicalAgentError.TREND_MISCLASSIFICATION: ErrorSeverity.HIGH
    }
    
    FUNDAMENTAL_AGENT_SEVERITY = {
        FundamentalAgentError.VALUATION_CALCULATION_ERROR: ErrorSeverity.HIGH,
        FundamentalAgentError.FINANCIAL_RATIO_MISINTERPRETATION: ErrorSeverity.MEDIUM,
        FundamentalAgentError.EARNINGS_HALLUCINATION: ErrorSeverity.FATAL,
        FundamentalAgentError.DEBT_ANALYSIS_ERROR: ErrorSeverity.HIGH,
        FundamentalAgentError.GROWTH_PROJECTION_UNREALISTIC: ErrorSeverity.MEDIUM,
        FundamentalAgentError.SECTOR_COMPARISON_INVALID: ErrorSeverity.LOW,
        FundamentalAgentError.MISSING_KEY_METRIC: ErrorSeverity.MEDIUM
    }
    
    STRATEGIST_AGENT_SEVERITY = {
        StrategistAgentError.CONFLICTING_SIGNALS_UNRESOLVED: ErrorSeverity.HIGH,
        StrategistAgentError.POSITION_SIZE_INAPPROPRIATE: ErrorSeverity.HIGH,
        StrategistAgentError.STOP_LOSS_MISSING: ErrorSeverity.FATAL,
        StrategistAgentError.TAKE_PROFIT_UNREALISTIC: ErrorSeverity.MEDIUM,
        StrategistAgentError.RISK_REWARD_MISCALCULATION: ErrorSeverity.HIGH,
        StrategistAgentError.PORTFOLIO_CONTEXT_IGNORED: ErrorSeverity.MEDIUM,
        StrategistAgentError.TIMING_INAPPROPRIATE: ErrorSeverity.MEDIUM
    }
    
    @classmethod
    def classify(cls, category: str, agent_type: Optional[str] = None) -> ErrorSeverity:
        """
        Classify error severity based on category and agent type
        
        Args:
            category: Error category (string)
            agent_type: Agent type (news, technical, fundamental, strategist)
        
        Returns:
            ErrorSeverity
        """
        # Try agent-specific mapping first
        if agent_type == 'news':
            try:
                error_enum = NewsAgentError(category)
                return cls.NEWS_AGENT_SEVERITY.get(error_enum, ErrorSeverity.MEDIUM)
            except ValueError:
                pass
        
        elif agent_type == 'technical':
            try:
                error_enum = TechnicalAgentError(category)
                return cls.TECHNICAL_AGENT_SEVERITY.get(error_enum, ErrorSeverity.MEDIUM)
            except ValueError:
                pass
        
        elif agent_type == 'fundamental':
            try:
                error_enum = FundamentalAgentError(category)
                return cls.FUNDAMENTAL_AGENT_SEVERITY.get(error_enum, ErrorSeverity.MEDIUM)
            except ValueError:
                pass
        
        elif agent_type == 'strategist':
            try:
                error_enum = StrategistAgentError(category)
                return cls.STRATEGIST_AGENT_SEVERITY.get(error_enum, ErrorSeverity.MEDIUM)
            except ValueError:
                pass
        
        # Fall back to general category mapping
        try:
            error_enum = ErrorCategory(category)
            return cls.CATEGORY_SEVERITY_MAP.get(error_enum, ErrorSeverity.MEDIUM)
        except ValueError:
            # Unknown category, default to MEDIUM
            return ErrorSeverity.MEDIUM
    
    @classmethod
    def determine_overall_severity(cls, errors: List[ErrorInstance]) -> ErrorSeverity:
        """
        Determine overall severity from list of errors
        
        Rules:
        - Any FATAL error → Overall FATAL
        - 2+ HIGH errors → Overall FATAL
        - Any HIGH error → Overall HIGH
        - 3+ MEDIUM errors → Overall HIGH
        - Any MEDIUM error → Overall MEDIUM
        - Otherwise → Lowest severity
        """
        if not errors:
            return ErrorSeverity.NEGLIGIBLE
        
        severity_counts = {
            ErrorSeverity.FATAL: 0,
            ErrorSeverity.HIGH: 0,
            ErrorSeverity.MEDIUM: 0,
            ErrorSeverity.LOW: 0,
            ErrorSeverity.NEGLIGIBLE: 0
        }
        
        for error in errors:
            severity_counts[error.severity] += 1
        
        # Rules
        if severity_counts[ErrorSeverity.FATAL] > 0:
            return ErrorSeverity.FATAL
        
        if severity_counts[ErrorSeverity.HIGH] >= 2:
            return ErrorSeverity.FATAL
        
        if severity_counts[ErrorSeverity.HIGH] > 0:
            return ErrorSeverity.HIGH
        
        if severity_counts[ErrorSeverity.MEDIUM] >= 3:
            return ErrorSeverity.HIGH
        
        if severity_counts[ErrorSeverity.MEDIUM] > 0:
            return ErrorSeverity.MEDIUM
        
        if severity_counts[ErrorSeverity.LOW] > 0:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.NEGLIGIBLE
    
    @classmethod
    def is_acceptable(cls, overall_severity: ErrorSeverity) -> bool:
        """
        Determine if output is acceptable given overall severity
        
        Rules:
        - FATAL: Not acceptable
        - HIGH: Not acceptable
        - MEDIUM: Acceptable with warnings
        - LOW: Acceptable
        - NEGLIGIBLE: Acceptable
        """
        return overall_severity in [
            ErrorSeverity.MEDIUM,
            ErrorSeverity.LOW,
            ErrorSeverity.NEGLIGIBLE
        ]


class ErrorTaxonomyManager:
    """
    Manages error taxonomy and provides utilities
    """
    
    @staticmethod
    def get_all_error_categories(agent_type: str) -> List[str]:
        """Get all possible error categories for an agent type"""
        general_categories = [e.value for e in ErrorCategory]
        
        if agent_type == 'news':
            agent_categories = [e.value for e in NewsAgentError]
        elif agent_type == 'technical':
            agent_categories = [e.value for e in TechnicalAgentError]
        elif agent_type == 'fundamental':
            agent_categories = [e.value for e in FundamentalAgentError]
        elif agent_type == 'strategist':
            agent_categories = [e.value for e in StrategistAgentError]
        else:
            agent_categories = []
        
        return general_categories + agent_categories
    
    @staticmethod
    def get_error_description(category: str, agent_type: str) -> str:
        """Get human-readable description of error category"""
        descriptions = {
            # General
            'hallucination': 'Fabricated or made-up data not present in source',
            'data_inaccuracy': 'Incorrect numerical values or facts',
            'missing_data': 'Required data points are absent',
            'logical_fallacy': 'Flawed reasoning or logical errors',
            'incomplete_analysis': 'Missing key factors or considerations',
            'contradictory': 'Self-contradicting statements',
            'unsupported_claim': 'Claims without evidence or justification',
            'wrong_recommendation': 'Incorrect action recommendation',
            'inappropriate_confidence': 'Confidence level not justified by analysis',
            'missing_risk_assessment': 'No risk analysis provided',
            'calculation_error': 'Mathematical or computational mistakes',
            'indicator_misuse': 'Incorrect use of technical indicators',
            'timeframe_mismatch': 'Wrong time period or timeframe',
            'formatting_error': 'Structural or formatting issues',
            'missing_field': 'Required field is absent',
            'invalid_value': 'Value is out of acceptable range',
            
            # News Agent
            'sentiment_miscalculation': 'Sentiment score does not match news content',
            'outdated_news': 'Using old or stale news',
            'news_hallucination': 'Fabricated news events or sources',
            'missing_source': 'No source provided for news',
            'sentiment_justification_missing': 'No explanation for sentiment score',
            'wrong_event_impact': 'Misassessed impact of news event',
            'conflating_companies': 'Mixing up different companies',
            
            # Technical Agent
            'indicator_calculation_error': 'Incorrect indicator calculation',
            'wrong_signal_interpretation': 'Misinterpreted technical signal',
            'timeframe_confusion': 'Confused timeframes (e.g., daily vs weekly)',
            'support_resistance_hallucination': 'Fabricated support/resistance levels',
            'pattern_misidentification': 'Incorrectly identified chart pattern',
            'volume_analysis_error': 'Wrong volume interpretation',
            'trend_misclassification': 'Incorrectly classified trend direction',
            
            # Fundamental Agent
            'valuation_calculation_error': 'Incorrect valuation calculation',
            'financial_ratio_misinterpretation': 'Misinterpreted financial ratio',
            'earnings_hallucination': 'Fabricated earnings data',
            'debt_analysis_error': 'Incorrect debt analysis',
            'growth_projection_unrealistic': 'Unrealistic growth projections',
            'sector_comparison_invalid': 'Invalid sector comparison',
            'missing_key_metric': 'Key financial metric missing',
            
            # Strategist Agent
            'conflicting_signals_unresolved': 'Conflicting signals not addressed',
            'position_size_inappropriate': 'Position size not appropriate for risk',
            'stop_loss_missing': 'No stop loss defined',
            'take_profit_unrealistic': 'Unrealistic take profit target',
            'risk_reward_miscalculation': 'Incorrect risk-reward ratio',
            'portfolio_context_ignored': 'Portfolio context not considered',
            'timing_inappropriate': 'Inappropriate timing for entry/exit'
        }
        
        return descriptions.get(category, 'Unknown error category')
    
    @staticmethod
    def get_suggested_fix(category: str, agent_type: str) -> str:
        """Get suggested fix for error category"""
        fixes = {
            # General
            'hallucination': 'Verify all data against reliable sources. Do not fabricate.',
            'data_inaccuracy': 'Double-check numerical values and facts.',
            'missing_data': 'Include all required data points.',
            'logical_fallacy': 'Review reasoning for logical consistency.',
            'incomplete_analysis': 'Consider all relevant factors.',
            'contradictory': 'Ensure statements are consistent.',
            'unsupported_claim': 'Provide evidence for all claims.',
            'wrong_recommendation': 'Re-evaluate recommendation based on analysis.',
            'inappropriate_confidence': 'Calibrate confidence to match analysis quality.',
            'missing_risk_assessment': 'Include comprehensive risk analysis.',
            'calculation_error': 'Verify all calculations.',
            'indicator_misuse': 'Use indicators correctly per their definition.',
            'timeframe_mismatch': 'Ensure consistent timeframe usage.',
            'formatting_error': 'Fix formatting to match required structure.',
            'missing_field': 'Include all required fields.',
            'invalid_value': 'Ensure value is within acceptable range.',
            
            # News Agent
            'sentiment_miscalculation': 'Recalculate sentiment based on news content.',
            'outdated_news': 'Use recent news only.',
            'news_hallucination': 'Only use verified news sources. Do not fabricate.',
            'missing_source': 'Cite source for all news.',
            'sentiment_justification_missing': 'Explain sentiment score calculation.',
            'wrong_event_impact': 'Reassess event impact on stock price.',
            'conflating_companies': 'Verify company identity. Do not mix up companies.',
            
            # Technical Agent
            'indicator_calculation_error': 'Recalculate indicator using correct formula.',
            'wrong_signal_interpretation': 'Review signal interpretation guidelines.',
            'timeframe_confusion': 'Clarify and use consistent timeframe.',
            'support_resistance_hallucination': 'Identify support/resistance from actual price data.',
            'pattern_misidentification': 'Verify pattern against chart pattern definitions.',
            'volume_analysis_error': 'Reanalyze volume in context of price action.',
            'trend_misclassification': 'Reclassify trend based on price movement.',
            
            # Fundamental Agent
            'valuation_calculation_error': 'Recalculate valuation using correct method.',
            'financial_ratio_misinterpretation': 'Review ratio interpretation guidelines.',
            'earnings_hallucination': 'Use only verified earnings data.',
            'debt_analysis_error': 'Reanalyze debt levels and coverage ratios.',
            'growth_projection_unrealistic': 'Use realistic growth assumptions.',
            'sector_comparison_invalid': 'Compare only within same sector.',
            'missing_key_metric': 'Include key metrics (P/E, ROE, etc.).',
            
            # Strategist Agent
            'conflicting_signals_unresolved': 'Address and resolve conflicting signals.',
            'position_size_inappropriate': 'Adjust position size based on risk.',
            'stop_loss_missing': 'Define stop loss level.',
            'take_profit_unrealistic': 'Set realistic take profit target.',
            'risk_reward_miscalculation': 'Recalculate risk-reward ratio.',
            'portfolio_context_ignored': 'Consider portfolio allocation and diversification.',
            'timing_inappropriate': 'Reassess entry/exit timing.'
        }
        
        return fixes.get(category, 'Review and correct the error.')


if __name__ == '__main__':
    # Test
    classifier = ErrorSeverityClassifier()
    
    # Test severity classification
    severity = classifier.classify('hallucination')
    print(f"Hallucination severity: {severity.value}")
    
    severity = classifier.classify('news_hallucination', 'news')
    print(f"News hallucination severity: {severity.value}")
    
    # Test error report
    errors = [
        ErrorInstance(
            category='news_hallucination',
            severity=ErrorSeverity.FATAL,
            description='Fabricated earnings announcement',
            location='key_events',
            suggested_fix='Use only verified news sources',
            evidence='No such announcement found in official sources'
        ),
        ErrorInstance(
            category='sentiment_miscalculation',
            severity=ErrorSeverity.MEDIUM,
            description='Sentiment score too high for negative news',
            location='sentiment_score',
            suggested_fix='Recalculate sentiment based on news content'
        )
    ]
    
    overall = classifier.determine_overall_severity(errors)
    print(f"Overall severity: {overall.value}")
    
    acceptable = classifier.is_acceptable(overall)
    print(f"Acceptable: {acceptable}")
    
    report = ErrorReport(
        agent_type='news',
        errors=errors,
        overall_severity=overall,
        is_acceptable=acceptable,
        summary='Found 1 fatal error (news hallucination) and 1 medium error'
    )
    
    print(f"\nError Report:")
    print(f"Agent: {report.agent_type}")
    print(f"Total errors: {len(report.errors)}")
    print(f"Fatal errors: {len(report.get_fatal_errors())}")
    print(f"Overall severity: {report.overall_severity.value}")
    print(f"Acceptable: {report.is_acceptable}")
