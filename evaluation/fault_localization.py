"""
Fault Localization Engine

Automatically detects and localizes errors in agent outputs.

For each agent type, provides specific fault detection rules:
- News Agent: Sentiment calculation, source verification, hallucination detection
- Technical Agent: Indicator calculation, signal interpretation, pattern recognition
- Fundamental Agent: Valuation calculation, ratio interpretation, data verification
- Strategist Agent: Risk management, position sizing, signal resolution

Key Features:
1. Agent-specific fault detection
2. Precise error localization (field-level)
3. Evidence extraction
4. Suggested fixes
5. Severity classification
"""

import re
from typing import Dict, List, Optional, Any
from loguru import logger

from evaluation.error_taxonomy import (
    ErrorInstance,
    ErrorReport,
    ErrorSeverity,
    ErrorCategory,
    NewsAgentError,
    TechnicalAgentError,
    FundamentalAgentError,
    StrategistAgentError,
    ErrorSeverityClassifier
)


class FaultDetector:
    """Base class for fault detection"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.classifier = ErrorSeverityClassifier()
    
    def detect(self, agent_output: Dict, ground_truth: Optional[Dict] = None) -> ErrorReport:
        """
        Detect faults in agent output
        
        Args:
            agent_output: Agent's output
            ground_truth: Optional ground truth for comparison
        
        Returns:
            ErrorReport with detected errors
        """
        raise NotImplementedError


class NewsAgentFaultDetector(FaultDetector):
    """Fault detector for News Agent"""
    
    def __init__(self):
        super().__init__('news')
    
    def detect(self, agent_output: Dict, ground_truth: Optional[Dict] = None) -> ErrorReport:
        """Detect faults in News Agent output"""
        errors = []
        
        # Check required fields
        errors.extend(self._check_required_fields(agent_output))
        
        # Check sentiment calculation
        errors.extend(self._check_sentiment(agent_output))
        
        # Check sources
        errors.extend(self._check_sources(agent_output))
        
        # Check for hallucinations
        errors.extend(self._check_hallucinations(agent_output, ground_truth))
        
        # Check event impact
        errors.extend(self._check_event_impact(agent_output))
        
        # Determine overall severity
        overall_severity = self.classifier.determine_overall_severity(errors)
        is_acceptable = self.classifier.is_acceptable(overall_severity)
        
        # Generate summary
        summary = self._generate_summary(errors)
        
        return ErrorReport(
            agent_type=self.agent_type,
            errors=errors,
            overall_severity=overall_severity,
            is_acceptable=is_acceptable,
            summary=summary
        )
    
    def _check_required_fields(self, output: Dict) -> List[ErrorInstance]:
        """Check if all required fields are present"""
        errors = []
        required_fields = ['sentiment_score', 'confidence', 'key_events', 'reasoning']
        
        for field in required_fields:
            if field not in output or output[field] is None:
                errors.append(ErrorInstance(
                    category=ErrorCategory.MISSING_FIELD.value,
                    severity=self.classifier.classify(ErrorCategory.MISSING_FIELD.value, self.agent_type),
                    description=f'Required field "{field}" is missing',
                    location=field,
                    suggested_fix=f'Include {field} in output'
                ))
        
        return errors
    
    def _check_sentiment(self, output: Dict) -> List[ErrorInstance]:
        """Check sentiment calculation"""
        errors = []
        
        sentiment = output.get('sentiment_score')
        reasoning = output.get('reasoning', '')
        
        if sentiment is None:
            return errors
        
        # Check range
        if not (-2.0 <= sentiment <= 2.0):
            errors.append(ErrorInstance(
                category=ErrorCategory.INVALID_VALUE.value,
                severity=self.classifier.classify(ErrorCategory.INVALID_VALUE.value, self.agent_type),
                description=f'Sentiment score {sentiment} is out of range [-2, 2]',
                location='sentiment_score',
                suggested_fix='Ensure sentiment score is between -2 and 2',
                evidence=f'sentiment_score = {sentiment}'
            ))
        
        # Check justification
        if not reasoning or len(reasoning) < 50:
            errors.append(ErrorInstance(
                category=NewsAgentError.SENTIMENT_JUSTIFICATION_MISSING.value,
                severity=self.classifier.classify(NewsAgentError.SENTIMENT_JUSTIFICATION_MISSING.value, self.agent_type),
                description='Sentiment score lacks sufficient justification',
                location='reasoning',
                suggested_fix='Provide detailed explanation for sentiment score',
                evidence=f'reasoning length = {len(reasoning)}'
            ))
        
        # Check consistency (simple heuristic)
        positive_words = ['positive', 'bullish', 'good', 'strong', 'growth', 'beat', 'exceed']
        negative_words = ['negative', 'bearish', 'bad', 'weak', 'decline', 'miss', 'below']
        
        reasoning_lower = reasoning.lower()
        positive_count = sum(1 for word in positive_words if word in reasoning_lower)
        negative_count = sum(1 for word in negative_words if word in reasoning_lower)
        
        if sentiment > 0.5 and negative_count > positive_count:
            errors.append(ErrorInstance(
                category=NewsAgentError.SENTIMENT_MISCALCULATION.value,
                severity=self.classifier.classify(NewsAgentError.SENTIMENT_MISCALCULATION.value, self.agent_type),
                description='Positive sentiment score but reasoning is mostly negative',
                location='sentiment_score',
                suggested_fix='Recalculate sentiment to match reasoning',
                evidence=f'sentiment={sentiment}, positive_words={positive_count}, negative_words={negative_count}'
            ))
        
        elif sentiment < -0.5 and positive_count > negative_count:
            errors.append(ErrorInstance(
                category=NewsAgentError.SENTIMENT_MISCALCULATION.value,
                severity=self.classifier.classify(NewsAgentError.SENTIMENT_MISCALCULATION.value, self.agent_type),
                description='Negative sentiment score but reasoning is mostly positive',
                location='sentiment_score',
                suggested_fix='Recalculate sentiment to match reasoning',
                evidence=f'sentiment={sentiment}, positive_words={positive_count}, negative_words={negative_count}'
            ))
        
        return errors
    
    def _check_sources(self, output: Dict) -> List[ErrorInstance]:
        """Check if sources are provided"""
        errors = []
        
        key_events = output.get('key_events', [])
        
        if not key_events:
            return errors
        
        # Check if events have sources
        events_without_sources = []
        for i, event in enumerate(key_events):
            if isinstance(event, dict):
                if 'source' not in event or not event['source']:
                    events_without_sources.append(i)
            elif isinstance(event, str):
                # String events should ideally have source info
                if 'source:' not in event.lower():
                    events_without_sources.append(i)
        
        if events_without_sources:
            errors.append(ErrorInstance(
                category=NewsAgentError.MISSING_SOURCE.value,
                severity=self.classifier.classify(NewsAgentError.MISSING_SOURCE.value, self.agent_type),
                description=f'{len(events_without_sources)} events lack source attribution',
                location='key_events',
                suggested_fix='Cite source for all news events',
                evidence=f'Events without sources: {events_without_sources}'
            ))
        
        return errors
    
    def _check_hallucinations(self, output: Dict, ground_truth: Optional[Dict]) -> List[ErrorInstance]:
        """Check for hallucinated news"""
        errors = []
        
        # If no ground truth, can't verify
        if not ground_truth:
            return errors
        
        key_events = output.get('key_events', [])
        true_events = ground_truth.get('verified_events', [])
        
        # Simple check: if output has events but ground truth has none
        if key_events and not true_events:
            errors.append(ErrorInstance(
                category=NewsAgentError.NEWS_HALLUCINATION.value,
                severity=self.classifier.classify(NewsAgentError.NEWS_HALLUCINATION.value, self.agent_type),
                description='Reported news events that do not exist',
                location='key_events',
                suggested_fix='Only report verified news events. Do not fabricate.',
                evidence=f'Reported {len(key_events)} events, but 0 verified events exist'
            ))
        
        return errors
    
    def _check_event_impact(self, output: Dict) -> List[ErrorInstance]:
        """Check if event impact assessment is reasonable"""
        errors = []
        
        key_events = output.get('key_events', [])
        sentiment = output.get('sentiment_score', 0)
        
        # If high sentiment but no events, or vice versa
        if abs(sentiment) > 1.5 and not key_events:
            errors.append(ErrorInstance(
                category=NewsAgentError.WRONG_EVENT_IMPACT.value,
                severity=self.classifier.classify(NewsAgentError.WRONG_EVENT_IMPACT.value, self.agent_type),
                description='High sentiment score but no key events provided',
                location='key_events',
                suggested_fix='Provide key events that justify sentiment score',
                evidence=f'sentiment={sentiment}, key_events={len(key_events)}'
            ))
        
        return errors
    
    def _generate_summary(self, errors: List[ErrorInstance]) -> str:
        """Generate error summary"""
        if not errors:
            return 'No errors detected'
        
        fatal = len([e for e in errors if e.severity == ErrorSeverity.FATAL])
        high = len([e for e in errors if e.severity == ErrorSeverity.HIGH])
        medium = len([e for e in errors if e.severity == ErrorSeverity.MEDIUM])
        
        summary = f'Found {len(errors)} errors: '
        parts = []
        if fatal > 0:
            parts.append(f'{fatal} fatal')
        if high > 0:
            parts.append(f'{high} high')
        if medium > 0:
            parts.append(f'{medium} medium')
        
        summary += ', '.join(parts)
        
        return summary


class TechnicalAgentFaultDetector(FaultDetector):
    """Fault detector for Technical Agent"""
    
    def __init__(self):
        super().__init__('technical')
    
    def detect(self, agent_output: Dict, ground_truth: Optional[Dict] = None) -> ErrorReport:
        """Detect faults in Technical Agent output"""
        errors = []
        
        # Check required fields
        errors.extend(self._check_required_fields(agent_output))
        
        # Check indicators
        errors.extend(self._check_indicators(agent_output, ground_truth))
        
        # Check signal interpretation
        errors.extend(self._check_signals(agent_output))
        
        # Check support/resistance
        errors.extend(self._check_support_resistance(agent_output, ground_truth))
        
        # Determine overall severity
        overall_severity = self.classifier.determine_overall_severity(errors)
        is_acceptable = self.classifier.is_acceptable(overall_severity)
        
        summary = self._generate_summary(errors)
        
        return ErrorReport(
            agent_type=self.agent_type,
            errors=errors,
            overall_severity=overall_severity,
            is_acceptable=is_acceptable,
            summary=summary
        )
    
    def _check_required_fields(self, output: Dict) -> List[ErrorInstance]:
        """Check required fields"""
        errors = []
        required_fields = ['signal', 'confidence', 'indicators', 'reasoning']
        
        for field in required_fields:
            if field not in output or output[field] is None:
                errors.append(ErrorInstance(
                    category=ErrorCategory.MISSING_FIELD.value,
                    severity=self.classifier.classify(ErrorCategory.MISSING_FIELD.value, self.agent_type),
                    description=f'Required field "{field}" is missing',
                    location=field,
                    suggested_fix=f'Include {field} in output'
                ))
        
        return errors
    
    def _check_indicators(self, output: Dict, ground_truth: Optional[Dict]) -> List[ErrorInstance]:
        """Check indicator calculations"""
        errors = []
        
        indicators = output.get('indicators', {})
        
        if not indicators:
            return errors
        
        # Check RSI range
        if 'RSI' in indicators:
            rsi = indicators['RSI']
            if not (0 <= rsi <= 100):
                errors.append(ErrorInstance(
                    category=TechnicalAgentError.INDICATOR_CALCULATION_ERROR.value,
                    severity=self.classifier.classify(TechnicalAgentError.INDICATOR_CALCULATION_ERROR.value, self.agent_type),
                    description=f'RSI value {rsi} is out of valid range [0, 100]',
                    location='indicators.RSI',
                    suggested_fix='Recalculate RSI using correct formula',
                    evidence=f'RSI = {rsi}'
                ))
        
        # If ground truth available, compare
        if ground_truth and 'indicators' in ground_truth:
            true_indicators = ground_truth['indicators']
            
            for indicator_name, true_value in true_indicators.items():
                if indicator_name in indicators:
                    output_value = indicators[indicator_name]
                    
                    # Allow 5% tolerance
                    if abs(output_value - true_value) / abs(true_value) > 0.05:
                        errors.append(ErrorInstance(
                            category=TechnicalAgentError.INDICATOR_CALCULATION_ERROR.value,
                            severity=self.classifier.classify(TechnicalAgentError.INDICATOR_CALCULATION_ERROR.value, self.agent_type),
                            description=f'{indicator_name} calculation error: {output_value} vs {true_value}',
                            location=f'indicators.{indicator_name}',
                            suggested_fix=f'Recalculate {indicator_name} correctly',
                            evidence=f'output={output_value}, expected={true_value}'
                        ))
        
        return errors
    
    def _check_signals(self, output: Dict) -> List[ErrorInstance]:
        """Check signal interpretation"""
        errors = []
        
        signal = output.get('signal')
        indicators = output.get('indicators', {})
        reasoning = output.get('reasoning', '')
        
        if not signal or not indicators:
            return errors
        
        # Check RSI signal consistency
        if 'RSI' in indicators:
            rsi = indicators['RSI']
            
            if signal == 'bullish' and rsi > 70:
                errors.append(ErrorInstance(
                    category=TechnicalAgentError.WRONG_SIGNAL_INTERPRETATION.value,
                    severity=self.classifier.classify(TechnicalAgentError.WRONG_SIGNAL_INTERPRETATION.value, self.agent_type),
                    description=f'Bullish signal but RSI is overbought ({rsi})',
                    location='signal',
                    suggested_fix='Reconsider signal given overbought RSI',
                    evidence=f'signal={signal}, RSI={rsi}'
                ))
            
            elif signal == 'bearish' and rsi < 30:
                errors.append(ErrorInstance(
                    category=TechnicalAgentError.WRONG_SIGNAL_INTERPRETATION.value,
                    severity=self.classifier.classify(TechnicalAgentError.WRONG_SIGNAL_INTERPRETATION.value, self.agent_type),
                    description=f'Bearish signal but RSI is oversold ({rsi})',
                    location='signal',
                    suggested_fix='Reconsider signal given oversold RSI',
                    evidence=f'signal={signal}, RSI={rsi}'
                ))
        
        return errors
    
    def _check_support_resistance(self, output: Dict, ground_truth: Optional[Dict]) -> List[ErrorInstance]:
        """Check support/resistance levels"""
        errors = []
        
        if not ground_truth:
            return errors
        
        support = output.get('support_level')
        resistance = output.get('resistance_level')
        current_price = ground_truth.get('current_price')
        
        if support and current_price and support > current_price:
            errors.append(ErrorInstance(
                category=TechnicalAgentError.SUPPORT_RESISTANCE_HALLUCINATION.value,
                severity=self.classifier.classify(TechnicalAgentError.SUPPORT_RESISTANCE_HALLUCINATION.value, self.agent_type),
                description=f'Support level ({support}) is above current price ({current_price})',
                location='support_level',
                suggested_fix='Support must be below current price',
                evidence=f'support={support}, current_price={current_price}'
            ))
        
        if resistance and current_price and resistance < current_price:
            errors.append(ErrorInstance(
                category=TechnicalAgentError.SUPPORT_RESISTANCE_HALLUCINATION.value,
                severity=self.classifier.classify(TechnicalAgentError.SUPPORT_RESISTANCE_HALLUCINATION.value, self.agent_type),
                description=f'Resistance level ({resistance}) is below current price ({current_price})',
                location='resistance_level',
                suggested_fix='Resistance must be above current price',
                evidence=f'resistance={resistance}, current_price={current_price}'
            ))
        
        return errors
    
    def _generate_summary(self, errors: List[ErrorInstance]) -> str:
        """Generate error summary"""
        if not errors:
            return 'No errors detected'
        
        fatal = len([e for e in errors if e.severity == ErrorSeverity.FATAL])
        high = len([e for e in errors if e.severity == ErrorSeverity.HIGH])
        medium = len([e for e in errors if e.severity == ErrorSeverity.MEDIUM])
        
        summary = f'Found {len(errors)} errors: '
        parts = []
        if fatal > 0:
            parts.append(f'{fatal} fatal')
        if high > 0:
            parts.append(f'{high} high')
        if medium > 0:
            parts.append(f'{medium} medium')
        
        summary += ', '.join(parts)
        
        return summary


class FundamentalAgentFaultDetector(FaultDetector):
    """Fault detector for Fundamental Agent"""
    
    def __init__(self):
        super().__init__('fundamental')
    
    def detect(self, agent_output: Dict, ground_truth: Optional[Dict] = None) -> ErrorReport:
        """Detect faults in Fundamental Agent output"""
        errors = []
        
        # Check required fields
        errors.extend(self._check_required_fields(agent_output))
        
        # Check valuation
        errors.extend(self._check_valuation(agent_output, ground_truth))
        
        # Check financial ratios
        errors.extend(self._check_ratios(agent_output, ground_truth))
        
        # Determine overall severity
        overall_severity = self.classifier.determine_overall_severity(errors)
        is_acceptable = self.classifier.is_acceptable(overall_severity)
        
        summary = self._generate_summary(errors)
        
        return ErrorReport(
            agent_type=self.agent_type,
            errors=errors,
            overall_severity=overall_severity,
            is_acceptable=is_acceptable,
            summary=summary
        )
    
    def _check_required_fields(self, output: Dict) -> List[ErrorInstance]:
        """Check required fields"""
        errors = []
        required_fields = ['valuation', 'confidence', 'key_metrics', 'reasoning']
        
        for field in required_fields:
            if field not in output or output[field] is None:
                errors.append(ErrorInstance(
                    category=ErrorCategory.MISSING_FIELD.value,
                    severity=self.classifier.classify(ErrorCategory.MISSING_FIELD.value, self.agent_type),
                    description=f'Required field "{field}" is missing',
                    location=field,
                    suggested_fix=f'Include {field} in output'
                ))
        
        return errors
    
    def _check_valuation(self, output: Dict, ground_truth: Optional[Dict]) -> List[ErrorInstance]:
        """Check valuation calculation"""
        errors = []
        
        valuation = output.get('valuation')
        
        if not valuation:
            return errors
        
        # Check if valuation is one of expected values
        valid_valuations = ['undervalued', 'fairly_valued', 'overvalued']
        if valuation not in valid_valuations:
            errors.append(ErrorInstance(
                category=ErrorCategory.INVALID_VALUE.value,
                severity=self.classifier.classify(ErrorCategory.INVALID_VALUE.value, self.agent_type),
                description=f'Invalid valuation "{valuation}"',
                location='valuation',
                suggested_fix=f'Use one of: {valid_valuations}',
                evidence=f'valuation = {valuation}'
            ))
        
        return errors
    
    def _check_ratios(self, output: Dict, ground_truth: Optional[Dict]) -> List[ErrorInstance]:
        """Check financial ratios"""
        errors = []
        
        key_metrics = output.get('key_metrics', {})
        
        if not key_metrics:
            errors.append(ErrorInstance(
                category=FundamentalAgentError.MISSING_KEY_METRIC.value,
                severity=self.classifier.classify(FundamentalAgentError.MISSING_KEY_METRIC.value, self.agent_type),
                description='No key metrics provided',
                location='key_metrics',
                suggested_fix='Include key financial metrics (P/E, ROE, etc.)'
            ))
            return errors
        
        # Check P/E ratio range
        if 'PE' in key_metrics:
            pe = key_metrics['PE']
            if pe < 0:
                errors.append(ErrorInstance(
                    category=FundamentalAgentError.FINANCIAL_RATIO_MISINTERPRETATION.value,
                    severity=self.classifier.classify(FundamentalAgentError.FINANCIAL_RATIO_MISINTERPRETATION.value, self.agent_type),
                    description=f'P/E ratio is negative ({pe})',
                    location='key_metrics.PE',
                    suggested_fix='Check earnings. Negative P/E indicates negative earnings.',
                    evidence=f'PE = {pe}'
                ))
            elif pe > 1000:
                errors.append(ErrorInstance(
                    category=FundamentalAgentError.VALUATION_CALCULATION_ERROR.value,
                    severity=self.classifier.classify(FundamentalAgentError.VALUATION_CALCULATION_ERROR.value, self.agent_type),
                    description=f'P/E ratio is unrealistically high ({pe})',
                    location='key_metrics.PE',
                    suggested_fix='Verify P/E calculation',
                    evidence=f'PE = {pe}'
                ))
        
        return errors
    
    def _generate_summary(self, errors: List[ErrorInstance]) -> str:
        """Generate error summary"""
        if not errors:
            return 'No errors detected'
        
        fatal = len([e for e in errors if e.severity == ErrorSeverity.FATAL])
        high = len([e for e in errors if e.severity == ErrorSeverity.HIGH])
        medium = len([e for e in errors if e.severity == ErrorSeverity.MEDIUM])
        
        summary = f'Found {len(errors)} errors: '
        parts = []
        if fatal > 0:
            parts.append(f'{fatal} fatal')
        if high > 0:
            parts.append(f'{high} high')
        if medium > 0:
            parts.append(f'{medium} medium')
        
        summary += ', '.join(parts)
        
        return summary


class StrategistAgentFaultDetector(FaultDetector):
    """Fault detector for Strategist Agent"""
    
    def __init__(self):
        super().__init__('strategist')
    
    def detect(self, agent_output: Dict, ground_truth: Optional[Dict] = None) -> ErrorReport:
        """Detect faults in Strategist Agent output"""
        errors = []
        
        # Check required fields
        errors.extend(self._check_required_fields(agent_output))
        
        # Check risk management
        errors.extend(self._check_risk_management(agent_output))
        
        # Check position sizing
        errors.extend(self._check_position_sizing(agent_output))
        
        # Determine overall severity
        overall_severity = self.classifier.determine_overall_severity(errors)
        is_acceptable = self.classifier.is_acceptable(overall_severity)
        
        summary = self._generate_summary(errors)
        
        return ErrorReport(
            agent_type=self.agent_type,
            errors=errors,
            overall_severity=overall_severity,
            is_acceptable=is_acceptable,
            summary=summary
        )
    
    def _check_required_fields(self, output: Dict) -> List[ErrorInstance]:
        """Check required fields"""
        errors = []
        required_fields = ['recommendation', 'confidence', 'position_size', 'reasoning']
        
        for field in required_fields:
            if field not in output or output[field] is None:
                errors.append(ErrorInstance(
                    category=ErrorCategory.MISSING_FIELD.value,
                    severity=self.classifier.classify(ErrorCategory.MISSING_FIELD.value, self.agent_type),
                    description=f'Required field "{field}" is missing',
                    location=field,
                    suggested_fix=f'Include {field} in output'
                ))
        
        return errors
    
    def _check_risk_management(self, output: Dict) -> List[ErrorInstance]:
        """Check risk management"""
        errors = []
        
        recommendation = output.get('recommendation')
        stop_loss = output.get('stop_loss')
        take_profit = output.get('take_profit')
        
        # If buy/sell recommendation, must have stop loss
        if recommendation in ['buy', 'sell', 'strong_buy', 'strong_sell']:
            if not stop_loss:
                errors.append(ErrorInstance(
                    category=StrategistAgentError.STOP_LOSS_MISSING.value,
                    severity=self.classifier.classify(StrategistAgentError.STOP_LOSS_MISSING.value, self.agent_type),
                    description='Stop loss is missing for buy/sell recommendation',
                    location='stop_loss',
                    suggested_fix='Define stop loss level for risk management',
                    evidence=f'recommendation={recommendation}, stop_loss=None'
                ))
        
        return errors
    
    def _check_position_sizing(self, output: Dict) -> List[ErrorInstance]:
        """Check position sizing"""
        errors = []
        
        position_size = output.get('position_size')
        
        if position_size is None:
            return errors
        
        # Check range
        if not (0 <= position_size <= 1):
            errors.append(ErrorInstance(
                category=ErrorCategory.INVALID_VALUE.value,
                severity=self.classifier.classify(ErrorCategory.INVALID_VALUE.value, self.agent_type),
                description=f'Position size {position_size} is out of range [0, 1]',
                location='position_size',
                suggested_fix='Position size should be between 0 and 1 (0-100%)',
                evidence=f'position_size = {position_size}'
            ))
        
        # Check if too large (>20% is risky)
        elif position_size > 0.2:
            errors.append(ErrorInstance(
                category=StrategistAgentError.POSITION_SIZE_INAPPROPRIATE.value,
                severity=self.classifier.classify(StrategistAgentError.POSITION_SIZE_INAPPROPRIATE.value, self.agent_type),
                description=f'Position size {position_size:.1%} is very large (>20%)',
                location='position_size',
                suggested_fix='Consider reducing position size for better risk management',
                evidence=f'position_size = {position_size:.1%}'
            ))
        
        return errors
    
    def _generate_summary(self, errors: List[ErrorInstance]) -> str:
        """Generate error summary"""
        if not errors:
            return 'No errors detected'
        
        fatal = len([e for e in errors if e.severity == ErrorSeverity.FATAL])
        high = len([e for e in errors if e.severity == ErrorSeverity.HIGH])
        medium = len([e for e in errors if e.severity == ErrorSeverity.MEDIUM])
        
        summary = f'Found {len(errors)} errors: '
        parts = []
        if fatal > 0:
            parts.append(f'{fatal} fatal')
        if high > 0:
            parts.append(f'{high} high')
        if medium > 0:
            parts.append(f'{medium} medium')
        
        summary += ', '.join(parts)
        
        return summary


class FaultLocalizationEngine:
    """
    Main engine for fault localization
    
    Routes to appropriate detector based on agent type
    """
    
    def __init__(self):
        self.detectors = {
            'news': NewsAgentFaultDetector(),
            'technical': TechnicalAgentFaultDetector(),
            'fundamental': FundamentalAgentFaultDetector(),
            'strategist': StrategistAgentFaultDetector()
        }
    
    def detect_faults(
        self,
        agent_type: str,
        agent_output: Dict,
        ground_truth: Optional[Dict] = None
    ) -> ErrorReport:
        """
        Detect faults in agent output
        
        Args:
            agent_type: Type of agent (news, technical, fundamental, strategist)
            agent_output: Agent's output
            ground_truth: Optional ground truth for comparison
        
        Returns:
            ErrorReport with detected errors
        """
        if agent_type not in self.detectors:
            raise ValueError(f'Unknown agent type: {agent_type}')
        
        detector = self.detectors[agent_type]
        return detector.detect(agent_output, ground_truth)


if __name__ == '__main__':
    # Test
    engine = FaultLocalizationEngine()
    
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
    
    report = engine.detect_faults('news', news_output)
    
    print(f"News Agent Report:")
    print(f"Errors: {len(report.errors)}")
    print(f"Overall Severity: {report.overall_severity.value}")
    print(f"Acceptable: {report.is_acceptable}")
    print(f"Summary: {report.summary}")
    
    for error in report.errors:
        print(f"\n- {error.category} ({error.severity.value})")
        print(f"  Location: {error.location}")
        print(f"  Description: {error.description}")
        print(f"  Fix: {error.suggested_fix}")
