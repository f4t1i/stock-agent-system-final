"""
Tests for Error Taxonomy System
"""

import pytest
from evaluation.error_taxonomy import (
    ErrorSeverity,
    ErrorCategory,
    NewsAgentError,
    TechnicalAgentError,
    FundamentalAgentError,
    StrategistAgentError,
    ErrorInstance,
    ErrorReport,
    ErrorSeverityClassifier,
    ErrorTaxonomyManager
)
from evaluation.fault_localization import (
    FaultLocalizationEngine,
    NewsAgentFaultDetector,
    TechnicalAgentFaultDetector,
    FundamentalAgentFaultDetector,
    StrategistAgentFaultDetector
)
from evaluation.taxonomy_guided_judge import TaxonomyGuidedJudge


class TestErrorSeverity:
    """Test error severity classification"""
    
    def test_severity_enum(self):
        """Test severity enum values"""
        assert ErrorSeverity.NEGLIGIBLE.value == 'negligible'
        assert ErrorSeverity.LOW.value == 'low'
        assert ErrorSeverity.MEDIUM.value == 'medium'
        assert ErrorSeverity.HIGH.value == 'high'
        assert ErrorSeverity.FATAL.value == 'fatal'
    
    def test_severity_classifier_general(self):
        """Test severity classification for general categories"""
        classifier = ErrorSeverityClassifier()
        
        # Hallucination should be FATAL
        severity = classifier.classify(ErrorCategory.HALLUCINATION.value)
        assert severity == ErrorSeverity.FATAL
        
        # Wrong recommendation should be HIGH
        severity = classifier.classify(ErrorCategory.WRONG_RECOMMENDATION.value)
        assert severity == ErrorSeverity.HIGH
        
        # Formatting error should be NEGLIGIBLE
        severity = classifier.classify(ErrorCategory.FORMATTING_ERROR.value)
        assert ErrorSeverity.NEGLIGIBLE
    
    def test_severity_classifier_agent_specific(self):
        """Test severity classification for agent-specific errors"""
        classifier = ErrorSeverityClassifier()
        
        # News agent: news hallucination should be FATAL
        severity = classifier.classify(NewsAgentError.NEWS_HALLUCINATION.value, 'news')
        assert severity == ErrorSeverity.FATAL
        
        # Technical agent: indicator calculation error should be HIGH
        severity = classifier.classify(TechnicalAgentError.INDICATOR_CALCULATION_ERROR.value, 'technical')
        assert severity == ErrorSeverity.HIGH
        
        # Strategist agent: stop loss missing should be FATAL
        severity = classifier.classify(StrategistAgentError.STOP_LOSS_MISSING.value, 'strategist')
        assert severity == ErrorSeverity.FATAL
    
    def test_overall_severity_determination(self):
        """Test overall severity determination from error list"""
        classifier = ErrorSeverityClassifier()
        
        # Any FATAL error → Overall FATAL
        errors = [
            ErrorInstance('error1', ErrorSeverity.FATAL, 'desc', 'loc', 'fix'),
            ErrorInstance('error2', ErrorSeverity.LOW, 'desc', 'loc', 'fix')
        ]
        overall = classifier.determine_overall_severity(errors)
        assert overall == ErrorSeverity.FATAL
        
        # 2+ HIGH errors → Overall FATAL
        errors = [
            ErrorInstance('error1', ErrorSeverity.HIGH, 'desc', 'loc', 'fix'),
            ErrorInstance('error2', ErrorSeverity.HIGH, 'desc', 'loc', 'fix')
        ]
        overall = classifier.determine_overall_severity(errors)
        assert overall == ErrorSeverity.FATAL
        
        # Any HIGH error → Overall HIGH
        errors = [
            ErrorInstance('error1', ErrorSeverity.HIGH, 'desc', 'loc', 'fix'),
            ErrorInstance('error2', ErrorSeverity.LOW, 'desc', 'loc', 'fix')
        ]
        overall = classifier.determine_overall_severity(errors)
        assert overall == ErrorSeverity.HIGH
        
        # 3+ MEDIUM errors → Overall HIGH
        errors = [
            ErrorInstance('error1', ErrorSeverity.MEDIUM, 'desc', 'loc', 'fix'),
            ErrorInstance('error2', ErrorSeverity.MEDIUM, 'desc', 'loc', 'fix'),
            ErrorInstance('error3', ErrorSeverity.MEDIUM, 'desc', 'loc', 'fix')
        ]
        overall = classifier.determine_overall_severity(errors)
        assert overall == ErrorSeverity.HIGH
    
    def test_acceptability(self):
        """Test acceptability determination"""
        classifier = ErrorSeverityClassifier()
        
        # FATAL not acceptable
        assert not classifier.is_acceptable(ErrorSeverity.FATAL)
        
        # HIGH not acceptable
        assert not classifier.is_acceptable(ErrorSeverity.HIGH)
        
        # MEDIUM acceptable
        assert classifier.is_acceptable(ErrorSeverity.MEDIUM)
        
        # LOW acceptable
        assert classifier.is_acceptable(ErrorSeverity.LOW)
        
        # NEGLIGIBLE acceptable
        assert classifier.is_acceptable(ErrorSeverity.NEGLIGIBLE)


class TestErrorReport:
    """Test error report"""
    
    def test_error_report_creation(self):
        """Test error report creation"""
        errors = [
            ErrorInstance('error1', ErrorSeverity.FATAL, 'desc1', 'loc1', 'fix1'),
            ErrorInstance('error2', ErrorSeverity.HIGH, 'desc2', 'loc2', 'fix2'),
            ErrorInstance('error3', ErrorSeverity.MEDIUM, 'desc3', 'loc3', 'fix3')
        ]
        
        report = ErrorReport(
            agent_type='news',
            errors=errors,
            overall_severity=ErrorSeverity.FATAL,
            is_acceptable=False,
            summary='Found 3 errors: 1 fatal, 1 high, 1 medium'
        )
        
        assert report.agent_type == 'news'
        assert len(report.errors) == 3
        assert report.overall_severity == ErrorSeverity.FATAL
        assert not report.is_acceptable
    
    def test_get_errors_by_severity(self):
        """Test filtering errors by severity"""
        errors = [
            ErrorInstance('error1', ErrorSeverity.FATAL, 'desc1', 'loc1', 'fix1'),
            ErrorInstance('error2', ErrorSeverity.HIGH, 'desc2', 'loc2', 'fix2'),
            ErrorInstance('error3', ErrorSeverity.FATAL, 'desc3', 'loc3', 'fix3')
        ]
        
        report = ErrorReport(
            agent_type='news',
            errors=errors,
            overall_severity=ErrorSeverity.FATAL,
            is_acceptable=False,
            summary='Test'
        )
        
        fatal_errors = report.get_errors_by_severity(ErrorSeverity.FATAL)
        assert len(fatal_errors) == 2
        
        high_errors = report.get_errors_by_severity(ErrorSeverity.HIGH)
        assert len(high_errors) == 1
    
    def test_has_fatal_errors(self):
        """Test fatal error detection"""
        errors = [
            ErrorInstance('error1', ErrorSeverity.FATAL, 'desc1', 'loc1', 'fix1')
        ]
        
        report = ErrorReport(
            agent_type='news',
            errors=errors,
            overall_severity=ErrorSeverity.FATAL,
            is_acceptable=False,
            summary='Test'
        )
        
        assert report.has_fatal_errors()


class TestNewsAgentFaultDetector:
    """Test News Agent fault detection"""
    
    def test_missing_fields(self):
        """Test detection of missing required fields"""
        detector = NewsAgentFaultDetector()
        
        output = {
            'sentiment_score': 1.0,
            # Missing: confidence, key_events, reasoning
        }
        
        report = detector.detect(output)
        
        assert len(report.errors) >= 3  # At least 3 missing fields
        assert any('confidence' in e.description for e in report.errors)
    
    def test_sentiment_out_of_range(self):
        """Test detection of sentiment out of range"""
        detector = NewsAgentFaultDetector()
        
        output = {
            'sentiment_score': 3.0,  # Out of range [-2, 2]
            'confidence': 0.8,
            'key_events': [],
            'reasoning': 'Test reasoning with enough length to pass the check.'
        }
        
        report = detector.detect(output)
        
        assert any('out of range' in e.description for e in report.errors)
    
    def test_sentiment_justification_missing(self):
        """Test detection of missing sentiment justification"""
        detector = NewsAgentFaultDetector()
        
        output = {
            'sentiment_score': 1.5,
            'confidence': 0.8,
            'key_events': [],
            'reasoning': 'Too short'  # Too short
        }
        
        report = detector.detect(output)
        
        assert any('justification' in e.description.lower() for e in report.errors)
    
    def test_sentiment_consistency(self):
        """Test detection of sentiment-reasoning inconsistency"""
        detector = NewsAgentFaultDetector()
        
        output = {
            'sentiment_score': 1.5,  # Positive
            'confidence': 0.8,
            'key_events': [],
            'reasoning': 'Very negative news. Bad earnings. Weak guidance. Bearish outlook. Decline expected.'
        }
        
        report = detector.detect(output)
        
        # Should detect inconsistency
        assert any('miscalculation' in e.category.lower() for e in report.errors)


class TestTechnicalAgentFaultDetector:
    """Test Technical Agent fault detection"""
    
    def test_rsi_out_of_range(self):
        """Test detection of RSI out of range"""
        detector = TechnicalAgentFaultDetector()
        
        output = {
            'signal': 'bullish',
            'confidence': 0.8,
            'indicators': {
                'RSI': 150  # Out of range [0, 100]
            },
            'reasoning': 'Test'
        }
        
        report = detector.detect(output)
        
        assert any('RSI' in e.description and 'out of' in e.description for e in report.errors)
    
    def test_wrong_signal_interpretation(self):
        """Test detection of wrong signal interpretation"""
        detector = TechnicalAgentFaultDetector()
        
        output = {
            'signal': 'bullish',
            'confidence': 0.8,
            'indicators': {
                'RSI': 85  # Overbought, shouldn't be bullish
            },
            'reasoning': 'Test'
        }
        
        report = detector.detect(output)
        
        assert any('overbought' in e.description.lower() for e in report.errors)
    
    def test_support_resistance_hallucination(self):
        """Test detection of invalid support/resistance levels"""
        detector = TechnicalAgentFaultDetector()
        
        output = {
            'signal': 'bullish',
            'confidence': 0.8,
            'indicators': {},
            'reasoning': 'Test',
            'support_level': 150,  # Above current price!
            'resistance_level': 140
        }
        
        ground_truth = {
            'current_price': 145
        }
        
        report = detector.detect(output, ground_truth)
        
        # Support above current price is invalid
        assert any('support' in e.location.lower() for e in report.errors)


class TestStrategistAgentFaultDetector:
    """Test Strategist Agent fault detection"""
    
    def test_stop_loss_missing(self):
        """Test detection of missing stop loss"""
        detector = StrategistAgentFaultDetector()
        
        output = {
            'recommendation': 'buy',
            'confidence': 0.8,
            'position_size': 0.1,
            'reasoning': 'Test',
            # Missing: stop_loss
        }
        
        report = detector.detect(output)
        
        assert any('stop_loss' in e.category.lower() for e in report.errors)
        assert report.has_fatal_errors()  # Stop loss missing is FATAL
    
    def test_position_size_out_of_range(self):
        """Test detection of position size out of range"""
        detector = StrategistAgentFaultDetector()
        
        output = {
            'recommendation': 'buy',
            'confidence': 0.8,
            'position_size': 1.5,  # Out of range [0, 1]
            'reasoning': 'Test',
            'stop_loss': 100
        }
        
        report = detector.detect(output)
        
        assert any('out of range' in e.description for e in report.errors)
    
    def test_position_size_too_large(self):
        """Test detection of inappropriately large position size"""
        detector = StrategistAgentFaultDetector()
        
        output = {
            'recommendation': 'buy',
            'confidence': 0.8,
            'position_size': 0.25,  # >20%, risky
            'reasoning': 'Test',
            'stop_loss': 100
        }
        
        report = detector.detect(output)
        
        assert any('large' in e.description.lower() for e in report.errors)


class TestFaultLocalizationEngine:
    """Test fault localization engine"""
    
    def test_engine_routing(self):
        """Test that engine routes to correct detector"""
        engine = FaultLocalizationEngine()
        
        # Test News Agent
        news_output = {
            'sentiment_score': 1.0,
            'confidence': 0.8,
            'key_events': [],
            'reasoning': 'Test reasoning with sufficient length for validation.'
        }
        
        report = engine.detect_faults('news', news_output)
        assert report.agent_type == 'news'
        
        # Test Technical Agent
        tech_output = {
            'signal': 'bullish',
            'confidence': 0.8,
            'indicators': {'RSI': 50},
            'reasoning': 'Test'
        }
        
        report = engine.detect_faults('technical', tech_output)
        assert report.agent_type == 'technical'
    
    def test_unknown_agent_type(self):
        """Test handling of unknown agent type"""
        engine = FaultLocalizationEngine()
        
        with pytest.raises(ValueError):
            engine.detect_faults('unknown_agent', {})


class TestTaxonomyGuidedJudge:
    """Test taxonomy-guided judge"""
    
    def test_evaluate_without_llm(self):
        """Test evaluation without LLM"""
        judge = TaxonomyGuidedJudge()
        
        news_output = {
            'sentiment_score': 1.5,
            'confidence': 0.8,
            'key_events': [],
            'reasoning': 'Short'  # Too short
        }
        
        report = judge.evaluate('news', news_output, use_llm=False)
        
        assert report.agent_type == 'news'
        assert len(report.errors) > 0
    
    def test_generate_sft_feedback(self):
        """Test SFT feedback generation"""
        judge = TaxonomyGuidedJudge()
        
        errors = [
            ErrorInstance(
                'news_hallucination',
                ErrorSeverity.FATAL,
                'Fabricated news event',
                'key_events',
                'Use only verified sources'
            ),
            ErrorInstance(
                'sentiment_miscalculation',
                ErrorSeverity.MEDIUM,
                'Sentiment too high',
                'sentiment_score',
                'Recalculate sentiment'
            )
        ]
        
        report = ErrorReport(
            agent_type='news',
            errors=errors,
            overall_severity=ErrorSeverity.FATAL,
            is_acceptable=False,
            summary='Test'
        )
        
        feedback = judge.generate_sft_feedback(report)
        
        assert 'FATAL' in feedback
        assert 'news_hallucination' in feedback
        assert 'sentiment_miscalculation' in feedback
    
    def test_generate_training_example(self):
        """Test training example generation"""
        judge = TaxonomyGuidedJudge()
        
        news_output = {
            'sentiment_score': 1.5,
            'confidence': 0.8,
            'key_events': [],
            'reasoning': 'Short'
        }
        
        report = judge.evaluate('news', news_output, use_llm=False)
        
        training_example = judge.generate_training_example('news', news_output, report)
        
        assert 'agent_type' in training_example
        assert 'output' in training_example
        assert 'feedback' in training_example
        assert 'errors' in training_example
        assert training_example['agent_type'] == 'news'


class TestErrorTaxonomyManager:
    """Test error taxonomy manager"""
    
    def test_get_all_error_categories(self):
        """Test getting all error categories for agent type"""
        manager = ErrorTaxonomyManager()
        
        # News agent
        categories = manager.get_all_error_categories('news')
        assert len(categories) > 0
        assert 'hallucination' in categories
        assert 'news_hallucination' in categories
        
        # Technical agent
        categories = manager.get_all_error_categories('technical')
        assert 'indicator_calculation_error' in categories
    
    def test_get_error_description(self):
        """Test getting error description"""
        manager = ErrorTaxonomyManager()
        
        desc = manager.get_error_description('hallucination', 'news')
        assert len(desc) > 0
        assert 'fabricated' in desc.lower() or 'made-up' in desc.lower()
    
    def test_get_suggested_fix(self):
        """Test getting suggested fix"""
        manager = ErrorTaxonomyManager()
        
        fix = manager.get_suggested_fix('hallucination', 'news')
        assert len(fix) > 0
        assert 'verify' in fix.lower() or 'source' in fix.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
