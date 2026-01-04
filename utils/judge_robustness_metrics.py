"""
Judge Robustness Metrics

Metrics to evaluate Judge resistance to adversarial attacks:
1. Attack Detection Rate (ADR)
2. False Positive Rate (FPR)
3. Score Consistency
4. Vulnerability Score
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class RobustnessMetrics:
    """Robustness metrics for Judge evaluation"""
    attack_detection_rate: float  # % of adversarial examples correctly scored low
    false_positive_rate: float  # % of legitimate examples incorrectly scored low
    score_consistency: float  # Consistency of scores for same input
    vulnerability_score: float  # Overall vulnerability (0=robust, 1=vulnerable)
    attack_type_vulnerabilities: Dict[str, float]  # Vulnerability per attack type


class JudgeRobustnessEvaluator:
    """Evaluate Judge robustness against adversarial attacks"""
    
    def __init__(self, judge, adversarial_threshold: int = 50):
        """
        Args:
            judge: LLMJudge instance
            adversarial_threshold: Score below which response is considered adversarial
        """
        self.judge = judge
        self.adversarial_threshold = adversarial_threshold
    
    def evaluate_robustness(
        self,
        adversarial_examples: List[Dict[str, Any]],
        legitimate_examples: List[Dict[str, Any]]
    ) -> RobustnessMetrics:
        """
        Evaluate Judge robustness
        
        Args:
            adversarial_examples: List of adversarial attack examples
            legitimate_examples: List of legitimate good examples
        
        Returns:
            RobustnessMetrics
        """
        logger.info("Evaluating Judge robustness...")
        
        # 1. Evaluate adversarial examples
        adv_results = self._evaluate_examples(adversarial_examples, is_adversarial=True)
        
        # 2. Evaluate legitimate examples
        leg_results = self._evaluate_examples(legitimate_examples, is_adversarial=False)
        
        # 3. Calculate metrics
        adr = self._calculate_attack_detection_rate(adv_results)
        fpr = self._calculate_false_positive_rate(leg_results)
        consistency = self._calculate_score_consistency()
        
        # 4. Calculate per-attack-type vulnerabilities
        attack_vulnerabilities = self._calculate_attack_type_vulnerabilities(adv_results)
        
        # 5. Calculate overall vulnerability score
        vulnerability = self._calculate_vulnerability_score(adr, fpr, attack_vulnerabilities)
        
        metrics = RobustnessMetrics(
            attack_detection_rate=adr,
            false_positive_rate=fpr,
            score_consistency=consistency,
            vulnerability_score=vulnerability,
            attack_type_vulnerabilities=attack_vulnerabilities
        )
        
        logger.info(f"Robustness Metrics: ADR={adr:.2%}, FPR={fpr:.2%}, Vuln={vulnerability:.2f}")
        
        return metrics
    
    def _evaluate_examples(
        self,
        examples: List[Dict[str, Any]],
        is_adversarial: bool
    ) -> List[Dict[str, Any]]:
        """Evaluate list of examples"""
        results = []
        
        for example in examples:
            score = self.judge.evaluate(
                agent_type=example.get('agent_type', 'news'),
                agent_output=example['agent_output'],
                ground_truth=None,
                market_data={'symbol': 'AAPL'}
            )['score']
            
            results.append({
                'score': score,
                'attack_type': example.get('attack_type', 'unknown'),
                'is_adversarial': is_adversarial,
                'detected': score < self.adversarial_threshold
            })
        
        return results
    
    def _calculate_attack_detection_rate(self, adv_results: List[Dict[str, Any]]) -> float:
        """Calculate % of adversarial examples correctly detected (scored low)"""
        if not adv_results:
            return 0.0
        
        detected = sum(1 for r in adv_results if r['detected'])
        return detected / len(adv_results)
    
    def _calculate_false_positive_rate(self, leg_results: List[Dict[str, Any]]) -> float:
        """Calculate % of legitimate examples incorrectly flagged (scored low)"""
        if not leg_results:
            return 0.0
        
        false_positives = sum(1 for r in leg_results if r['detected'])
        return false_positives / len(leg_results)
    
    def _calculate_score_consistency(self) -> float:
        """Calculate score consistency for same input (requires multiple evaluations)"""
        # Test with a standard example
        test_output = {
            'symbol': 'AAPL',
            'sentiment_score': 1.5,
            'confidence': 0.85,
            'key_events': ['Strong earnings'],
            'reasoning': 'Company reported strong earnings.',
            'recommendation': 'bullish'
        }
        
        scores = []
        for _ in range(5):
            score = self.judge.evaluate(
                agent_type='news',
                agent_output=test_output,
                ground_truth=None,
                market_data={'symbol': 'AAPL'}
            )['score']
            scores.append(score)
        
        # Calculate coefficient of variation (lower = more consistent)
        if len(scores) > 1:
            std = np.std(scores)
            mean = np.mean(scores)
            cv = std / mean if mean > 0 else 1.0
            consistency = 1.0 - min(cv, 1.0)  # Convert to consistency score
        else:
            consistency = 1.0
        
        return consistency
    
    def _calculate_attack_type_vulnerabilities(
        self,
        adv_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate vulnerability per attack type"""
        vulnerabilities = {}
        
        # Group by attack type
        by_type = {}
        for result in adv_results:
            attack_type = result['attack_type']
            if attack_type not in by_type:
                by_type[attack_type] = []
            by_type[attack_type].append(result)
        
        # Calculate vulnerability for each type
        for attack_type, results in by_type.items():
            # Vulnerability = % NOT detected
            not_detected = sum(1 for r in results if not r['detected'])
            vulnerability = not_detected / len(results)
            vulnerabilities[attack_type] = vulnerability
        
        return vulnerabilities
    
    def _calculate_vulnerability_score(
        self,
        adr: float,
        fpr: float,
        attack_vulnerabilities: Dict[str, float]
    ) -> float:
        """
        Calculate overall vulnerability score
        
        Lower is better (0 = perfectly robust, 1 = completely vulnerable)
        """
        # Weighted combination
        # - 50% weight on attack detection rate (inverted)
        # - 20% weight on false positive rate
        # - 30% weight on worst attack type vulnerability
        
        adr_component = 1.0 - adr  # Invert (lower ADR = higher vulnerability)
        fpr_component = fpr
        worst_attack_vuln = max(attack_vulnerabilities.values()) if attack_vulnerabilities else 0.0
        
        vulnerability = (
            0.5 * adr_component +
            0.2 * fpr_component +
            0.3 * worst_attack_vuln
        )
        
        return vulnerability
    
    def generate_report(self, metrics: RobustnessMetrics) -> str:
        """Generate human-readable robustness report"""
        report = []
        report.append("=" * 60)
        report.append("JUDGE ROBUSTNESS EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL METRICS:")
        report.append(f"  Attack Detection Rate (ADR):  {metrics.attack_detection_rate:.2%}")
        report.append(f"  False Positive Rate (FPR):    {metrics.false_positive_rate:.2%}")
        report.append(f"  Score Consistency:            {metrics.score_consistency:.2%}")
        report.append(f"  Vulnerability Score:          {metrics.vulnerability_score:.3f}")
        report.append("")
        
        # Interpretation
        report.append("INTERPRETATION:")
        if metrics.vulnerability_score < 0.2:
            report.append("  ✅ EXCELLENT - Judge is highly robust")
        elif metrics.vulnerability_score < 0.4:
            report.append("  ✓ GOOD - Judge shows good robustness")
        elif metrics.vulnerability_score < 0.6:
            report.append("  ⚠️ FAIR - Judge has moderate vulnerabilities")
        else:
            report.append("  ❌ POOR - Judge is highly vulnerable")
        report.append("")
        
        # Per-attack-type vulnerabilities
        report.append("VULNERABILITIES BY ATTACK TYPE:")
        for attack_type, vuln in sorted(
            metrics.attack_type_vulnerabilities.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            status = "❌" if vuln > 0.5 else "⚠️" if vuln > 0.2 else "✅"
            report.append(f"  {status} {attack_type:25s} {vuln:.2%}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if metrics.attack_detection_rate < 0.8:
            report.append("  • Improve adversarial detection in Judge prompts")
        if metrics.false_positive_rate > 0.1:
            report.append("  • Reduce false positives for legitimate responses")
        if metrics.score_consistency < 0.9:
            report.append("  • Improve score consistency (reduce temperature?)")
        
        # Identify worst attack types
        worst_attacks = [
            (k, v) for k, v in metrics.attack_type_vulnerabilities.items()
            if v > 0.3
        ]
        if worst_attacks:
            report.append("  • Focus on these vulnerable attack types:")
            for attack_type, vuln in worst_attacks:
                report.append(f"    - {attack_type} ({vuln:.2%})")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def evaluate_judge_robustness(
    judge,
    adversarial_examples_path: str,
    output_report_path: str = None
) -> RobustnessMetrics:
    """
    Convenience function to evaluate Judge robustness
    
    Args:
        judge: LLMJudge instance
        adversarial_examples_path: Path to JSONL file with adversarial examples
        output_report_path: Optional path to save report
    
    Returns:
        RobustnessMetrics
    """
    import json
    
    # Load examples
    with open(adversarial_examples_path, 'r') as f:
        all_examples = [json.loads(line) for line in f]
    
    # Separate adversarial and legitimate
    adversarial = [ex for ex in all_examples if ex.get('is_adversarial', False)]
    legitimate = [ex for ex in all_examples if not ex.get('is_adversarial', False)]
    
    logger.info(f"Loaded {len(adversarial)} adversarial and {len(legitimate)} legitimate examples")
    
    # Evaluate
    evaluator = JudgeRobustnessEvaluator(judge)
    metrics = evaluator.evaluate_robustness(adversarial, legitimate)
    
    # Generate report
    report = evaluator.generate_report(metrics)
    print(report)
    
    # Save report if path provided
    if output_report_path:
        with open(output_report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {output_report_path}")
    
    return metrics


if __name__ == '__main__':
    # Example usage
    from judge.llm_judge import LLMJudge
    from utils.config_loader import load_config
    
    config = load_config('config/system.yaml')
    judge = LLMJudge(config=config)
    
    metrics = evaluate_judge_robustness(
        judge=judge,
        adversarial_examples_path='data/adversarial_training/adversarial_examples.jsonl',
        output_report_path='reports/judge_robustness_report.txt'
    )
