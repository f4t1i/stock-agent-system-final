"""
Judge Runner - Orchestrates hard rules + LLM judge for signal evaluation

Combines:
1. Format validation (Signal Contract)
2. Fact checking (hard rules from rubrics)
3. LLM scoring (qualitative assessment)
4. Final composite judgment
5. CI/CD integration (exit codes for blocking)

Usage:
    python evaluation/judge_runner.py signal.json
    make judge-signal SIGNAL=signal.json  # CI integration
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.signal_validator import SignalValidator, SignalValidationError
from evaluation.llm_judge import LLMJudge, LLMJudgeResult


@dataclass
class JudgeRunnerResult:
    """Complete judgment result from Judge Runner"""
    signal_id: str
    timestamp: str

    # Step 1: Format validation
    format_validation_passed: bool
    format_errors: List[str]

    # Step 2: Fact checking
    fact_check_passed: bool
    fact_check_errors: List[str]

    # Step 3: LLM judgment
    llm_result: Optional[Dict]

    # Final
    final_judgment: str  # "ACCEPT" or "REJECT"
    rejection_reason: Optional[str]
    warnings: List[str]

    # Metadata
    execution_time_ms: float


class JudgeRunner:
    """
    Orchestrates complete signal judgment process.

    Implements 4-step judgment flow from judge_rubrics.yaml:
    1. Format validation (Signal Contract)
    2. Fact checking (hard rules)
    3. LLM assessment (qualitative)
    4. Final composite judgment
    """

    def __init__(
        self,
        rubrics_path: Optional[str] = None,
        signal_validator: Optional[SignalValidator] = None,
        llm_judge: Optional[LLMJudge] = None,
        strict: bool = True
    ):
        """
        Initialize Judge Runner.

        Args:
            rubrics_path: Path to judge_rubrics.yaml
            signal_validator: Signal validator instance
            llm_judge: LLM judge instance
            strict: If True, reject on any warnings
        """
        # Load rubrics
        if rubrics_path is None:
            rubrics_path = Path(__file__).parent / "judge_rubrics.yaml"

        with open(rubrics_path, 'r') as f:
            self.rubrics = yaml.safe_load(f)

        # Initialize components
        self.signal_validator = signal_validator or SignalValidator()
        self.llm_judge = llm_judge or LLMJudge(rubrics_path=rubrics_path)
        self.strict = strict

        logger.info("Judge Runner initialized")

    def judge(self, signal: Dict) -> JudgeRunnerResult:
        """
        Run complete judgment process on a signal.

        Args:
            signal: Trading signal dictionary

        Returns:
            JudgeRunnerResult with all judgment steps
        """
        start_time = datetime.now()

        signal_id = signal.get('metadata', {}).get('symbol', 'unknown')

        logger.info(f"Judging signal: {signal_id}")

        # Step 1: Format validation
        logger.debug("Step 1: Format validation")
        format_passed, format_errors = self._validate_format(signal)

        if not format_passed:
            logger.warning(f"Format validation FAILED: {len(format_errors)} errors")
            return self._reject_result(
                signal_id,
                format_passed, format_errors,
                False, [],
                None,
                "Format validation failed",
                (datetime.now() - start_time).total_seconds() * 1000
            )

        # Step 2: Fact checking
        logger.debug("Step 2: Fact checking")
        fact_passed, fact_errors = self._check_facts(signal)

        if not fact_passed:
            logger.warning(f"Fact checking FAILED: {len(fact_errors)} errors")
            return self._reject_result(
                signal_id,
                format_passed, format_errors,
                fact_passed, fact_errors,
                None,
                "Fact checking failed",
                (datetime.now() - start_time).total_seconds() * 1000
            )

        # Step 3: LLM assessment
        logger.debug("Step 3: LLM assessment")
        llm_result = self.llm_judge.judge_signal(signal, explain=True)

        # Step 4: Final composite judgment
        logger.debug("Step 4: Composite judgment")
        final_judgment, rejection_reason, warnings = self._make_final_judgment(
            format_passed,
            fact_passed,
            llm_result
        )

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        result = JudgeRunnerResult(
            signal_id=signal_id,
            timestamp=datetime.now().isoformat(),
            format_validation_passed=format_passed,
            format_errors=format_errors,
            fact_check_passed=fact_passed,
            fact_check_errors=fact_errors,
            llm_result=self.llm_judge.to_dict(llm_result),
            final_judgment=final_judgment,
            rejection_reason=rejection_reason,
            warnings=warnings,
            execution_time_ms=execution_time
        )

        logger.info(
            f"Signal {signal_id}: {final_judgment} "
            f"(LLM: {llm_result.composite_score:.2f}/10, {llm_result.judgment})"
        )

        return result

    def _validate_format(self, signal: Dict) -> Tuple[bool, List[str]]:
        """Step 1: Validate signal format using Signal Validator"""
        try:
            is_valid, errors = self.signal_validator.validate(signal, strict=False)
            return is_valid, errors
        except Exception as e:
            logger.error(f"Format validation error: {e}")
            return False, [str(e)]

    def _check_facts(self, signal: Dict) -> Tuple[bool, List[str]]:
        """Step 2: Check hard fact-checking rules"""
        errors = []

        fact_rules = self.rubrics['fact_check_rules']

        # Price sanity (skip if no current price)
        current_price = signal.get('metadata', {}).get('current_price')
        if current_price:
            # Already checked by SignalValidator, but we can add extra checks here
            pass

        # Risk parameters
        risk_rules = fact_rules['risk_parameters']['rules']
        for rule in risk_rules:
            if rule['auto_fail']:
                error = self._check_risk_rule(signal, rule)
                if error:
                    errors.append(error)

        # Position sizing logic
        sizing_rules = fact_rules['position_sizing_logic']['rules']
        for rule in sizing_rules:
            if rule['auto_fail']:
                error = self._check_sizing_rule(signal, rule)
                if error:
                    errors.append(error)

        return len(errors) == 0, errors

    def _check_risk_rule(self, signal: Dict, rule: Dict) -> Optional[str]:
        """Check a single risk parameter rule"""
        rule_name = rule['name']

        if 'Risk/reward ratio' in rule_name:
            # Check R/R ratio
            current_price = signal.get('metadata', {}).get('current_price', 0)
            stop_loss = signal.get('risk', {}).get('stop_loss', 0)
            take_profit = signal.get('risk', {}).get('take_profit', 0)

            if current_price > 0 and stop_loss > 0 and take_profit > 0:
                if signal['signal'] == 'buy':
                    risk = current_price - stop_loss
                    reward = take_profit - current_price

                    if risk > 0:
                        rr_ratio = reward / risk
                        minimum = rule.get('minimum', 1.0)

                        if rr_ratio < minimum:
                            return f"Risk/reward ratio ({rr_ratio:.2f}) below minimum ({minimum})"

        return None

    def _check_sizing_rule(self, signal: Dict, rule: Dict) -> Optional[str]:
        """Check a single position sizing rule"""
        rule_name = rule['name']

        if 'Hold signal position size' in rule_name:
            if signal['signal'] == 'hold':
                position_size = signal.get('sizing', {}).get('position_size', 0)
                if position_size > 0:
                    return f"Hold signal has non-zero position_size ({position_size})"

        elif 'Buy/sell position size' in rule_name:
            if signal['signal'] in ['buy', 'sell']:
                position_size = signal.get('sizing', {}).get('position_size', 0)
                if position_size == 0:
                    return f"{signal['signal'].upper()} signal has zero position_size"

        return None

    def _make_final_judgment(
        self,
        format_passed: bool,
        fact_passed: bool,
        llm_result: LLMJudgeResult
    ) -> Tuple[str, Optional[str], List[str]]:
        """
        Step 4: Make final composite judgment.

        Logic from judge_rubrics.yaml:
        - If hard rules failed → REJECT
        - If LLM score >= 8.5 → ACCEPT (excellent)
        - If LLM score >= 7.0 → ACCEPT (good)
        - If LLM score >= 6.0 → ACCEPT (acceptable, with warnings)
        - If LLM score < 6.0 → REJECT (poor quality)

        Returns:
            Tuple of (judgment, rejection_reason, warnings)
        """
        warnings = []

        # Hard rules always override
        if not format_passed or not fact_passed:
            return "REJECT", "Hard rule violations", []

        # LLM judgment
        score = llm_result.composite_score

        thresholds = self.rubrics['llm_judge_rubric']['composite_score_calculation']['thresholds']

        if score >= thresholds['excellent']:
            return "ACCEPT", None, []

        elif score >= thresholds['good']:
            return "ACCEPT", None, []

        elif score >= thresholds['acceptable']:
            warnings.append(f"LLM score ({score:.2f}) is acceptable but not good (< {thresholds['good']})")

            if self.strict:
                return "REJECT", f"Strict mode: score ({score:.2f}) < {thresholds['good']}", warnings
            else:
                return "ACCEPT", None, warnings

        else:
            # Score < 6.0 → poor quality
            return "REJECT", f"Poor LLM score ({score:.2f}) < {thresholds['acceptable']}", warnings

    def _reject_result(
        self,
        signal_id: str,
        format_passed: bool,
        format_errors: List[str],
        fact_passed: bool,
        fact_errors: List[str],
        llm_result: Optional[Dict],
        rejection_reason: str,
        execution_time_ms: float
    ) -> JudgeRunnerResult:
        """Create rejection result"""
        return JudgeRunnerResult(
            signal_id=signal_id,
            timestamp=datetime.now().isoformat(),
            format_validation_passed=format_passed,
            format_errors=format_errors,
            fact_check_passed=fact_passed,
            fact_check_errors=fact_errors,
            llm_result=llm_result,
            final_judgment="REJECT",
            rejection_reason=rejection_reason,
            warnings=[],
            execution_time_ms=execution_time_ms
        )

    def to_dict(self, result: JudgeRunnerResult) -> Dict:
        """Convert result to dictionary"""
        return asdict(result)

    def get_exit_code(self, result: JudgeRunnerResult) -> int:
        """
        Get exit code for CI/CD integration.

        Returns:
            0 if ACCEPT, 1 if REJECT
        """
        return 0 if result.final_judgment == "ACCEPT" else 1


def main():
    """CLI entry point for judge runner"""
    import argparse

    parser = argparse.ArgumentParser(description='Judge trading signals')
    parser.add_argument('signal_file', help='Path to signal JSON file')
    parser.add_argument('--strict', action='store_true', help='Strict mode (reject warnings)')
    parser.add_argument('--output', help='Output file for judgment result')

    args = parser.parse_args()

    # Load signal
    with open(args.signal_file, 'r') as f:
        signal = json.load(f)

    # Run judgment
    runner = JudgeRunner(strict=args.strict)
    result = runner.judge(signal)

    # Print result
    print("\n" + "="*60)
    print("SIGNAL JUDGMENT")
    print("="*60)
    print(f"\nSignal: {result.signal_id}")
    print(f"Judgment: {result.final_judgment}")

    if result.final_judgment == "REJECT":
        print(f"Reason: {result.rejection_reason}")

        if result.format_errors:
            print(f"\nFormat Errors ({len(result.format_errors)}):")
            for err in result.format_errors:
                print(f"  - {err}")

        if result.fact_check_errors:
            print(f"\nFact Check Errors ({len(result.fact_check_errors)}):")
            for err in result.fact_check_errors:
                print(f"  - {err}")

    if result.llm_result:
        print(f"\nLLM Score: {result.llm_result['composite_score']:.2f}/10.0 ({result.llm_result['judgment']})")

    if result.warnings:
        print(f"\nWarnings:")
        for warn in result.warnings:
            print(f"  ⚠️  {warn}")

    print(f"\nExecution Time: {result.execution_time_ms:.1f}ms")
    print("="*60 + "\n")

    # Save result
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(runner.to_dict(result), f, indent=2)
        print(f"Result saved to: {args.output}")

    # Exit with appropriate code for CI/CD
    sys.exit(runner.get_exit_code(result))


if __name__ == "__main__":
    main()
