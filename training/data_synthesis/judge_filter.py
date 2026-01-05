"""
Judge Filter - Applies judge validation to experiences for dataset quality control

Purpose:
    Filter experiences using the judge system before dataset synthesis.
    Updates experience store with judge approval status and scores.

Features:
    - Batch processing of experiences through judge
    - Configurable quality thresholds
    - Statistics tracking (pass rate, avg score)
    - Incremental processing (skip already judged)

Usage:
    filter = JudgeFilter(experience_store=store)

    results = filter.filter_experiences(
        min_score=6.0,
        skip_already_judged=True
    )

    print(f"Passed: {results.num_passed}/{results.num_total}")
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
from tqdm import tqdm

from training.data_synthesis.experience_store import Experience, ExperienceStore
from evaluation.judge_runner import JudgeRunner


@dataclass
class JudgeFilterResults:
    """Results from judge filtering"""

    num_total: int
    num_passed: int
    num_failed: int
    num_skipped: int  # Already judged

    pass_rate: float
    avg_score_passed: float
    avg_score_failed: float

    # Rejection reasons breakdown
    rejection_reasons: Dict[str, int]

    # Score distribution
    score_distribution: Dict[str, int]  # excellent, good, acceptable, poor


class JudgeFilter:
    """
    Applies judge validation to experiences for quality control

    Workflow:
    1. Load experiences from store
    2. For each experience:
       - Extract signal
       - Run through JudgeRunner
       - Update experience with judge results
    3. Return filtering statistics
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        judge_runner: Optional[JudgeRunner] = None
    ):
        self.store = experience_store
        self.judge = judge_runner or JudgeRunner()

        logger.info("JudgeFilter initialized")

    def filter_experiences(
        self,
        min_score: float = 6.0,
        skip_already_judged: bool = True,
        symbol: Optional[str] = None,
        backtest_id: Optional[str] = None,
        batch_size: int = 100
    ) -> JudgeFilterResults:
        """
        Filter experiences using judge validation

        Args:
            min_score: Minimum judge score for acceptance (0-10)
            skip_already_judged: Skip experiences that already have judge scores
            symbol: Filter by symbol (optional)
            backtest_id: Filter by backtest ID (optional)
            batch_size: Process in batches of N experiences

        Returns:
            Filtering results with statistics
        """
        logger.info(f"Starting judge filtering (min_score={min_score})")

        # Query experiences
        experiences = self.store.query(
            symbol=symbol,
            backtest_id=backtest_id
        )

        if not experiences:
            logger.warning("No experiences found to filter")
            return JudgeFilterResults(
                num_total=0, num_passed=0, num_failed=0, num_skipped=0,
                pass_rate=0.0, avg_score_passed=0.0, avg_score_failed=0.0,
                rejection_reasons={}, score_distribution={}
            )

        logger.info(f"Found {len(experiences)} experiences to process")

        # Filter out already judged if requested
        if skip_already_judged:
            experiences_to_judge = [e for e in experiences if e.judge_score is None]
            num_skipped = len(experiences) - len(experiences_to_judge)
            logger.info(f"Skipping {num_skipped} already-judged experiences")
        else:
            experiences_to_judge = experiences
            num_skipped = 0

        if not experiences_to_judge:
            logger.info("No experiences to judge (all already judged)")
            return self._compute_results(experiences, num_skipped, min_score)

        # Process in batches
        num_passed = 0
        num_failed = 0
        passed_scores = []
        failed_scores = []
        rejection_reasons = {}
        score_levels = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0}

        logger.info(f"Processing {len(experiences_to_judge)} experiences through judge...")

        for i in tqdm(range(0, len(experiences_to_judge), batch_size), desc="Filtering"):
            batch = experiences_to_judge[i:i+batch_size]

            for exp in batch:
                # Run through judge
                result = self.judge.judge(exp.signal)

                # Extract score
                if result.llm_result:
                    score = result.llm_result["composite_score"]
                    score_level = result.llm_result["judgment"]
                else:
                    # Use heuristic score if LLM not available
                    score = 5.0  # Default neutral score
                    score_level = "acceptable"

                # Determine pass/fail
                passed = (result.final_judgment == "ACCEPT" and score >= min_score)

                # Update experience in store
                exp.judge_approved = passed
                exp.judge_score = score

                # Update statistics
                if passed:
                    num_passed += 1
                    passed_scores.append(score)
                else:
                    num_failed += 1
                    failed_scores.append(score)

                    # Track rejection reason
                    reason = result.rejection_reason or "score_below_threshold"
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

                # Track score distribution
                score_levels[score_level] = score_levels.get(score_level, 0) + 1

        # Compute final statistics
        avg_score_passed = sum(passed_scores) / len(passed_scores) if passed_scores else 0.0
        avg_score_failed = sum(failed_scores) / len(failed_scores) if failed_scores else 0.0
        pass_rate = num_passed / len(experiences_to_judge) if experiences_to_judge else 0.0

        results = JudgeFilterResults(
            num_total=len(experiences),
            num_passed=num_passed,
            num_failed=num_failed,
            num_skipped=num_skipped,
            pass_rate=pass_rate,
            avg_score_passed=avg_score_passed,
            avg_score_failed=avg_score_failed,
            rejection_reasons=rejection_reasons,
            score_distribution=score_levels
        )

        logger.info(f"Judge filtering complete:")
        logger.info(f"  Total: {results.num_total}")
        logger.info(f"  Passed: {results.num_passed} ({results.pass_rate:.1%})")
        logger.info(f"  Failed: {results.num_failed}")
        logger.info(f"  Skipped: {results.num_skipped}")
        logger.info(f"  Avg score (passed): {results.avg_score_passed:.2f}")
        logger.info(f"  Avg score (failed): {results.avg_score_failed:.2f}")

        return results

    def _compute_results(
        self,
        experiences: List[Experience],
        num_skipped: int,
        min_score: float
    ) -> JudgeFilterResults:
        """Compute results from already-judged experiences"""

        passed = [e for e in experiences if e.judge_approved and (e.judge_score or 0) >= min_score]
        failed = [e for e in experiences if not e.judge_approved or (e.judge_score or 0) < min_score]

        passed_scores = [e.judge_score for e in passed if e.judge_score is not None]
        failed_scores = [e.judge_score for e in failed if e.judge_score is not None]

        avg_score_passed = sum(passed_scores) / len(passed_scores) if passed_scores else 0.0
        avg_score_failed = sum(failed_scores) / len(failed_scores) if failed_scores else 0.0
        pass_rate = len(passed) / len(experiences) if experiences else 0.0

        return JudgeFilterResults(
            num_total=len(experiences),
            num_passed=len(passed),
            num_failed=len(failed),
            num_skipped=num_skipped,
            pass_rate=pass_rate,
            avg_score_passed=avg_score_passed,
            avg_score_failed=avg_score_failed,
            rejection_reasons={},
            score_distribution={}
        )

    def export_filtered_experiences(
        self,
        output_path: Path,
        min_score: float = 6.0,
        symbol: Optional[str] = None
    ):
        """
        Export judge-approved experiences to file

        Args:
            output_path: Output file path
            min_score: Minimum judge score
            symbol: Filter by symbol (optional)
        """
        # Query approved experiences
        experiences = self.store.query(
            symbol=symbol,
            judge_approved_only=True,
            min_judge_score=min_score
        )

        if not experiences:
            logger.warning("No approved experiences found to export")
            return

        # Export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for exp in experiences:
                data = {
                    "experience_id": exp.experience_id,
                    "symbol": exp.symbol,
                    "signal": exp.signal,
                    "action": exp.action,
                    "outcome": exp.outcome,
                    "reward": exp.reward,
                    "judge_approved": exp.judge_approved,
                    "judge_score": exp.judge_score,
                    "timestamp": exp.timestamp
                }
                f.write(json.dumps(data) + "\n")

        logger.info(f"Exported {len(experiences)} approved experiences to {output_path}")


# CLI Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Judge Filter - Apply judge validation to experiences")
    parser.add_argument("--storage-dir", type=Path, default=Path("data/experiences"), help="Experience store directory")
    parser.add_argument("--min-score", type=float, default=6.0, help="Minimum judge score (0-10)")
    parser.add_argument("--symbol", type=str, help="Filter by symbol")
    parser.add_argument("--backtest-id", type=str, help="Filter by backtest ID")
    parser.add_argument("--skip-already-judged", action="store_true", default=True, help="Skip already-judged experiences")
    parser.add_argument("--export", type=Path, help="Export approved experiences to file")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")

    args = parser.parse_args()

    # Initialize store
    from training.data_synthesis.experience_store import ExperienceStoreConfig
    store_config = ExperienceStoreConfig(storage_dir=args.storage_dir)
    store = ExperienceStore(store_config)

    # Initialize filter
    filter = JudgeFilter(experience_store=store)

    # Filter experiences
    results = filter.filter_experiences(
        min_score=args.min_score,
        skip_already_judged=args.skip_already_judged,
        symbol=args.symbol,
        backtest_id=args.backtest_id,
        batch_size=args.batch_size
    )

    # Print results
    print("\n=== Judge Filtering Results ===")
    print(f"Total experiences: {results.num_total}")
    print(f"Passed: {results.num_passed} ({results.pass_rate:.1%})")
    print(f"Failed: {results.num_failed}")
    print(f"Skipped (already judged): {results.num_skipped}")
    print(f"\nAverage score (passed): {results.avg_score_passed:.2f}/10.0")
    print(f"Average score (failed): {results.avg_score_failed:.2f}/10.0")

    if results.rejection_reasons:
        print("\nRejection reasons:")
        for reason, count in sorted(results.rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    if results.score_distribution:
        print("\nScore distribution:")
        for level, count in results.score_distribution.items():
            print(f"  {level}: {count}")

    # Export if requested
    if args.export:
        filter.export_filtered_experiences(
            output_path=args.export,
            min_score=args.min_score,
            symbol=args.symbol
        )
        print(f"\n✅ Approved experiences exported to {args.export}")

    store.close()
