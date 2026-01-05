#!/usr/bin/env python
"""
Generate Backtest Report - Enhanced standardized reports with Pass/Fail gates

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --format html
    python scripts/generate_report.py --with-gates
    make report
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from evaluation.pass_fail_gates import PassFailGatesEvaluator


def find_latest_backtest_results(results_dir: str = "backtest_results") -> Optional[Path]:
    """Find the most recent backtest metrics file"""
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return None

    # Find all metrics files
    metrics_files = list(results_path.glob("backtest_metrics_*.json"))

    if not metrics_files:
        logger.error(f"No backtest results found in {results_dir}")
        return None

    # Return most recent
    latest = max(metrics_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest results: {latest}")

    return latest


def load_backtest_results(filepath: Path) -> Dict:
    """Load backtest results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_markdown_report(results: Dict, output_path: Optional[Path] = None, with_gates: bool = True) -> str:
    """Generate enhanced markdown report with comprehensive metrics and gates"""

    # Evaluate gates if enabled
    gates_result = None
    if with_gates:
        try:
            evaluator = PassFailGatesEvaluator()
            gates_result = evaluator.evaluate(results)
        except Exception as e:
            logger.warning(f"Gates evaluation failed: {e}")

    # Header with overall status
    if gates_result:
        status_emoji = "✅" if gates_result.final_judgment == "PASS" else "❌"
        status_text = f"{status_emoji} **{gates_result.final_judgment}**"
    else:
        status_text = "⚠️ **NO GATES EVALUATION**"

    report_lines = [
        "# Backtest Report",
        "",
        f"**Status:** {status_text}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Configuration",
        "",
        f"- **Symbols:** {', '.join(results['config']['symbols'])}",
        f"- **Date Range:** {results['start_date']} to {results['end_date']}",
        f"- **Initial Capital:** ${results['initial_capital']:,.2f}",
        f"- **Commission Rate:** {results['config']['commission_rate']:.3%}",
        f"- **Slippage:** {results['config']['slippage_bps']} bps",
        f"- **Random Seed:** {results['config']['random_seed']}",
        "",
        "### Institutional Controls",
        "",
        f"- **Survivorship Bias Guard:** {'✅ Enabled' if results.get('survivorship_bias_guarded') else '❌ Disabled'}",
        f"- **Corporate Actions:** {'✅ Enabled' if results.get('corporate_actions_handled') else '❌ Disabled'}",
        f"- **Signal Validation:** {'✅ Enabled' if results['config']['validate_signals'] else '❌ Disabled'}",
        "",
        "---",
        "",
        "## 📊 Performance Summary",
        "",
        "| Metric | Value | Description |",
        "|--------|-------|-------------|",
        f"| **Initial Capital** | ${results['initial_capital']:,.2f} | Starting portfolio value |",
        f"| **Final Value** | ${results['final_value']:,.2f} | Ending portfolio value |",
        f"| **Total Return** | {results['total_return']:.2%} | Overall return |",
        f"| **Annualized Return** | {results.get('annualized_return', 0):.2%} | CAGR |",
        "",
        "### Risk-Adjusted Returns",
        "",
        "| Metric | Value | Benchmark | Description |",
        "|--------|-------|-----------|-------------|",
        f"| **Sharpe Ratio** | {results.get('sharpe_ratio', 0):.3f} | ≥ 1.0 | Risk-adjusted returns |",
        f"| **Sortino Ratio** | {results.get('sortino_ratio', 0):.3f} | ≥ 1.0 | Downside risk-adjusted |",
        f"| **Calmar Ratio** | {results.get('calmar_ratio', 0):.3f} | ≥ 1.0 | Return/Max DD |",
        "",
        "---",
        "",
        "## ⚠️ Risk Metrics",
        "",
        "| Metric | Value | Limit | Description |",
        "|--------|-------|-------|-------------|",
        f"| **Volatility (Annual)** | {results.get('volatility', 0):.2%} | ≤ 40% | Portfolio volatility |",
        f"| **Downside Deviation** | {results.get('downside_deviation', 0):.2%} | ≤ 30% | Downside risk only |",
        f"| **Max Drawdown** | {results.get('max_drawdown', 0):.2%} | ≤ 20% | Largest peak-to-trough decline |",
        f"| **Max DD Duration** | {results.get('max_drawdown_duration_days', 0)} days | ≤ 180 days | Longest drawdown period |",
        "",
        "---",
        "",
        "## 💼 Trading Activity",
        "",
        "### Trade Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Total Trades** | {results.get('total_trades', results.get('num_trades', 0))} |",
        f"| **Trading Days** | {results['trading_days']} |",
        f"| **Winning Trades** | {results.get('winning_trades', 0)} |",
        f"| **Losing Trades** | {results.get('losing_trades', 0)} |",
        "",
        "### Trade Performance",
        "",
        "| Metric | Value | Benchmark |",
        "|--------|-------|-----------|",
        f"| **Win Rate** | {results.get('win_rate', 0):.2%} | ≥ 50% |",
        f"| **Profit Factor** | {results.get('profit_factor', 0):.2f} | ≥ 1.5 |",
        f"| **Avg Trade P&L** | ${results.get('avg_trade_pnl', 0):.2f} | > $0 |",
        f"| **Avg Win** | ${results.get('avg_win', 0):.2f} | - |",
        f"| **Avg Loss** | ${results.get('avg_loss', 0):.2f} | - |",
        "",
    ]

    # Add delisted symbols section
    if results.get('delisted_symbols'):
        report_lines.extend([
            "---",
            "",
            "## 🚫 Survivorship Bias Detection",
            "",
            f"**Delisted Symbols:** {len(results['delisted_symbols'])} detected and positions force-closed",
            "",
        ])
        for symbol in results['delisted_symbols']:
            report_lines.append(f"- **{symbol}** - Delisted during backtest period")
        report_lines.append("")

    # Add failed signals section
    if results.get('failed_signals', 0) > 0:
        failed_rate = results['failed_signals'] / results.get('num_signals', 1) if results.get('num_signals', 0) > 0 else 0
        report_lines.extend([
            "---",
            "",
            "## ❌ Signal Validation Failures",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Failed Signals** | {results['failed_signals']} |",
            f"| **Total Signals** | {results.get('num_signals', 0)} |",
            f"| **Failure Rate** | {failed_rate:.2%} |",
            "",
            "*Failed signals were excluded from trading to maintain signal quality.*",
            "",
        ])

    # Add comprehensive pass/fail gates
    if gates_result:
        report_lines.extend([
            "---",
            "",
            "## ✅ Quality Gates Evaluation",
            "",
            f"**Final Judgment:** {gates_result.final_judgment}",
            "",
            f"**Gates Summary:** {gates_result.passed_gates}/{gates_result.total_gates} passed",
            "",
            f"- **Critical Failures:** {gates_result.critical_failures}",
            f"- **High Severity Failures:** {gates_result.high_failures}",
            f"- **Medium Severity Failures:** {gates_result.medium_failures}",
            "",
        ])

        if gates_result.final_judgment == "FAIL":
            report_lines.extend([
                f"**❌ Rejection Reason:** {gates_result.rejection_reason}",
                "",
            ])

        # Group gates by category
        categories = {}
        for gate in gates_result.gate_results:
            if gate.category not in categories:
                categories[gate.category] = []
            categories[gate.category].append(gate)

        # Display gates by category
        for category, gates in categories.items():
            report_lines.extend([
                f"### {category.replace('_', ' ').title()} Gates",
                "",
                "| Gate | Threshold | Actual | Status | Severity |",
                "|------|-----------|--------|--------|----------|",
            ])

            for gate in gates:
                status = "✅ PASS" if gate.passed else "❌ FAIL"
                inst_std = f" (Inst: {gate.institutional_standard})" if gate.institutional_standard else ""

                # Format values based on gate type
                if 'ratio' in gate.gate_name or 'factor' in gate.gate_name:
                    threshold_str = f"{gate.threshold:.2f}{inst_std}"
                    actual_str = f"{gate.actual_value:.3f}"
                elif 'rate' in gate.gate_name or 'drawdown' in gate.gate_name or 'volatility' in gate.gate_name or 'deviation' in gate.gate_name:
                    threshold_str = f"{gate.threshold:.1%}{inst_std}"
                    actual_str = f"{gate.actual_value:.2%}"
                elif 'days' in gate.gate_name or 'trades' in gate.gate_name:
                    threshold_str = f"{int(gate.threshold)}{inst_std}"
                    actual_str = f"{int(gate.actual_value)}"
                else:
                    threshold_str = f"{gate.threshold:.2f}{inst_std}"
                    actual_str = f"{gate.actual_value:.2f}"

                report_lines.append(
                    f"| {gate.gate_name.replace('_', ' ').title()} | {gate.comparison} {threshold_str} | {actual_str} | {status} | {gate.severity} |"
                )

            report_lines.append("")

        # Add warnings if any
        if gates_result.warnings:
            report_lines.extend([
                "### ⚠️ Warnings",
                "",
            ])
            for warning in gates_result.warnings:
                report_lines.append(f"- {warning}")
            report_lines.append("")

    # Footer
    report_lines.extend([
        "---",
        "",
        "## 📝 Report Metadata",
        "",
        f"- **Report Version:** 2.0.0 (Enhanced)",
        f"- **Generated By:** Stock Agent Trading System",
        f"- **Timestamp:** {datetime.now().isoformat()}",
        f"- **Gates Evaluation:** {'Enabled' if with_gates else 'Disabled'}",
        "",
        "---",
        "",
        "*This report was generated with institutional-grade metrics and quality gates.*",
        "",
    ])

    report = "\n".join(report_lines)

    # Save to file if output_path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {output_path}")

    return report


def generate_html_report(markdown_report: str, output_path: Optional[Path] = None) -> str:
    """Generate HTML report from markdown (basic conversion)"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Backtest Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            line-height: 1.6;
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f5f5f5;
            font-weight: 600;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 1px solid #ecf0f1; padding-bottom: 8px; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        hr {{ border: none; border-top: 2px solid #ecf0f1; margin: 30px 0; }}
        .pass {{ color: #27ae60; font-weight: 600; }}
        .fail {{ color: #e74c3c; font-weight: 600; }}
    </style>
</head>
<body>
{markdown_to_html_simple(markdown_report)}
</body>
</html>"""

    if output_path:
        with open(output_path, 'w') as f:
            f.write(html)
        logger.info(f"HTML report saved to: {output_path}")

    return html


def markdown_to_html_simple(markdown: str) -> str:
    """Simple markdown to HTML conversion"""
    import re

    html = markdown

    # Headers
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)

    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

    # Tables (keep markdown tables, browser will render)
    # Lists
    html = re.sub(r'^\- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

    # Horizontal rules
    html = html.replace('---', '<hr>')

    # Line breaks
    html = html.replace('\n\n', '<br><br>')

    return html


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate enhanced backtest report with quality gates')

    parser.add_argument(
        '--format',
        type=str,
        default='markdown',
        choices=['markdown', 'html', 'pdf'],
        help='Report format (default: markdown)'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Path to backtest results JSON file (default: auto-detect latest)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: auto-generate in backtest_results/)'
    )

    parser.add_argument(
        '--with-gates',
        action='store_true',
        default=True,
        help='Include pass/fail gates evaluation (default: True)'
    )

    parser.add_argument(
        '--no-gates',
        action='store_true',
        help='Disable pass/fail gates evaluation'
    )

    return parser.parse_args()


def main():
    """Main report generation"""
    args = parse_args()

    # Determine if gates should be used
    with_gates = not args.no_gates if hasattr(args, 'no_gates') else args.with_gates

    # Find input file
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_latest_backtest_results()

    if not input_path or not input_path.exists():
        logger.error("No backtest results found")
        return 1

    # Load results
    logger.info(f"Loading results from: {input_path}")
    results = load_backtest_results(input_path)

    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.format == 'markdown':
        output_path = Path(args.output) if args.output else Path(f"backtest_results/report_{timestamp}.md")
        report = generate_markdown_report(results, output_path, with_gates=with_gates)

        if not args.output:
            print(report)

        logger.success(f"✅ Enhanced markdown report generated: {output_path}")
        if with_gates:
            logger.info("   Report includes quality gates evaluation")

    elif args.format == 'html':
        # First generate markdown with gates
        markdown_report = generate_markdown_report(results, with_gates=with_gates)

        # Convert to HTML
        output_path = Path(args.output) if args.output else Path(f"backtest_results/report_{timestamp}.html")
        html_report = generate_html_report(markdown_report, output_path)

        logger.success(f"✅ Enhanced HTML report generated: {output_path}")
        if with_gates:
            logger.info("   Report includes quality gates evaluation")

    elif args.format == 'pdf':
        logger.error("PDF generation not yet implemented (requires reportlab or weasyprint)")
        logger.info("Generate HTML report and print to PDF instead:")
        logger.info("  python scripts/generate_report.py --format html")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
