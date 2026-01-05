#!/usr/bin/env python
"""
Generate Backtest Report - Create standardized reports from backtest results

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --format html
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


def generate_markdown_report(results: Dict, output_path: Optional[Path] = None) -> str:
    """Generate markdown report"""

    report_lines = [
        "# Backtest Report",
        "",
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
        "### Controls",
        "",
        f"- **Survivorship Bias Guard:** {'✅ Enabled' if results.get('survivorship_bias_guarded') else '❌ Disabled'}",
        f"- **Corporate Actions:** {'✅ Enabled' if results.get('corporate_actions_handled') else '❌ Disabled'}",
        f"- **Signal Validation:** {'✅ Enabled' if results['config']['validate_signals'] else '❌ Disabled'}",
        "",
        "---",
        "",
        "## Performance Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Initial Capital** | ${results['initial_capital']:,.2f} |",
        f"| **Final Value** | ${results['final_value']:,.2f} |",
        f"| **Total Return** | {results['total_return']:.2%} |",
        f"| **Sharpe Ratio** | {results.get('sharpe_ratio', 0):.3f} |",
        f"| **Max Drawdown** | {results.get('max_drawdown', 0):.2%} |",
        f"| **Win Rate** | {results.get('win_rate', 0):.2%} |",
        f"| **Profit Factor** | {results.get('profit_factor', 0):.2f} |",
        "",
        "---",
        "",
        "## Trading Activity",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Total Trades** | {results['num_trades']} |",
        f"| **Trading Days** | {results['trading_days']} |",
        f"| **Winning Trades** | {results.get('winning_trades', 0)} |",
        f"| **Losing Trades** | {results.get('losing_trades', 0)} |",
        f"| **Avg Trade P&L** | ${results.get('avg_trade_pnl', 0):.2f} |",
        "",
        "---",
        "",
        "## Risk Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Volatility (Annual)** | {results.get('volatility', 0):.2%} |",
        f"| **Downside Deviation** | {results.get('downside_deviation', 0):.2%} |",
        f"| **Sortino Ratio** | {results.get('sortino_ratio', 0):.3f} |",
        f"| **Calmar Ratio** | {results.get('calmar_ratio', 0):.3f} |",
        "",
    ]

    # Add delisted symbols if any
    if results.get('delisted_symbols'):
        report_lines.extend([
            "---",
            "",
            "## ⚠️ Delisted Symbols",
            "",
            f"The following symbols were delisted during the backtest period:",
            "",
        ])
        for symbol in results['delisted_symbols']:
            report_lines.append(f"- {symbol}")
        report_lines.append("")

    # Add failed signals if any
    if results.get('failed_signals', 0) > 0:
        report_lines.extend([
            "---",
            "",
            "## ❌ Failed Signals",
            "",
            f"**Total Failed Signals:** {results['failed_signals']}",
            "",
            "Signals that failed validation were excluded from trading.",
            "",
        ])

    # Add pass/fail gates section (placeholder for Phase A0 Week 2)
    report_lines.extend([
        "---",
        "",
        "## ✅ Pass/Fail Gates",
        "",
        "| Gate | Threshold | Result | Status |",
        "|------|-----------|--------|--------|",
    ])

    # Example gates (will be configurable later)
    sharpe_threshold = 1.0
    sharpe = results.get('sharpe_ratio', 0)
    sharpe_status = "✅ PASS" if sharpe >= sharpe_threshold else "❌ FAIL"

    max_dd_threshold = 0.20  # 20%
    max_dd = abs(results.get('max_drawdown', 0))
    max_dd_status = "✅ PASS" if max_dd <= max_dd_threshold else "❌ FAIL"

    win_rate_threshold = 0.50  # 50%
    win_rate = results.get('win_rate', 0)
    win_rate_status = "✅ PASS" if win_rate >= win_rate_threshold else "❌ FAIL"

    report_lines.extend([
        f"| Sharpe Ratio | ≥ {sharpe_threshold} | {sharpe:.2f} | {sharpe_status} |",
        f"| Max Drawdown | ≤ {max_dd_threshold:.0%} | {max_dd:.2%} | {max_dd_status} |",
        f"| Win Rate | ≥ {win_rate_threshold:.0%} | {win_rate:.2%} | {win_rate_status} |",
        "",
    ])

    # Footer
    report_lines.extend([
        "---",
        "",
        "*Report generated by Stock Agent Trading System v1.0.0*",
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
    parser = argparse.ArgumentParser(description='Generate backtest report')

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

    return parser.parse_args()


def main():
    """Main report generation"""
    args = parse_args()

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
        report = generate_markdown_report(results, output_path)

        if not args.output:
            print(report)

        logger.success(f"✅ Markdown report generated: {output_path}")

    elif args.format == 'html':
        # First generate markdown
        markdown_report = generate_markdown_report(results)

        # Convert to HTML
        output_path = Path(args.output) if args.output else Path(f"backtest_results/report_{timestamp}.html")
        html_report = generate_html_report(markdown_report, output_path)

        logger.success(f"✅ HTML report generated: {output_path}")

    elif args.format == 'pdf':
        logger.error("PDF generation not yet implemented (requires reportlab or weasyprint)")
        logger.info("Generate HTML report and print to PDF instead:")
        logger.info("  python scripts/generate_report.py --format html")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
