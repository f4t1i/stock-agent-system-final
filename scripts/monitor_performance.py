#!/usr/bin/env python3
"""
Performance Monitoring Dashboard

Real-time monitoring of system performance with:
- Live metrics display
- Progress towards goals
- Performance charts
- Alert system
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.warning("Rich library not available, using simple output")


class PerformanceMonitor:
    """
    Real-time performance monitoring dashboard.

    Displays:
    - Current metrics (Sharpe, Win Rate, etc.)
    - Progress towards goals
    - Recent performance history
    - Alerts and warnings
    """

    def __init__(
        self,
        state_file: str = "continuous_training/training_state.json",
        refresh_interval: int = 5
    ):
        """
        Initialize monitor.

        Args:
            state_file: Path to training state file
            refresh_interval: Refresh interval in seconds
        """
        self.state_file = Path(state_file)
        self.refresh_interval = refresh_interval
        self.console = Console() if RICH_AVAILABLE else None

        logger.info("Performance Monitor initialized")

    def load_state(self) -> Dict:
        """Load current training state"""
        if not self.state_file.exists():
            return {
                'total_trajectories': 0,
                'training_iterations': 0,
                'best_sharpe': 0.0,
                'best_win_rate': 0.0,
                'benchmarks': []
            }

        with open(self.state_file, 'r') as f:
            return json.load(f)

    def create_dashboard(self, state: Dict, targets: Dict) -> Layout:
        """Create dashboard layout"""
        if not RICH_AVAILABLE:
            return None

        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        # Header
        layout["header"].update(
            Panel(
                f"[bold]Stock Agent System - Performance Dashboard[/bold]\n"
                f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                style="bold blue"
            )
        )

        # Body
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        # Left: Current metrics
        layout["left"].update(self._create_metrics_table(state, targets))

        # Right: Progress and history
        layout["right"].split_column(
            Layout(name="progress"),
            Layout(name="history")
        )

        layout["right"]["progress"].update(self._create_progress_panel(state, targets))
        layout["right"]["history"].update(self._create_history_table(state))

        # Footer
        status = self._get_status_message(state, targets)
        layout["footer"].update(Panel(status, style="bold"))

        return layout

    def _create_metrics_table(self, state: Dict, targets: Dict) -> Table:
        """Create current metrics table"""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="Current Performance", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="magenta")
        table.add_column("Target", style="green")
        table.add_column("Status", style="bold")

        # Get latest benchmark
        latest = state['benchmarks'][-1] if state['benchmarks'] else {}

        # Trajectories
        traj_current = state['total_trajectories']
        traj_target = targets['trajectories']
        traj_status = "✅" if traj_current >= traj_target else "⏳"
        table.add_row(
            "Trajectories",
            f"{traj_current:,}",
            f"{traj_target:,}",
            traj_status
        )

        # Sharpe
        sharpe_current = latest.get('sharpe_ratio', state['best_sharpe'])
        sharpe_target = targets['sharpe']
        sharpe_status = "✅" if sharpe_current >= sharpe_target else "⏳"
        table.add_row(
            "Sharpe Ratio",
            f"{sharpe_current:.3f}",
            f">{sharpe_target}",
            sharpe_status
        )

        # Win Rate
        win_rate_current = latest.get('win_rate', state['best_win_rate'])
        win_rate_target = targets['win_rate']
        win_rate_status = "✅" if win_rate_current >= win_rate_target else "⏳"
        table.add_row(
            "Win Rate",
            f"{win_rate_current*100:.1f}%",
            f">{win_rate_target*100:.1f}%",
            win_rate_status
        )

        # Additional metrics
        if latest:
            table.add_row(
                "Total Return",
                f"{latest.get('total_return', 0)*100:.1f}%",
                "-",
                "ℹ️"
            )

            table.add_row(
                "Max Drawdown",
                f"{latest.get('max_drawdown', 0)*100:.1f}%",
                "<15%",
                "✅" if abs(latest.get('max_drawdown', 0)) < 0.15 else "⚠️"
            )

        return table

    def _create_progress_panel(self, state: Dict, targets: Dict) -> Panel:
        """Create progress panel"""
        if not RICH_AVAILABLE:
            return Panel("Progress unavailable")

        # Calculate progress percentages
        traj_progress = min(state['total_trajectories'] / targets['trajectories'] * 100, 100)

        latest = state['benchmarks'][-1] if state['benchmarks'] else {}
        sharpe_current = latest.get('sharpe_ratio', state['best_sharpe'])
        sharpe_progress = min(sharpe_current / targets['sharpe'] * 100, 100)

        win_rate_current = latest.get('win_rate', state['best_win_rate'])
        win_rate_progress = min(win_rate_current / targets['win_rate'] * 100, 100)

        content = f"""[bold]Progress Towards Goals[/bold]

Trajectories: {traj_progress:.1f}%
{'█' * int(traj_progress/5)}{'░' * (20 - int(traj_progress/5))}

Sharpe Ratio: {sharpe_progress:.1f}%
{'█' * int(sharpe_progress/5)}{'░' * (20 - int(sharpe_progress/5))}

Win Rate: {win_rate_progress:.1f}%
{'█' * int(win_rate_progress/5)}{'░' * (20 - int(win_rate_progress/5))}

Training Iterations: {state['training_iterations']}
"""

        return Panel(content, style="green")

    def _create_history_table(self, state: Dict) -> Table:
        """Create performance history table"""
        if not RICH_AVAILABLE:
            return None

        table = Table(title="Recent Performance", show_header=True)
        table.add_column("Date", style="cyan")
        table.add_column("Sharpe", style="magenta")
        table.add_column("Win Rate", style="green")
        table.add_column("Return", style="yellow")

        # Show last 5 benchmarks
        recent = state['benchmarks'][-5:] if state['benchmarks'] else []

        for bm in recent:
            date = datetime.fromisoformat(bm['evaluated_at']).strftime('%m-%d %H:%M')
            table.add_row(
                date,
                f"{bm['sharpe_ratio']:.3f}",
                f"{bm['win_rate']*100:.1f}%",
                f"{bm['total_return']*100:+.1f}%"
            )

        if not recent:
            table.add_row("No data", "-", "-", "-")

        return table

    def _get_status_message(self, state: Dict, targets: Dict) -> str:
        """Get status message"""
        # Check goal achievement
        traj_ok = state['total_trajectories'] >= targets['trajectories']

        latest = state['benchmarks'][-1] if state['benchmarks'] else {}
        sharpe_current = latest.get('sharpe_ratio', state['best_sharpe'])
        win_rate_current = latest.get('win_rate', state['best_win_rate'])

        sharpe_ok = sharpe_current >= targets['sharpe']
        win_rate_ok = win_rate_current >= targets['win_rate']

        if traj_ok and sharpe_ok and win_rate_ok:
            return "[bold green]🎉 ALL GOALS ACHIEVED! 🎉[/bold green]"

        # Count pending goals
        pending = sum([not traj_ok, not sharpe_ok, not win_rate_ok])

        if pending == 3:
            return f"[yellow]⏳ All goals pending - Keep training![/yellow]"
        elif pending == 2:
            return f"[yellow]⏳ {pending} goals remaining - Good progress![/yellow]"
        elif pending == 1:
            return f"[green]🎯 Almost there! {pending} goal remaining[/green]"

        return "[blue]Monitoring...[/blue]"

    def monitor_live(
        self,
        target_trajectories: int = 10000,
        target_sharpe: float = 1.5,
        target_win_rate: float = 0.55
    ):
        """Run live monitoring dashboard"""
        targets = {
            'trajectories': target_trajectories,
            'sharpe': target_sharpe,
            'win_rate': target_win_rate
        }

        if RICH_AVAILABLE:
            logger.info("Starting live dashboard (Press Ctrl+C to exit)")

            with Live(refresh_per_second=1) as live:
                try:
                    while True:
                        state = self.load_state()
                        dashboard = self.create_dashboard(state, targets)
                        live.update(dashboard)
                        time.sleep(self.refresh_interval)

                except KeyboardInterrupt:
                    logger.info("\nMonitoring stopped")

        else:
            # Fallback to simple text output
            logger.info("Starting simple monitoring (Press Ctrl+C to exit)")

            try:
                while True:
                    state = self.load_state()
                    self._print_simple_dashboard(state, targets)
                    time.sleep(self.refresh_interval)

            except KeyboardInterrupt:
                logger.info("\nMonitoring stopped")

    def _print_simple_dashboard(self, state: Dict, targets: Dict):
        """Print simple text dashboard"""
        os.system('clear' if os.name == 'posix' else 'cls')

        print("="*70)
        print("STOCK AGENT SYSTEM - PERFORMANCE DASHBOARD")
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        latest = state['benchmarks'][-1] if state['benchmarks'] else {}

        print("\nCurrent Performance:")
        print(f"  Trajectories: {state['total_trajectories']:,}/{targets['trajectories']:,}")
        print(f"  Sharpe Ratio: {latest.get('sharpe_ratio', state['best_sharpe']):.3f} (target: >{targets['sharpe']})")
        print(f"  Win Rate: {latest.get('win_rate', state['best_win_rate'])*100:.1f}% (target: >{targets['win_rate']*100:.1f}%)")

        if latest:
            print(f"  Total Return: {latest.get('total_return', 0)*100:.1f}%")
            print(f"  Max Drawdown: {latest.get('max_drawdown', 0)*100:.1f}%")

        print(f"\nTraining Iterations: {state['training_iterations']}")

        print("\n" + "="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Monitor system performance in real-time"
    )

    parser.add_argument('--state-file', type=str,
                       default='continuous_training/training_state.json',
                       help='Path to training state file')
    parser.add_argument('--refresh', type=int, default=5,
                       help='Refresh interval in seconds (default: 5)')

    parser.add_argument('--target-trajectories', type=int, default=10000,
                       help='Target trajectory count')
    parser.add_argument('--target-sharpe', type=float, default=1.5,
                       help='Target Sharpe ratio')
    parser.add_argument('--target-win-rate', type=float, default=0.55,
                       help='Target win rate')

    args = parser.parse_args()

    monitor = PerformanceMonitor(
        state_file=args.state_file,
        refresh_interval=args.refresh
    )

    monitor.monitor_live(
        target_trajectories=args.target_trajectories,
        target_sharpe=args.target_sharpe,
        target_win_rate=args.target_win_rate
    )


if __name__ == '__main__':
    main()
