"""
Performance Metrics Calculator - Institutional-grade trading metrics

Computes:
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside risk-adjusted returns)
- Max Drawdown (largest peak-to-trough decline)
- Calmar Ratio (return/max drawdown)
- Win Rate (% of profitable trades)
- Profit Factor (gross profit / gross loss)
- Volatility (annual standard deviation)
- Downside Deviation (downside volatility only)

Usage:
    calculator = MetricsCalculator(equity_curve, trades, risk_free_rate=0.02)
    metrics = calculator.calculate_all_metrics()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class TradeResult:
    """Single trade result"""
    entry_date: str
    exit_date: str
    symbol: str
    pnl: float
    return_pct: float
    holding_period_days: int


@dataclass
class PerformanceMetrics:
    """Complete performance metrics"""
    # Returns
    total_return: float
    annualized_return: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_duration_days: int

    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    winning_trades: int
    losing_trades: int
    total_trades: int

    # Duration
    trading_days: int
    years: float


class MetricsCalculator:
    """
    Calculate institutional-grade performance metrics.

    Formulas:
    - Sharpe Ratio = (R_p - R_f) / σ_p
    - Sortino Ratio = (R_p - R_f) / σ_downside
    - Max Drawdown = max((Peak - Trough) / Peak)
    - Calmar Ratio = Annualized Return / Max Drawdown
    - Win Rate = Winning Trades / Total Trades
    - Profit Factor = Gross Profit / Gross Loss
    """

    def __init__(
        self,
        equity_curve: pd.Series,
        trades: Optional[List[TradeResult]] = None,
        risk_free_rate: float = 0.02  # 2% annual risk-free rate (10-year Treasury)
    ):
        """
        Initialize metrics calculator.

        Args:
            equity_curve: Time series of portfolio value (daily)
            trades: List of individual trade results (optional)
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino
        """
        self.equity_curve = equity_curve
        self.trades = trades or []
        self.risk_free_rate = risk_free_rate

        # Convert equity curve to returns
        self.returns = self.equity_curve.pct_change().dropna()

        # Trading days and years
        self.trading_days = len(self.equity_curve)
        self.years = self.trading_days / 252.0  # 252 trading days per year

        logger.debug(f"Initialized MetricsCalculator: {self.trading_days} days, {self.years:.2f} years")

    def calculate_all_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics"""

        # Returns
        total_return = self.calculate_total_return()
        annualized_return = self.calculate_annualized_return(total_return)

        # Risk metrics
        volatility = self.calculate_volatility()
        downside_deviation = self.calculate_downside_deviation()
        max_dd, max_dd_duration = self.calculate_max_drawdown()

        # Risk-adjusted metrics
        sharpe_ratio = self.calculate_sharpe_ratio(annualized_return, volatility)
        sortino_ratio = self.calculate_sortino_ratio(annualized_return, downside_deviation)
        calmar_ratio = self.calculate_calmar_ratio(annualized_return, max_dd)

        # Trading metrics
        trade_metrics = self.calculate_trade_metrics()

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_duration,
            win_rate=trade_metrics['win_rate'],
            profit_factor=trade_metrics['profit_factor'],
            avg_trade_pnl=trade_metrics['avg_trade_pnl'],
            avg_win=trade_metrics['avg_win'],
            avg_loss=trade_metrics['avg_loss'],
            winning_trades=trade_metrics['winning_trades'],
            losing_trades=trade_metrics['losing_trades'],
            total_trades=trade_metrics['total_trades'],
            trading_days=self.trading_days,
            years=self.years
        )

    def calculate_total_return(self) -> float:
        """Calculate total return"""
        if len(self.equity_curve) == 0:
            return 0.0

        initial_value = self.equity_curve.iloc[0]
        final_value = self.equity_curve.iloc[-1]

        if initial_value == 0:
            return 0.0

        return (final_value - initial_value) / initial_value

    def calculate_annualized_return(self, total_return: float) -> float:
        """
        Calculate annualized return (CAGR).

        Formula: (1 + total_return)^(1/years) - 1
        """
        if self.years == 0:
            return 0.0

        return (1 + total_return) ** (1 / self.years) - 1

    def calculate_volatility(self) -> float:
        """
        Calculate annualized volatility (standard deviation).

        Formula: σ_annual = σ_daily * sqrt(252)
        """
        if len(self.returns) == 0:
            return 0.0

        daily_vol = self.returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        return annual_vol

    def calculate_downside_deviation(self, target_return: float = 0.0) -> float:
        """
        Calculate annualized downside deviation (semi-deviation).

        Only considers returns below target (default: 0).

        Formula: σ_downside = sqrt(mean((min(R - target, 0))^2)) * sqrt(252)
        """
        if len(self.returns) == 0:
            return 0.0

        # Daily target return
        daily_target = target_return / 252.0

        # Downside returns only
        downside_returns = self.returns[self.returns < daily_target]

        if len(downside_returns) == 0:
            return 0.0

        # Downside deviation
        downside_diff = downside_returns - daily_target
        daily_downside_dev = np.sqrt(np.mean(downside_diff ** 2))
        annual_downside_dev = daily_downside_dev * np.sqrt(252)

        return annual_downside_dev

    def calculate_sharpe_ratio(self, annualized_return: float, volatility: float) -> float:
        """
        Calculate Sharpe Ratio.

        Formula: (R_p - R_f) / σ_p

        Where:
        - R_p = portfolio return
        - R_f = risk-free rate
        - σ_p = portfolio volatility
        """
        if volatility == 0:
            return 0.0

        excess_return = annualized_return - self.risk_free_rate
        sharpe = excess_return / volatility

        return sharpe

    def calculate_sortino_ratio(self, annualized_return: float, downside_deviation: float) -> float:
        """
        Calculate Sortino Ratio.

        Formula: (R_p - R_f) / σ_downside

        Like Sharpe, but only penalizes downside volatility.
        """
        if downside_deviation == 0:
            return 0.0

        excess_return = annualized_return - self.risk_free_rate
        sortino = excess_return / downside_deviation

        return sortino

    def calculate_max_drawdown(self) -> tuple[float, int]:
        """
        Calculate maximum drawdown and duration.

        Returns:
            Tuple of (max_drawdown, duration_days)

        Max Drawdown = max((Peak - Trough) / Peak)
        """
        if len(self.equity_curve) == 0:
            return 0.0, 0

        # Calculate running maximum (peak)
        running_max = self.equity_curve.expanding().max()

        # Calculate drawdown at each point
        drawdown = (self.equity_curve - running_max) / running_max

        # Max drawdown (most negative value)
        max_dd = drawdown.min()

        # Duration: days from peak to recovery
        # Find the index of max drawdown
        max_dd_idx = drawdown.idxmin()

        # Find previous peak
        peak_value = running_max.loc[max_dd_idx]
        peak_idx = (self.equity_curve == peak_value).idxmax()

        # Find recovery (if any)
        after_max_dd = self.equity_curve.loc[max_dd_idx:]
        recovery_mask = after_max_dd >= peak_value

        if recovery_mask.any():
            recovery_idx = after_max_dd[recovery_mask].index[0]
            duration = (recovery_idx - peak_idx).days if hasattr(peak_idx, 'days') else int(recovery_idx - peak_idx)
        else:
            # Still in drawdown at end
            duration = len(self.equity_curve) - int(peak_idx) if isinstance(peak_idx, (int, np.integer)) else (self.equity_curve.index[-1] - peak_idx).days

        return abs(max_dd), duration

    def calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar Ratio.

        Formula: Annualized Return / |Max Drawdown|

        Measures return per unit of downside risk.
        """
        if max_drawdown == 0:
            return 0.0

        calmar = annualized_return / abs(max_drawdown)

        return calmar

    def calculate_trade_metrics(self) -> Dict:
        """Calculate trade-level metrics"""

        if not self.trades or len(self.trades) == 0:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_trades': 0
            }

        # Extract P&L
        pnls = [trade.pnl for trade in self.trades]

        # Winning and losing trades
        winning_pnls = [pnl for pnl in pnls if pnl > 0]
        losing_pnls = [pnl for pnl in pnls if pnl < 0]

        winning_trades = len(winning_pnls)
        losing_trades = len(losing_pnls)
        total_trades = len(pnls)

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Averages
        avg_trade_pnl = np.mean(pnls) if pnls else 0.0
        avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0

        # Profit factor
        gross_profit = sum(winning_pnls) if winning_pnls else 0.0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_trades': total_trades
        }


def metrics_to_dict(metrics: PerformanceMetrics) -> Dict:
    """Convert PerformanceMetrics to dictionary"""
    return {
        'total_return': metrics.total_return,
        'annualized_return': metrics.annualized_return,
        'sharpe_ratio': metrics.sharpe_ratio,
        'sortino_ratio': metrics.sortino_ratio,
        'calmar_ratio': metrics.calmar_ratio,
        'volatility': metrics.volatility,
        'downside_deviation': metrics.downside_deviation,
        'max_drawdown': metrics.max_drawdown,
        'max_drawdown_duration_days': metrics.max_drawdown_duration_days,
        'win_rate': metrics.win_rate,
        'profit_factor': metrics.profit_factor,
        'avg_trade_pnl': metrics.avg_trade_pnl,
        'avg_win': metrics.avg_win,
        'avg_loss': metrics.avg_loss,
        'winning_trades': metrics.winning_trades,
        'losing_trades': metrics.losing_trades,
        'total_trades': metrics.total_trades,
        'trading_days': metrics.trading_days,
        'years': metrics.years
    }


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from datetime import datetime, timedelta

    print("\n" + "="*60)
    print("METRICS CALCULATOR TEST")
    print("="*60 + "\n")

    # Create sample equity curve (starts at $100k, grows to $150k)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # Simulate growing equity with some volatility
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% daily return, 2% volatility
    equity_values = [100000.0]

    for ret in returns[1:]:
        equity_values.append(equity_values[-1] * (1 + ret))

    equity_curve = pd.Series(equity_values, index=dates)

    # Create sample trades
    trades = [
        TradeResult(entry_date='2023-01-15', exit_date='2023-01-20', symbol='AAPL', pnl=500, return_pct=0.05, holding_period_days=5),
        TradeResult(entry_date='2023-02-10', exit_date='2023-02-15', symbol='MSFT', pnl=-200, return_pct=-0.02, holding_period_days=5),
        TradeResult(entry_date='2023-03-05', exit_date='2023-03-10', symbol='GOOGL', pnl=1000, return_pct=0.10, holding_period_days=5),
        TradeResult(entry_date='2023-04-12', exit_date='2023-04-17', symbol='AAPL', pnl=300, return_pct=0.03, holding_period_days=5),
        TradeResult(entry_date='2023-05-20', exit_date='2023-05-25', symbol='TSLA', pnl=-500, return_pct=-0.05, holding_period_days=5),
    ]

    # Calculate metrics
    calculator = MetricsCalculator(equity_curve, trades, risk_free_rate=0.02)
    metrics = calculator.calculate_all_metrics()

    # Print results
    print("RETURNS:")
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Annualized Return: {metrics.annualized_return:.2%}")

    print("\nRISK-ADJUSTED RETURNS:")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.3f}")

    print("\nRISK METRICS:")
    print(f"  Volatility (Annual): {metrics.volatility:.2%}")
    print(f"  Downside Deviation: {metrics.downside_deviation:.2%}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Max DD Duration: {metrics.max_drawdown_duration_days} days")

    print("\nTRADING METRICS:")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Winning Trades: {metrics.winning_trades}")
    print(f"  Losing Trades: {metrics.losing_trades}")
    print(f"  Win Rate: {metrics.win_rate:.2%}")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Avg Trade P&L: ${metrics.avg_trade_pnl:.2f}")
    print(f"  Avg Win: ${metrics.avg_win:.2f}")
    print(f"  Avg Loss: ${metrics.avg_loss:.2f}")

    print("\nDURATION:")
    print(f"  Trading Days: {metrics.trading_days}")
    print(f"  Years: {metrics.years:.2f}")

    print("\n" + "="*60 + "\n")
