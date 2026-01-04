"""
Performance Metrics - Calculate trading and portfolio metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
    annualize: bool = True
) -> float:
    """
    Calculate Sharpe Ratio

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate (default 2%)
        annualize: Annualize the ratio (default True)

    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)

    # Mean excess return
    mean_return = np.mean(returns_array)

    # Adjust risk-free rate to period
    if annualize:
        periods_per_year = 252  # Trading days
        period_rf_rate = risk_free_rate / periods_per_year
    else:
        period_rf_rate = risk_free_rate

    excess_return = mean_return - period_rf_rate

    # Standard deviation
    std_dev = np.std(returns_array, ddof=1)

    if std_dev == 0:
        return 0.0

    # Sharpe ratio
    sharpe = excess_return / std_dev

    # Annualize if requested
    if annualize:
        sharpe = sharpe * np.sqrt(periods_per_year)

    return float(sharpe)


def calculate_sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
    annualize: bool = True
) -> float:
    """
    Calculate Sortino Ratio (like Sharpe but only penalizes downside volatility)

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        annualize: Annualize the ratio

    Returns:
        Sortino ratio
    """
    if not returns or len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)

    mean_return = np.mean(returns_array)

    if annualize:
        periods_per_year = 252
        period_rf_rate = risk_free_rate / periods_per_year
    else:
        period_rf_rate = risk_free_rate

    excess_return = mean_return - period_rf_rate

    # Downside deviation (only negative returns)
    downside_returns = returns_array[returns_array < 0]

    if len(downside_returns) == 0:
        return float('inf')  # No downside risk

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0:
        return 0.0

    sortino = excess_return / downside_std

    if annualize:
        sortino = sortino * np.sqrt(periods_per_year)

    return float(sortino)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate Maximum Drawdown

    Args:
        equity_curve: List of portfolio values over time

    Returns:
        Max drawdown as decimal (e.g., 0.15 for 15%)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    equity_array = np.array(equity_curve)

    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_array)

    # Calculate drawdown at each point
    drawdown = (equity_array - running_max) / running_max

    # Maximum drawdown (most negative)
    max_dd = np.min(drawdown)

    return float(abs(max_dd))


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    Calculate Win Rate

    Args:
        trades: List of trade dicts with 'pnl' field

    Returns:
        Win rate as decimal (e.g., 0.55 for 55%)
    """
    if not trades:
        return 0.0

    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)

    return winning_trades / len(trades)


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Calculate Profit Factor (gross profit / gross loss)

    Args:
        trades: List of trade dicts with 'pnl' field

    Returns:
        Profit factor
    """
    if not trades:
        return 0.0

    gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
    gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_calmar_ratio(
    returns: List[float],
    equity_curve: List[float],
    annualize: bool = True
) -> float:
    """
    Calculate Calmar Ratio (Annual Return / Max Drawdown)

    Args:
        returns: List of period returns
        equity_curve: List of portfolio values
        annualize: Annualize the return

    Returns:
        Calmar ratio
    """
    if not returns or not equity_curve:
        return 0.0

    # Total return
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    # Annualize if requested
    if annualize:
        periods = len(returns)
        periods_per_year = 252
        years = periods / periods_per_year
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = total_return

    # Max drawdown
    max_dd = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return float('inf') if annual_return > 0 else 0.0

    return annual_return / max_dd


def calculate_portfolio_metrics(
    returns: List[float],
    equity_curve: List[float],
    trades: Optional[List[Dict]] = None,
    benchmark_returns: Optional[List[float]] = None
) -> Dict:
    """
    Calculate comprehensive portfolio metrics

    Args:
        returns: List of period returns
        equity_curve: Portfolio values over time
        trades: Optional list of trades
        benchmark_returns: Optional benchmark returns for comparison

    Returns:
        Dict with all metrics
    """
    metrics = {
        'total_return': 0.0,
        'annualized_return': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'max_drawdown': 0.0,
        'calmar_ratio': 0.0,
        'volatility': 0.0
    }

    if not returns or not equity_curve:
        return metrics

    # Total return
    metrics['total_return'] = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    # Annualized return
    periods = len(returns)
    periods_per_year = 252
    years = periods / periods_per_year
    if years > 0:
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (1 / years) - 1

    # Sharpe ratio
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)

    # Sortino ratio
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns)

    # Max drawdown
    metrics['max_drawdown'] = calculate_max_drawdown(equity_curve)

    # Calmar ratio
    metrics['calmar_ratio'] = calculate_calmar_ratio(returns, equity_curve)

    # Volatility
    metrics['volatility'] = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))

    # Trade-based metrics
    if trades:
        metrics['win_rate'] = calculate_win_rate(trades)
        metrics['profit_factor'] = calculate_profit_factor(trades)
        metrics['total_trades'] = len(trades)
        metrics['avg_win'] = np.mean([t['pnl'] for t in trades if t.get('pnl', 0) > 0]) if any(t.get('pnl', 0) > 0 for t in trades) else 0
        metrics['avg_loss'] = np.mean([t['pnl'] for t in trades if t.get('pnl', 0) < 0]) if any(t.get('pnl', 0) < 0 for t in trades) else 0

    # Benchmark comparison
    if benchmark_returns and len(benchmark_returns) == len(returns):
        metrics['alpha'] = metrics['annualized_return'] - np.mean(benchmark_returns) * periods_per_year
        metrics['beta'] = np.cov(returns, benchmark_returns)[0][1] / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0

    return metrics


if __name__ == "__main__":
    # Example usage
    example_returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.03, -0.002, 0.012]
    example_equity = [100000, 101000, 100495, 102505, 104043, 103003, 106093, 105881, 107152]

    metrics = calculate_portfolio_metrics(example_returns, example_equity)

    print("Portfolio Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'return' in key or 'drawdown' in key or 'volatility' in key or 'rate' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
