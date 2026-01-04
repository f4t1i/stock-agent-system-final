"""
Backtester - Simulate trading strategy on historical data
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from loguru import logger

from utils.metrics import calculate_portfolio_metrics


class Backtester:
    """
    Backtest trading strategies on historical data.

    Simulates trades, calculates P&L, and computes performance metrics.
    """

    def __init__(
        self,
        coordinator,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000
    ):
        """
        Args:
            coordinator: SystemCoordinator instance
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_capital: Starting capital
        """
        self.coordinator = coordinator
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        # Portfolio state
        self.cash = initial_capital
        self.positions = {}
        self.equity_curve = [initial_capital]
        self.trades = []
        self.daily_returns = []

    def run(self, symbols: List[str]) -> Dict:
        """
        Run backtest on list of symbols

        Args:
            symbols: List of stock symbols

        Returns:
            Dict with backtest results and metrics
        """
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        logger.info(f"Symbols: {', '.join(symbols)}")

        # Get date range
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)

        # Download historical data for all symbols
        logger.info("Downloading historical data...")
        price_data = self._download_data(symbols, start, end)

        if not price_data:
            logger.error("Failed to download price data")
            return {'error': 'Failed to download data'}

        # Get trading days
        all_dates = sorted(set(
            date for symbol_data in price_data.values()
            for date in symbol_data.index
        ))

        logger.info(f"Running backtest for {len(all_dates)} trading days")

        # Simulate trading day by day
        for i, date in enumerate(all_dates):
            if i % 20 == 0:
                logger.info(f"Processing day {i+1}/{len(all_dates)}: {date.date()}")

            self._simulate_day(date, symbols, price_data)

        # Calculate final metrics
        metrics = self._calculate_metrics()

        logger.info("Backtest complete")

        return metrics

    def _download_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Download historical price data"""

        price_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end)

                if not df.empty:
                    price_data[symbol] = df

            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")

        return price_data

    def _simulate_day(
        self,
        date: datetime,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame]
    ):
        """Simulate one trading day"""

        # Get prices for this day
        current_prices = {}
        for symbol in symbols:
            if symbol in price_data and date in price_data[symbol].index:
                current_prices[symbol] = float(price_data[symbol].loc[date, 'Close'])

        if not current_prices:
            return

        # Update portfolio value
        portfolio_value = self.cash
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                portfolio_value += quantity * current_prices[symbol]

        # Calculate daily return
        if len(self.equity_curve) > 0:
            daily_return = (portfolio_value - self.equity_curve[-1]) / self.equity_curve[-1]
            self.daily_returns.append(daily_return)

        self.equity_curve.append(portfolio_value)

        # Make trading decisions (simplified - only every N days)
        if len(self.equity_curve) % 5 == 0:  # Trade every 5 days
            self._execute_strategy(date, symbols, current_prices)

    def _execute_strategy(
        self,
        date: datetime,
        symbols: List[str],
        current_prices: Dict[str, float]
    ):
        """Execute trading strategy"""

        # For each symbol, get recommendation
        for symbol in symbols:
            if symbol not in current_prices:
                continue

            try:
                # Get analysis from coordinator
                analysis = self.coordinator.analyze_symbol(symbol, use_supervisor=False)

                decision = analysis.get('recommendation', 'hold')
                confidence = analysis.get('confidence', 0.0)
                position_size = analysis.get('position_size', 0.0)

                # Execute based on decision
                if decision == 'buy' and confidence > 0.6:
                    self._execute_buy(symbol, current_prices[symbol], position_size, date)

                elif decision == 'sell' and symbol in self.positions:
                    self._execute_sell(symbol, current_prices[symbol], date)

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

    def _execute_buy(
        self,
        symbol: str,
        price: float,
        position_size: float,
        date: datetime
    ):
        """Execute buy order"""

        # Calculate shares to buy
        max_investment = self.cash * min(position_size, 0.20)  # Max 20% per position
        shares = int(max_investment / price)

        if shares > 0 and shares * price <= self.cash:
            cost = shares * price

            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0) + shares

            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'buy',
                'quantity': shares,
                'price': price,
                'cost': cost,
                'pnl': 0
            })

            logger.debug(f"BUY {shares} shares of {symbol} at ${price:.2f}")

    def _execute_sell(self, symbol: str, price: float, date: datetime):
        """Execute sell order"""

        if symbol not in self.positions:
            return

        shares = self.positions[symbol]
        proceeds = shares * price

        # Calculate P&L
        buy_trades = [t for t in self.trades if t['symbol'] == symbol and t['action'] == 'buy']
        avg_cost = sum(t['cost'] for t in buy_trades) / sum(t['quantity'] for t in buy_trades) if buy_trades else price
        pnl = proceeds - (shares * avg_cost)

        self.cash += proceeds
        del self.positions[symbol]

        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'sell',
            'quantity': shares,
            'price': price,
            'proceeds': proceeds,
            'pnl': pnl
        })

        logger.debug(f"SELL {shares} shares of {symbol} at ${price:.2f} (P&L: ${pnl:.2f})")

    def _calculate_metrics(self) -> Dict:
        """Calculate backtest metrics"""

        if len(self.equity_curve) < 2:
            return {'error': 'Insufficient data'}

        # Calculate comprehensive metrics
        metrics = calculate_portfolio_metrics(
            returns=self.daily_returns,
            equity_curve=self.equity_curve,
            trades=self.trades
        )

        # Add backtest-specific info
        metrics.update({
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_value': self.equity_curve[-1],
            'num_trades': len(self.trades),
            'final_cash': self.cash,
            'final_positions': len(self.positions),
            'trading_days': len(self.equity_curve) - 1
        })

        return metrics


if __name__ == "__main__":
    # Example usage
    from orchestration.coordinator import SystemCoordinator

    coordinator = SystemCoordinator()

    backtester = Backtester(
        coordinator,
        start_date="2023-01-01",
        end_date="2023-12-31",
        initial_capital=100000
    )

    results = backtester.run(['AAPL', 'MSFT', 'GOOGL'])

    print("\n=== Backtest Results ===")
    for key, value in results.items():
        if isinstance(value, float):
            if 'return' in key or 'drawdown' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
