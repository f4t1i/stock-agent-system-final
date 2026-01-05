"""
Backtest Harness v2 - Deterministic backtesting with institutional-grade controls

Enhancements over v1:
- Survivorship Bias Guards (delisting detection)
- Corporate Actions Handling (splits, dividends)
- Standardized Input/Output Contract
- Fail-Fast on missing data
- Signal Contract integration
- Reproducible results with seed control

Based on:
- TradingAgents backtest harness
- Qlib evaluation framework
- LEAN backtesting engine standards
"""

import json
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import yfinance as yf
import numpy as np
from loguru import logger

from contracts.signal_validator import SignalValidator, SignalValidationError
from utils.metrics import calculate_portfolio_metrics


@dataclass
class BacktestConfig:
    """Standardized backtest configuration contract"""
    # Universe
    symbols: List[str]
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD

    # Capital
    initial_capital: float = 100000.0

    # Costs
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_bps: float = 5.0  # 5 basis points

    # Controls
    enable_survivorship_bias_guard: bool = True
    enable_corporate_actions: bool = True
    fail_fast_on_missing_data: bool = True
    validate_signals: bool = True

    # Reproducibility
    random_seed: Optional[int] = 42

    # Output
    output_dir: str = "backtest_results"
    save_trades: bool = True
    save_signals: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []

        # Date validation
        try:
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            if start >= end:
                errors.append(f"start_date ({self.start_date}) must be < end_date ({self.end_date})")
        except Exception as e:
            errors.append(f"Invalid date format: {e}")

        # Symbol validation
        if not self.symbols:
            errors.append("symbols list is empty")

        for symbol in self.symbols:
            if not symbol.isupper() or len(symbol) > 5:
                errors.append(f"Invalid symbol: {symbol} (must be uppercase, max 5 chars)")

        # Capital validation
        if self.initial_capital <= 0:
            errors.append(f"initial_capital must be > 0, got {self.initial_capital}")

        # Commission/slippage validation
        if self.commission_rate < 0 or self.commission_rate > 0.1:
            errors.append(f"commission_rate must be in [0, 0.1], got {self.commission_rate}")

        if self.slippage_bps < 0 or self.slippage_bps > 100:
            errors.append(f"slippage_bps must be in [0, 100], got {self.slippage_bps}")

        return len(errors) == 0, errors


@dataclass
class CorporateAction:
    """Corporate action (split, dividend)"""
    symbol: str
    date: str
    action_type: str  # 'split' or 'dividend'
    ratio: Optional[float] = None  # For splits (e.g., 2.0 for 2:1 split)
    amount: Optional[float] = None  # For dividends ($ per share)


class SurvivorshipBiasGuard:
    """
    Detect and handle delisted stocks to prevent survivorship bias.

    In backtesting, using only currently-listed stocks creates survivorship bias
    (ignoring failed companies). This guard detects delistings and handles them.
    """

    def __init__(self):
        self.delisted_stocks = {}  # symbol -> delisting_date
        logger.info("Survivorship Bias Guard initialized")

    def check_delisting(
        self,
        symbol: str,
        current_date: datetime,
        price_data: pd.DataFrame
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if stock is delisted at current_date.

        Args:
            symbol: Stock symbol
            current_date: Current backtest date
            price_data: Historical price data

        Returns:
            Tuple of (is_delisted, reason)
        """
        # Check if already marked as delisted
        if symbol in self.delisted_stocks:
            delisting_date = pd.to_datetime(self.delisted_stocks[symbol])
            if current_date >= delisting_date:
                return True, f"Stock delisted on {delisting_date.date()}"

        # Check for data gaps (potential delisting indicator)
        if price_data.empty:
            return False, None

        # Get last available date
        last_available_date = price_data.index[-1]

        # If current_date is > 30 days past last available data, likely delisted
        days_since_last_data = (current_date - last_available_date).days

        if days_since_last_data > 30:
            # Mark as delisted
            self.delisted_stocks[symbol] = last_available_date.isoformat()
            logger.warning(
                f"Survivorship Bias Guard: {symbol} likely delisted around {last_available_date.date()} "
                f"(no data for {days_since_last_data} days)"
            )
            return True, f"No data for {days_since_last_data} days (likely delisted)"

        # Check for zero volume (trading halted)
        recent_data = price_data[price_data.index >= current_date - timedelta(days=5)]
        if not recent_data.empty and (recent_data['Volume'] == 0).all():
            logger.warning(f"Survivorship Bias Guard: {symbol} has zero volume (trading halted)")
            return True, "Zero volume - trading halted"

        return False, None

    def get_delisted_stocks(self) -> Dict[str, str]:
        """Get all delisted stocks"""
        return self.delisted_stocks.copy()


class CorporateActionsHandler:
    """
    Handle corporate actions (stock splits, dividends).

    Critical for accurate backtesting - splits/dividends affect prices and positions.
    """

    def __init__(self):
        self.actions = []  # List of CorporateAction
        logger.info("Corporate Actions Handler initialized")

    def add_action(self, action: CorporateAction):
        """Add a corporate action"""
        self.actions.append(action)
        logger.debug(f"Added corporate action: {action.symbol} {action.action_type} on {action.date}")

    def get_actions_for_date(self, symbol: str, date: datetime) -> List[CorporateAction]:
        """Get corporate actions for symbol on date"""
        date_str = date.strftime('%Y-%m-%d')

        return [
            action for action in self.actions
            if action.symbol == symbol and action.date == date_str
        ]

    def adjust_position_for_split(
        self,
        symbol: str,
        quantity: float,
        split_ratio: float
    ) -> float:
        """
        Adjust position quantity for stock split.

        Args:
            symbol: Stock symbol
            quantity: Current quantity
            split_ratio: Split ratio (e.g., 2.0 for 2:1 split)

        Returns:
            Adjusted quantity
        """
        adjusted_quantity = quantity * split_ratio

        logger.info(
            f"Corporate Action: {symbol} {split_ratio}:1 split. "
            f"Position adjusted: {quantity} -> {adjusted_quantity} shares"
        )

        return adjusted_quantity

    def adjust_cash_for_dividend(
        self,
        symbol: str,
        quantity: float,
        dividend_amount: float,
        cash: float
    ) -> float:
        """
        Adjust cash for dividend payment.

        Args:
            symbol: Stock symbol
            quantity: Number of shares held
            dividend_amount: Dividend per share
            cash: Current cash

        Returns:
            Adjusted cash
        """
        dividend_payment = quantity * dividend_amount
        adjusted_cash = cash + dividend_payment

        logger.info(
            f"Corporate Action: {symbol} dividend ${dividend_amount}/share. "
            f"Payment: ${dividend_payment:.2f} ({quantity} shares)"
        )

        return adjusted_cash

    def detect_splits_from_data(
        self,
        symbol: str,
        price_data: pd.DataFrame
    ):
        """
        Auto-detect stock splits from price data.

        Large price gaps + volume spikes typically indicate splits.
        """
        if len(price_data) < 2:
            return

        # Calculate daily price change ratios
        price_ratios = price_data['Close'].pct_change()

        # Detect large drops (potential splits)
        # E.g., 2:1 split causes ~50% price drop
        split_threshold = -0.4  # 40% drop

        potential_splits = price_data[price_ratios < split_threshold]

        for date, row in potential_splits.iterrows():
            split_ratio = 1.0 / (1.0 + price_ratios[date])  # Estimate ratio

            # Round to common split ratios (2:1, 3:1, 4:1)
            if 1.8 <= split_ratio <= 2.2:
                split_ratio = 2.0
            elif 2.8 <= split_ratio <= 3.2:
                split_ratio = 3.0
            elif 3.8 <= split_ratio <= 4.2:
                split_ratio = 4.0
            else:
                continue  # Not a standard split

            action = CorporateAction(
                symbol=symbol,
                date=date.strftime('%Y-%m-%d'),
                action_type='split',
                ratio=split_ratio
            )

            self.add_action(action)


class BacktesterV2:
    """
    Backtest Harness v2 - Deterministic, institutional-grade backtesting.

    Features:
    - Survivorship Bias Guards
    - Corporate Actions Handling
    - Signal Contract validation
    - Fail-Fast on missing data
    - Reproducible results
    - Comprehensive reporting
    """

    def __init__(
        self,
        config: BacktestConfig,
        coordinator=None,
        signal_validator: Optional[SignalValidator] = None
    ):
        """
        Initialize Backtester v2.

        Args:
            config: Backtest configuration
            coordinator: SystemCoordinator for signal generation
            signal_validator: Signal validator (creates new if None)
        """
        self.config = config
        self.coordinator = coordinator
        self.signal_validator = signal_validator or SignalValidator()

        # Validate config
        is_valid, errors = config.validate()
        if not is_valid:
            error_msg = f"Invalid backtest config:\n" + "\n".join(f"  - {err}" for err in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize components
        self.survivorship_guard = SurvivorshipBiasGuard() if config.enable_survivorship_bias_guard else None
        self.corporate_actions = CorporateActionsHandler() if config.enable_corporate_actions else None

        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        # Portfolio state
        self.cash = config.initial_capital
        self.positions = {}  # symbol -> quantity
        self.equity_curve = [config.initial_capital]
        self.trades = []
        self.signals = []  # Store all signals
        self.daily_returns = []

        # Tracking
        self.failed_signals = []  # Signals that failed validation
        self.delisted_symbols = set()

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Backtester v2 initialized with config: {config.to_dict()}")

    def run(self) -> Dict:
        """
        Run backtest.

        Returns:
            Backtest results dictionary
        """
        logger.info(
            f"Starting backtest: {self.config.start_date} to {self.config.end_date}, "
            f"symbols: {', '.join(self.config.symbols)}"
        )

        # 1. Download historical data
        logger.info("Downloading historical data...")
        price_data = self._download_data()

        if not price_data:
            error_msg = "Failed to download price data"
            logger.error(error_msg)
            return {'error': error_msg}

        # 2. Detect corporate actions
        if self.corporate_actions:
            logger.info("Detecting corporate actions...")
            for symbol in self.config.symbols:
                if symbol in price_data:
                    self.corporate_actions.detect_splits_from_data(symbol, price_data[symbol])

        # 3. Get trading days
        all_dates = sorted(set(
            date for symbol_data in price_data.values()
            for date in symbol_data.index
        ))

        logger.info(f"Running backtest for {len(all_dates)} trading days")

        # 4. Simulate day by day
        for i, date in enumerate(all_dates):
            if i % 20 == 0:
                progress = (i / len(all_dates)) * 100
                logger.info(f"Progress: {progress:.1f}% - Day {i+1}/{len(all_dates)}: {date.date()}")

            self._simulate_day(date, price_data)

        # 5. Close all positions at end
        self._close_all_positions(all_dates[-1], price_data)

        # 6. Calculate metrics
        metrics = self._calculate_metrics()

        # 7. Save results
        self._save_results(metrics)

        logger.info("Backtest complete")

        return metrics

    def _download_data(self) -> Dict[str, pd.DataFrame]:
        """Download historical price data with error handling"""
        price_data = {}

        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)

        for symbol in self.config.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end)

                if df.empty:
                    msg = f"No data available for {symbol}"

                    if self.config.fail_fast_on_missing_data:
                        logger.error(msg)
                        raise ValueError(msg)
                    else:
                        logger.warning(msg)
                        continue

                price_data[symbol] = df
                logger.debug(f"Downloaded {len(df)} days of data for {symbol}")

            except Exception as e:
                msg = f"Error downloading {symbol}: {e}"

                if self.config.fail_fast_on_missing_data:
                    logger.error(msg)
                    raise
                else:
                    logger.warning(msg)

        return price_data

    def _simulate_day(
        self,
        date: datetime,
        price_data: Dict[str, pd.DataFrame]
    ):
        """Simulate one trading day"""

        # Get prices for this day
        current_prices = {}
        for symbol in self.config.symbols:
            # Skip if no data
            if symbol not in price_data or date not in price_data[symbol].index:
                continue

            # Survivorship Bias Guard
            if self.survivorship_guard:
                is_delisted, reason = self.survivorship_guard.check_delisting(
                    symbol, date, price_data[symbol]
                )

                if is_delisted:
                    # Force close position if delisted
                    if symbol in self.positions:
                        logger.warning(f"Force closing {symbol} position due to delisting: {reason}")
                        self._force_close_position(symbol, date, price_data[symbol])

                    self.delisted_symbols.add(symbol)
                    continue

            current_prices[symbol] = float(price_data[symbol].loc[date, 'Close'])

        if not current_prices:
            return

        # Handle corporate actions
        if self.corporate_actions:
            for symbol in list(self.positions.keys()):
                actions = self.corporate_actions.get_actions_for_date(symbol, date)

                for action in actions:
                    if action.action_type == 'split' and action.ratio:
                        self.positions[symbol] = self.corporate_actions.adjust_position_for_split(
                            symbol, self.positions[symbol], action.ratio
                        )

                    elif action.action_type == 'dividend' and action.amount:
                        self.cash = self.corporate_actions.adjust_cash_for_dividend(
                            symbol, self.positions[symbol], action.amount, self.cash
                        )

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

        # Trading decisions (every N days)
        if len(self.equity_curve) % 5 == 0:  # Trade every 5 days
            self._execute_strategy(date, current_prices)

    def _execute_strategy(
        self,
        date: datetime,
        current_prices: Dict[str, float]
    ):
        """Execute trading strategy with signal validation"""

        for symbol in current_prices.keys():
            # Skip delisted stocks
            if symbol in self.delisted_symbols:
                continue

            try:
                # Get analysis/signal from coordinator
                if not self.coordinator:
                    continue

                analysis = self.coordinator.analyze_symbol(symbol, use_supervisor=False)

                # Build signal (simplified - in production, coordinator returns full signal)
                signal = {
                    "analysis": analysis.get('analysis', {}),
                    "signal": analysis.get('recommendation', 'hold'),
                    "sizing": {
                        "position_size": analysis.get('position_size', 0.0),
                        "rationale": analysis.get('reasoning', 'N/A')[:100]
                    },
                    "risk": {
                        "stop_loss": analysis.get('stop_loss', 0.0),
                        "take_profit": analysis.get('take_profit', 0.0),
                        "max_drawdown": 0.05
                    },
                    "rationale": analysis.get('reasoning', 'N/A')[:500],
                    "evidence": {
                        "sources": [],
                        "confidence": analysis.get('confidence', 0.0)
                    },
                    "metadata": {
                        "symbol": symbol,
                        "timestamp": date.isoformat(),
                        "version": "1.0.0",
                        "current_price": current_prices[symbol]
                    }
                }

                # Validate signal
                if self.config.validate_signals:
                    try:
                        is_valid, errors = self.signal_validator.validate(signal, strict=False)

                        if not is_valid:
                            logger.warning(
                                f"Signal validation failed for {symbol}: {len(errors)} errors. Skipping."
                            )
                            self.failed_signals.append({
                                'date': date,
                                'symbol': symbol,
                                'signal': signal,
                                'errors': errors
                            })
                            continue

                    except Exception as e:
                        logger.error(f"Signal validation error for {symbol}: {e}")
                        continue

                # Store signal
                if self.config.save_signals:
                    self.signals.append({
                        'date': date.isoformat(),
                        'symbol': symbol,
                        'signal': signal
                    })

                # Execute based on decision
                decision = signal['signal']
                confidence = signal['evidence']['confidence']
                position_size = signal['sizing']['position_size']

                if decision == 'buy' and confidence > 0.6:
                    self._execute_buy(symbol, current_prices[symbol], position_size, date)

                elif decision == 'sell' and symbol in self.positions:
                    self._execute_sell(symbol, current_prices[symbol], date)

            except Exception as e:
                logger.error(f"Error executing strategy for {symbol} on {date.date()}: {e}")

    def _execute_buy(
        self,
        symbol: str,
        price: float,
        position_size: float,
        date: datetime
    ):
        """Execute buy order with commission and slippage"""

        # Apply slippage (price goes up when buying)
        slippage_factor = 1.0 + (self.config.slippage_bps / 10000.0)
        execution_price = price * slippage_factor

        # Calculate shares to buy
        max_investment = self.cash * min(position_size, 0.20)  # Max 20% per position
        shares = int(max_investment / execution_price)

        if shares > 0:
            cost = shares * execution_price
            commission = cost * self.config.commission_rate
            total_cost = cost + commission

            if total_cost <= self.cash:
                self.cash -= total_cost
                self.positions[symbol] = self.positions.get(symbol, 0) + shares

                self.trades.append({
                    'date': date.isoformat(),
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': shares,
                    'price': execution_price,
                    'cost': cost,
                    'commission': commission,
                    'slippage_bps': self.config.slippage_bps,
                    'pnl': 0
                })

                logger.debug(
                    f"BUY {shares} {symbol} @ ${execution_price:.2f} "
                    f"(slippage: {self.config.slippage_bps}bps, commission: ${commission:.2f})"
                )

    def _execute_sell(self, symbol: str, price: float, date: datetime):
        """Execute sell order with commission and slippage"""

        if symbol not in self.positions:
            return

        # Apply slippage (price goes down when selling)
        slippage_factor = 1.0 - (self.config.slippage_bps / 10000.0)
        execution_price = price * slippage_factor

        shares = self.positions[symbol]
        proceeds = shares * execution_price
        commission = proceeds * self.config.commission_rate
        net_proceeds = proceeds - commission

        # Calculate P&L
        buy_trades = [t for t in self.trades if t['symbol'] == symbol and t['action'] == 'buy']
        total_cost = sum(t['cost'] + t.get('commission', 0) for t in buy_trades)
        pnl = net_proceeds - total_cost

        self.cash += net_proceeds
        del self.positions[symbol]

        self.trades.append({
            'date': date.isoformat(),
            'symbol': symbol,
            'action': 'sell',
            'quantity': shares,
            'price': execution_price,
            'proceeds': proceeds,
            'commission': commission,
            'slippage_bps': self.config.slippage_bps,
            'pnl': pnl
        })

        logger.debug(
            f"SELL {shares} {symbol} @ ${execution_price:.2f} "
            f"(P&L: ${pnl:.2f}, commission: ${commission:.2f})"
        )

    def _force_close_position(
        self,
        symbol: str,
        date: datetime,
        price_data: pd.DataFrame
    ):
        """Force close position (e.g., due to delisting) at last available price"""

        if symbol not in self.positions:
            return

        # Get last available price
        last_price = float(price_data.loc[price_data.index <= date, 'Close'].iloc[-1])

        self._execute_sell(symbol, last_price, date)

        logger.warning(f"Force closed {symbol} position at ${last_price:.2f}")

    def _close_all_positions(
        self,
        date: datetime,
        price_data: Dict[str, pd.DataFrame]
    ):
        """Close all remaining positions at end of backtest"""

        for symbol in list(self.positions.keys()):
            if symbol in price_data and date in price_data[symbol].index:
                price = float(price_data[symbol].loc[date, 'Close'])
                self._execute_sell(symbol, price, date)

    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics"""

        if len(self.equity_curve) < 2:
            return {'error': 'Insufficient data'}

        # Use existing metrics calculator
        metrics = calculate_portfolio_metrics(
            returns=self.daily_returns,
            equity_curve=self.equity_curve,
            trades=self.trades
        )

        # Add backtest-specific info
        metrics.update({
            'config': self.config.to_dict(),
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'initial_capital': self.config.initial_capital,
            'final_value': self.equity_curve[-1],
            'total_return': (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0],
            'num_trades': len(self.trades),
            'num_signals': len(self.signals),
            'failed_signals': len(self.failed_signals),
            'delisted_symbols': list(self.delisted_symbols),
            'trading_days': len(self.equity_curve) - 1,
            'survivorship_bias_guarded': self.config.enable_survivorship_bias_guard,
            'corporate_actions_handled': self.config.enable_corporate_actions
        })

        return metrics

    def _save_results(self, metrics: Dict):
        """Save backtest results to files"""

        # Save metrics
        metrics_file = self.output_dir / f"backtest_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved metrics to {metrics_file}")

        # Save trades
        if self.config.save_trades and self.trades:
            trades_file = self.output_dir / f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(trades_file, 'w') as f:
                json.dump(self.trades, f, indent=2, default=str)
            logger.info(f"Saved {len(self.trades)} trades to {trades_file}")

        # Save signals
        if self.config.save_signals and self.signals:
            signals_file = self.output_dir / f"backtest_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(signals_file, 'w') as f:
                json.dump(self.signals, f, indent=2, default=str)
            logger.info(f"Saved {len(self.signals)} signals to {signals_file}")

        # Save failed signals
        if self.failed_signals:
            failed_file = self.output_dir / f"backtest_failed_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(failed_file, 'w') as f:
                json.dump(self.failed_signals, f, indent=2, default=str)
            logger.warning(f"Saved {len(self.failed_signals)} failed signals to {failed_file}")


if __name__ == "__main__":
    # Example usage

    config = BacktestConfig(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_capital=100000,
        commission_rate=0.001,
        slippage_bps=5.0,
        enable_survivorship_bias_guard=True,
        enable_corporate_actions=True,
        fail_fast_on_missing_data=False,
        validate_signals=False,  # Set to True when coordinator returns full signals
        random_seed=42
    )

    backtester = BacktesterV2(config)

    results = backtester.run()

    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60 + "\n")

    for key, value in results.items():
        if key in ['config', 'trades', 'signals']:
            continue  # Skip large objects

        if isinstance(value, float):
            if 'return' in key or 'drawdown' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
