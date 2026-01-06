/**
 * Portfolio Analytics Component
 *
 * Comprehensive portfolio performance dashboard with:
 * - Performance metrics (returns, Sharpe ratio, volatility)
 * - Asset allocation visualization
 * - Sector breakdown
 * - Top holdings
 * - Risk analysis
 */

import React, { useMemo } from 'react';
import { AreaChart } from '../charts/AreaChart';
import { LineChart, PerformanceChart } from '../charts/LineChart';

export interface PortfolioPosition {
  symbol: string;
  shares: number;
  avgCost: number;
  currentPrice: number;
  sector: string;
  assetClass: string;
}

export interface PortfolioPerformancePoint {
  timestamp: string;
  value: number;
  returns: number;
  benchmark: number;
}

export interface PortfolioAnalyticsProps {
  positions: PortfolioPosition[];
  performance: PortfolioPerformancePoint[];
  benchmarkName?: string;
  className?: string;
}

interface MetricCardProps {
  label: string;
  value: string | number;
  change?: number;
  prefix?: string;
  suffix?: string;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

const MetricCard = ({
  label,
  value,
  change,
  prefix = '',
  suffix = '',
  trend,
  className = '',
}: MetricCardProps) => {
  const changeColor =
    trend === 'up'
      ? 'text-green-400'
      : trend === 'down'
      ? 'text-red-400'
      : 'text-gray-400';

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <p className="text-gray-400 text-sm mb-2">{label}</p>
      <p className="text-white text-2xl font-bold">
        {prefix}
        {typeof value === 'number' ? value.toLocaleString() : value}
        {suffix}
      </p>
      {change !== undefined && (
        <p className={`text-sm mt-1 ${changeColor}`}>
          {change >= 0 ? '+' : ''}
          {change.toFixed(2)}%
        </p>
      )}
    </div>
  );
};

export function PortfolioAnalytics({
  positions,
  performance,
  benchmarkName = 'S&P 500',
  className = '',
}: PortfolioAnalyticsProps) {
  // Calculate portfolio metrics
  const metrics = useMemo(() => {
    if (!positions || positions.length === 0) {
      return {
        totalValue: 0,
        totalCost: 0,
        totalGainLoss: 0,
        totalReturn: 0,
        dayChange: 0,
        dayChangePercent: 0,
      };
    }

    const totalValue = positions.reduce(
      (sum, pos) => sum + pos.shares * pos.currentPrice,
      0
    );

    const totalCost = positions.reduce(
      (sum, pos) => sum + pos.shares * pos.avgCost,
      0
    );

    const totalGainLoss = totalValue - totalCost;
    const totalReturn = totalCost > 0 ? (totalGainLoss / totalCost) * 100 : 0;

    // Calculate day change from performance data
    const dayChange =
      performance && performance.length >= 2
        ? performance[performance.length - 1].value -
          performance[performance.length - 2].value
        : 0;

    const dayChangePercent =
      performance && performance.length >= 2
        ? ((dayChange / performance[performance.length - 2].value) * 100)
        : 0;

    return {
      totalValue,
      totalCost,
      totalGainLoss,
      totalReturn,
      dayChange,
      dayChangePercent,
    };
  }, [positions, performance]);

  // Calculate asset allocation
  const assetAllocation = useMemo(() => {
    if (!positions || positions.length === 0) return [];

    const totalValue = positions.reduce(
      (sum, pos) => sum + pos.shares * pos.currentPrice,
      0
    );

    const allocation = positions.reduce((acc, pos) => {
      const value = pos.shares * pos.currentPrice;
      const percentage = totalValue > 0 ? (value / totalValue) * 100 : 0;

      if (!acc[pos.assetClass]) {
        acc[pos.assetClass] = { value: 0, percentage: 0 };
      }

      acc[pos.assetClass].value += value;
      acc[pos.assetClass].percentage += percentage;

      return acc;
    }, {} as Record<string, { value: number; percentage: number }>);

    return Object.entries(allocation)
      .map(([assetClass, data]) => ({
        assetClass,
        ...data,
      }))
      .sort((a, b) => b.value - a.value);
  }, [positions]);

  // Calculate sector breakdown
  const sectorBreakdown = useMemo(() => {
    if (!positions || positions.length === 0) return [];

    const totalValue = positions.reduce(
      (sum, pos) => sum + pos.shares * pos.currentPrice,
      0
    );

    const sectors = positions.reduce((acc, pos) => {
      const value = pos.shares * pos.currentPrice;
      const percentage = totalValue > 0 ? (value / totalValue) * 100 : 0;

      if (!acc[pos.sector]) {
        acc[pos.sector] = { value: 0, percentage: 0 };
      }

      acc[pos.sector].value += value;
      acc[pos.sector].percentage += percentage;

      return acc;
    }, {} as Record<string, { value: number; percentage: number }>);

    return Object.entries(sectors)
      .map(([sector, data]) => ({
        sector,
        ...data,
      }))
      .sort((a, b) => b.value - a.value);
  }, [positions]);

  // Top holdings
  const topHoldings = useMemo(() => {
    if (!positions || positions.length === 0) return [];

    const totalValue = positions.reduce(
      (sum, pos) => sum + pos.shares * pos.currentPrice,
      0
    );

    return positions
      .map((pos) => {
        const value = pos.shares * pos.currentPrice;
        const cost = pos.shares * pos.avgCost;
        const gainLoss = value - cost;
        const returnPercent = cost > 0 ? (gainLoss / cost) * 100 : 0;
        const allocation = totalValue > 0 ? (value / totalValue) * 100 : 0;

        return {
          symbol: pos.symbol,
          shares: pos.shares,
          value,
          cost,
          gainLoss,
          returnPercent,
          allocation,
          currentPrice: pos.currentPrice,
        };
      })
      .sort((a, b) => b.value - a.value)
      .slice(0, 10);
  }, [positions]);

  // Risk metrics
  const riskMetrics = useMemo(() => {
    if (!performance || performance.length < 2) {
      return {
        volatility: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        beta: 0,
      };
    }

    // Calculate daily returns
    const returns = performance.slice(1).map((point, i) => {
      const prevValue = performance[i].value;
      return ((point.value - prevValue) / prevValue) * 100;
    });

    // Volatility (annualized standard deviation)
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance =
      returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance) * Math.sqrt(252); // Annualized

    // Sharpe Ratio (assuming 2% risk-free rate)
    const riskFreeRate = 2.0;
    const annualizedReturn = mean * 252;
    const sharpeRatio =
      volatility > 0 ? (annualizedReturn - riskFreeRate) / volatility : 0;

    // Max Drawdown
    let peak = performance[0].value;
    let maxDrawdown = 0;

    performance.forEach((point) => {
      if (point.value > peak) {
        peak = point.value;
      }
      const drawdown = ((point.value - peak) / peak) * 100;
      if (drawdown < maxDrawdown) {
        maxDrawdown = drawdown;
      }
    });

    // Beta (correlation with benchmark)
    const benchmarkReturns = performance.slice(1).map((point, i) => {
      const prevBenchmark = performance[i].benchmark;
      return ((point.benchmark - prevBenchmark) / prevBenchmark) * 100;
    });

    const portfolioMean = mean;
    const benchmarkMean =
      benchmarkReturns.reduce((sum, r) => sum + r, 0) / benchmarkReturns.length;

    const covariance =
      returns.reduce((sum, r, i) => {
        return sum + (r - portfolioMean) * (benchmarkReturns[i] - benchmarkMean);
      }, 0) / returns.length;

    const benchmarkVariance =
      benchmarkReturns.reduce((sum, r) => sum + Math.pow(r - benchmarkMean, 2), 0) /
      benchmarkReturns.length;

    const beta = benchmarkVariance > 0 ? covariance / benchmarkVariance : 0;

    return {
      volatility,
      sharpeRatio,
      maxDrawdown,
      beta,
    };
  }, [performance]);

  if (!positions || positions.length === 0) {
    return (
      <div className={`bg-gray-900 rounded-lg p-8 text-center ${className}`}>
        <p className="text-gray-400 text-lg">No portfolio data available</p>
        <p className="text-gray-500 text-sm mt-2">
          Add positions to see analytics
        </p>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Summary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Portfolio Value"
          value={metrics.totalValue.toFixed(2)}
          prefix="$"
          change={metrics.dayChangePercent}
          trend={metrics.dayChangePercent >= 0 ? 'up' : 'down'}
        />
        <MetricCard
          label="Total Return"
          value={metrics.totalReturn.toFixed(2)}
          suffix="%"
          trend={metrics.totalReturn >= 0 ? 'up' : 'down'}
        />
        <MetricCard
          label="Gain/Loss"
          value={metrics.totalGainLoss.toFixed(2)}
          prefix={metrics.totalGainLoss >= 0 ? '+$' : '-$'}
          trend={metrics.totalGainLoss >= 0 ? 'up' : 'down'}
        />
        <MetricCard
          label="Day Change"
          value={metrics.dayChange.toFixed(2)}
          prefix={metrics.dayChange >= 0 ? '+$' : '-$'}
          trend={metrics.dayChange >= 0 ? 'up' : 'down'}
        />
      </div>

      {/* Risk Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Volatility"
          value={riskMetrics.volatility.toFixed(2)}
          suffix="%"
          trend="neutral"
        />
        <MetricCard
          label="Sharpe Ratio"
          value={riskMetrics.sharpeRatio.toFixed(2)}
          trend={riskMetrics.sharpeRatio >= 1 ? 'up' : 'neutral'}
        />
        <MetricCard
          label="Max Drawdown"
          value={riskMetrics.maxDrawdown.toFixed(2)}
          suffix="%"
          trend="down"
        />
        <MetricCard
          label="Beta"
          value={riskMetrics.beta.toFixed(2)}
          trend="neutral"
        />
      </div>

      {/* Performance Chart */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-white text-lg font-semibold mb-4">
          Performance vs {benchmarkName}
        </h3>
        {performance && performance.length > 0 && (
          <PerformanceChart data={performance} height={300} />
        )}
      </div>

      {/* Asset Allocation & Sector Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Asset Allocation */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-white text-lg font-semibold mb-4">
            Asset Allocation
          </h3>
          <div className="space-y-3">
            {assetAllocation.map((asset, index) => (
              <div key={index}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-300">{asset.assetClass}</span>
                  <span className="text-white font-mono">
                    {asset.percentage.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{ width: `${asset.percentage}%` }}
                  />
                </div>
                <p className="text-gray-400 text-xs mt-1">
                  ${asset.value.toLocaleString()}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Sector Breakdown */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-white text-lg font-semibold mb-4">
            Sector Breakdown
          </h3>
          <div className="space-y-3">
            {sectorBreakdown.map((sector, index) => {
              const colors = [
                'bg-blue-500',
                'bg-green-500',
                'bg-yellow-500',
                'bg-red-500',
                'bg-purple-500',
                'bg-pink-500',
                'bg-indigo-500',
                'bg-cyan-500',
              ];
              const color = colors[index % colors.length];

              return (
                <div key={index}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-300">{sector.sector}</span>
                    <span className="text-white font-mono">
                      {sector.percentage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                      className={`${color} h-2 rounded-full transition-all`}
                      style={{ width: `${sector.percentage}%` }}
                    />
                  </div>
                  <p className="text-gray-400 text-xs mt-1">
                    ${sector.value.toLocaleString()}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Top Holdings */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-white text-lg font-semibold mb-4">Top Holdings</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                <th className="pb-2">Symbol</th>
                <th className="pb-2 text-right">Shares</th>
                <th className="pb-2 text-right">Price</th>
                <th className="pb-2 text-right">Value</th>
                <th className="pb-2 text-right">Allocation</th>
                <th className="pb-2 text-right">Gain/Loss</th>
                <th className="pb-2 text-right">Return</th>
              </tr>
            </thead>
            <tbody>
              {topHoldings.map((holding, index) => (
                <tr
                  key={index}
                  className="text-sm border-b border-gray-700 hover:bg-gray-750"
                >
                  <td className="py-3 text-white font-semibold">
                    {holding.symbol}
                  </td>
                  <td className="py-3 text-right text-gray-300 font-mono">
                    {holding.shares.toLocaleString()}
                  </td>
                  <td className="py-3 text-right text-gray-300 font-mono">
                    ${holding.currentPrice.toFixed(2)}
                  </td>
                  <td className="py-3 text-right text-white font-mono">
                    ${holding.value.toLocaleString()}
                  </td>
                  <td className="py-3 text-right text-gray-300 font-mono">
                    {holding.allocation.toFixed(1)}%
                  </td>
                  <td
                    className={`py-3 text-right font-mono ${
                      holding.gainLoss >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}
                  >
                    {holding.gainLoss >= 0 ? '+' : ''}$
                    {holding.gainLoss.toFixed(2)}
                  </td>
                  <td
                    className={`py-3 text-right font-mono ${
                      holding.returnPercent >= 0
                        ? 'text-green-400'
                        : 'text-red-400'
                    }`}
                  >
                    {holding.returnPercent >= 0 ? '+' : ''}
                    {holding.returnPercent.toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
