/**
 * Main Dashboard Component
 *
 * Complete dashboard layout showcasing all visualization components:
 * - Portfolio analytics
 * - Real-time charts
 * - Technical analysis
 * - Market heatmaps
 * - Performance metrics
 */

import React, { useState } from 'react';
import {
  DashboardGrid,
  GridItem,
  Widget,
  MetricGrid,
  Tabs,
  StatusBadge,
  ProgressBar,
} from './DashboardGrid';
import { RealTimePriceDisplay, RealTimeCandlestickChart } from '../charts/RealTimeChart';
import { TechnicalChart } from '../charts/TechnicalChart';
import { PortfolioAnalytics } from '../analytics/PortfolioAnalytics';
import {
  SectorPerformanceHeatmap,
  CorrelationMatrix,
  MarketOverviewHeatmap,
} from '../charts/Heatmap';

export interface DashboardProps {
  className?: string;
}

export function Dashboard({ className = '' }: DashboardProps) {
  const [activeTab, setActiveTab] = useState('overview');

  // Mock data for demonstration
  const portfolioMetrics = [
    {
      label: 'Portfolio Value',
      value: '$127,543.21',
      change: 2.34,
      trend: 'up' as const,
    },
    {
      label: 'Total Return',
      value: '24.8%',
      change: 1.2,
      trend: 'up' as const,
    },
    {
      label: 'Day Change',
      value: '+$2,847.32',
      change: 2.34,
      trend: 'up' as const,
    },
    {
      label: 'Cash Balance',
      value: '$15,234.00',
      trend: 'neutral' as const,
    },
  ];

  const systemMetrics = [
    {
      label: 'Active Agents',
      value: 5,
      prefix: '',
      suffix: '/5',
      trend: 'up' as const,
    },
    {
      label: 'Analysis Today',
      value: 127,
      trend: 'up' as const,
    },
    {
      label: 'Win Rate',
      value: '68.5%',
      trend: 'up' as const,
    },
    {
      label: 'Avg Confidence',
      value: '84.2%',
      trend: 'up' as const,
    },
  ];

  const watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'];

  return (
    <div className={`min-h-screen bg-gray-900 p-6 ${className}`}>
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-white">Stock Agent Dashboard</h1>
            <p className="text-gray-400 mt-1">
              AI-Powered Trading System - Real-time Analytics
            </p>
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge status="success" text="All Systems Operational" />
            <StatusBadge status="info" text="5 Agents Active" />
          </div>
        </div>

        <Tabs
          tabs={[
            { id: 'overview', label: 'Overview', badge: 12 },
            { id: 'portfolio', label: 'Portfolio' },
            { id: 'analysis', label: 'Technical Analysis' },
            { id: 'market', label: 'Market' },
            { id: 'agents', label: 'AI Agents' },
          ]}
          activeTab={activeTab}
          onChange={setActiveTab}
        />
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Portfolio Metrics */}
          <MetricGrid metrics={portfolioMetrics} columns={4} />

          {/* System Metrics */}
          <Widget title="System Status" subtitle="AI Agent Performance">
            <MetricGrid metrics={systemMetrics} columns={4} />
          </Widget>

          {/* Main Grid */}
          <DashboardGrid columns={12} gap={6}>
            {/* Watchlist Prices */}
            <GridItem colSpan={12} className="lg:col-span-4">
              <Widget title="Watchlist" subtitle="Real-time prices">
                <div className="space-y-3">
                  {watchlist.map((symbol) => (
                    <RealTimePriceDisplay key={symbol} symbol={symbol} />
                  ))}
                </div>
              </Widget>
            </GridItem>

            {/* Main Chart */}
            <GridItem colSpan={12} className="lg:col-span-8">
              <Widget title="AAPL - Apple Inc." padding={false}>
                <RealTimeCandlestickChart
                  symbol="AAPL"
                  height={400}
                  showVolume={true}
                  showMA={true}
                />
              </Widget>
            </GridItem>
          </DashboardGrid>

          {/* Market Overview Heatmap */}
          <Widget
            title="Market Overview"
            subtitle="Performance heatmap of major stocks"
          >
            <MarketOverviewHeatmap
              stocks={[
                { symbol: 'AAPL', change: 2.3, marketCap: 3000000000000 },
                { symbol: 'MSFT', change: 1.8, marketCap: 2800000000000 },
                { symbol: 'GOOGL', change: -0.5, marketCap: 1900000000000 },
                { symbol: 'AMZN', change: 3.2, marketCap: 1700000000000 },
                { symbol: 'TSLA', change: -1.2, marketCap: 800000000000 },
                { symbol: 'META', change: 0.8, marketCap: 900000000000 },
                { symbol: 'NVDA', change: 4.5, marketCap: 2100000000000 },
                { symbol: 'BRK.B', change: 0.3, marketCap: 800000000000 },
                { symbol: 'JPM', change: -0.8, marketCap: 500000000000 },
              ]}
            />
          </Widget>

          {/* Agent Performance */}
          <Widget title="AI Agent Performance" subtitle="Individual agent metrics">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-300">News Agent</span>
                  <span className="text-green-400">92% Accuracy</span>
                </div>
                <ProgressBar value={92} color="green" />
              </div>

              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-300">Technical Agent</span>
                  <span className="text-green-400">88% Accuracy</span>
                </div>
                <ProgressBar value={88} color="green" />
              </div>

              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-300">Fundamental Agent</span>
                  <span className="text-blue-400">85% Accuracy</span>
                </div>
                <ProgressBar value={85} color="blue" />
              </div>

              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-300">Sentiment Agent</span>
                  <span className="text-blue-400">79% Accuracy</span>
                </div>
                <ProgressBar value={79} color="blue" />
              </div>

              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-300">Strategist Agent</span>
                  <span className="text-yellow-400">73% Accuracy</span>
                </div>
                <ProgressBar value={73} color="yellow" />
              </div>
            </div>
          </Widget>
        </div>
      )}

      {/* Portfolio Tab */}
      {activeTab === 'portfolio' && (
        <div className="space-y-6">
          <PortfolioAnalytics
            positions={[
              {
                symbol: 'AAPL',
                shares: 100,
                avgCost: 150,
                currentPrice: 185,
                sector: 'Technology',
                assetClass: 'Stocks',
              },
              {
                symbol: 'MSFT',
                shares: 50,
                avgCost: 300,
                currentPrice: 375,
                sector: 'Technology',
                assetClass: 'Stocks',
              },
              {
                symbol: 'GOOGL',
                shares: 75,
                avgCost: 120,
                currentPrice: 140,
                sector: 'Technology',
                assetClass: 'Stocks',
              },
            ]}
            performance={[
              {
                timestamp: '2024-01-01',
                value: 100000,
                returns: 0,
                benchmark: 0,
              },
              {
                timestamp: '2024-02-01',
                value: 105000,
                returns: 5,
                benchmark: 3,
              },
              {
                timestamp: '2024-03-01',
                value: 112000,
                returns: 12,
                benchmark: 8,
              },
            ]}
          />
        </div>
      )}

      {/* Technical Analysis Tab */}
      {activeTab === 'analysis' && (
        <div className="space-y-6">
          <Widget title="Technical Analysis" subtitle="AAPL with indicators">
            <TechnicalChart
              data={[
                {
                  timestamp: '2024-01-01T00:00:00Z',
                  open: 180,
                  high: 185,
                  low: 178,
                  close: 183,
                  volume: 50000000,
                },
                {
                  timestamp: '2024-01-02T00:00:00Z',
                  open: 183,
                  high: 188,
                  low: 182,
                  close: 186,
                  volume: 55000000,
                },
                // Add more data points...
              ]}
              indicators={{
                sma: [20, 50],
                ema: [12, 26],
                bollinger: true,
                rsi: true,
                macd: true,
                volume: true,
              }}
              height={700}
            />
          </Widget>
        </div>
      )}

      {/* Market Tab */}
      {activeTab === 'market' && (
        <div className="space-y-6">
          <DashboardGrid columns={2} gap={6}>
            <GridItem colSpan={1}>
              <Widget
                title="Sector Performance"
                subtitle="7-day performance by sector"
              >
                <SectorPerformanceHeatmap
                  sectors={[
                    'Technology',
                    'Healthcare',
                    'Finance',
                    'Energy',
                    'Consumer',
                  ]}
                  timeframes={['1D', '1W', '1M', '3M']}
                  performance={[
                    [2.3, 5.1, 12.4, 18.7],
                    [0.8, 2.1, 8.3, 15.2],
                    [-0.5, 1.2, 6.8, 11.4],
                    [3.2, 6.7, 15.2, 22.8],
                    [1.1, 3.4, 9.1, 14.6],
                  ]}
                />
              </Widget>
            </GridItem>

            <GridItem colSpan={1}>
              <Widget
                title="Correlation Matrix"
                subtitle="Asset correlation analysis"
              >
                <CorrelationMatrix
                  symbols={['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']}
                  correlations={[
                    [1.0, 0.85, 0.72, 0.68, 0.45],
                    [0.85, 1.0, 0.79, 0.71, 0.52],
                    [0.72, 0.79, 1.0, 0.76, 0.48],
                    [0.68, 0.71, 0.76, 1.0, 0.51],
                    [0.45, 0.52, 0.48, 0.51, 1.0],
                  ]}
                />
              </Widget>
            </GridItem>
          </DashboardGrid>
        </div>
      )}

      {/* Agents Tab */}
      {activeTab === 'agents' && (
        <div className="space-y-6">
          <Widget title="AI Agent Status" subtitle="Multi-agent system overview">
            <div className="space-y-4">
              {[
                { name: 'News Agent', status: 'active', uptime: 99.8 },
                { name: 'Technical Agent', status: 'active', uptime: 99.5 },
                { name: 'Fundamental Agent', status: 'active', uptime: 98.9 },
                { name: 'Sentiment Agent', status: 'active', uptime: 99.2 },
                { name: 'Strategist Agent', status: 'active', uptime: 99.7 },
              ].map((agent, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-4 bg-gray-750 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                    <div>
                      <p className="text-white font-semibold">{agent.name}</p>
                      <p className="text-gray-400 text-sm">
                        Uptime: {agent.uptime}%
                      </p>
                    </div>
                  </div>
                  <StatusBadge status="success" text={agent.status} size="sm" />
                </div>
              ))}
            </div>
          </Widget>
        </div>
      )}
    </div>
  );
}
