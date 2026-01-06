/**
 * Chart Components - Barrel Export
 *
 * Centralized export for all chart and visualization components.
 */

// Basic Charts
export { CandlestickChart } from './CandlestickChart';
export type { CandlestickData, CandlestickChartProps } from './CandlestickChart';

export { LineChart, PriceComparisonChart, PerformanceChart, MetricsChart } from './LineChart';
export type { LineDataPoint, LineSeries, LineChartProps } from './LineChart';

export { AreaChart, PortfolioAllocationChart, CumulativeReturnsChart, SectorAllocationChart, VolumeProfileChart } from './AreaChart';
export type { AreaDataPoint, AreaSeries, AreaChartProps } from './AreaChart';

// Heatmaps
export { Heatmap, CorrelationMatrix, SectorPerformanceHeatmap, MarketOverviewHeatmap, RiskHeatmap } from './Heatmap';
export type { HeatmapCell, HeatmapProps } from './Heatmap';

// Real-Time Charts
export { RealTimeChart, RealTimeCandlestickChart, RealTimeLineChart, RealTimePriceDisplay } from './RealTimeChart';
export type { RealTimeChartProps } from './RealTimeChart';

// Technical Analysis
export { TechnicalChart } from './TechnicalChart';
export type { TechnicalChartProps } from './TechnicalChart';
