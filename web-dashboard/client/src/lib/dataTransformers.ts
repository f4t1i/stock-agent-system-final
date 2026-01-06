/**
 * Data Transformation Utilities
 *
 * Helper functions for transforming data between different formats:
 * - API responses to chart data
 * - Data aggregation and grouping
 * - Time series transformations
 * - Statistical calculations
 */

import { CandlestickData } from '../components/charts/CandlestickChart';

/**
 * Transform raw price data to candlestick format
 */
export interface RawPriceData {
  timestamp: string | Date;
  open?: number;
  high?: number;
  low?: number;
  close: number;
  volume?: number;
}

export function toCandlestickData(data: RawPriceData[]): CandlestickData[] {
  return data.map((item) => ({
    timestamp: typeof item.timestamp === 'string' ? item.timestamp : item.timestamp.toISOString(),
    open: item.open ?? item.close,
    high: item.high ?? item.close,
    low: item.low ?? item.close,
    close: item.close,
    volume: item.volume ?? 0,
  }));
}

/**
 * Aggregate data by time interval
 */
export type TimeInterval = '1min' | '5min' | '15min' | '1hour' | '1day' | '1week' | '1month';

export function aggregateByInterval(
  data: CandlestickData[],
  interval: TimeInterval
): CandlestickData[] {
  if (data.length === 0) return [];

  const intervalMs = {
    '1min': 60 * 1000,
    '5min': 5 * 60 * 1000,
    '15min': 15 * 60 * 1000,
    '1hour': 60 * 60 * 1000,
    '1day': 24 * 60 * 60 * 1000,
    '1week': 7 * 24 * 60 * 60 * 1000,
    '1month': 30 * 24 * 60 * 60 * 1000,
  }[interval];

  const aggregated: CandlestickData[] = [];
  let currentBucket: CandlestickData[] = [];
  let bucketStart = new Date(data[0].timestamp).getTime();

  data.forEach((candle) => {
    const timestamp = new Date(candle.timestamp).getTime();

    if (timestamp < bucketStart + intervalMs) {
      currentBucket.push(candle);
    } else {
      // Aggregate current bucket
      if (currentBucket.length > 0) {
        aggregated.push(aggregateBucket(currentBucket, bucketStart));
      }

      // Start new bucket
      bucketStart = Math.floor(timestamp / intervalMs) * intervalMs;
      currentBucket = [candle];
    }
  });

  // Aggregate remaining bucket
  if (currentBucket.length > 0) {
    aggregated.push(aggregateBucket(currentBucket, bucketStart));
  }

  return aggregated;
}

function aggregateBucket(candles: CandlestickData[], timestamp: number): CandlestickData {
  return {
    timestamp: new Date(timestamp).toISOString(),
    open: candles[0].open,
    high: Math.max(...candles.map((c) => c.high)),
    low: Math.min(...candles.map((c) => c.low)),
    close: candles[candles.length - 1].close,
    volume: candles.reduce((sum, c) => sum + c.volume, 0),
  };
}

/**
 * Calculate returns (percentage change)
 */
export interface ReturnData {
  timestamp: string;
  value: number;
  returns: number;
  cumulativeReturns: number;
}

export function calculateReturns(
  data: Array<{ timestamp: string; value: number }>
): ReturnData[] {
  if (data.length === 0) return [];

  const results: ReturnData[] = [];
  let cumulativeReturns = 0;

  data.forEach((point, index) => {
    if (index === 0) {
      results.push({
        timestamp: point.timestamp,
        value: point.value,
        returns: 0,
        cumulativeReturns: 0,
      });
    } else {
      const prevValue = data[index - 1].value;
      const returns = ((point.value - prevValue) / prevValue) * 100;
      cumulativeReturns += returns;

      results.push({
        timestamp: point.timestamp,
        value: point.value,
        returns,
        cumulativeReturns,
      });
    }
  });

  return results;
}

/**
 * Normalize data to percentage scale (0-100)
 */
export function normalizeToPercentage(
  data: Array<{ timestamp: string; value: number }>,
  baseValue?: number
): Array<{ timestamp: string; value: number; normalizedValue: number }> {
  if (data.length === 0) return [];

  const base = baseValue ?? data[0].value;

  return data.map((point) => ({
    ...point,
    normalizedValue: ((point.value - base) / base) * 100,
  }));
}

/**
 * Fill missing time periods with interpolated values
 */
export function fillMissingPeriods(
  data: Array<{ timestamp: string; value: number }>,
  interval: TimeInterval
): Array<{ timestamp: string; value: number; interpolated: boolean }> {
  if (data.length < 2) return data.map((d) => ({ ...d, interpolated: false }));

  const intervalMs = {
    '1min': 60 * 1000,
    '5min': 5 * 60 * 1000,
    '15min': 15 * 60 * 1000,
    '1hour': 60 * 60 * 1000,
    '1day': 24 * 60 * 60 * 1000,
    '1week': 7 * 24 * 60 * 60 * 1000,
    '1month': 30 * 24 * 60 * 60 * 1000,
  }[interval];

  const filled: Array<{ timestamp: string; value: number; interpolated: boolean }> = [];

  for (let i = 0; i < data.length - 1; i++) {
    const current = data[i];
    const next = data[i + 1];

    filled.push({ ...current, interpolated: false });

    const currentTime = new Date(current.timestamp).getTime();
    const nextTime = new Date(next.timestamp).getTime();
    const gap = nextTime - currentTime;

    // If gap is larger than interval, fill with interpolated values
    if (gap > intervalMs * 1.5) {
      const steps = Math.floor(gap / intervalMs);

      for (let j = 1; j < steps; j++) {
        const interpolatedTime = currentTime + j * intervalMs;
        const ratio = j / steps;
        const interpolatedValue = current.value + (next.value - current.value) * ratio;

        filled.push({
          timestamp: new Date(interpolatedTime).toISOString(),
          value: interpolatedValue,
          interpolated: true,
        });
      }
    }
  }

  // Add last point
  filled.push({ ...data[data.length - 1], interpolated: false });

  return filled;
}

/**
 * Calculate moving statistics
 */
export interface MovingStats {
  timestamp: string;
  mean: number;
  median: number;
  stdDev: number;
  min: number;
  max: number;
}

export function calculateMovingStats(
  data: Array<{ timestamp: string; value: number }>,
  window: number
): MovingStats[] {
  const results: MovingStats[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < window - 1) {
      results.push({
        timestamp: data[i].timestamp,
        mean: NaN,
        median: NaN,
        stdDev: NaN,
        min: NaN,
        max: NaN,
      });
      continue;
    }

    const windowData = data.slice(i - window + 1, i + 1).map((d) => d.value);

    // Mean
    const mean = windowData.reduce((sum, val) => sum + val, 0) / window;

    // Median
    const sorted = [...windowData].sort((a, b) => a - b);
    const median =
      window % 2 === 0
        ? (sorted[window / 2 - 1] + sorted[window / 2]) / 2
        : sorted[Math.floor(window / 2)];

    // Standard Deviation
    const variance =
      windowData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / window;
    const stdDev = Math.sqrt(variance);

    // Min/Max
    const min = Math.min(...windowData);
    const max = Math.max(...windowData);

    results.push({
      timestamp: data[i].timestamp,
      mean,
      median,
      stdDev,
      min,
      max,
    });
  }

  return results;
}

/**
 * Group data by category
 */
export function groupByCategory<T extends Record<string, any>>(
  data: T[],
  categoryKey: keyof T
): Map<string, T[]> {
  const grouped = new Map<string, T[]>();

  data.forEach((item) => {
    const category = String(item[categoryKey]);

    if (!grouped.has(category)) {
      grouped.set(category, []);
    }

    grouped.get(category)!.push(item);
  });

  return grouped;
}

/**
 * Calculate correlation between two series
 */
export function calculateCorrelation(series1: number[], series2: number[]): number {
  if (series1.length !== series2.length || series1.length === 0) {
    return NaN;
  }

  const n = series1.length;

  const mean1 = series1.reduce((sum, val) => sum + val, 0) / n;
  const mean2 = series2.reduce((sum, val) => sum + val, 0) / n;

  let numerator = 0;
  let sum1Sq = 0;
  let sum2Sq = 0;

  for (let i = 0; i < n; i++) {
    const diff1 = series1[i] - mean1;
    const diff2 = series2[i] - mean2;

    numerator += diff1 * diff2;
    sum1Sq += diff1 * diff1;
    sum2Sq += diff2 * diff2;
  }

  const denominator = Math.sqrt(sum1Sq * sum2Sq);

  return denominator === 0 ? NaN : numerator / denominator;
}

/**
 * Calculate correlation matrix
 */
export function calculateCorrelationMatrix(
  data: Map<string, number[]>
): { symbols: string[]; matrix: number[][] } {
  const symbols = Array.from(data.keys());
  const matrix: number[][] = [];

  symbols.forEach((symbol1) => {
    const row: number[] = [];
    const series1 = data.get(symbol1)!;

    symbols.forEach((symbol2) => {
      const series2 = data.get(symbol2)!;
      row.push(calculateCorrelation(series1, series2));
    });

    matrix.push(row);
  });

  return { symbols, matrix };
}

/**
 * Resample data to target number of points
 */
export function resampleData<T extends { timestamp: string }>(
  data: T[],
  targetPoints: number
): T[] {
  if (data.length <= targetPoints) return data;

  const step = data.length / targetPoints;
  const resampled: T[] = [];

  for (let i = 0; i < targetPoints; i++) {
    const index = Math.floor(i * step);
    resampled.push(data[index]);
  }

  return resampled;
}

/**
 * Filter data by date range
 */
export function filterByDateRange<T extends { timestamp: string }>(
  data: T[],
  startDate: Date,
  endDate: Date
): T[] {
  const startTime = startDate.getTime();
  const endTime = endDate.getTime();

  return data.filter((item) => {
    const time = new Date(item.timestamp).getTime();
    return time >= startTime && time <= endTime;
  });
}

/**
 * Calculate percentage breakdown
 */
export interface PercentageBreakdown {
  category: string;
  value: number;
  percentage: number;
}

export function calculatePercentageBreakdown<T extends Record<string, any>>(
  data: T[],
  categoryKey: keyof T,
  valueKey: keyof T
): PercentageBreakdown[] {
  const grouped = groupByCategory(data, categoryKey);
  const breakdown: PercentageBreakdown[] = [];

  let total = 0;
  grouped.forEach((items) => {
    const sum = items.reduce((acc, item) => acc + Number(item[valueKey]), 0);
    total += sum;
  });

  grouped.forEach((items, category) => {
    const value = items.reduce((acc, item) => acc + Number(item[valueKey]), 0);
    const percentage = total > 0 ? (value / total) * 100 : 0;

    breakdown.push({ category, value, percentage });
  });

  return breakdown.sort((a, b) => b.value - a.value);
}

/**
 * Calculate drawdown series
 */
export interface DrawdownData {
  timestamp: string;
  value: number;
  peak: number;
  drawdown: number;
  drawdownPercent: number;
}

export function calculateDrawdown(
  data: Array<{ timestamp: string; value: number }>
): DrawdownData[] {
  const results: DrawdownData[] = [];
  let peak = -Infinity;

  data.forEach((point) => {
    if (point.value > peak) {
      peak = point.value;
    }

    const drawdown = point.value - peak;
    const drawdownPercent = peak > 0 ? (drawdown / peak) * 100 : 0;

    results.push({
      timestamp: point.timestamp,
      value: point.value,
      peak,
      drawdown,
      drawdownPercent,
    });
  });

  return results;
}

/**
 * Rolling window transformation
 */
export function rollingWindow<T, R>(
  data: T[],
  windowSize: number,
  transform: (window: T[]) => R
): (R | null)[] {
  const results: (R | null)[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < windowSize - 1) {
      results.push(null);
    } else {
      const window = data.slice(i - windowSize + 1, i + 1);
      results.push(transform(window));
    }
  }

  return results;
}
