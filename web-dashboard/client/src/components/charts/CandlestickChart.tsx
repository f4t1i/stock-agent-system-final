/**
 * Candlestick Chart Component
 *
 * Interactive candlestick chart with zoom, pan, and technical indicators.
 * Uses Recharts for rendering with custom candlestick shapes.
 */

import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Line,
  Bar,
  Cell,
  Legend,
} from 'recharts';
import { format } from 'date-fns';

export interface CandlestickData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface CandlestickChartProps {
  data: CandlestickData[];
  showVolume?: boolean;
  showMA?: boolean;
  maPeriods?: number[];
  height?: number;
  className?: string;
}

interface ProcessedCandle extends CandlestickData {
  date: Date;
  color: string;
  bodyHeight: number;
  bodyY: number;
  wickTop: number;
  wickBottom: number;
  ma20?: number;
  ma50?: number;
}

// Custom Candlestick Shape
const CandlestickShape = (props: any) => {
  const { x, y, width, height, payload } = props;

  if (!payload) return null;

  const candleWidth = Math.max(width * 0.6, 2);
  const wickWidth = Math.max(width * 0.1, 1);
  const xCenter = x + width / 2;

  const isGreen = payload.close >= payload.open;
  const color = isGreen ? '#10b981' : '#ef4444';

  const high = payload.high;
  const low = payload.low;
  const open = payload.open;
  const close = payload.close;

  // Calculate y positions (higher price = lower y coordinate)
  const scale = height / (high - low);
  const yHigh = y;
  const yLow = y + height;
  const yOpen = y + (high - open) * scale;
  const yClose = y + (high - close) * scale;

  const bodyTop = Math.min(yOpen, yClose);
  const bodyHeight = Math.abs(yOpen - yClose);

  return (
    <g>
      {/* Upper wick */}
      <line
        x1={xCenter}
        y1={yHigh}
        x2={xCenter}
        y2={bodyTop}
        stroke={color}
        strokeWidth={wickWidth}
      />

      {/* Body */}
      <rect
        x={xCenter - candleWidth / 2}
        y={bodyTop}
        width={candleWidth}
        height={Math.max(bodyHeight, 1)}
        fill={color}
        stroke={color}
        strokeWidth={1}
      />

      {/* Lower wick */}
      <line
        x1={xCenter}
        y1={bodyTop + bodyHeight}
        x2={xCenter}
        y2={yLow}
        stroke={color}
        strokeWidth={wickWidth}
      />
    </g>
  );
};

// Calculate Moving Average
const calculateMA = (data: CandlestickData[], period: number): number[] => {
  const ma: number[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      ma.push(NaN);
      continue;
    }

    const sum = data
      .slice(i - period + 1, i + 1)
      .reduce((acc, candle) => acc + candle.close, 0);
    ma.push(sum / period);
  }

  return ma;
};

// Custom Tooltip
const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload || !payload[0]) return null;

  const data = payload[0].payload as CandlestickData;
  const isGreen = data.close >= data.open;
  const change = data.close - data.open;
  const changePercent = (change / data.open) * 100;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-lg">
      <p className="text-gray-300 text-sm mb-2">
        {format(new Date(data.timestamp), 'MMM dd, yyyy HH:mm')}
      </p>

      <div className="space-y-1 text-sm">
        <div className="flex justify-between gap-4">
          <span className="text-gray-400">Open:</span>
          <span className="text-white font-mono">${data.open.toFixed(2)}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-gray-400">High:</span>
          <span className="text-green-400 font-mono">${data.high.toFixed(2)}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-gray-400">Low:</span>
          <span className="text-red-400 font-mono">${data.low.toFixed(2)}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-gray-400">Close:</span>
          <span className="text-white font-mono">${data.close.toFixed(2)}</span>
        </div>

        <div className="flex justify-between gap-4 pt-2 border-t border-gray-700">
          <span className="text-gray-400">Change:</span>
          <span
            className={`font-mono ${
              isGreen ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {change >= 0 ? '+' : ''}
            {change.toFixed(2)} ({changePercent >= 0 ? '+' : ''}
            {changePercent.toFixed(2)}%)
          </span>
        </div>

        <div className="flex justify-between gap-4">
          <span className="text-gray-400">Volume:</span>
          <span className="text-blue-400 font-mono">
            {(data.volume / 1e6).toFixed(2)}M
          </span>
        </div>
      </div>
    </div>
  );
};

export function CandlestickChart({
  data,
  showVolume = true,
  showMA = true,
  maPeriods = [20, 50],
  height = 400,
  className = '',
}: CandlestickChartProps) {
  // Process data with moving averages
  const processedData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const ma20 = showMA && maPeriods.includes(20) ? calculateMA(data, 20) : [];
    const ma50 = showMA && maPeriods.includes(50) ? calculateMA(data, 50) : [];

    return data.map((candle, i) => ({
      ...candle,
      date: new Date(candle.timestamp),
      ma20: ma20[i],
      ma50: ma50[i],
    }));
  }, [data, showMA, maPeriods]);

  if (!processedData || processedData.length === 0) {
    return (
      <div
        className={`flex items-center justify-center bg-gray-800 rounded-lg ${className}`}
        style={{ height }}
      >
        <p className="text-gray-400">No data available</p>
      </div>
    );
  }

  // Calculate price domain for proper scaling
  const priceExtent = useMemo(() => {
    const highs = data.map((d) => d.high);
    const lows = data.map((d) => d.low);
    const max = Math.max(...highs);
    const min = Math.min(...lows);
    const padding = (max - min) * 0.1;
    return [min - padding, max + padding];
  }, [data]);

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={processedData}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />

          <XAxis
            dataKey="timestamp"
            tickFormatter={(value) => format(new Date(value), 'MMM dd')}
            stroke="#9ca3af"
            style={{ fontSize: '12px' }}
          />

          <YAxis
            yAxisId="price"
            domain={priceExtent}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
            stroke="#9ca3af"
            style={{ fontSize: '12px' }}
          />

          {showVolume && (
            <YAxis
              yAxisId="volume"
              orientation="right"
              tickFormatter={(value) => `${(value / 1e6).toFixed(0)}M`}
              stroke="#9ca3af"
              style={{ fontSize: '12px' }}
            />
          )}

          <Tooltip content={<CustomTooltip />} />

          <Legend />

          {/* Candlesticks */}
          <Bar
            yAxisId="price"
            dataKey="high"
            fill="transparent"
            shape={<CandlestickShape />}
            name="Price"
          />

          {/* Moving Averages */}
          {showMA && maPeriods.includes(20) && (
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="ma20"
              stroke="#f59e0b"
              strokeWidth={1.5}
              dot={false}
              name="MA 20"
              connectNulls
            />
          )}

          {showMA && maPeriods.includes(50) && (
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="ma50"
              stroke="#8b5cf6"
              strokeWidth={1.5}
              dot={false}
              name="MA 50"
              connectNulls
            />
          )}

          {/* Volume */}
          {showVolume && (
            <Bar
              yAxisId="volume"
              dataKey="volume"
              fill="url(#volumeGradient)"
              opacity={0.3}
              name="Volume"
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
