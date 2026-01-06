/**
 * Line Chart Component
 *
 * Multi-line chart for comparing multiple stocks or metrics over time.
 * Supports multiple series, tooltips, legends, and custom styling.
 */

import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import { format } from 'date-fns';

export interface LineDataPoint {
  timestamp: string;
  [key: string]: number | string; // Dynamic keys for multiple series
}

export interface LineSeries {
  dataKey: string;
  name: string;
  color: string;
  strokeWidth?: number;
  strokeDasharray?: string;
}

export interface LineChartProps {
  data: LineDataPoint[];
  series: LineSeries[];
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
  referenceLines?: Array<{ value: number; label: string; color?: string }>;
  yAxisLabel?: string;
  yAxisFormatter?: (value: number) => string;
  className?: string;
}

// Custom Tooltip
const CustomTooltip = ({
  active,
  payload,
  label,
  series,
}: {
  active?: boolean;
  payload?: any[];
  label?: string;
  series: LineSeries[];
}) => {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-lg">
      <p className="text-gray-300 text-sm mb-2">
        {format(new Date(label || ''), 'MMM dd, yyyy HH:mm')}
      </p>

      <div className="space-y-1">
        {payload.map((entry, index) => {
          const seriesConfig = series.find((s) => s.dataKey === entry.dataKey);
          if (!seriesConfig) return null;

          return (
            <div key={index} className="flex justify-between gap-4 text-sm">
              <span className="flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: entry.color }}
                />
                <span className="text-gray-400">{seriesConfig.name}:</span>
              </span>
              <span className="text-white font-mono">
                {typeof entry.value === 'number'
                  ? entry.value.toFixed(2)
                  : entry.value}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export function LineChart({
  data,
  series,
  height = 400,
  showGrid = true,
  showLegend = true,
  referenceLines = [],
  yAxisLabel,
  yAxisFormatter = (value) => value.toFixed(2),
  className = '',
}: LineChartProps) {
  // Calculate Y-axis domain
  const yDomain = useMemo(() => {
    if (!data || data.length === 0) return [0, 100];

    const allValues: number[] = [];
    data.forEach((point) => {
      series.forEach((s) => {
        const value = point[s.dataKey];
        if (typeof value === 'number' && !isNaN(value)) {
          allValues.push(value);
        }
      });
    });

    if (allValues.length === 0) return [0, 100];

    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const padding = (max - min) * 0.1;

    return [Math.max(0, min - padding), max + padding];
  }, [data, series]);

  if (!data || data.length === 0) {
    return (
      <div
        className={`flex items-center justify-center bg-gray-800 rounded-lg ${className}`}
        style={{ height }}
      >
        <p className="text-gray-400">No data available</p>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <ResponsiveContainer width="100%" height={height}>
        <RechartsLineChart
          data={data}
          margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
        >
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#374151" />}

          <XAxis
            dataKey="timestamp"
            tickFormatter={(value) => format(new Date(value), 'MMM dd')}
            stroke="#9ca3af"
            style={{ fontSize: '12px' }}
          />

          <YAxis
            domain={yDomain}
            tickFormatter={yAxisFormatter}
            stroke="#9ca3af"
            style={{ fontSize: '12px' }}
            label={
              yAxisLabel
                ? {
                    value: yAxisLabel,
                    angle: -90,
                    position: 'insideLeft',
                    style: { fill: '#9ca3af', fontSize: '12px' },
                  }
                : undefined
            }
          />

          <Tooltip content={(props) => <CustomTooltip {...props} series={series} />} />

          {showLegend && <Legend />}

          {/* Reference Lines */}
          {referenceLines.map((line, index) => (
            <ReferenceLine
              key={index}
              y={line.value}
              label={line.label}
              stroke={line.color || '#6b7280'}
              strokeDasharray="3 3"
            />
          ))}

          {/* Data Lines */}
          {series.map((s, index) => (
            <Line
              key={index}
              type="monotone"
              dataKey={s.dataKey}
              name={s.name}
              stroke={s.color}
              strokeWidth={s.strokeWidth || 2}
              strokeDasharray={s.strokeDasharray}
              dot={false}
              activeDot={{ r: 6 }}
              connectNulls
            />
          ))}
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Preset Configurations for Common Use Cases
 */

export const PriceComparisonChart = ({
  data,
  symbols,
  height = 400,
  className = '',
}: {
  data: LineDataPoint[];
  symbols: string[];
  height?: number;
  className?: string;
}) => {
  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  const series: LineSeries[] = symbols.map((symbol, index) => ({
    dataKey: symbol,
    name: symbol,
    color: colors[index % colors.length],
    strokeWidth: 2,
  }));

  return (
    <LineChart
      data={data}
      series={series}
      height={height}
      className={className}
      yAxisFormatter={(value) => `$${value.toFixed(2)}`}
    />
  );
};

export const PerformanceChart = ({
  data,
  height = 400,
  className = '',
}: {
  data: LineDataPoint[];
  height?: number;
  className?: string;
}) => {
  const series: LineSeries[] = [
    {
      dataKey: 'portfolio',
      name: 'Portfolio',
      color: '#3b82f6',
      strokeWidth: 3,
    },
    {
      dataKey: 'benchmark',
      name: 'S&P 500',
      color: '#6b7280',
      strokeWidth: 2,
      strokeDasharray: '5 5',
    },
  ];

  return (
    <LineChart
      data={data}
      series={series}
      height={height}
      className={className}
      yAxisFormatter={(value) => `${value.toFixed(1)}%`}
      yAxisLabel="Return (%)"
      referenceLines={[{ value: 0, label: '0%', color: '#6b7280' }]}
    />
  );
};

export const MetricsChart = ({
  data,
  metrics,
  height = 400,
  className = '',
}: {
  data: LineDataPoint[];
  metrics: Array<{ key: string; name: string; color: string }>;
  height?: number;
  className?: string;
}) => {
  const series: LineSeries[] = metrics.map((m) => ({
    dataKey: m.key,
    name: m.name,
    color: m.color,
    strokeWidth: 2,
  }));

  return (
    <LineChart
      data={data}
      series={series}
      height={height}
      className={className}
      showLegend={true}
    />
  );
};
