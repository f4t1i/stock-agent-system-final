/**
 * Area Chart Component
 *
 * Stacked or overlapping area charts for visualizing cumulative data,
 * portfolio allocation, and trend patterns with filled regions.
 */

import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart as RechartsAreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { format } from 'date-fns';

export interface AreaDataPoint {
  timestamp: string;
  [key: string]: number | string;
}

export interface AreaSeries {
  dataKey: string;
  name: string;
  color: string;
  stackId?: string;
  fillOpacity?: number;
}

export interface AreaChartProps {
  data: AreaDataPoint[];
  series: AreaSeries[];
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
  stacked?: boolean;
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
  stacked,
}: {
  active?: boolean;
  payload?: any[];
  label?: string;
  series: AreaSeries[];
  stacked?: boolean;
}) => {
  if (!active || !payload || payload.length === 0) return null;

  // Calculate total for stacked charts
  const total = stacked
    ? payload.reduce((sum, entry) => sum + (entry.value || 0), 0)
    : null;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-lg">
      <p className="text-gray-300 text-sm mb-2">
        {format(new Date(label || ''), 'MMM dd, yyyy HH:mm')}
      </p>

      <div className="space-y-1">
        {payload
          .slice()
          .reverse()
          .map((entry, index) => {
            const seriesConfig = series.find((s) => s.dataKey === entry.dataKey);
            if (!seriesConfig) return null;

            const percentage =
              stacked && total ? ((entry.value / total) * 100).toFixed(1) : null;

            return (
              <div key={index} className="flex justify-between gap-4 text-sm">
                <span className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded"
                    style={{ backgroundColor: entry.color }}
                  />
                  <span className="text-gray-400">{seriesConfig.name}:</span>
                </span>
                <span className="text-white font-mono">
                  {typeof entry.value === 'number'
                    ? entry.value.toFixed(2)
                    : entry.value}
                  {percentage && (
                    <span className="text-gray-400 ml-1">({percentage}%)</span>
                  )}
                </span>
              </div>
            );
          })}

        {stacked && total !== null && (
          <div className="flex justify-between gap-4 pt-2 border-t border-gray-700 text-sm font-semibold">
            <span className="text-gray-400">Total:</span>
            <span className="text-white font-mono">{total.toFixed(2)}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export function AreaChart({
  data,
  series,
  height = 400,
  showGrid = true,
  showLegend = true,
  stacked = false,
  yAxisLabel,
  yAxisFormatter = (value) => value.toFixed(2),
  className = '',
}: AreaChartProps) {
  // Calculate Y-axis domain
  const yDomain = useMemo(() => {
    if (!data || data.length === 0) return [0, 100];

    if (stacked) {
      // For stacked charts, calculate sum at each timestamp
      const maxSum = Math.max(
        ...data.map((point) => {
          return series.reduce((sum, s) => {
            const value = point[s.dataKey];
            return sum + (typeof value === 'number' ? value : 0);
          }, 0);
        })
      );
      return [0, maxSum * 1.1];
    } else {
      // For overlapping charts, find max individual value
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
    }
  }, [data, series, stacked]);

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
        <RechartsAreaChart
          data={data}
          margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
        >
          <defs>
            {series.map((s, index) => (
              <linearGradient
                key={index}
                id={`gradient-${s.dataKey}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop
                  offset="5%"
                  stopColor={s.color}
                  stopOpacity={s.fillOpacity || 0.8}
                />
                <stop
                  offset="95%"
                  stopColor={s.color}
                  stopOpacity={s.fillOpacity ? s.fillOpacity * 0.2 : 0.1}
                />
              </linearGradient>
            ))}
          </defs>

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

          <Tooltip
            content={(props) => (
              <CustomTooltip {...props} series={series} stacked={stacked} />
            )}
          />

          {showLegend && <Legend />}

          {/* Area Series */}
          {series.map((s, index) => (
            <Area
              key={index}
              type="monotone"
              dataKey={s.dataKey}
              name={s.name}
              stackId={stacked ? 'stack' : undefined}
              stroke={s.color}
              strokeWidth={2}
              fill={`url(#gradient-${s.dataKey})`}
              fillOpacity={1}
              connectNulls
            />
          ))}
        </RechartsAreaChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Preset Configurations for Common Use Cases
 */

export const PortfolioAllocationChart = ({
  data,
  height = 400,
  className = '',
}: {
  data: AreaDataPoint[];
  height?: number;
  className?: string;
}) => {
  const series: AreaSeries[] = [
    { dataKey: 'stocks', name: 'Stocks', color: '#3b82f6', fillOpacity: 0.8 },
    { dataKey: 'bonds', name: 'Bonds', color: '#10b981', fillOpacity: 0.8 },
    { dataKey: 'cash', name: 'Cash', color: '#f59e0b', fillOpacity: 0.8 },
    { dataKey: 'crypto', name: 'Crypto', color: '#8b5cf6', fillOpacity: 0.8 },
  ];

  return (
    <AreaChart
      data={data}
      series={series}
      height={height}
      className={className}
      stacked={true}
      yAxisFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
      yAxisLabel="Portfolio Value"
    />
  );
};

export const CumulativeReturnsChart = ({
  data,
  height = 400,
  className = '',
}: {
  data: AreaDataPoint[];
  height?: number;
  className?: string;
}) => {
  const series: AreaSeries[] = [
    {
      dataKey: 'returns',
      name: 'Cumulative Returns',
      color: '#3b82f6',
      fillOpacity: 0.6,
    },
  ];

  return (
    <AreaChart
      data={data}
      series={series}
      height={height}
      className={className}
      yAxisFormatter={(value) => `${value.toFixed(1)}%`}
      yAxisLabel="Return (%)"
    />
  );
};

export const SectorAllocationChart = ({
  data,
  sectors,
  height = 400,
  className = '',
}: {
  data: AreaDataPoint[];
  sectors: Array<{ key: string; name: string; color: string }>;
  height?: number;
  className?: string;
}) => {
  const series: AreaSeries[] = sectors.map((sector) => ({
    dataKey: sector.key,
    name: sector.name,
    color: sector.color,
    fillOpacity: 0.7,
  }));

  return (
    <AreaChart
      data={data}
      series={series}
      height={height}
      className={className}
      stacked={true}
      yAxisFormatter={(value) => `${value.toFixed(0)}%`}
      yAxisLabel="Allocation (%)"
    />
  );
};

export const VolumeProfileChart = ({
  data,
  height = 400,
  className = '',
}: {
  data: AreaDataPoint[];
  height?: number;
  className?: string;
}) => {
  const series: AreaSeries[] = [
    {
      dataKey: 'volume',
      name: 'Volume',
      color: '#3b82f6',
      fillOpacity: 0.5,
    },
  ];

  return (
    <AreaChart
      data={data}
      series={series}
      height={height}
      className={className}
      yAxisFormatter={(value) => `${(value / 1e6).toFixed(1)}M`}
      yAxisLabel="Volume"
    />
  );
};
