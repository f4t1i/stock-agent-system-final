/**
 * Heatmap Component
 *
 * Interactive heatmap for visualizing:
 * - Sector performance
 * - Correlation matrices
 * - Market overviews
 * - Risk matrices
 */

import React, { useState, useMemo } from 'react';

export interface HeatmapCell {
  row: string;
  col: string;
  value: number;
  label?: string;
}

export interface HeatmapProps {
  data: HeatmapCell[];
  rows: string[];
  cols: string[];
  minValue?: number;
  maxValue?: number;
  colorScale?: 'diverging' | 'sequential' | 'performance';
  valueFormatter?: (value: number) => string;
  showValues?: boolean;
  showLabels?: boolean;
  cellSize?: number;
  className?: string;
}

interface TooltipData {
  cell: HeatmapCell;
  x: number;
  y: number;
}

// Color interpolation helpers
const interpolateColor = (
  value: number,
  min: number,
  max: number,
  colorScale: 'diverging' | 'sequential' | 'performance'
): string => {
  const normalized = (value - min) / (max - min);

  if (colorScale === 'diverging') {
    // Red (negative) -> White (neutral) -> Green (positive)
    if (normalized < 0.5) {
      // Red to White
      const t = normalized * 2;
      const r = 239; // ef4444
      const g = Math.round(68 + (255 - 68) * t);
      const b = Math.round(68 + (255 - 68) * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // White to Green
      const t = (normalized - 0.5) * 2;
      const r = Math.round(255 - (255 - 16) * t);
      const g = Math.round(255 - (255 - 185) * t);
      const b = Math.round(255 - (255 - 129) * t);
      return `rgb(${r}, ${g}, ${b})`;
    }
  } else if (colorScale === 'performance') {
    // Performance: Red (bad) -> Yellow (neutral) -> Green (good)
    if (normalized < 0.5) {
      // Red to Yellow
      const t = normalized * 2;
      const r = 239;
      const g = Math.round(68 + (234 - 68) * t);
      const b = 68;
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Yellow to Green
      const t = (normalized - 0.5) * 2;
      const r = Math.round(234 - (234 - 16) * t);
      const g = Math.round(234 - (234 - 185) * t);
      const b = Math.round(68 + (129 - 68) * t);
      return `rgb(${r}, ${g}, ${b})`;
    }
  } else {
    // Sequential: Light Blue -> Dark Blue
    const r = Math.round(219 - (219 - 59) * normalized);
    const g = Math.round(234 - (234 - 130) * normalized);
    const b = Math.round(254 - (254 - 246) * normalized);
    return `rgb(${r}, ${g}, ${b})`;
  }
};

const getTextColor = (bgColor: string): string => {
  // Extract RGB values
  const match = bgColor.match(/\d+/g);
  if (!match) return '#ffffff';

  const [r, g, b] = match.map(Number);

  // Calculate luminance
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;

  return luminance > 0.6 ? '#000000' : '#ffffff';
};

export function Heatmap({
  data,
  rows,
  cols,
  minValue,
  maxValue,
  colorScale = 'diverging',
  valueFormatter = (v) => v.toFixed(2),
  showValues = true,
  showLabels = true,
  cellSize = 80,
  className = '',
}: HeatmapProps) {
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);

  // Calculate min/max if not provided
  const { min, max } = useMemo(() => {
    if (minValue !== undefined && maxValue !== undefined) {
      return { min: minValue, max: maxValue };
    }

    const values = data.map((cell) => cell.value);
    return {
      min: minValue ?? Math.min(...values),
      max: maxValue ?? Math.max(...values),
    };
  }, [data, minValue, maxValue]);

  // Create cell lookup map
  const cellMap = useMemo(() => {
    const map = new Map<string, HeatmapCell>();
    data.forEach((cell) => {
      map.set(`${cell.row}-${cell.col}`, cell);
    });
    return map;
  }, [data]);

  const handleMouseEnter = (cell: HeatmapCell, event: React.MouseEvent) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setTooltip({
      cell,
      x: rect.left + rect.width / 2,
      y: rect.top,
    });
  };

  const handleMouseLeave = () => {
    setTooltip(null);
  };

  if (!data || data.length === 0) {
    return (
      <div
        className={`flex items-center justify-center bg-gray-800 rounded-lg ${className}`}
        style={{ height: cellSize * Math.max(rows.length, 3) }}
      >
        <p className="text-gray-400">No data available</p>
      </div>
    );
  }

  const labelWidth = showLabels ? 120 : 0;
  const labelHeight = showLabels ? 40 : 0;

  return (
    <div className={`bg-gray-800 rounded-lg p-4 overflow-auto ${className}`}>
      <div
        style={{
          width: labelWidth + cols.length * cellSize,
          height: labelHeight + rows.length * cellSize,
          position: 'relative',
        }}
      >
        {/* Column Labels */}
        {showLabels && (
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: labelWidth,
              display: 'flex',
            }}
          >
            {cols.map((col, index) => (
              <div
                key={index}
                style={{
                  width: cellSize,
                  height: labelHeight,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
                className="text-gray-300 text-sm font-medium"
              >
                {col}
              </div>
            ))}
          </div>
        )}

        {/* Row Labels */}
        {showLabels && (
          <div
            style={{
              position: 'absolute',
              top: labelHeight,
              left: 0,
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {rows.map((row, index) => (
              <div
                key={index}
                style={{
                  width: labelWidth,
                  height: cellSize,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'flex-end',
                  paddingRight: '8px',
                }}
                className="text-gray-300 text-sm font-medium"
              >
                {row}
              </div>
            ))}
          </div>
        )}

        {/* Heatmap Grid */}
        <div
          style={{
            position: 'absolute',
            top: labelHeight,
            left: labelWidth,
          }}
        >
          {rows.map((row, rowIndex) => (
            <div key={rowIndex} style={{ display: 'flex' }}>
              {cols.map((col, colIndex) => {
                const cell = cellMap.get(`${row}-${col}`);
                if (!cell) {
                  return (
                    <div
                      key={colIndex}
                      style={{
                        width: cellSize,
                        height: cellSize,
                        border: '1px solid #374151',
                      }}
                      className="bg-gray-700"
                    />
                  );
                }

                const bgColor = interpolateColor(cell.value, min, max, colorScale);
                const textColor = getTextColor(bgColor);

                return (
                  <div
                    key={colIndex}
                    style={{
                      width: cellSize,
                      height: cellSize,
                      backgroundColor: bgColor,
                      border: '1px solid #374151',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                    }}
                    className="hover:opacity-80"
                    onMouseEnter={(e) => handleMouseEnter(cell, e)}
                    onMouseLeave={handleMouseLeave}
                  >
                    {showValues && (
                      <span
                        style={{ color: textColor }}
                        className="text-xs font-mono font-semibold"
                      >
                        {cell.label || valueFormatter(cell.value)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          style={{
            position: 'fixed',
            left: tooltip.x,
            top: tooltip.y - 10,
            transform: 'translate(-50%, -100%)',
            pointerEvents: 'none',
            zIndex: 1000,
          }}
          className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-lg"
        >
          <div className="space-y-1 text-sm">
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Row:</span>
              <span className="text-white font-semibold">{tooltip.cell.row}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Col:</span>
              <span className="text-white font-semibold">{tooltip.cell.col}</span>
            </div>
            <div className="flex justify-between gap-4 pt-1 border-t border-gray-700">
              <span className="text-gray-400">Value:</span>
              <span className="text-white font-mono font-semibold">
                {valueFormatter(tooltip.cell.value)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Preset Configurations for Common Use Cases
 */

export const CorrelationMatrix = ({
  symbols,
  correlations,
  className = '',
}: {
  symbols: string[];
  correlations: number[][];
  className?: string;
}) => {
  const data: HeatmapCell[] = [];

  symbols.forEach((row, i) => {
    symbols.forEach((col, j) => {
      data.push({
        row,
        col,
        value: correlations[i][j],
      });
    });
  });

  return (
    <Heatmap
      data={data}
      rows={symbols}
      cols={symbols}
      minValue={-1}
      maxValue={1}
      colorScale="diverging"
      valueFormatter={(v) => v.toFixed(2)}
      showValues={true}
      showLabels={true}
      cellSize={70}
      className={className}
    />
  );
};

export const SectorPerformanceHeatmap = ({
  sectors,
  timeframes,
  performance,
  className = '',
}: {
  sectors: string[];
  timeframes: string[];
  performance: number[][];
  className?: string;
}) => {
  const data: HeatmapCell[] = [];

  sectors.forEach((sector, i) => {
    timeframes.forEach((timeframe, j) => {
      data.push({
        row: sector,
        col: timeframe,
        value: performance[i][j],
      });
    });
  });

  return (
    <Heatmap
      data={data}
      rows={sectors}
      cols={timeframes}
      colorScale="performance"
      valueFormatter={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`}
      showValues={true}
      showLabels={true}
      cellSize={90}
      className={className}
    />
  );
};

export const MarketOverviewHeatmap = ({
  stocks,
  className = '',
}: {
  stocks: Array<{
    symbol: string;
    change: number;
    marketCap: number;
  }>;
  className?: string;
}) => {
  // Create a grid layout based on market cap
  const sortedStocks = [...stocks].sort((a, b) => b.marketCap - a.marketCap);

  // Calculate grid dimensions
  const gridSize = Math.ceil(Math.sqrt(sortedStocks.length));
  const rows: string[] = [];
  const cols: string[] = [];

  for (let i = 0; i < gridSize; i++) {
    rows.push(`R${i}`);
    cols.push(`C${i}`);
  }

  const data: HeatmapCell[] = sortedStocks.map((stock, index) => {
    const row = Math.floor(index / gridSize);
    const col = index % gridSize;

    return {
      row: `R${row}`,
      col: `C${col}`,
      value: stock.change,
      label: stock.symbol,
    };
  });

  return (
    <Heatmap
      data={data}
      rows={rows}
      cols={cols}
      colorScale="performance"
      valueFormatter={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`}
      showValues={false}
      showLabels={false}
      cellSize={60}
      className={className}
    />
  );
};

export const RiskHeatmap = ({
  assets,
  riskMetrics,
  riskData,
  className = '',
}: {
  assets: string[];
  riskMetrics: string[];
  riskData: number[][];
  className?: string;
}) => {
  const data: HeatmapCell[] = [];

  assets.forEach((asset, i) => {
    riskMetrics.forEach((metric, j) => {
      data.push({
        row: asset,
        col: metric,
        value: riskData[i][j],
      });
    });
  });

  return (
    <Heatmap
      data={data}
      rows={assets}
      cols={riskMetrics}
      colorScale="performance"
      valueFormatter={(v) => v.toFixed(2)}
      showValues={true}
      showLabels={true}
      cellSize={100}
      className={className}
    />
  );
};
