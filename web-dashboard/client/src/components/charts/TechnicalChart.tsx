/**
 * Technical Chart Component
 *
 * Advanced charting component with technical indicator overlays:
 * - Price chart with candlesticks
 * - Bollinger Bands, SMA/EMA overlays
 * - RSI panel
 * - MACD panel
 * - Volume panel
 * - Stochastic oscillator
 */

import React, { useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Line,
  Bar,
  ReferenceLine,
  Legend,
} from 'recharts';
import { format } from 'date-fns';
import { CandlestickData } from './CandlestickChart';
import {
  calculateSMA,
  calculateEMA,
  calculateRSI,
  calculateMACD,
  calculateBollingerBands,
  calculateStochastic,
  calculateATR,
  PriceData,
} from '../../lib/technicalIndicators';

export interface TechnicalChartProps {
  data: CandlestickData[];
  indicators?: {
    sma?: number[];
    ema?: number[];
    bollinger?: boolean;
    rsi?: boolean;
    macd?: boolean;
    stochastic?: boolean;
    volume?: boolean;
  };
  height?: number;
  className?: string;
}

interface ProcessedData extends CandlestickData {
  sma20?: number;
  sma50?: number;
  sma200?: number;
  ema12?: number;
  ema26?: number;
  bbUpper?: number;
  bbMiddle?: number;
  bbLower?: number;
  rsi?: number;
  macd?: number;
  macdSignal?: number;
  macdHistogram?: number;
  stochK?: number;
  stochD?: number;
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

  const scale = height / (high - low);
  const yHigh = y;
  const yLow = y + height;
  const yOpen = y + (high - open) * scale;
  const yClose = y + (high - close) * scale;

  const bodyTop = Math.min(yOpen, yClose);
  const bodyHeight = Math.abs(yOpen - yClose);

  return (
    <g>
      <line x1={xCenter} y1={yHigh} x2={xCenter} y2={bodyTop} stroke={color} strokeWidth={wickWidth} />
      <rect
        x={xCenter - candleWidth / 2}
        y={bodyTop}
        width={candleWidth}
        height={Math.max(bodyHeight, 1)}
        fill={color}
        stroke={color}
        strokeWidth={1}
      />
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

export function TechnicalChart({
  data,
  indicators = {
    sma: [20, 50],
    ema: [12, 26],
    bollinger: true,
    rsi: true,
    macd: true,
    volume: true,
  },
  height = 600,
  className = '',
}: TechnicalChartProps) {
  const [selectedIndicators, setSelectedIndicators] = useState(indicators);

  // Process data with all indicators
  const processedData = useMemo((): ProcessedData[] => {
    if (!data || data.length === 0) return [];

    const closes = data.map((d) => d.close);
    const priceData: PriceData[] = data.map((d) => ({
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
      volume: d.volume,
    }));

    // Calculate SMAs
    const sma20 = selectedIndicators.sma?.includes(20)
      ? calculateSMA(closes, 20)
      : [];
    const sma50 = selectedIndicators.sma?.includes(50)
      ? calculateSMA(closes, 50)
      : [];
    const sma200 = selectedIndicators.sma?.includes(200)
      ? calculateSMA(closes, 200)
      : [];

    // Calculate EMAs
    const ema12 = selectedIndicators.ema?.includes(12)
      ? calculateEMA(closes, 12)
      : [];
    const ema26 = selectedIndicators.ema?.includes(26)
      ? calculateEMA(closes, 26)
      : [];

    // Calculate Bollinger Bands
    const bollinger = selectedIndicators.bollinger
      ? calculateBollingerBands(closes, 20, 2)
      : null;

    // Calculate RSI
    const rsi = selectedIndicators.rsi ? calculateRSI(closes, 14) : [];

    // Calculate MACD
    const macd = selectedIndicators.macd
      ? calculateMACD(closes, 12, 26, 9)
      : null;

    // Calculate Stochastic
    const stochastic = selectedIndicators.stochastic
      ? calculateStochastic(priceData, 14, 3)
      : null;

    // Combine all data
    return data.map((candle, i) => ({
      ...candle,
      sma20: sma20[i],
      sma50: sma50[i],
      sma200: sma200[i],
      ema12: ema12[i],
      ema26: ema26[i],
      bbUpper: bollinger?.upper[i],
      bbMiddle: bollinger?.middle[i],
      bbLower: bollinger?.lower[i],
      rsi: rsi[i],
      macd: macd?.macd[i],
      macdSignal: macd?.signal[i],
      macdHistogram: macd?.histogram[i],
      stochK: stochastic?.k[i],
      stochD: stochastic?.d[i],
    }));
  }, [data, selectedIndicators]);

  // Calculate domains
  const priceDomain = useMemo(() => {
    if (!processedData || processedData.length === 0) return [0, 100];

    const values = processedData.flatMap((d) => [
      d.high,
      d.low,
      d.bbUpper || 0,
      d.bbLower || 0,
    ]);
    const min = Math.min(...values.filter((v) => !isNaN(v)));
    const max = Math.max(...values.filter((v) => !isNaN(v)));
    const padding = (max - min) * 0.05;

    return [min - padding, max + padding];
  }, [processedData]);

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

  // Calculate panel heights
  const mainChartHeight = height * 0.5;
  const indicatorHeight = height * 0.15;
  const volumeHeight = height * 0.1;

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      {/* Indicator Toggles */}
      <div className="mb-4 flex flex-wrap gap-2">
        <button
          onClick={() =>
            setSelectedIndicators((prev) => ({
              ...prev,
              bollinger: !prev.bollinger,
            }))
          }
          className={`px-3 py-1 rounded text-xs font-medium ${
            selectedIndicators.bollinger
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300'
          }`}
        >
          Bollinger Bands
        </button>
        <button
          onClick={() =>
            setSelectedIndicators((prev) => ({ ...prev, rsi: !prev.rsi }))
          }
          className={`px-3 py-1 rounded text-xs font-medium ${
            selectedIndicators.rsi
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300'
          }`}
        >
          RSI
        </button>
        <button
          onClick={() =>
            setSelectedIndicators((prev) => ({ ...prev, macd: !prev.macd }))
          }
          className={`px-3 py-1 rounded text-xs font-medium ${
            selectedIndicators.macd
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300'
          }`}
        >
          MACD
        </button>
        <button
          onClick={() =>
            setSelectedIndicators((prev) => ({ ...prev, volume: !prev.volume }))
          }
          className={`px-3 py-1 rounded text-xs font-medium ${
            selectedIndicators.volume
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300'
          }`}
        >
          Volume
        </button>
      </div>

      <div className="space-y-2">
        {/* Main Price Chart */}
        <ResponsiveContainer width="100%" height={mainChartHeight}>
          <ComposedChart
            data={processedData}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(value) => format(new Date(value), 'MMM dd')}
              stroke="#9ca3af"
              style={{ fontSize: '12px' }}
            />
            <YAxis
              yAxisId="price"
              domain={priceDomain}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
              stroke="#9ca3af"
              style={{ fontSize: '12px' }}
            />
            <Tooltip />
            <Legend />

            {/* Bollinger Bands */}
            {selectedIndicators.bollinger && (
              <>
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="bbUpper"
                  stroke="#6b7280"
                  strokeWidth={1}
                  dot={false}
                  name="BB Upper"
                  strokeDasharray="3 3"
                  connectNulls
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="bbMiddle"
                  stroke="#9ca3af"
                  strokeWidth={1}
                  dot={false}
                  name="BB Middle"
                  connectNulls
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="bbLower"
                  stroke="#6b7280"
                  strokeWidth={1}
                  dot={false}
                  name="BB Lower"
                  strokeDasharray="3 3"
                  connectNulls
                />
              </>
            )}

            {/* SMAs */}
            {selectedIndicators.sma?.includes(20) && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="sma20"
                stroke="#f59e0b"
                strokeWidth={1.5}
                dot={false}
                name="SMA 20"
                connectNulls
              />
            )}
            {selectedIndicators.sma?.includes(50) && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="sma50"
                stroke="#8b5cf6"
                strokeWidth={1.5}
                dot={false}
                name="SMA 50"
                connectNulls
              />
            )}

            {/* EMAs */}
            {selectedIndicators.ema?.includes(12) && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="ema12"
                stroke="#3b82f6"
                strokeWidth={1.5}
                dot={false}
                name="EMA 12"
                connectNulls
              />
            )}
            {selectedIndicators.ema?.includes(26) && (
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="ema26"
                stroke="#10b981"
                strokeWidth={1.5}
                dot={false}
                name="EMA 26"
                connectNulls
              />
            )}

            {/* Candlesticks */}
            <Bar
              yAxisId="price"
              dataKey="high"
              fill="transparent"
              shape={<CandlestickShape />}
              name="Price"
            />
          </ComposedChart>
        </ResponsiveContainer>

        {/* Volume Panel */}
        {selectedIndicators.volume && (
          <ResponsiveContainer width="100%" height={volumeHeight}>
            <ComposedChart
              data={processedData}
              margin={{ top: 0, right: 30, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(value) => format(new Date(value), 'MMM dd')}
                stroke="#9ca3af"
                style={{ fontSize: '10px' }}
              />
              <YAxis
                tickFormatter={(value) => `${(value / 1e6).toFixed(0)}M`}
                stroke="#9ca3af"
                style={{ fontSize: '10px' }}
              />
              <Bar dataKey="volume" fill="#3b82f6" opacity={0.5} name="Volume" />
            </ComposedChart>
          </ResponsiveContainer>
        )}

        {/* RSI Panel */}
        {selectedIndicators.rsi && (
          <ResponsiveContainer width="100%" height={indicatorHeight}>
            <ComposedChart
              data={processedData}
              margin={{ top: 0, right: 30, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(value) => format(new Date(value), 'MMM dd')}
                stroke="#9ca3af"
                style={{ fontSize: '10px' }}
              />
              <YAxis domain={[0, 100]} stroke="#9ca3af" style={{ fontSize: '10px' }} />
              <Tooltip />
              <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" />
              <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" />
              <Line
                type="monotone"
                dataKey="rsi"
                stroke="#8b5cf6"
                strokeWidth={2}
                dot={false}
                name="RSI"
                connectNulls
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}

        {/* MACD Panel */}
        {selectedIndicators.macd && (
          <ResponsiveContainer width="100%" height={indicatorHeight}>
            <ComposedChart
              data={processedData}
              margin={{ top: 0, right: 30, left: 0, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(value) => format(new Date(value), 'MMM dd')}
                stroke="#9ca3af"
                style={{ fontSize: '10px' }}
              />
              <YAxis stroke="#9ca3af" style={{ fontSize: '10px' }} />
              <Tooltip />
              <ReferenceLine y={0} stroke="#6b7280" />
              <Bar
                dataKey="macdHistogram"
                fill="#3b82f6"
                opacity={0.6}
                name="Histogram"
              />
              <Line
                type="monotone"
                dataKey="macd"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
                name="MACD"
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="macdSignal"
                stroke="#ef4444"
                strokeWidth={2}
                dot={false}
                name="Signal"
                connectNulls
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
