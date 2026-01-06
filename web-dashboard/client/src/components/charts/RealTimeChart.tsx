/**
 * Real-Time Chart Component
 *
 * Wrapper component that connects any chart to WebSocket data stream
 * for real-time updates. Handles data buffering, throttling, and state management.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import type { Channel, PriceUpdateMessage } from '../../lib/websocket';

export interface RealTimeChartProps<T> {
  symbols: string[];
  channel?: Channel;
  maxDataPoints?: number;
  updateInterval?: number;
  initialData?: T[];
  dataTransformer: (updates: PriceUpdateMessage[], currentData: T[]) => T[];
  children: (data: T[], isConnected: boolean, isLoading: boolean) => React.ReactNode;
  onError?: (error: Error) => void;
}

export function RealTimeChart<T>({
  symbols,
  channel = 'prices',
  maxDataPoints = 100,
  updateInterval = 1000,
  initialData = [],
  dataTransformer,
  children,
  onError,
}: RealTimeChartProps<T>) {
  const [data, setData] = useState<T[]>(initialData);
  const [isLoading, setIsLoading] = useState(true);
  const updateBuffer = useRef<PriceUpdateMessage[]>([]);
  const lastUpdateTime = useRef<number>(Date.now());

  const { isConnected, subscribe, unsubscribe, connectionState } = useWebSocket({
    onPriceUpdate: useCallback((message: PriceUpdateMessage) => {
      // Buffer updates
      updateBuffer.current.push(message);

      // Throttle updates based on updateInterval
      const now = Date.now();
      if (now - lastUpdateTime.current >= updateInterval) {
        processBufferedUpdates();
        lastUpdateTime.current = now;
      }
    }, [updateInterval]),

    onError: useCallback((error: string) => {
      if (onError) {
        onError(new Error(error));
      }
      console.error('WebSocket error:', error);
    }, [onError]),
  });

  const processBufferedUpdates = useCallback(() => {
    if (updateBuffer.current.length === 0) return;

    setData((currentData) => {
      try {
        // Transform buffered updates into new data points
        const newData = dataTransformer(updateBuffer.current, currentData);

        // Clear buffer
        updateBuffer.current = [];

        // Limit data points
        if (newData.length > maxDataPoints) {
          return newData.slice(-maxDataPoints);
        }

        return newData;
      } catch (error) {
        console.error('Error transforming data:', error);
        if (onError) {
          onError(error as Error);
        }
        return currentData;
      }
    });
  }, [dataTransformer, maxDataPoints, onError]);

  // Subscribe to WebSocket channel on mount
  useEffect(() => {
    if (isConnected) {
      subscribe(channel, symbols);
      setIsLoading(false);
    }

    return () => {
      if (isConnected) {
        unsubscribe(channel);
      }
    };
  }, [isConnected, channel, symbols, subscribe, unsubscribe]);

  // Process any remaining buffered updates on unmount
  useEffect(() => {
    return () => {
      if (updateBuffer.current.length > 0) {
        processBufferedUpdates();
      }
    };
  }, [processBufferedUpdates]);

  // Set up interval to process buffered updates
  useEffect(() => {
    const interval = setInterval(() => {
      if (updateBuffer.current.length > 0) {
        processBufferedUpdates();
      }
    }, updateInterval);

    return () => clearInterval(interval);
  }, [processBufferedUpdates, updateInterval]);

  return <>{children(data, isConnected, isLoading)}</>;
}

/**
 * Preset Real-Time Chart Components
 */

import { CandlestickChart, CandlestickData } from './CandlestickChart';
import { LineChart, LineSeries } from './LineChart';
import { format } from 'date-fns';

export const RealTimeCandlestickChart = ({
  symbol,
  height = 400,
  showVolume = true,
  showMA = true,
  className = '',
}: {
  symbol: string;
  height?: number;
  showVolume?: boolean;
  showMA?: boolean;
  className?: string;
}) => {
  // Transform price updates into candlestick data (1-minute candles)
  const transformer = useCallback(
    (updates: PriceUpdateMessage[], currentData: CandlestickData[]): CandlestickData[] => {
      const newData = [...currentData];

      updates.forEach((update) => {
        if (update.symbol !== symbol) return;

        const timestamp = new Date(update.timestamp);
        const minuteTimestamp = new Date(
          timestamp.getFullYear(),
          timestamp.getMonth(),
          timestamp.getDate(),
          timestamp.getHours(),
          timestamp.getMinutes(),
          0,
          0
        ).toISOString();

        // Find existing candle for this minute
        const existingIndex = newData.findIndex(
          (candle) => candle.timestamp === minuteTimestamp
        );

        if (existingIndex >= 0) {
          // Update existing candle
          const candle = newData[existingIndex];
          candle.high = Math.max(candle.high, update.data.price);
          candle.low = Math.min(candle.low, update.data.low || update.data.price);
          candle.close = update.data.price;
          candle.volume += update.data.volume || 0;
        } else {
          // Create new candle
          newData.push({
            timestamp: minuteTimestamp,
            open: update.data.open || update.data.price,
            high: update.data.high || update.data.price,
            low: update.data.low || update.data.price,
            close: update.data.price,
            volume: update.data.volume || 0,
          });
        }
      });

      // Sort by timestamp
      return newData.sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
    },
    [symbol]
  );

  return (
    <RealTimeChart
      symbols={[symbol]}
      channel="prices"
      maxDataPoints={100}
      updateInterval={1000}
      dataTransformer={transformer}
    >
      {(data, isConnected, isLoading) => {
        if (isLoading) {
          return (
            <div
              className={`flex items-center justify-center bg-gray-800 rounded-lg ${className}`}
              style={{ height }}
            >
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4" />
                <p className="text-gray-400">Connecting to live data...</p>
              </div>
            </div>
          );
        }

        if (!isConnected) {
          return (
            <div
              className={`flex items-center justify-center bg-gray-800 rounded-lg ${className}`}
              style={{ height }}
            >
              <div className="text-center">
                <p className="text-red-400 mb-2">Disconnected</p>
                <p className="text-gray-400 text-sm">Attempting to reconnect...</p>
              </div>
            </div>
          );
        }

        return (
          <div className={className}>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-white font-semibold">{symbol} - Live</h3>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-green-400 text-sm">Live</span>
              </div>
            </div>
            <CandlestickChart
              data={data}
              showVolume={showVolume}
              showMA={showMA}
              height={height}
            />
          </div>
        );
      }}
    </RealTimeChart>
  );
};

export const RealTimeLineChart = ({
  symbols,
  colors,
  height = 400,
  yAxisFormatter,
  className = '',
}: {
  symbols: string[];
  colors?: string[];
  height?: number;
  yAxisFormatter?: (value: number) => string;
  className?: string;
}) => {
  const defaultColors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];
  const seriesColors = colors || defaultColors;

  const series: LineSeries[] = symbols.map((symbol, index) => ({
    dataKey: symbol,
    name: symbol,
    color: seriesColors[index % seriesColors.length],
    strokeWidth: 2,
  }));

  // Transform price updates into line chart data
  const transformer = useCallback(
    (updates: PriceUpdateMessage[], currentData: any[]): any[] => {
      const newData = [...currentData];

      // Group updates by timestamp
      const updatesByTime = new Map<string, Map<string, number>>();

      updates.forEach((update) => {
        if (!symbols.includes(update.symbol)) return;

        const timestamp = new Date(update.timestamp).toISOString();

        if (!updatesByTime.has(timestamp)) {
          updatesByTime.set(timestamp, new Map());
        }

        updatesByTime.get(timestamp)!.set(update.symbol, update.data.price);
      });

      // Add new data points
      updatesByTime.forEach((symbolPrices, timestamp) => {
        const existingIndex = newData.findIndex((d) => d.timestamp === timestamp);

        if (existingIndex >= 0) {
          // Update existing point
          symbolPrices.forEach((price, symbol) => {
            newData[existingIndex][symbol] = price;
          });
        } else {
          // Create new point
          const point: any = { timestamp };
          symbolPrices.forEach((price, symbol) => {
            point[symbol] = price;
          });
          newData.push(point);
        }
      });

      // Sort by timestamp
      return newData.sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
      );
    },
    [symbols]
  );

  return (
    <RealTimeChart
      symbols={symbols}
      channel="prices"
      maxDataPoints={100}
      updateInterval={1000}
      dataTransformer={transformer}
    >
      {(data, isConnected, isLoading) => {
        if (isLoading) {
          return (
            <div
              className={`flex items-center justify-center bg-gray-800 rounded-lg ${className}`}
              style={{ height }}
            >
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4" />
                <p className="text-gray-400">Connecting to live data...</p>
              </div>
            </div>
          );
        }

        return (
          <div className={className}>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-white font-semibold">
                {symbols.join(', ')} - Live Comparison
              </h3>
              {isConnected && (
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="text-green-400 text-sm">Live</span>
                </div>
              )}
            </div>
            <LineChart
              data={data}
              series={series}
              height={height}
              yAxisFormatter={yAxisFormatter}
            />
          </div>
        );
      }}
    </RealTimeChart>
  );
};

export const RealTimePriceDisplay = ({
  symbol,
  className = '',
}: {
  symbol: string;
  className?: string;
}) => {
  const [price, setPrice] = useState<number | null>(null);
  const [change, setChange] = useState<number>(0);
  const [changePercent, setChangePercent] = useState<number>(0);
  const previousPrice = useRef<number | null>(null);

  const { isConnected, subscribe, unsubscribe } = useWebSocket({
    onPriceUpdate: useCallback((message: PriceUpdateMessage) => {
      if (message.symbol !== symbol) return;

      const newPrice = message.data.price;

      if (previousPrice.current !== null) {
        const diff = newPrice - previousPrice.current;
        const pct = (diff / previousPrice.current) * 100;
        setChange(diff);
        setChangePercent(pct);
      }

      setPrice(newPrice);
      previousPrice.current = newPrice;
    }, [symbol]),
  });

  useEffect(() => {
    if (isConnected) {
      subscribe('prices', [symbol]);
    }

    return () => {
      if (isConnected) {
        unsubscribe('prices');
      }
    };
  }, [isConnected, symbol, subscribe, unsubscribe]);

  if (!isConnected || price === null) {
    return (
      <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
        <p className="text-gray-400 text-sm">Connecting...</p>
      </div>
    );
  }

  const isPositive = change >= 0;

  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-gray-400 text-sm">{symbol}</h3>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <span className="text-green-400 text-xs">Live</span>
        </div>
      </div>

      <div className="flex items-baseline gap-3">
        <p className="text-white text-3xl font-bold">${price.toFixed(2)}</p>
        <div
          className={`text-sm font-mono ${
            isPositive ? 'text-green-400' : 'text-red-400'
          }`}
        >
          {isPositive ? '+' : ''}
          {change.toFixed(2)} ({isPositive ? '+' : ''}
          {changePercent.toFixed(2)}%)
        </div>
      </div>
    </div>
  );
};
