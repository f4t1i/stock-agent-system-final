/**
 * TradingChart Component
 * 
 * Professional candlestick chart using TradingView Lightweight Charts
 */
import { useEffect, useRef, useState } from "react";
import { 
  createChart, 
  IChartApi, 
  ISeriesApi, 
  Time,
  CandlestickSeries,
  CandlestickData
} from "lightweight-charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface TradingChartProps {
  symbol: string;
  data?: CandlestickData[];
  className?: string;
}

// Generate mock candlestick data
function generateMockData(days: number = 100): CandlestickData[] {
  const data: CandlestickData[] = [];
  let basePrice = 150;
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);

  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    
    // Skip weekends
    if (date.getDay() === 0 || date.getDay() === 6) continue;

    const timestamp = Math.floor(date.getTime() / 1000) as Time;
    const open = basePrice + (Math.random() - 0.5) * 5;
    const close = open + (Math.random() - 0.5) * 10;
    const high = Math.max(open, close) + Math.random() * 5;
    const low = Math.min(open, close) - Math.random() * 5;

    data.push({
      time: timestamp,
      open,
      high,
      low,
      close,
    });

    basePrice = close;
  }

  return data;
}

function getDaysForTimeframe(tf: string): number {
  switch (tf) {
    case "1D": return 1;
    case "1W": return 7;
    case "1M": return 30;
    case "3M": return 90;
    case "1Y": return 365;
    default: return 90;
  }
}

export function TradingChart({ symbol, data, className }: TradingChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const [timeframe, setTimeframe] = useState<"1D" | "1W" | "1M" | "3M" | "1Y">("3M");
  const [chartData, setChartData] = useState<CandlestickData[]>([]);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: "transparent" },
        textColor: "#9CA3AF",
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.05)" },
        horzLines: { color: "rgba(255, 255, 255, 0.05)" },
      },
      width: chartContainerRef.current.clientWidth,
      height: 400,
      timeScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
      },
    });

    chartRef.current = chart;

    // Add candlestick series using v5 API
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#10B981",
      downColor: "#EF4444",
      borderUpColor: "#10B981",
      borderDownColor: "#EF4444",
      wickUpColor: "#10B981",
      wickDownColor: "#EF4444",
    });

    candlestickSeriesRef.current = candlestickSeries;

    // Set initial data
    const initialData = data || generateMockData(getDaysForTimeframe(timeframe));
    setChartData(initialData);
    candlestickSeries.setData(initialData);

    // Fit content
    chart.timeScale().fitContent();

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, []);

  // Update data when timeframe or external data changes
  useEffect(() => {
    if (candlestickSeriesRef.current) {
      const newData = data || generateMockData(getDaysForTimeframe(timeframe));
      setChartData(newData);
      candlestickSeriesRef.current.setData(newData);
      chartRef.current?.timeScale().fitContent();
    }
  }, [data, timeframe]);

  const handleTimeframeChange = (tf: "1D" | "1W" | "1M" | "3M" | "1Y") => {
    setTimeframe(tf);
  };

  const latestData = chartData[chartData.length - 1] || { open: 0, high: 0, low: 0, close: 0, time: 0 as Time };
  const priceChange = latestData ? latestData.close - latestData.open : 0;
  const priceChangePercent = latestData && latestData.open !== 0 ? (priceChange / latestData.open) * 100 : 0;

  return (
    <Card className={cn("bg-card border-border", className)}>
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <CardTitle>{symbol}</CardTitle>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold">
                ${latestData?.close.toFixed(2)}
              </span>
              <Badge
                className={cn(
                  "text-xs",
                  priceChange >= 0
                    ? "bg-green-500/10 text-green-500 border-green-500/20"
                    : "bg-red-500/10 text-red-500 border-red-500/20"
                )}
              >
                {priceChange >= 0 ? "+" : ""}
                {priceChange.toFixed(2)} ({priceChangePercent.toFixed(2)}%)
              </Badge>
            </div>
          </div>

          {/* Timeframe Controls */}
          <div className="flex gap-1">
            {(["1D", "1W", "1M", "3M", "1Y"] as const).map((tf) => (
              <Button
                key={tf}
                variant={timeframe === tf ? "default" : "ghost"}
                size="sm"
                onClick={() => handleTimeframeChange(tf)}
                className={cn(
                  "text-xs",
                  timeframe === tf && "bg-primary text-primary-foreground"
                )}
              >
                {tf}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div ref={chartContainerRef} className="w-full" />
        
        {/* Chart Info */}
        <div className="mt-4 grid grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Open</p>
            <p className="font-semibold">${latestData?.open.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-muted-foreground">High</p>
            <p className="font-semibold text-green-500">${latestData?.high.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Low</p>
            <p className="font-semibold text-red-500">${latestData?.low.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Close</p>
            <p className="font-semibold">${latestData?.close.toFixed(2)}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
