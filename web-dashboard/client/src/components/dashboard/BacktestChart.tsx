/**
 * BacktestChart Component
 * 
 * Displays backtest equity curve using lightweight-charts
 */
import { useEffect, useRef } from "react";
import { createChart, IChartApi, Time, LineSeries } from "lightweight-charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { History } from "lucide-react";

interface BacktestChartProps {
  data?: Array<{ time: Time; value: number }>;
  className?: string;
}

// Generate mock backtest equity curve
function generateBacktestData(days: number = 365) {
  const data = [];
  let equity = 10000; // Starting capital
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);

  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    
    // Skip weekends
    if (date.getDay() === 0 || date.getDay() === 6) continue;

    const timestamp = Math.floor(date.getTime() / 1000) as Time;
    
    // Simulate trading with wins and losses
    if (Math.random() > 0.3) { // 70% win rate
      equity += Math.random() * 300 + 100; // Win
    } else {
      equity -= Math.random() * 200 + 50; // Loss
    }

    data.push({
      time: timestamp,
      value: Math.max(equity, 5000), // Don't go below 5k
    });
  }

  return data;
}

export function BacktestChart({ data, className }: BacktestChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

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
      height: 300,
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

    // Add line series for equity curve
    const lineSeries = chart.addSeries(LineSeries, {
      color: "#10B981",
      lineWidth: 2,
    });

    // Set data
    const chartData = data || generateBacktestData(365);
    lineSeries.setData(chartData);

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
  }, [data]);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <History className="h-5 w-5 text-primary" />
          Equity Curve
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div ref={chartContainerRef} className="w-full" />
      </CardContent>
    </Card>
  );
}
