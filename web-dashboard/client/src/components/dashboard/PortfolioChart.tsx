/**
 * PortfolioChart Component
 * 
 * Displays portfolio performance over time using lightweight-charts
 */
import { useEffect, useRef } from "react";
import { createChart, IChartApi, Time, AreaSeries } from "lightweight-charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";

interface PortfolioChartProps {
  className?: string;
}

// Generate mock portfolio performance data
function generatePortfolioData(days: number = 90) {
  const data = [];
  let value = 100000; // Starting portfolio value
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - days);

  for (let i = 0; i < days; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    
    // Skip weekends
    if (date.getDay() === 0 || date.getDay() === 6) continue;

    const timestamp = Math.floor(date.getTime() / 1000) as Time;
    // Simulate portfolio growth with some volatility
    value += (Math.random() - 0.4) * 1000 + 200; // Slight upward bias

    data.push({
      time: timestamp,
      value: Math.max(value, 80000), // Don't go below 80k
    });
  }

  return data;
}

export function PortfolioChart({ className }: PortfolioChartProps) {
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

    // Add area series for portfolio value
    const areaSeries = chart.addSeries(AreaSeries, {
      lineColor: "#00D4FF",
      topColor: "rgba(0, 212, 255, 0.4)",
      bottomColor: "rgba(0, 212, 255, 0.0)",
      lineWidth: 2,
    });

    // Set data
    const chartData = generatePortfolioData(90);
    areaSeries.setData(chartData);

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

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-primary" />
          Portfolio Performance
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div ref={chartContainerRef} className="w-full" />
      </CardContent>
    </Card>
  );
}
