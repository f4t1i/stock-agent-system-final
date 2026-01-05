/**
 * TrainingChart Component
 * 
 * Displays training loss and accuracy curves using lightweight-charts
 */
import { useEffect, useRef } from "react";
import { createChart, IChartApi, Time, LineSeries } from "lightweight-charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { GraduationCap } from "lucide-react";

interface TrainingChartProps {
  className?: string;
}

// Generate mock training data
function generateTrainingData(epochs: number = 100) {
  const lossData = [];
  const accuracyData = [];
  
  for (let i = 0; i < epochs; i++) {
    // Simulate decreasing loss
    const loss = 1.0 * Math.exp(-i / 20) + Math.random() * 0.1;
    
    // Simulate increasing accuracy
    const accuracy = 100 * (1 - Math.exp(-i / 15)) + Math.random() * 2;
    
    lossData.push({
      time: i as Time,
      value: loss,
    });
    
    accuracyData.push({
      time: i as Time,
      value: Math.min(accuracy, 95), // Cap at 95%
    });
  }

  return { lossData, accuracyData };
}

export function TrainingChart({ className }: TrainingChartProps) {
  const lossChartRef = useRef<HTMLDivElement>(null);
  const accuracyChartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!lossChartRef.current || !accuracyChartRef.current) return;

    const { lossData, accuracyData } = generateTrainingData(100);

    // Create loss chart
    const lossChart = createChart(lossChartRef.current, {
      layout: {
        background: { color: "transparent" },
        textColor: "#9CA3AF",
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.05)" },
        horzLines: { color: "rgba(255, 255, 255, 0.05)" },
      },
      width: lossChartRef.current.clientWidth,
      height: 200,
      timeScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
        visible: false,
      },
      rightPriceScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
      },
    });

    const lossSeries = lossChart.addSeries(LineSeries, {
      color: "#EF4444",
      lineWidth: 2,
      title: "Loss",
    });
    lossSeries.setData(lossData);
    lossChart.timeScale().fitContent();

    // Create accuracy chart
    const accuracyChart = createChart(accuracyChartRef.current, {
      layout: {
        background: { color: "transparent" },
        textColor: "#9CA3AF",
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.05)" },
        horzLines: { color: "rgba(255, 255, 255, 0.05)" },
      },
      width: accuracyChartRef.current.clientWidth,
      height: 200,
      timeScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
        visible: true,
      },
      rightPriceScale: {
        borderColor: "rgba(255, 255, 255, 0.1)",
      },
    });

    const accuracySeries = accuracyChart.addSeries(LineSeries, {
      color: "#10B981",
      lineWidth: 2,
      title: "Accuracy",
    });
    accuracySeries.setData(accuracyData);
    accuracyChart.timeScale().fitContent();

    // Handle resize
    const handleResize = () => {
      if (lossChartRef.current && accuracyChartRef.current) {
        lossChart.applyOptions({
          width: lossChartRef.current.clientWidth,
        });
        accuracyChart.applyOptions({
          width: accuracyChartRef.current.clientWidth,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      lossChart.remove();
      accuracyChart.remove();
    };
  }, []);

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <GraduationCap className="h-5 w-5 text-primary" />
          Training Progress
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div>
          <p className="text-sm text-muted-foreground mb-2">Training Loss</p>
          <div ref={lossChartRef} className="w-full" />
        </div>
        <div>
          <p className="text-sm text-muted-foreground mb-2">Accuracy (%)</p>
          <div ref={accuracyChartRef} className="w-full" />
        </div>
      </CardContent>
    </Card>
  );
}
