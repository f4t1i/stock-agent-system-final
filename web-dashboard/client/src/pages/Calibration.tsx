import { trpc } from "@/lib/trpc";
import { CalibrationDashboard } from "@/components/calibration/CalibrationDashboard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { RefreshCw } from "lucide-react";

export default function Calibration() {
  const { data: metrics, refetch: refetchMetrics } = trpc.calibration.getMetrics.useQuery();
  const { data: reliabilityData } = trpc.calibration.getReliabilityDiagram.useQuery();

  return (
    <div className="container mx-auto py-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Confidence Calibration</h1>
          <p className="text-muted-foreground mt-2">
            Monitor and improve AI confidence accuracy
          </p>
        </div>
        <Button onClick={() => refetchMetrics()}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {metrics && (
        <CalibrationDashboard
          metrics={metrics}
          reliabilityData={reliabilityData}
        />
      )}

      <Card>
        <CardHeader>
          <CardTitle>About Calibration</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground space-y-2">
          <p>
            <strong>Expected Calibration Error (ECE):</strong> Measures how well confidence scores match actual accuracy. Lower is better.
          </p>
          <p>
            <strong>Maximum Calibration Error (MCE):</strong> Worst-case calibration error across all confidence bins.
          </p>
          <p>
            <strong>Brier Score:</strong> Measures prediction accuracy. Lower is better.
          </p>
          <p>
            <strong>Reliability Diagram:</strong> Shows how predicted confidence aligns with actual accuracy. Bars should match the confidence level.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
