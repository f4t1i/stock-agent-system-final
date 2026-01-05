import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { TrendingUp, TrendingDown, AlertCircle } from "lucide-react";

interface CalibrationMetrics {
  ece: number;
  mce: number;
  brier_score: number;
  accuracy: number;
}

interface CalibrationDashboardProps {
  metrics: CalibrationMetrics;
  reliabilityData?: {
    bin_centers: number[];
    bin_accuracies: number[];
    bin_counts: number[];
  };
}

export function CalibrationDashboard({ metrics, reliabilityData }: CalibrationDashboardProps) {
  const getECEStatus = (ece: number) => {
    if (ece < 0.05) return { label: "Excellent", color: "bg-green-500" };
    if (ece < 0.1) return { label: "Good", color: "bg-blue-500" };
    if (ece < 0.15) return { label: "Fair", color: "bg-yellow-500" };
    return { label: "Poor", color: "bg-red-500" };
  };

  const eceStatus = getECEStatus(metrics.ece);

  return (
    <div className="space-y-6">
      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Expected Calibration Error
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.ece.toFixed(4)}</div>
            <Badge className={`mt-2 ${eceStatus.color}`}>{eceStatus.label}</Badge>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Maximum Calibration Error
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.mce.toFixed(4)}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Brier Score
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.brier_score.toFixed(4)}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Accuracy
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(metrics.accuracy * 100).toFixed(1)}%</div>
          </CardContent>
        </Card>
      </div>

      {/* Reliability Diagram */}
      {reliabilityData && (
        <Card>
          <CardHeader>
            <CardTitle>Reliability Diagram</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {reliabilityData.bin_centers.map((center, idx) => (
                <div key={idx} className="flex items-center gap-4">
                  <div className="w-20 text-sm text-muted-foreground">
                    {(center * 100).toFixed(0)}%
                  </div>
                  <div className="flex-1">
                    <Progress
                      value={reliabilityData.bin_accuracies[idx] * 100}
                      className="h-6"
                    />
                  </div>
                  <div className="w-16 text-sm text-right">
                    {(reliabilityData.bin_accuracies[idx] * 100).toFixed(1)}%
                  </div>
                  <div className="w-16 text-xs text-muted-foreground text-right">
                    n={reliabilityData.bin_counts[idx]}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
