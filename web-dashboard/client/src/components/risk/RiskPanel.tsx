import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { AlertTriangle, CheckCircle, XCircle, Info } from "lucide-react";

interface RiskCheck {
  name: string;
  status: "pass" | "warn" | "fail";
  message: string;
  value?: number;
  limit?: number;
}

interface RiskPanelProps {
  checks: RiskCheck[];
  riskLevel: "low" | "medium" | "high" | "critical";
  approved: boolean;
}

export function RiskPanel({ checks, riskLevel, approved }: RiskPanelProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "pass":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "warn":
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      case "fail":
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Info className="h-5 w-5 text-gray-500" />;
    }
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case "low":
        return "bg-green-500";
      case "medium":
        return "bg-yellow-500";
      case "high":
        return "bg-orange-500";
      case "critical":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Risk Assessment</CardTitle>
          <Badge variant={approved ? "default" : "destructive"}>
            {approved ? "Approved" : "Rejected"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Risk Level */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Risk Level</span>
            <Badge className={getRiskLevelColor(riskLevel)}>
              {riskLevel.toUpperCase()}
            </Badge>
          </div>
        </div>

        {/* Risk Checks */}
        <div className="space-y-3">
          {checks.map((check, idx) => (
            <div key={idx} className="flex items-start gap-3 p-3 rounded-lg border">
              {getStatusIcon(check.status)}
              <div className="flex-1 min-w-0">
                <div className="font-medium capitalize">{check.name.replace(/_/g, " ")}</div>
                <div className="text-sm text-muted-foreground">{check.message}</div>
                {check.value !== undefined && check.limit !== undefined && (
                  <Progress
                    value={(check.value / check.limit) * 100}
                    className="mt-2"
                  />
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
