import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";

interface Factor {
  name: string;
  importance: number;
  value: any;
  description: string;
}

interface ReasoningVisualizationProps {
  factors: Factor[];
  title?: string;
}

export function ReasoningVisualization({
  factors,
  title = "Decision Factors",
}: ReasoningVisualizationProps) {
  // Sort factors by importance
  const sortedFactors = [...factors].sort((a, b) => b.importance - a.importance);

  // Get color for importance level
  const getImportanceColor = (importance: number) => {
    if (importance >= 0.8) return "bg-emerald-500";
    if (importance >= 0.6) return "bg-green-500";
    if (importance >= 0.4) return "bg-yellow-500";
    if (importance >= 0.2) return "bg-orange-500";
    return "bg-red-500";
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {sortedFactors.map((factor, idx) => (
            <div key={idx} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">{factor.name}</span>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <Info className="h-3.5 w-3.5 text-muted-foreground" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-xs text-xs">{factor.description}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">
                    {typeof factor.value === "number"
                      ? factor.value.toFixed(2)
                      : factor.value}
                  </span>
                  <span className="text-xs font-medium">
                    {(factor.importance * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Horizontal bar chart */}
              <div className="relative h-6 bg-muted rounded-md overflow-hidden">
                <div
                  className={`h-full ${getImportanceColor(
                    factor.importance
                  )} transition-all duration-500 ease-out`}
                  style={{ width: `${factor.importance * 100}%` }}
                />
              </div>
            </div>
          ))}

          {sortedFactors.length === 0 && (
            <p className="text-sm text-muted-foreground text-center py-4">
              No factors available
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
