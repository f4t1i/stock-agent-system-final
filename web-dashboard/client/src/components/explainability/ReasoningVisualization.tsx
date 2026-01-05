import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info, TrendingUp, TrendingDown, Minus } from "lucide-react";

interface Factor {
  name: string;
  importance: number;
  value: any;
  description: string;
}

interface ConfidencePoint {
  timestamp: Date;
  confidence: number;
  label?: string;
}

interface DecisionNode {
  id: string;
  label: string;
  value?: string;
  children?: DecisionNode[];
  isLeaf?: boolean;
}

interface ReasoningVisualizationProps {
  factors: Factor[];
  confidenceHistory?: ConfidencePoint[];
  decisionTree?: DecisionNode;
  title?: string;
}

export function ReasoningVisualization({
  factors,
  confidenceHistory = [],
  decisionTree,
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

  // Render decision tree node
  const renderTreeNode = (node: DecisionNode, depth: number = 0) => {
    const indent = depth * 24;
    
    return (
      <div key={node.id} className="space-y-2">
        <div
          className="flex items-center gap-2 p-2 rounded-md hover:bg-muted/50 transition-colors"
          style={{ marginLeft: `${indent}px` }}
        >
          {!node.isLeaf && (
            <div className="w-4 h-4 rounded-full border-2 border-primary bg-background" />
          )}
          {node.isLeaf && (
            <div className="w-4 h-4 rounded-full bg-primary" />
          )}
          <span className="text-sm font-medium">{node.label}</span>
          {node.value && (
            <span className="text-xs text-muted-foreground ml-auto">
              {node.value}
            </span>
          )}
        </div>
        {node.children && node.children.length > 0 && (
          <div className="space-y-1">
            {node.children.map((child) => renderTreeNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="factors" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="factors">Factors</TabsTrigger>
            <TabsTrigger value="timeline">Timeline</TabsTrigger>
            <TabsTrigger value="tree">Decision Tree</TabsTrigger>
          </TabsList>

          {/* Factor Importance Bar Chart */}
          <TabsContent value="factors" className="space-y-4 mt-4">
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
          </TabsContent>

          {/* Confidence Timeline Graph */}
          <TabsContent value="timeline" className="mt-4">
            {confidenceHistory.length > 0 ? (
              <div className="space-y-4">
                <div className="relative h-48 border rounded-md p-4">
                  {/* Y-axis labels */}
                  <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-xs text-muted-foreground pr-2">
                    <span>100%</span>
                    <span>75%</span>
                    <span>50%</span>
                    <span>25%</span>
                    <span>0%</span>
                  </div>

                  {/* Graph area */}
                  <div className="ml-8 h-full relative">
                    {/* Grid lines */}
                    {[0, 25, 50, 75, 100].map((percent) => (
                      <div
                        key={percent}
                        className="absolute left-0 right-0 border-t border-muted"
                        style={{ top: `${100 - percent}%` }}
                      />
                    ))}

                    {/* Line chart */}
                    <svg className="absolute inset-0 w-full h-full">
                      <polyline
                        fill="none"
                        stroke="hsl(var(--primary))"
                        strokeWidth="2"
                        points={confidenceHistory
                          .map((point, idx) => {
                            const x = (idx / (confidenceHistory.length - 1)) * 100;
                            const y = 100 - point.confidence * 100;
                            return `${x}%,${y}%`;
                          })
                          .join(" ")}
                      />
                      {/* Data points */}
                      {confidenceHistory.map((point, idx) => {
                        const x = (idx / (confidenceHistory.length - 1)) * 100;
                        const y = 100 - point.confidence * 100;
                        return (
                          <TooltipProvider key={idx}>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <circle
                                  cx={`${x}%`}
                                  cy={`${y}%`}
                                  r="4"
                                  fill="hsl(var(--primary))"
                                  className="cursor-pointer hover:r-6 transition-all"
                                />
                              </TooltipTrigger>
                              <TooltipContent>
                                <div className="text-xs space-y-1">
                                  <p className="font-medium">
                                    {(point.confidence * 100).toFixed(1)}%
                                  </p>
                                  <p className="text-muted-foreground">
                                    {new Date(point.timestamp).toLocaleString()}
                                  </p>
                                  {point.label && <p>{point.label}</p>}
                                </div>
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        );
                      })}
                    </svg>
                  </div>
                </div>

                {/* Timeline legend */}
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>
                    {new Date(confidenceHistory[0].timestamp).toLocaleDateString()}
                  </span>
                  <span>
                    {new Date(
                      confidenceHistory[confidenceHistory.length - 1].timestamp
                    ).toLocaleDateString()}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-8">
                No confidence history available
              </p>
            )}
          </TabsContent>

          {/* Decision Tree Visualization */}
          <TabsContent value="tree" className="mt-4">
            {decisionTree ? (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {renderTreeNode(decisionTree)}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-8">
                No decision tree available
              </p>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
