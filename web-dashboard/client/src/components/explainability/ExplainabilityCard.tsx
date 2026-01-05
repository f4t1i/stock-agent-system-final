import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ChevronDown, ChevronUp, TrendingUp, TrendingDown, Minus } from "lucide-react";

interface FactorImportance {
  name: string;
  importance: number;
  value: any;
  description: string;
}

interface AlternativeScenario {
  scenario: string;
  recommendation: string;
  confidence: number;
  reasoning: string;
}

interface ExplainabilityCardProps {
  decisionId: string;
  symbol: string;
  agentName: string;
  recommendation: "BUY" | "SELL" | "HOLD";
  confidence: number;
  reasoning: string;
  keyFactors: FactorImportance[];
  alternatives?: AlternativeScenario[];
  timestamp: Date;
}

export function ExplainabilityCard({
  decisionId,
  symbol,
  agentName,
  recommendation,
  confidence,
  reasoning,
  keyFactors,
  alternatives = [],
  timestamp,
}: ExplainabilityCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Get recommendation color and icon
  const getRecommendationStyle = () => {
    switch (recommendation) {
      case "BUY":
        return {
          color: "text-green-600 dark:text-green-400",
          bg: "bg-green-50 dark:bg-green-950",
          icon: <TrendingUp className="h-5 w-5" />,
        };
      case "SELL":
        return {
          color: "text-red-600 dark:text-red-400",
          bg: "bg-red-50 dark:bg-red-950",
          icon: <TrendingDown className="h-5 w-5" />,
        };
      case "HOLD":
        return {
          color: "text-yellow-600 dark:text-yellow-400",
          bg: "bg-yellow-50 dark:bg-yellow-950",
          icon: <Minus className="h-5 w-5" />,
        };
    }
  };

  // Get confidence color
  const getConfidenceColor = () => {
    if (confidence >= 0.85) return "text-emerald-600 dark:text-emerald-400";
    if (confidence >= 0.7) return "text-green-600 dark:text-green-400";
    if (confidence >= 0.5) return "text-yellow-600 dark:text-yellow-400";
    if (confidence >= 0.3) return "text-orange-600 dark:text-orange-400";
    return "text-red-600 dark:text-red-400";
  };

  // Get confidence label
  const getConfidenceLabel = () => {
    if (confidence >= 0.85) return "Very High";
    if (confidence >= 0.7) return "High";
    if (confidence >= 0.5) return "Medium";
    if (confidence >= 0.3) return "Low";
    return "Very Low";
  };

  // Format agent name
  const formatAgentName = (name: string) => {
    return name
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  const style = getRecommendationStyle();

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${style.bg}`}>
              <div className={style.color}>{style.icon}</div>
            </div>
            <div>
              <CardTitle className="text-lg">
                {symbol} - {formatAgentName(agentName)}
              </CardTitle>
              <CardDescription>
                {new Date(timestamp).toLocaleString()}
              </CardDescription>
            </div>
          </div>
          <Badge variant="outline" className={`${style.color} font-semibold`}>
            {recommendation}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Confidence Meter */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">Confidence</span>
            <span className={`font-semibold ${getConfidenceColor()}`}>
              {(confidence * 100).toFixed(0)}% - {getConfidenceLabel()}
            </span>
          </div>
          <Progress value={confidence * 100} className="h-2" />
        </div>

        {/* Reasoning */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Reasoning</h4>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {reasoning}
          </p>
        </div>

        {/* Key Factors (Top 3 when collapsed) */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium">Key Factors</h4>
          <div className="space-y-2">
            {keyFactors.slice(0, isExpanded ? undefined : 3).map((factor, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">{factor.name}</span>
                  <span className="text-muted-foreground">
                    {(factor.importance * 100).toFixed(0)}%
                  </span>
                </div>
                <Progress value={factor.importance * 100} className="h-1.5" />
                <p className="text-xs text-muted-foreground">
                  {factor.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Expand/Collapse Button */}
        {(keyFactors.length > 3 || alternatives.length > 0) && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="h-4 w-4 mr-2" />
                Show Less
              </>
            ) : (
              <>
                <ChevronDown className="h-4 w-4 mr-2" />
                Show More
              </>
            )}
          </Button>
        )}

        {/* Alternative Scenarios (when expanded) */}
        {isExpanded && alternatives.length > 0 && (
          <div className="space-y-2 pt-2 border-t">
            <h4 className="text-sm font-medium">Alternative Scenarios</h4>
            <div className="space-y-3">
              {alternatives.map((alt, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-lg bg-muted/50 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{alt.scenario}</span>
                    <Badge variant="secondary">{alt.recommendation}</Badge>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {alt.reasoning}
                  </p>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-muted-foreground">Confidence:</span>
                    <span className="font-medium">
                      {(alt.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Decision ID (when expanded) */}
        {isExpanded && (
          <div className="pt-2 border-t">
            <p className="text-xs text-muted-foreground">
              Decision ID: <code className="text-xs">{decisionId}</code>
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
