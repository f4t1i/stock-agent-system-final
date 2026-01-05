/**
 * RecommendationCard Component
 * 
 * Displays AI recommendation with confidence and reasoning
 */
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TrendingUp, TrendingDown, Minus, Clock, Brain } from "lucide-react";
import { cn } from "@/lib/utils";
import { StockAnalysisResponse } from "@/lib/api";

interface RecommendationCardProps {
  analysis?: StockAnalysisResponse;
  className?: string;
  onAnalyze?: () => void;
}

export function RecommendationCard({ 
  analysis, 
  className,
  onAnalyze 
}: RecommendationCardProps) {
  if (!analysis) {
    return (
      <Card className={cn("bg-card border-border", className)}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            AI Recommendation
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center py-8 space-y-4">
          <p className="text-muted-foreground text-center">
            No analysis available. Analyze a stock to see AI recommendations.
          </p>
          {onAnalyze && (
            <Button onClick={onAnalyze} className="bg-primary hover:bg-primary/90">
              Analyze Stock
            </Button>
          )}
        </CardContent>
      </Card>
    );
  }

  const { supervisor_recommendation, confidence, reasoning, symbol, timestamp } = analysis;
  
  const RecommendationIcon = 
    supervisor_recommendation === 'BUY' ? TrendingUp :
    supervisor_recommendation === 'SELL' ? TrendingDown :
    Minus;

  const recommendationColor = 
    supervisor_recommendation === 'BUY' ? 'text-green-500 bg-green-500/10 border-green-500/20' :
    supervisor_recommendation === 'SELL' ? 'text-red-500 bg-red-500/10 border-red-500/20' :
    'text-yellow-500 bg-yellow-500/10 border-yellow-500/20';

  const confidencePercent = Math.round(confidence * 100);
  const confidenceColor = 
    confidencePercent >= 80 ? 'text-green-500' :
    confidencePercent >= 60 ? 'text-yellow-500' :
    'text-red-500';

  const timeAgo = new Date(timestamp).toLocaleString();

  return (
    <Card className={cn("bg-card border-border", className)}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            AI Recommendation
          </div>
          <Badge variant="outline" className="text-xs">
            {symbol}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Main Recommendation */}
        <div className="flex items-center justify-between p-4 rounded-lg bg-secondary/50 border border-border">
          <div className="flex items-center gap-3">
            <div className={cn(
              "p-3 rounded-full",
              supervisor_recommendation === 'BUY' ? 'bg-green-500/20' :
              supervisor_recommendation === 'SELL' ? 'bg-red-500/20' :
              'bg-yellow-500/20'
            )}>
              <RecommendationIcon className={cn("h-6 w-6", recommendationColor.split(' ')[0])} />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Action</p>
              <p className="text-2xl font-bold text-foreground">
                {supervisor_recommendation}
              </p>
            </div>
          </div>
          
          <div className="text-right">
            <p className="text-sm text-muted-foreground">Confidence</p>
            <p className={cn("text-2xl font-bold", confidenceColor)}>
              {confidencePercent}%
            </p>
          </div>
        </div>

        {/* Reasoning */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-foreground">Analysis Reasoning</h4>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {reasoning}
          </p>
        </div>

        {/* Timestamp */}
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" />
          <span>Analyzed: {timeAgo}</span>
        </div>

        {/* Agent Consensus */}
        <div className="pt-4 border-t border-border">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Agent Consensus</span>
            <div className="flex gap-1">
              {analysis.agent_analyses.map((agent) => (
                <Badge
                  key={agent.agent_name}
                  variant="outline"
                  className={cn(
                    "text-xs px-2 py-0.5",
                    agent.recommendation === 'BUY' ? 'bg-green-500/10 text-green-500 border-green-500/20' :
                    agent.recommendation === 'SELL' ? 'bg-red-500/10 text-red-500 border-red-500/20' :
                    'bg-yellow-500/10 text-yellow-500 border-yellow-500/20'
                  )}
                >
                  {agent.recommendation}
                </Badge>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
