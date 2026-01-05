/**
 * AgentStatus Component
 * 
 * Displays the status and confidence of AI agents
 */
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Brain, TrendingUp, Newspaper, LineChart, Target } from "lucide-react";
import { cn } from "@/lib/utils";
import { AgentAnalysis } from "@/lib/api";

interface AgentStatusProps {
  agents?: AgentAnalysis[];
  className?: string;
}

const agentIcons: Record<string, typeof Brain> = {
  'News Agent': Newspaper,
  'Technical Agent': LineChart,
  'Fundamental Agent': TrendingUp,
  'Strategist Agent': Target,
  'Supervisor': Brain,
};

const recommendationColors: Record<string, string> = {
  'BUY': 'bg-green-500/10 text-green-500 border-green-500/20',
  'SELL': 'bg-red-500/10 text-red-500 border-red-500/20',
  'HOLD': 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20',
};

export function AgentStatus({ agents, className }: AgentStatusProps) {
  // Default agents if none provided
  const defaultAgents: AgentAnalysis[] = [
    {
      agent_name: 'News Agent',
      recommendation: 'HOLD',
      confidence: 0,
      reasoning: 'Waiting for analysis...',
      timestamp: new Date().toISOString(),
    },
    {
      agent_name: 'Technical Agent',
      recommendation: 'HOLD',
      confidence: 0,
      reasoning: 'Waiting for analysis...',
      timestamp: new Date().toISOString(),
    },
    {
      agent_name: 'Fundamental Agent',
      recommendation: 'HOLD',
      confidence: 0,
      reasoning: 'Waiting for analysis...',
      timestamp: new Date().toISOString(),
    },
    {
      agent_name: 'Strategist Agent',
      recommendation: 'HOLD',
      confidence: 0,
      reasoning: 'Waiting for analysis...',
      timestamp: new Date().toISOString(),
    },
    {
      agent_name: 'Supervisor',
      recommendation: 'HOLD',
      confidence: 0,
      reasoning: 'Waiting for analysis...',
      timestamp: new Date().toISOString(),
    },
  ];

  const displayAgents = agents || defaultAgents;

  return (
    <Card className={cn("bg-card border-border", className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-primary" />
          AI Agent Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {displayAgents.map((agent) => {
          const Icon = agentIcons[agent.agent_name] || Brain;
          const confidencePercent = Math.round(agent.confidence * 100);
          
          return (
            <div key={agent.agent_name} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Icon className="h-4 w-4 text-primary" />
                  <span className="text-sm font-medium text-foreground">
                    {agent.agent_name}
                  </span>
                </div>
                <Badge 
                  variant="outline" 
                  className={cn(
                    "text-xs",
                    recommendationColors[agent.recommendation]
                  )}
                >
                  {agent.recommendation}
                </Badge>
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Confidence</span>
                  <span className="font-medium text-foreground">{confidencePercent}%</span>
                </div>
                <Progress 
                  value={confidencePercent} 
                  className="h-2"
                />
              </div>
              
              {agent.reasoning && confidencePercent > 0 && (
                <p className="text-xs text-muted-foreground line-clamp-2">
                  {agent.reasoning}
                </p>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
