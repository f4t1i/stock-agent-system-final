/**
 * MetricCard Component
 * 
 * Displays key metrics with trend indicators
 */
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowDown, ArrowUp, LucideIcon, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon?: LucideIcon;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export function MetricCard({
  title,
  value,
  change,
  changeLabel,
  icon: Icon,
  trend,
  className,
}: MetricCardProps) {
  // Determine trend from change if not explicitly provided
  const effectiveTrend = trend || (change && change > 0 ? 'up' : change && change < 0 ? 'down' : 'neutral');
  
  const TrendIcon = effectiveTrend === 'up' ? ArrowUp : effectiveTrend === 'down' ? ArrowDown : Minus;
  
  const trendColor = 
    effectiveTrend === 'up' ? 'text-green-500' : 
    effectiveTrend === 'down' ? 'text-red-500' : 
    'text-muted-foreground';

  return (
    <Card className={cn("bg-card border-border hover:border-primary/50 transition-colors", className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        {Icon && <Icon className="h-4 w-4 text-primary" />}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold text-foreground">{value}</div>
        {(change !== undefined || changeLabel) && (
          <div className={cn("flex items-center gap-1 text-xs mt-1", trendColor)}>
            <TrendIcon className="h-3 w-3" />
            <span>
              {change !== undefined && (
                <span className="font-medium">
                  {change > 0 ? '+' : ''}{change.toFixed(2)}%
                </span>
              )}
              {changeLabel && <span className="ml-1 text-muted-foreground">{changeLabel}</span>}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
