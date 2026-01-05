/**
 * RecentTrades Component
 * 
 * Displays recent trading activity and recommendations
 */
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowUpRight, ArrowDownRight, Clock } from "lucide-react";
import { cn } from "@/lib/utils";

interface Trade {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  price: number;
  quantity: number;
  timestamp: string;
  confidence: number;
  profit?: number;
}

// Mock trades data
const mockTrades: Trade[] = [
  {
    id: '1',
    symbol: 'AAPL',
    action: 'BUY',
    price: 178.50,
    quantity: 10,
    timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
    confidence: 0.87,
    profit: 125.50,
  },
  {
    id: '2',
    symbol: 'TSLA',
    action: 'SELL',
    price: 238.50,
    quantity: 5,
    timestamp: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
    confidence: 0.92,
    profit: -45.20,
  },
  {
    id: '3',
    symbol: 'MSFT',
    action: 'BUY',
    price: 365.25,
    quantity: 8,
    timestamp: new Date(Date.now() - 1000 * 60 * 120).toISOString(),
    confidence: 0.85,
    profit: 234.80,
  },
  {
    id: '4',
    symbol: 'GOOGL',
    action: 'HOLD',
    price: 138.75,
    quantity: 15,
    timestamp: new Date(Date.now() - 1000 * 60 * 180).toISOString(),
    confidence: 0.78,
  },
  {
    id: '5',
    symbol: 'NVDA',
    action: 'BUY',
    price: 495.75,
    quantity: 6,
    timestamp: new Date(Date.now() - 1000 * 60 * 240).toISOString(),
    confidence: 0.91,
    profit: 345.60,
  },
];

export function RecentTrades() {
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else if (diffMins < 1440) {
      return `${Math.floor(diffMins / 60)}h ago`;
    } else {
      return `${Math.floor(diffMins / 1440)}d ago`;
    }
  };

  return (
    <Card className="bg-card border-border">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clock className="h-5 w-5 text-primary" />
          Recent Activity
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {mockTrades.map((trade) => (
            <div
              key={trade.id}
              className="flex items-center justify-between p-3 rounded-lg bg-secondary/50 hover:bg-secondary/70 transition-colors"
            >
              <div className="flex items-center gap-3 flex-1">
                {/* Action Badge */}
                <Badge
                  className={cn(
                    "text-xs font-semibold min-w-[60px] justify-center",
                    trade.action === 'BUY'
                      ? "bg-green-500/10 text-green-500 border-green-500/20"
                      : trade.action === 'SELL'
                      ? "bg-red-500/10 text-red-500 border-red-500/20"
                      : "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
                  )}
                >
                  {trade.action}
                </Badge>

                {/* Symbol and Details */}
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">{trade.symbol}</span>
                    <span className="text-xs text-muted-foreground">
                      {trade.quantity} @ ${trade.price.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span>{formatTime(trade.timestamp)}</span>
                    <span>•</span>
                    <span>{Math.round(trade.confidence * 100)}% confidence</span>
                  </div>
                </div>

                {/* Profit/Loss */}
                {trade.profit !== undefined && (
                  <div className="flex items-center gap-1">
                    {trade.profit >= 0 ? (
                      <>
                        <ArrowUpRight className="h-4 w-4 text-green-500" />
                        <span className="text-sm font-semibold text-green-500">
                          +${trade.profit.toFixed(2)}
                        </span>
                      </>
                    ) : (
                      <>
                        <ArrowDownRight className="h-4 w-4 text-red-500" />
                        <span className="text-sm font-semibold text-red-500">
                          ${trade.profit.toFixed(2)}
                        </span>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
