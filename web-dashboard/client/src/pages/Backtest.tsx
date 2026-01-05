/**
 * Backtest Page
 * 
 * Run historical backtests on AI trading strategies
 */
import { useState } from "react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { BacktestChart } from "@/components/dashboard/BacktestChart";
import {
  History,
  Play,
  TrendingUp,
  DollarSign,
  Target,
  Activity,
  Calendar,
} from "lucide-react";

function BacktestContent() {
  const [symbol, setSymbol] = useState("AAPL");
  const [startDate, setStartDate] = useState("2023-01-01");
  const [endDate, setEndDate] = useState("2024-01-01");
  const [initialCapital, setInitialCapital] = useState("10000");
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleRunBacktest = () => {
    setIsRunning(true);
    // Simulate backtest
    setTimeout(() => {
      setResults({
        symbol,
        startDate,
        endDate,
        initialCapital: parseFloat(initialCapital),
        finalCapital: 12543.75,
        totalReturn: 25.44,
        totalTrades: 24,
        winningTrades: 16,
        losingTrades: 8,
        winRate: 66.67,
        sharpeRatio: 1.85,
        maxDrawdown: -8.5,
      });
      setIsRunning(false);
    }, 2000);
  };

  return (
    <div className="container py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold tracking-tight">Backtest</h1>
        <p className="text-muted-foreground mt-2">
          Test AI trading strategies on historical data
        </p>
      </div>

      {/* Configuration */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle>Backtest Configuration</CardTitle>
          <CardDescription>
            Set parameters for historical strategy testing
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="symbol">Stock Symbol</Label>
              <Input
                id="symbol"
                placeholder="AAPL"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="bg-background border-border"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="capital">Initial Capital ($)</Label>
              <Input
                id="capital"
                type="number"
                placeholder="10000"
                value={initialCapital}
                onChange={(e) => setInitialCapital(e.target.value)}
                className="bg-background border-border"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="startDate">Start Date</Label>
              <Input
                id="startDate"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="bg-background border-border"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="endDate">End Date</Label>
              <Input
                id="endDate"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="bg-background border-border"
              />
            </div>
          </div>
          <Button
            onClick={handleRunBacktest}
            disabled={isRunning}
            className="bg-primary hover:bg-primary/90 w-full md:w-auto"
          >
            <Play className="mr-2 h-4 w-4" />
            {isRunning ? "Running Backtest..." : "Run Backtest"}
          </Button>
        </CardContent>
      </Card>

      {/* Results */}
      {results && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard
              title="Final Capital"
              value={`$${results.finalCapital.toLocaleString()}`}
              icon={DollarSign}
            />
            <MetricCard
              title="Total Return"
              value={`${results.totalReturn.toFixed(2)}%`}
              change={results.totalReturn}
              icon={TrendingUp}
              trend="up"
            />
            <MetricCard
              title="Win Rate"
              value={`${results.winRate.toFixed(1)}%`}
              icon={Target}
            />
            <MetricCard
              title="Sharpe Ratio"
              value={results.sharpeRatio.toFixed(2)}
              icon={Activity}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Trade Statistics */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Trade Statistics</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Total Trades</span>
                  <span className="font-semibold">{results.totalTrades}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Winning Trades</span>
                  <span className="font-semibold text-green-500">
                    {results.winningTrades}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Losing Trades</span>
                  <span className="font-semibold text-red-500">
                    {results.losingTrades}
                  </span>
                </div>
                <div className="flex items-center justify-between pt-4 border-t border-border">
                  <span className="text-muted-foreground">Max Drawdown</span>
                  <span className="font-semibold text-red-500">
                    {results.maxDrawdown.toFixed(2)}%
                  </span>
                </div>
              </CardContent>
            </Card>

            {/* Period Info */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Test Period</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Symbol</span>
                  <span className="font-semibold">{results.symbol}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Start Date</span>
                  <span className="font-semibold">{results.startDate}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">End Date</span>
                  <span className="font-semibold">{results.endDate}</span>
                </div>
                <div className="flex items-center justify-between pt-4 border-t border-border">
                  <span className="text-muted-foreground">Initial Capital</span>
                  <span className="font-semibold">
                    ${results.initialCapital.toLocaleString()}
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Equity Curve */}
          <BacktestChart />
        </>
      )}

      {/* Empty State */}
      {!results && (
        <Card className="bg-card border-border">
          <CardContent className="py-16 text-center space-y-4">
            <History className="h-16 w-16 text-primary mx-auto" />
            <h3 className="text-xl font-semibold">No Backtest Results</h3>
            <p className="text-muted-foreground max-w-md mx-auto">
              Configure the parameters above and run a backtest to see how the AI trading
              strategy would have performed historically.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default function Backtest() {
  return (
    <DashboardLayout>
      <BacktestContent />
    </DashboardLayout>
  );
}
