/**
 * Portfolio Page
 * 
 * View and manage stock portfolio with performance metrics
 */
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { PortfolioChart } from "@/components/dashboard/PortfolioChart";
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  BarChart3,
  Plus,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";

// Mock portfolio data
const portfolioData = {
  totalValue: 125430.50,
  totalCost: 112000,
  totalReturn: 13430.50,
  returnPercent: 11.99,
  positions: [
    {
      symbol: "AAPL",
      name: "Apple Inc.",
      shares: 50,
      avgCost: 150.25,
      currentPrice: 178.50,
      totalValue: 8925,
      totalCost: 7512.50,
      gain: 1412.50,
      gainPercent: 18.80,
    },
    {
      symbol: "MSFT",
      name: "Microsoft Corporation",
      shares: 30,
      avgCost: 320.00,
      currentPrice: 365.25,
      totalValue: 10957.50,
      totalCost: 9600,
      gain: 1357.50,
      gainPercent: 14.14,
    },
    {
      symbol: "GOOGL",
      name: "Alphabet Inc.",
      shares: 40,
      avgCost: 125.50,
      currentPrice: 138.75,
      totalValue: 5550,
      totalCost: 5020,
      gain: 530,
      gainPercent: 10.56,
    },
    {
      symbol: "TSLA",
      name: "Tesla Inc.",
      shares: 25,
      avgCost: 245.00,
      currentPrice: 238.50,
      totalValue: 5962.50,
      totalCost: 6125,
      gain: -162.50,
      gainPercent: -2.65,
    },
    {
      symbol: "NVDA",
      name: "NVIDIA Corporation",
      shares: 20,
      avgCost: 450.00,
      currentPrice: 495.75,
      totalValue: 9915,
      totalCost: 9000,
      gain: 915,
      gainPercent: 10.17,
    },
  ],
};

function PortfolioContent() {
  return (
    <div className="container py-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold tracking-tight">Portfolio</h1>
          <p className="text-muted-foreground mt-2">
            Track your stock holdings and performance
          </p>
        </div>
        <Button className="bg-primary hover:bg-primary/90">
          <Plus className="mr-2 h-4 w-4" />
          Add Position
        </Button>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Value"
          value={`$${portfolioData.totalValue.toLocaleString()}`}
          icon={DollarSign}
        />
        <MetricCard
          title="Total Return"
          value={`$${portfolioData.totalReturn.toLocaleString()}`}
          change={portfolioData.returnPercent}
          icon={TrendingUp}
          trend="up"
        />
        <MetricCard
          title="Total Cost"
          value={`$${portfolioData.totalCost.toLocaleString()}`}
          icon={BarChart3}
        />
        <MetricCard
          title="Positions"
          value={portfolioData.positions.length}
          icon={BarChart3}
        />
      </div>

      {/* Holdings */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle>Holdings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Table Header */}
            <div className="grid grid-cols-7 gap-4 pb-2 border-b border-border text-sm font-medium text-muted-foreground">
              <div className="col-span-2">Stock</div>
              <div className="text-right">Shares</div>
              <div className="text-right">Avg Cost</div>
              <div className="text-right">Current</div>
              <div className="text-right">Value</div>
              <div className="text-right">Gain/Loss</div>
            </div>

            {/* Table Rows */}
            {portfolioData.positions.map((position) => (
              <div
                key={position.symbol}
                className="grid grid-cols-7 gap-4 py-3 border-b border-border/50 hover:bg-secondary/50 rounded-lg transition-colors"
              >
                <div className="col-span-2">
                  <div className="font-semibold">{position.symbol}</div>
                  <div className="text-xs text-muted-foreground">{position.name}</div>
                </div>
                <div className="text-right">{position.shares}</div>
                <div className="text-right">${position.avgCost.toFixed(2)}</div>
                <div className="text-right">${position.currentPrice.toFixed(2)}</div>
                <div className="text-right font-semibold">
                  ${position.totalValue.toLocaleString()}
                </div>
                <div className="text-right">
                  <div
                    className={
                      position.gain >= 0 ? "text-green-500" : "text-red-500"
                    }
                  >
                    <div className="flex items-center justify-end gap-1">
                      {position.gain >= 0 ? (
                        <ArrowUpRight className="h-3 w-3" />
                      ) : (
                        <ArrowDownRight className="h-3 w-3" />
                      )}
                      <span className="font-semibold">
                        ${Math.abs(position.gain).toFixed(2)}
                      </span>
                    </div>
                    <div className="text-xs">
                      {position.gain >= 0 ? "+" : ""}
                      {position.gainPercent.toFixed(2)}%
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Performance Chart */}
      <PortfolioChart />
    </div>
  );
}

export default function Portfolio() {
  return (
    <DashboardLayout>
      <PortfolioContent />
    </DashboardLayout>
  );
}
