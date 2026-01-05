import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { ExplainabilityCard } from "@/components/explainability/ExplainabilityCard";
import { ConfidenceGauge } from "@/components/explainability/ConfidenceGauge";
import { ReasoningVisualization } from "@/components/explainability/ReasoningVisualization";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search, Filter } from "lucide-react";

export default function Explainability() {
  const [selectedAgent, setSelectedAgent] = useState<string>("all");
  const [searchSymbol, setSearchSymbol] = useState("");

  // Fetch recent decisions
  const { data: decisions, isLoading } = trpc.explainability.listRecent.useQuery({
    limit: 20,
    agentName: selectedAgent === "all" ? undefined : selectedAgent,
    symbol: searchSymbol || undefined,
  });

  // Fetch detailed decision for first one (for demo)
  const { data: detailedDecision } = trpc.explainability.getDecision.useQuery(
    { decisionId: decisions?.[0]?.decisionId || "demo" },
    { enabled: !!decisions?.[0] }
  );

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Explainability Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Understand how AI agents make trading decisions
        </p>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Filter className="h-4 w-4" />
            Filters
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row gap-4">
            {/* Agent Filter */}
            <div className="flex-1">
              <label className="text-sm font-medium mb-2 block">Agent</label>
              <Select value={selectedAgent} onValueChange={setSelectedAgent}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Agents</SelectItem>
                  <SelectItem value="news_agent">News Agent</SelectItem>
                  <SelectItem value="technical_agent">Technical Agent</SelectItem>
                  <SelectItem value="fundamental_agent">Fundamental Agent</SelectItem>
                  <SelectItem value="strategist">Strategist</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Symbol Search */}
            <div className="flex-1">
              <label className="text-sm font-medium mb-2 block">Symbol</label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by symbol..."
                  value={searchSymbol}
                  onChange={(e) => setSearchSymbol(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>

            {/* Clear Button */}
            <div className="flex items-end">
              <Button
                variant="outline"
                onClick={() => {
                  setSelectedAgent("all");
                  setSearchSymbol("");
                }}
              >
                Clear
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Featured Decision with Detailed View */}
      {detailedDecision && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Card */}
          <div className="lg:col-span-2">
            <ExplainabilityCard
              decisionId={detailedDecision.decisionId}
              symbol={detailedDecision.symbol}
              agentName={detailedDecision.agentName}
              recommendation={detailedDecision.recommendation}
              confidence={detailedDecision.confidence}
              reasoning={detailedDecision.reasoning}
              keyFactors={detailedDecision.keyFactors}
              alternatives={detailedDecision.alternatives}
              timestamp={detailedDecision.timestamp}
            />
          </div>

          {/* Sidebar with Visualizations */}
          <div className="space-y-4">
            {/* Confidence Gauge */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Confidence</CardTitle>
              </CardHeader>
              <CardContent className="flex justify-center">
                <ConfidenceGauge
                  confidence={detailedDecision.confidence}
                  size={140}
                />
              </CardContent>
            </Card>

            {/* Factor Visualization */}
            <ReasoningVisualization
              factors={detailedDecision.keyFactors}
              title="Factor Importance"
            />
          </div>
        </div>
      )}

      {/* Decision History */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Recent Decisions</h2>
        
        {isLoading ? (
          <div className="text-center py-12 text-muted-foreground">
            Loading decisions...
          </div>
        ) : decisions && decisions.length > 0 ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {decisions.map((decision) => (
              <Card key={decision.decisionId} className="cursor-pointer hover:border-primary transition-colors">
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="font-semibold text-lg">{decision.symbol}</h3>
                      <p className="text-sm text-muted-foreground">
                        {decision.agentName.replace("_", " ")}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${
                        decision.recommendation === "BUY"
                          ? "text-green-600"
                          : decision.recommendation === "SELL"
                          ? "text-red-600"
                          : "text-yellow-600"
                      }`}>
                        {decision.recommendation}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {(decision.confidence * 100).toFixed(0)}% confidence
                      </div>
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {new Date(decision.timestamp).toLocaleString()}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="py-12 text-center text-muted-foreground">
              No decisions found. Try adjusting your filters.
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
