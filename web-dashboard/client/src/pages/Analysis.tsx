/**
 * Analysis Page
 * 
 * Detailed stock analysis with agent insights and charts
 */
import { useState } from "react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AgentStatus } from "@/components/dashboard/AgentStatus";
import { RecommendationCard } from "@/components/dashboard/RecommendationCard";
import { useAnalysisStore } from "@/stores/analysisStore";
import { api } from "@/lib/api";
import {
  Search,
  Loader2,
  TrendingUp,
  Newspaper,
  LineChart,
  Target,
  Brain,
  Calendar,
  DollarSign,
  FileDown,
  FileText,
} from "lucide-react";
import { toast } from "sonner";
import { exportAnalysisToPDF, exportAnalysisToCSV } from "@/lib/export";

function AnalysisContent() {
  const [searchSymbol, setSearchSymbol] = useState("");
  const {
    currentAnalysis,
    analysisHistory,
    isAnalyzing,
    setCurrentAnalysis,
    setIsAnalyzing,
    addToHistory,
    setError,
  } = useAnalysisStore();

  const handleAnalyze = async (symbol: string) => {
    if (!symbol.trim()) {
      toast.error("Please enter a stock symbol");
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const result = await api.analyzeStock({ symbol: symbol.toUpperCase() });
      setCurrentAnalysis(result);
      addToHistory(result);
      toast.success(`Analysis complete for ${symbol.toUpperCase()}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Analysis failed";
      setError(message);
      toast.error(message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="container py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold tracking-tight">Stock Analysis</h1>
        <p className="text-muted-foreground mt-2">
          Deep dive into stock analysis with AI-powered insights from 5 specialized agents
        </p>
      </div>

      {/* Search */}
      <Card className="bg-card border-border">
        <CardContent className="pt-6">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Enter stock symbol (e.g., AAPL, TSLA, MSFT)"
                value={searchSymbol}
                onChange={(e) => setSearchSymbol(e.target.value.toUpperCase())}
                onKeyDown={(e) => e.key === 'Enter' && handleAnalyze(searchSymbol)}
                className="pl-10 bg-background border-border"
              />
            </div>
            <Button onClick={() => handleAnalyze(searchSymbol)} disabled={isAnalyzing}>
                {isAnalyzing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Search className="mr-2 h-4 w-4" />
                    Analyze
                  </>
                )}
              </Button>
              {currentAnalysis && (
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      exportAnalysisToPDF({
                        symbol: currentAnalysis.symbol,
                        date: new Date().toLocaleDateString(),
                        recommendation: currentAnalysis.supervisor_recommendation || "HOLD",
                        confidence: Math.round(currentAnalysis.confidence * 100),
                        targetPrice: undefined,
                        newsAgentResult: currentAnalysis.agent_analyses.find(a => a.agent_name === "News Agent")?.reasoning,
                        technicalAgentResult: currentAnalysis.agent_analyses.find(a => a.agent_name === "Technical Agent")?.reasoning,
                        fundamentalAgentResult: currentAnalysis.agent_analyses.find(a => a.agent_name === "Fundamental Agent")?.reasoning,
                        strategistAgentResult: currentAnalysis.agent_analyses.find(a => a.agent_name === "Strategist Agent")?.reasoning,
                        supervisorResult: currentAnalysis.agent_analyses.find(a => a.agent_name === "Supervisor")?.reasoning,
                      });
                      toast.success("PDF exported successfully");
                    }}
                  >
                    <FileDown className="mr-2 h-4 w-4" />
                    Export PDF
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      exportAnalysisToCSV([{
                        symbol: currentAnalysis.symbol,
                        date: new Date().toLocaleDateString(),
                        recommendation: currentAnalysis.supervisor_recommendation || "HOLD",
                        confidence: Math.round(currentAnalysis.confidence * 100),
                        targetPrice: undefined,
                      }]);
                      toast.success("CSV exported successfully");
                    }}
                  >
                    <FileText className="mr-2 h-4 w-4" />
                    Export CSV
                  </Button>
                </div>
              )}
          </div>
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {currentAnalysis && (
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="bg-secondary">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="agents">Agent Details</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1">
                <AgentStatus agents={currentAnalysis.agent_analyses} />
              </div>
              <div className="lg:col-span-2">
                <RecommendationCard analysis={currentAnalysis} />
              </div>
            </div>

            {/* Stock Info */}
            <Card className="bg-card border-border">
              <CardHeader>
                <CardTitle>Stock Information</CardTitle>
                <CardDescription>{currentAnalysis.symbol}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Calendar className="h-4 w-4" />
                      <span className="text-sm">Analysis Date</span>
                    </div>
                    <p className="text-lg font-semibold">
                      {new Date(currentAnalysis.timestamp).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <Brain className="h-4 w-4" />
                      <span className="text-sm">Confidence Score</span>
                    </div>
                    <p className="text-lg font-semibold">
                      {Math.round(currentAnalysis.confidence * 100)}%
                    </p>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-muted-foreground">
                      <TrendingUp className="h-4 w-4" />
                      <span className="text-sm">Recommendation</span>
                    </div>
                    <Badge
                      className={
                        currentAnalysis.supervisor_recommendation === 'BUY'
                          ? 'bg-green-500/10 text-green-500 border-green-500/20'
                          : currentAnalysis.supervisor_recommendation === 'SELL'
                          ? 'bg-red-500/10 text-red-500 border-red-500/20'
                          : 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20'
                      }
                    >
                      {currentAnalysis.supervisor_recommendation}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="agents" className="space-y-6">
            {currentAnalysis.agent_analyses.map((agent) => {
              const icons: Record<string, typeof Brain> = {
                'News Agent': Newspaper,
                'Technical Agent': LineChart,
                'Fundamental Agent': DollarSign,
                'Strategist Agent': Target,
                'Supervisor': Brain,
              };
              const Icon = icons[agent.agent_name] || Brain;

              return (
                <Card key={agent.agent_name} className="bg-card border-border">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-primary/10">
                          <Icon className="h-5 w-5 text-primary" />
                        </div>
                        <CardTitle>{agent.agent_name}</CardTitle>
                      </div>
                      <Badge
                        className={
                          agent.recommendation === 'BUY'
                            ? 'bg-green-500/10 text-green-500 border-green-500/20'
                            : agent.recommendation === 'SELL'
                            ? 'bg-red-500/10 text-red-500 border-red-500/20'
                            : 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20'
                        }
                      >
                        {agent.recommendation}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Confidence</p>
                      <p className="text-2xl font-bold">
                        {Math.round(agent.confidence * 100)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground mb-2">Reasoning</p>
                      <p className="text-sm leading-relaxed">{agent.reasoning}</p>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </TabsContent>

          <TabsContent value="history" className="space-y-4">
            {analysisHistory.length === 0 ? (
              <Card className="bg-card border-border">
                <CardContent className="py-12 text-center text-muted-foreground">
                  No analysis history yet. Analyze stocks to build your history.
                </CardContent>
              </Card>
            ) : (
              analysisHistory.map((analysis, index) => (
                <Card
                  key={index}
                  className="bg-card border-border hover:border-primary/50 transition-colors cursor-pointer"
                  onClick={() => setCurrentAnalysis(analysis)}
                >
                  <CardContent className="py-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className="text-xl font-bold">{analysis.symbol}</div>
                        <Badge
                          className={
                            analysis.supervisor_recommendation === 'BUY'
                              ? 'bg-green-500/10 text-green-500 border-green-500/20'
                              : analysis.supervisor_recommendation === 'SELL'
                              ? 'bg-red-500/10 text-red-500 border-red-500/20'
                              : 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20'
                          }
                        >
                          {analysis.supervisor_recommendation}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {Math.round(analysis.confidence * 100)}% confidence
                        </span>
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {new Date(analysis.timestamp).toLocaleString()}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </TabsContent>
        </Tabs>
      )}

      {/* Empty State */}
      {!currentAnalysis && (
        <Card className="bg-card border-border">
          <CardContent className="py-16 text-center space-y-4">
            <Brain className="h-16 w-16 text-primary mx-auto" />
            <h3 className="text-xl font-semibold">No Analysis Yet</h3>
            <p className="text-muted-foreground max-w-md mx-auto">
              Enter a stock symbol above to get started with AI-powered analysis from our 5
              specialized agents.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default function Analysis() {
  return (
    <DashboardLayout>
      <AnalysisContent />
    </DashboardLayout>
  );
}
