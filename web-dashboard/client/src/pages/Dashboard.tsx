/**
 * Dashboard Page
 * 
 * Main dashboard showing portfolio overview, agent status, and recommendations
 */
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { AgentStatus } from "@/components/dashboard/AgentStatus";
import { RecommendationCard } from "@/components/dashboard/RecommendationCard";
import { SystemHealthMonitor } from "@/components/dashboard/SystemHealthMonitor";
import { RecentTrades } from "@/components/dashboard/RecentTrades";
import { TradingChart } from "@/components/dashboard/TradingChart";
import { NaturalLanguageQuery } from "@/components/dashboard/NaturalLanguageQuery";
import { ExplainabilityCard } from "@/components/explainability/ExplainabilityCard";
import { trpc } from "@/lib/trpc";
import { Link } from "wouter";
import { ArrowRight } from "lucide-react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { useAnalysisStore } from "@/stores/analysisStore";
import { useWebSocket } from "@/hooks/useWebSocket";
import { api } from "@/lib/api";
import { motion } from "framer-motion";
import { 
  DollarSign, 
  TrendingUp, 
  Activity, 
  BarChart3, 
  Search,
  Loader2,
  Mic,
  Wifi
} from "lucide-react";
import { toast } from "sonner";

function RecentDecisionsSection() {
  // Fetch recent decisions
  const { data: decisions, isLoading } = trpc.explainability.listRecent.useQuery({
    limit: 3,
  });

  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
      >
        <Card>
          <CardHeader>
            <CardTitle>Recent Decisions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  if (!decisions || decisions.length === 0) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.35 }}
    >
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Recent Decisions</CardTitle>
            <Link href="/explainability">
              <Button variant="ghost" size="sm">
                View All
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {decisions.map((decision) => (
              <ExplainabilityCard
                key={decision.decisionId}
                decisionId={decision.decisionId}
                symbol={decision.symbol}
                agentName={decision.agentName}
                recommendation={decision.recommendation as "BUY" | "SELL" | "HOLD"}
                confidence={decision.confidence}
                reasoning="Click to view full explanation"
                keyFactors={[]}
                timestamp={new Date(decision.timestamp)}
              />
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function DashboardContent() {
  const [searchSymbol, setSearchSymbol] = useState("AAPL");
  const [isListening, setIsListening] = useState(false);
  const { isConnected: wsConnected } = useWebSocket();
  
  const {
    currentAnalysis,
    isAnalyzing,
    setCurrentAnalysis,
    setIsAnalyzing,
    addToHistory,
    setError,
  } = useAnalysisStore();

  // Handle stock analysis
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

  // Voice input support
  const handleVoiceInput = () => {
    if (!('webkitSpeechRecognition' in window)) {
      toast.error("Voice input not supported in this browser");
      return;
    }

    const recognition = new (window as any).webkitSpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
      setIsListening(true);
      toast.info("Listening... Speak a stock symbol");
    };

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setSearchSymbol(transcript.toUpperCase());
      toast.success(`Heard: ${transcript}`);
    };

    recognition.onerror = () => {
      setIsListening(false);
      toast.error("Voice input failed");
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.start();
  };

  return (
    <motion.div 
      className="container py-8 space-y-8"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header */}
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-4xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground mt-2">
            AI-powered stock analysis with 5 specialized agents
          </p>
        </div>

        {/* WebSocket Status */}
        {wsConnected && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-green-500/10 border border-green-500/20 text-green-500 text-sm w-fit">
              <Wifi className="h-4 w-4" />
              <span>Real-time updates active</span>
              <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            </div>
          </motion.div>
        )}
      </div>

      {/* Metrics */}
      <motion.div 
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
      >
        <MetricCard
          title="Portfolio Value"
          value="$125,430"
          change={12.5}
          icon={DollarSign}
          trend="up"
        />
        <MetricCard
          title="Today's P&L"
          value="$2,340"
          change={3.2}
          icon={TrendingUp}
          trend="up"
        />
        <MetricCard
          title="Active Positions"
          value="12"
          icon={BarChart3}
        />
        <MetricCard
          title="Win Rate"
          value="68%"
          change={5.1}
          icon={Activity}
          trend="up"
        />
      </motion.div>

      {/* Search */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
      >
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
              <Button
                variant="outline"
                size="icon"
                onClick={handleVoiceInput}
                disabled={isListening}
                className="border-border"
              >
                <Mic className={isListening ? "h-4 w-4 animate-pulse text-primary" : "h-4 w-4"} />
              </Button>
              <Button
                onClick={() => handleAnalyze(searchSymbol)}
                disabled={isAnalyzing}
                className="bg-primary hover:bg-primary/90 min-w-[140px]"
              >
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
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Natural Language Query */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <NaturalLanguageQuery />
      </motion.div>

      {/* TradingView Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.25 }}
      >
        <TradingChart symbol={searchSymbol || "AAPL"} />
      </motion.div>

      {/* Analysis Results */}
      {currentAnalysis && (
        <motion.div 
          className="grid grid-cols-1 lg:grid-cols-3 gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="lg:col-span-1">
            <AgentStatus agents={currentAnalysis.agent_analyses} />
          </div>
          <div className="lg:col-span-2">
            <RecommendationCard analysis={currentAnalysis} />
          </div>
        </motion.div>
      )}

      {/* Empty State */}
      {!currentAnalysis && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Card className="bg-card border-border">
            <CardContent className="py-16 text-center space-y-4">
              <BarChart3 className="h-16 w-16 text-primary mx-auto" />
              <h3 className="text-xl font-semibold">No Analysis Yet</h3>
              <p className="text-muted-foreground max-w-md mx-auto">
                Enter a stock symbol above to get started with AI-powered analysis from our 5
                specialized agents.
              </p>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Recent Decisions - Explainability */}
      <RecentDecisionsSection />

      {/* System Health & Recent Trades */}
      <motion.div 
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <SystemHealthMonitor />
        <RecentTrades />
      </motion.div>
    </motion.div>
  );
}

export default function Dashboard() {
  return (
    <DashboardLayout>
      <DashboardContent />
    </DashboardLayout>
  );
}
