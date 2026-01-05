import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { useLocation } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, BarChart3, Brain, TrendingUp } from "lucide-react";

export default function Home() {
  const { user, isAuthenticated } = useAuth();
  const [, setLocation] = useLocation();

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Hero Section */}
      <div className="container py-12">
        <div className="flex flex-col items-center text-center space-y-6 mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20">
            <Brain className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-primary">AI-Powered Stock Analysis</span>
          </div>
          
          <h1 className="text-5xl font-bold tracking-tight">
            Stock Agent Dashboard
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl">
            Professional multi-agent AI system for comprehensive stock analysis with 5 specialized agents: 
            News, Technical, Fundamental, Strategist, and Supervisor.
          </p>

          {isAuthenticated ? (
            <div className="flex gap-4 mt-8">
              <Button 
                size="lg" 
                className="bg-primary hover:bg-primary/90 text-primary-foreground"
                onClick={() => setLocation('/dashboard')}
              >
                <Activity className="w-5 h-5 mr-2" />
                View Dashboard
              </Button>
              <Button 
                size="lg" 
                variant="outline"
                onClick={() => setLocation('/dashboard')}
              >
                <BarChart3 className="w-5 h-5 mr-2" />
                Analyze Stock
              </Button>
            </div>
          ) : (
            <Button 
              size="lg" 
              className="bg-primary hover:bg-primary/90 text-primary-foreground mt-8"
              onClick={() => setLocation('/dashboard')}
            >
              Get Started
            </Button>
          )}
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-16">
          <Card className="bg-card border-border hover:border-primary/50 transition-colors">
            <CardHeader>
              <Brain className="w-10 h-10 text-primary mb-2" />
              <CardTitle>5 AI Agents</CardTitle>
              <CardDescription>
                Specialized agents for news, technical, fundamental analysis, strategy, and supervision
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="bg-card border-border hover:border-primary/50 transition-colors">
            <CardHeader>
              <TrendingUp className="w-10 h-10 text-primary mb-2" />
              <CardTitle>Real-Time Analysis</CardTitle>
              <CardDescription>
                Live market data integration with OpenBB for professional financial insights
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="bg-card border-border hover:border-primary/50 transition-colors">
            <CardHeader>
              <BarChart3 className="w-10 h-10 text-primary mb-2" />
              <CardTitle>Advanced Charts</CardTitle>
              <CardDescription>
                TradingView integration for professional candlestick charts and technical indicators
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="bg-card border-border hover:border-primary/50 transition-colors">
            <CardHeader>
              <Activity className="w-10 h-10 text-primary mb-2" />
              <CardTitle>Training System</CardTitle>
              <CardDescription>
                Continuous learning with SFT, RL, and online learning for improved predictions
              </CardDescription>
            </CardHeader>
          </Card>
        </div>

        {/* System Status */}
        <div className="mt-16">
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle>System Status</CardTitle>
              <CardDescription>Current agent and backend status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                  <div>
                    <p className="text-sm font-medium">Backend API</p>
                    <p className="text-xs text-muted-foreground">FastAPI Running</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                  <div>
                    <p className="text-sm font-medium">AI Agents</p>
                    <p className="text-xs text-muted-foreground">5/5 Active</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                  <div>
                    <p className="text-sm font-medium">Data Provider</p>
                    <p className="text-xs text-muted-foreground">OpenBB Connected</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
