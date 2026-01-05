/**
 * SystemHealthMonitor Component
 * 
 * Displays real-time system health and connection status
 */
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useWebSocket } from "@/hooks/useWebSocket";
import { api } from "@/lib/api";
import { 
  Activity, 
  Database, 
  Wifi, 
  WifiOff, 
  Brain,
  TrendingUp,
  CheckCircle2,
  XCircle,
  AlertCircle
} from "lucide-react";
import { cn } from "@/lib/utils";

interface HealthStatus {
  backend: 'healthy' | 'degraded' | 'down';
  database: 'healthy' | 'degraded' | 'down';
  openbb: 'healthy' | 'degraded' | 'down';
  agents: {
    [key: string]: 'active' | 'idle' | 'error';
  };
  websocket: 'connected' | 'disconnected';
}

export function SystemHealthMonitor() {
  const { isConnected: wsConnected } = useWebSocket();
  const [health, setHealth] = useState<HealthStatus>({
    backend: 'healthy',
    database: 'healthy',
    openbb: 'healthy',
    agents: {
      'News Agent': 'active',
      'Technical Agent': 'active',
      'Fundamental Agent': 'active',
      'Strategist Agent': 'active',
      'Supervisor': 'active',
    },
    websocket: 'disconnected',
  });

  useEffect(() => {
    // Update WebSocket status
    setHealth(prev => ({
      ...prev,
      websocket: wsConnected ? 'connected' : 'disconnected',
    }));
  }, [wsConnected]);

  useEffect(() => {
    // Check backend health periodically
    const checkHealth = async () => {
      try {
        const response = await api.health();
        setHealth(prev => ({
          ...prev,
          backend: response.status === 'healthy' ? 'healthy' : 'degraded',
          database: response.database === 'connected' ? 'healthy' : 'down',
          openbb: response.openbb === 'connected' ? 'healthy' : 'down',
          agents: Object.entries(response.agents).reduce((acc, [key, value]) => ({
            ...acc,
            [key]: value === 'active' ? 'active' : 'idle',
          }), {}),
        }));
      } catch (error) {
        setHealth(prev => ({
          ...prev,
          backend: 'down',
        }));
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000); // Check every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'active':
      case 'connected':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'degraded':
      case 'idle':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case 'down':
      case 'error':
      case 'disconnected':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Activity className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const className = cn(
      "text-xs",
      status === 'healthy' || status === 'active' || status === 'connected'
        ? "bg-green-500/10 text-green-500 border-green-500/20"
        : status === 'degraded' || status === 'idle'
        ? "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
        : "bg-red-500/10 text-red-500 border-red-500/20"
    );

    return (
      <Badge variant="outline" className={className}>
        {status}
      </Badge>
    );
  };

  return (
    <Card className="bg-card border-border">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-primary" />
          System Health
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Backend Status */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
          <div className="flex items-center gap-3">
            <TrendingUp className="h-5 w-5 text-primary" />
            <div>
              <p className="font-medium">Backend API</p>
              <p className="text-xs text-muted-foreground">FastAPI Server</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {getStatusIcon(health.backend)}
            {getStatusBadge(health.backend)}
          </div>
        </div>

        {/* Database Status */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
          <div className="flex items-center gap-3">
            <Database className="h-5 w-5 text-primary" />
            <div>
              <p className="font-medium">Database</p>
              <p className="text-xs text-muted-foreground">PostgreSQL + pgvector</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {getStatusIcon(health.database)}
            {getStatusBadge(health.database)}
          </div>
        </div>

        {/* OpenBB Status */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
          <div className="flex items-center gap-3">
            <TrendingUp className="h-5 w-5 text-primary" />
            <div>
              <p className="font-medium">OpenBB Platform</p>
              <p className="text-xs text-muted-foreground">Financial Data Provider</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {getStatusIcon(health.openbb)}
            {getStatusBadge(health.openbb)}
          </div>
        </div>

        {/* WebSocket Status */}
        <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
          <div className="flex items-center gap-3">
            {wsConnected ? (
              <Wifi className="h-5 w-5 text-primary" />
            ) : (
              <WifiOff className="h-5 w-5 text-muted-foreground" />
            )}
            <div>
              <p className="font-medium">Real-Time Updates</p>
              <p className="text-xs text-muted-foreground">WebSocket Connection</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {getStatusIcon(health.websocket)}
            {getStatusBadge(health.websocket)}
          </div>
        </div>

        {/* AI Agents Status */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Brain className="h-4 w-4 text-primary" />
            <span>AI Agents</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(health.agents).map(([agent, status]) => (
              <div
                key={agent}
                className="flex items-center justify-between p-2 rounded bg-secondary/30 text-xs"
              >
                <span className="truncate">{agent}</span>
                {getStatusIcon(status)}
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
