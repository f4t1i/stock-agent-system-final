/**
 * Settings Page
 * 
 * Configure dashboard and system settings
 */
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Settings as SettingsIcon,
  Save,
  Database,
  Zap,
  Bell,
  Shield,
} from "lucide-react";

function SettingsContent() {
  return (
    <div className="container py-8 space-y-8 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground mt-2">
          Configure your dashboard and system preferences
        </p>
      </div>

      {/* API Configuration */}
      <Card className="bg-card border-border">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            <CardTitle>API Configuration</CardTitle>
          </div>
          <CardDescription>
            Configure connection to FastAPI backend
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="apiUrl">Backend API URL</Label>
            <Input
              id="apiUrl"
              placeholder="http://localhost:8000"
              defaultValue="http://localhost:8000"
              className="bg-background border-border"
            />
            <p className="text-xs text-muted-foreground">
              The base URL for your FastAPI backend server
            </p>
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
            <div>
              <p className="font-medium">Connection Status</p>
              <p className="text-sm text-muted-foreground">Backend API health</p>
            </div>
            <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
              Connected
            </Badge>
          </div>
          <Button className="bg-primary hover:bg-primary/90">
            <Save className="mr-2 h-4 w-4" />
            Save API Settings
          </Button>
        </CardContent>
      </Card>

      {/* Database Settings */}
      <Card className="bg-card border-border">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5 text-primary" />
            <CardTitle>Database Settings</CardTitle>
          </div>
          <CardDescription>
            Configure database connections and storage
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
              <div>
                <p className="font-medium">PostgreSQL + pgvector</p>
                <p className="text-sm text-muted-foreground">Vector database for embeddings</p>
              </div>
              <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
                Active
              </Badge>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
              <div>
                <p className="font-medium">SQLite</p>
                <p className="text-sm text-muted-foreground">Local development database</p>
              </div>
              <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
                Active
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Notification Settings */}
      <Card className="bg-card border-border">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Bell className="h-5 w-5 text-primary" />
            <CardTitle>Notifications</CardTitle>
          </div>
          <CardDescription>
            Manage notification preferences
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Analysis Complete</p>
                <p className="text-sm text-muted-foreground">
                  Notify when stock analysis is complete
                </p>
              </div>
              <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
                Enabled
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Training Updates</p>
                <p className="text-sm text-muted-foreground">
                  Notify on training milestones
                </p>
              </div>
              <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
                Enabled
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Portfolio Alerts</p>
                <p className="text-sm text-muted-foreground">
                  Notify on significant portfolio changes
                </p>
              </div>
              <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
                Enabled
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Agent Configuration */}
      <Card className="bg-card border-border">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-primary" />
            <CardTitle>AI Agent Configuration</CardTitle>
          </div>
          <CardDescription>
            Configure AI agent behavior and parameters
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="confidence">Minimum Confidence Threshold</Label>
            <Input
              id="confidence"
              type="number"
              placeholder="0.7"
              defaultValue="0.7"
              min="0"
              max="1"
              step="0.1"
              className="bg-background border-border"
            />
            <p className="text-xs text-muted-foreground">
              Minimum confidence score for recommendations (0-1)
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="agents">Active Agents</Label>
            <div className="grid grid-cols-2 gap-2">
              {["News Agent", "Technical Agent", "Fundamental Agent", "Strategist Agent", "Supervisor"].map((agent) => (
                <div
                  key={agent}
                  className="flex items-center justify-between p-2 rounded-lg bg-secondary/50"
                >
                  <span className="text-sm">{agent}</span>
                  <Badge className="bg-green-500/10 text-green-500 border-green-500/20 text-xs">
                    Active
                  </Badge>
                </div>
              ))}
            </div>
          </div>
          <Button className="bg-primary hover:bg-primary/90">
            <Save className="mr-2 h-4 w-4" />
            Save Agent Settings
          </Button>
        </CardContent>
      </Card>

      {/* System Info */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle>System Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Dashboard Version</span>
            <span className="font-medium">1.0.0</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Backend Version</span>
            <span className="font-medium">1.0.0</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">OpenBB Integration</span>
            <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
              Active
            </Badge>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">LangGraph Version</span>
            <span className="font-medium">0.2.0</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function Settings() {
  return (
    <DashboardLayout>
      <SettingsContent />
    </DashboardLayout>
  );
}
