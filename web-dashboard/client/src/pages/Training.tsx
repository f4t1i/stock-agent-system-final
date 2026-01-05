/**
 * Training Page
 * 
 * Monitor and control AI agent training
 */
import { useState } from "react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { TrainingChart } from "@/components/dashboard/TrainingChart";
import {
  GraduationCap,
  Play,
  Pause,
  TrendingUp,
  Activity,
  Target,
  Zap,
  Brain,
} from "lucide-react";

function TrainingContent() {
  const [isTraining, setIsTraining] = useState(false);

  // Mock training data
  const trainingMetrics = {
    currentEpoch: 45,
    totalEpochs: 100,
    accuracy: 87.5,
    loss: 0.234,
    learningRate: 0.001,
    trainingTime: "2h 34m",
  };

  const agentMetrics = [
    { name: "News Agent", accuracy: 89.2, improvement: 5.3 },
    { name: "Technical Agent", accuracy: 91.5, improvement: 7.1 },
    { name: "Fundamental Agent", accuracy: 85.8, improvement: 4.2 },
    { name: "Strategist Agent", accuracy: 88.4, improvement: 6.5 },
    { name: "Supervisor", accuracy: 90.1, improvement: 5.8 },
  ];

  return (
    <div className="container py-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold tracking-tight">Training</h1>
          <p className="text-muted-foreground mt-2">
            Monitor and control AI agent training progress
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => setIsTraining(!isTraining)}
            className="border-border"
          >
            {isTraining ? (
              <>
                <Pause className="mr-2 h-4 w-4" />
                Pause Training
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Start Training
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Training Status */}
      <Card className="bg-card border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Training Status</CardTitle>
            <Badge
              className={
                isTraining
                  ? "bg-green-500/10 text-green-500 border-green-500/20"
                  : "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
              }
            >
              {isTraining ? "Training" : "Idle"}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">
                Epoch {trainingMetrics.currentEpoch} / {trainingMetrics.totalEpochs}
              </span>
              <span className="font-medium">
                {Math.round((trainingMetrics.currentEpoch / trainingMetrics.totalEpochs) * 100)}%
              </span>
            </div>
            <Progress
              value={(trainingMetrics.currentEpoch / trainingMetrics.totalEpochs) * 100}
              className="h-3"
            />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Accuracy</p>
              <p className="text-2xl font-bold text-green-500">
                {trainingMetrics.accuracy}%
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Loss</p>
              <p className="text-2xl font-bold">{trainingMetrics.loss}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Learning Rate</p>
              <p className="text-2xl font-bold">{trainingMetrics.learningRate}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Training Time</p>
              <p className="text-2xl font-bold">{trainingMetrics.trainingTime}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Overall Accuracy"
          value={`${trainingMetrics.accuracy}%`}
          change={5.2}
          icon={Target}
          trend="up"
        />
        <MetricCard
          title="Training Loss"
          value={trainingMetrics.loss}
          change={-12.3}
          icon={Activity}
          trend="up"
        />
        <MetricCard
          title="Epochs Completed"
          value={trainingMetrics.currentEpoch}
          icon={Zap}
        />
        <MetricCard
          title="Active Agents"
          value="5/5"
          icon={Brain}
        />
      </div>

      {/* Agent Performance */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle>Agent Performance</CardTitle>
          <CardDescription>Individual agent training metrics</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {agentMetrics.map((agent) => (
            <div key={agent.name} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-medium">{agent.name}</span>
                <div className="flex items-center gap-4">
                  <span className="text-sm text-green-500">
                    +{agent.improvement.toFixed(1)}%
                  </span>
                  <span className="font-semibold">{agent.accuracy}%</span>
                </div>
              </div>
              <Progress value={agent.accuracy} className="h-2" />
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Training History */}
      <TrainingChart />

      {/* Training Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle>Training Methods</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
              <div>
                <p className="font-medium">Supervised Fine-Tuning (SFT)</p>
                <p className="text-sm text-muted-foreground">Base model training</p>
              </div>
              <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
                Active
              </Badge>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
              <div>
                <p className="font-medium">Reinforcement Learning (RL)</p>
                <p className="text-sm text-muted-foreground">Reward-based optimization</p>
              </div>
              <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
                Active
              </Badge>
            </div>
            <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
              <div>
                <p className="font-medium">Online Learning</p>
                <p className="text-sm text-muted-foreground">Continuous adaptation</p>
              </div>
              <Badge className="bg-green-500/10 text-green-500 border-green-500/20">
                Active
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle>LLM Judge Metrics</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Reasoning Quality</span>
              <span className="font-semibold">8.7/10</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Recommendation Accuracy</span>
              <span className="font-semibold">89.3%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Confidence Calibration</span>
              <span className="font-semibold">92.1%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Consistency Score</span>
              <span className="font-semibold">87.5%</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default function Training() {
  return (
    <DashboardLayout>
      <TrainingContent />
    </DashboardLayout>
  );
}
