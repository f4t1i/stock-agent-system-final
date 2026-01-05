import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Bell, BellOff, TrendingUp, TrendingDown, Edit, Trash2 } from "lucide-react";

interface Alert {
  alert_id: string;
  symbol: string;
  alert_type: string;
  condition: string;
  threshold: number;
  notification_channels: string[];
  enabled: boolean;
  last_triggered?: Date;
  created_at: Date;
}

interface AlertsPanelProps {
  alerts: Alert[];
  onToggle: (alertId: string, enabled: boolean) => void;
  onEdit: (alertId: string) => void;
  onDelete: (alertId: string) => void;
  onCreateNew: () => void;
}

export function AlertsPanel({
  alerts,
  onToggle,
  onEdit,
  onDelete,
  onCreateNew,
}: AlertsPanelProps) {
  const [filter, setFilter] = useState<"all" | "active" | "triggered">("all");

  // Filter alerts
  const filteredAlerts = alerts.filter((alert) => {
    if (filter === "active") return alert.enabled;
    if (filter === "triggered") return alert.last_triggered;
    return true;
  });

  // Format alert type
  const formatAlertType = (type: string) => {
    return type
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  // Format condition
  const formatCondition = (condition: string) => {
    const map: Record<string, string> = {
      above: "Above",
      below: "Below",
      crosses_above: "Crosses Above",
      crosses_below: "Crosses Below",
      equals: "Equals",
    };
    return map[condition] || condition;
  };

  // Get condition icon
  const getConditionIcon = (condition: string) => {
    if (condition.includes("above")) {
      return <TrendingUp className="h-4 w-4" />;
    }
    return <TrendingDown className="h-4 w-4" />;
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Price Alerts</CardTitle>
            <CardDescription>
              {alerts.length} alert{alerts.length !== 1 ? "s" : ""} configured
            </CardDescription>
          </div>
          <Button onClick={onCreateNew}>
            <Bell className="h-4 w-4 mr-2" />
            New Alert
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Filter Tabs */}
        <div className="flex gap-2 border-b pb-2">
          <Button
            variant={filter === "all" ? "default" : "ghost"}
            size="sm"
            onClick={() => setFilter("all")}
          >
            All ({alerts.length})
          </Button>
          <Button
            variant={filter === "active" ? "default" : "ghost"}
            size="sm"
            onClick={() => setFilter("active")}
          >
            Active ({alerts.filter((a) => a.enabled).length})
          </Button>
          <Button
            variant={filter === "triggered" ? "default" : "ghost"}
            size="sm"
            onClick={() => setFilter("triggered")}
          >
            Triggered ({alerts.filter((a) => a.last_triggered).length})
          </Button>
        </div>

        {/* Alert List */}
        <div className="space-y-3">
          {filteredAlerts.length > 0 ? (
            filteredAlerts.map((alert) => (
              <div
                key={alert.alert_id}
                className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
              >
                {/* Left: Alert Info */}
                <div className="flex items-center gap-4 flex-1">
                  {/* Icon */}
                  <div className={`p-2 rounded-lg ${
                    alert.enabled ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"
                  }`}>
                    {alert.enabled ? (
                      <Bell className="h-5 w-5" />
                    ) : (
                      <BellOff className="h-5 w-5" />
                    )}
                  </div>

                  {/* Details */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold">{alert.symbol}</span>
                      <Badge variant="outline" className="text-xs">
                        {formatAlertType(alert.alert_type)}
                      </Badge>
                      {alert.last_triggered && (
                        <Badge variant="secondary" className="text-xs">
                          Triggered
                        </Badge>
                      )}
                    </div>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      {getConditionIcon(alert.condition)}
                      <span>
                        {formatCondition(alert.condition)} ${alert.threshold.toFixed(2)}
                      </span>
                    </div>
                    {alert.last_triggered && (
                      <div className="text-xs text-muted-foreground mt-1">
                        Last triggered: {new Date(alert.last_triggered).toLocaleString()}
                      </div>
                    )}
                  </div>
                </div>

                {/* Right: Actions */}
                <div className="flex items-center gap-3">
                  {/* Toggle */}
                  <div className="flex items-center gap-2">
                    <Switch
                      checked={alert.enabled}
                      onCheckedChange={(checked) => onToggle(alert.alert_id, checked)}
                    />
                    <span className="text-sm text-muted-foreground">
                      {alert.enabled ? "On" : "Off"}
                    </span>
                  </div>

                  {/* Edit */}
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => onEdit(alert.alert_id)}
                  >
                    <Edit className="h-4 w-4" />
                  </Button>

                  {/* Delete */}
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => onDelete(alert.alert_id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              {filter === "all" ? (
                <>
                  <Bell className="h-12 w-12 mx-auto mb-4 opacity-20" />
                  <p>No alerts configured</p>
                  <Button onClick={onCreateNew} className="mt-4">
                    Create Your First Alert
                  </Button>
                </>
              ) : (
                <p>No {filter} alerts</p>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
