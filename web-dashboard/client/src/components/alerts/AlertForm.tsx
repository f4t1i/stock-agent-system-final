import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";

interface AlertFormProps {
  onSubmit: (alert: {
    symbol: string;
    alert_type: string;
    condition: string;
    threshold: number;
    notification_channels: string[];
  }) => void;
  onCancel: () => void;
}

export function AlertForm({ onSubmit, onCancel }: AlertFormProps) {
  const [symbol, setSymbol] = useState("");
  const [alertType, setAlertType] = useState("price_threshold");
  const [condition, setCondition] = useState("crosses_above");
  const [threshold, setThreshold] = useState("");
  const [channels, setChannels] = useState<string[]>(["push"]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!symbol || !threshold) return;

    onSubmit({
      symbol: symbol.toUpperCase(),
      alert_type: alertType,
      condition,
      threshold: parseFloat(threshold),
      notification_channels: channels,
    });
  };

  const toggleChannel = (channel: string) => {
    setChannels((prev) =>
      prev.includes(channel)
        ? prev.filter((c) => c !== channel)
        : [...prev, channel]
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Create New Alert</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Symbol */}
          <div className="space-y-2">
            <Label htmlFor="symbol">Symbol</Label>
            <Input
              id="symbol"
              placeholder="AAPL"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              required
            />
          </div>

          {/* Alert Type */}
          <div className="space-y-2">
            <Label htmlFor="alert-type">Alert Type</Label>
            <Select value={alertType} onValueChange={setAlertType}>
              <SelectTrigger id="alert-type">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="price_threshold">Price Threshold</SelectItem>
                <SelectItem value="confidence_change">Confidence Change</SelectItem>
                <SelectItem value="recommendation_change">Recommendation Change</SelectItem>
                <SelectItem value="technical_signal">Technical Signal</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Condition */}
          <div className="space-y-2">
            <Label htmlFor="condition">Condition</Label>
            <Select value={condition} onValueChange={setCondition}>
              <SelectTrigger id="condition">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="above">Above</SelectItem>
                <SelectItem value="below">Below</SelectItem>
                <SelectItem value="crosses_above">Crosses Above</SelectItem>
                <SelectItem value="crosses_below">Crosses Below</SelectItem>
                <SelectItem value="equals">Equals</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Threshold */}
          <div className="space-y-2">
            <Label htmlFor="threshold">Threshold</Label>
            <Input
              id="threshold"
              type="number"
              step="0.01"
              placeholder="150.00"
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
              required
            />
          </div>

          {/* Notification Channels */}
          <div className="space-y-2">
            <Label>Notification Channels</Label>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="push"
                  checked={channels.includes("push")}
                  onCheckedChange={() => toggleChannel("push")}
                />
                <label htmlFor="push" className="text-sm cursor-pointer">
                  Push Notification
                </label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="email"
                  checked={channels.includes("email")}
                  onCheckedChange={() => toggleChannel("email")}
                />
                <label htmlFor="email" className="text-sm cursor-pointer">
                  Email
                </label>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-2 pt-4">
            <Button type="submit" className="flex-1">
              Create Alert
            </Button>
            <Button type="button" variant="outline" onClick={onCancel}>
              Cancel
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
