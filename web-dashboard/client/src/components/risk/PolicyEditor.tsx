import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";

interface PolicyEditorProps {
  onSave: (policy: any) => void;
  onCancel: () => void;
}

export function PolicyEditor({ onSave, onCancel }: PolicyEditorProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Edit Risk Policy</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label>Max Position Size (%)</Label>
          <Input type="number" defaultValue="10" />
        </div>
        <div className="space-y-2">
          <Label>Min Confidence</Label>
          <Input type="number" step="0.01" defaultValue="0.6" />
        </div>
        <div className="space-y-2">
          <Label>Max Volatility</Label>
          <Input type="number" step="0.01" defaultValue="0.5" />
        </div>
        <div className="flex items-center space-x-2">
          <Switch id="enabled" defaultChecked />
          <Label htmlFor="enabled">Policy Enabled</Label>
        </div>
        <div className="flex gap-2 pt-4">
          <Button onClick={() => onSave({})}>Save</Button>
          <Button variant="outline" onClick={onCancel}>Cancel</Button>
        </div>
      </CardContent>
    </Card>
  );
}
