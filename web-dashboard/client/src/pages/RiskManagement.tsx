import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { RiskPanel } from "@/components/risk/RiskPanel";
import { PolicyEditor } from "@/components/risk/PolicyEditor";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";

export default function RiskManagement() {
  const [showPolicyEditor, setShowPolicyEditor] = useState(false);
  const [testTrade, setTestTrade] = useState({
    symbol: "AAPL",
    action: "BUY" as const,
    quantity: 100,
    price: 150,
    confidence: 0.85,
  });
  const { toast } = useToast();

  const evaluateTrade = trpc.risk.evaluateTrade.useMutation({
    onSuccess: (result) => {
      toast({
        title: result.approved ? "Trade Approved" : "Trade Rejected",
        description: `Risk Level: ${result.risk_level}`,
      });
    },
  });

  const handleEvaluate = () => {
    evaluateTrade.mutate(testTrade);
  };

  return (
    <div className="container mx-auto py-6 space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Risk Management</h1>
        <p className="text-muted-foreground mt-2">
          Configure trading policies and evaluate risk
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Test Trade */}
        <Card>
          <CardHeader>
            <CardTitle>Test Trade Evaluation</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Symbol</Label>
                <Input
                  value={testTrade.symbol}
                  onChange={(e) => setTestTrade({ ...testTrade, symbol: e.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label>Quantity</Label>
                <Input
                  type="number"
                  value={testTrade.quantity}
                  onChange={(e) => setTestTrade({ ...testTrade, quantity: parseInt(e.target.value) })}
                />
              </div>
              <div className="space-y-2">
                <Label>Price</Label>
                <Input
                  type="number"
                  value={testTrade.price}
                  onChange={(e) => setTestTrade({ ...testTrade, price: parseFloat(e.target.value) })}
                />
              </div>
              <div className="space-y-2">
                <Label>Confidence</Label>
                <Input
                  type="number"
                  step="0.01"
                  value={testTrade.confidence}
                  onChange={(e) => setTestTrade({ ...testTrade, confidence: parseFloat(e.target.value) })}
                />
              </div>
            </div>
            <Button onClick={handleEvaluate} className="w-full">
              Evaluate Trade
            </Button>
          </CardContent>
        </Card>

        {/* Risk Result */}
        {evaluateTrade.data && (
          <RiskPanel
            checks={evaluateTrade.data.checks}
            riskLevel={evaluateTrade.data.risk_level}
            approved={evaluateTrade.data.approved}
          />
        )}
      </div>

      {/* Policy Editor */}
      {showPolicyEditor && (
        <PolicyEditor
          onSave={() => {
            setShowPolicyEditor(false);
            toast({ title: "Policy saved" });
          }}
          onCancel={() => setShowPolicyEditor(false)}
        />
      )}

      {!showPolicyEditor && (
        <Button onClick={() => setShowPolicyEditor(true)}>
          Edit Risk Policies
        </Button>
      )}
    </div>
  );
}
