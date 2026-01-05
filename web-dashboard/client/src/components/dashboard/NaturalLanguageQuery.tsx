/**
 * NaturalLanguageQuery Component
 * 
 * Allows users to ask questions in natural language using OpenBB
 * Example: "Should I buy AAPL?", "What's the outlook for TSLA?"
 */
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useOpenBBQuery } from "@/hooks/useOpenBB";
import { MessageSquare, Send, Loader2, Sparkles } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface QueryResult {
  query: string;
  answer: string;
  confidence?: number;
  timestamp: Date;
}

export function NaturalLanguageQuery() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<QueryResult[]>([]);
  const openbbQuery = useOpenBBQuery();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) return;

    try {
      const response = await openbbQuery.mutateAsync({ query: query.trim() });
      
      setResults(prev => [{
        query: query.trim(),
        answer: response.answer,
        confidence: response.confidence,
        timestamp: new Date(),
      }, ...prev]);
      
      setQuery("");
    } catch (error) {
      console.error('Query failed:', error);
    }
  };

  const exampleQueries = [
    "Should I buy AAPL?",
    "What's the outlook for TSLA?",
    "Is MSFT a good investment?",
    "Compare GOOGL and META",
  ];

  return (
    <Card className="bg-card border-border">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-primary" />
          Ask OpenBB AI
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Query Input */}
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            placeholder="Ask anything about stocks... (e.g., Should I buy AAPL?)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={openbbQuery.isPending}
            className="flex-1 bg-background border-border"
          />
          <Button
            type="submit"
            disabled={openbbQuery.isPending || !query.trim()}
            className="bg-primary hover:bg-primary/90"
          >
            {openbbQuery.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </form>

        {/* Example Queries */}
        {results.length === 0 && (
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Try asking:</p>
            <div className="flex flex-wrap gap-2">
              {exampleQueries.map((example) => (
                <Button
                  key={example}
                  variant="outline"
                  size="sm"
                  onClick={() => setQuery(example)}
                  className="text-xs border-border hover:border-primary"
                >
                  {example}
                </Button>
              ))}
            </div>
          </div>
        )}

        {/* Results */}
        <AnimatePresence mode="popLayout">
          {results.map((result, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="space-y-3 p-4 rounded-lg bg-secondary/50 border border-border"
            >
              {/* Question */}
              <div className="flex items-start gap-2">
                <MessageSquare className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                <div className="flex-1">
                  <p className="font-medium text-sm">{result.query}</p>
                  <p className="text-xs text-muted-foreground">
                    {result.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>

              {/* Answer */}
              <div className="pl-6 space-y-2">
                <p className="text-sm leading-relaxed">{result.answer}</p>
                {result.confidence !== undefined && (
                  <p className="text-xs text-muted-foreground">
                    Confidence: {Math.round(result.confidence * 100)}%
                  </p>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
}
