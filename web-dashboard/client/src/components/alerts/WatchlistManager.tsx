import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Eye, Plus, X, Search } from "lucide-react";

interface Watchlist {
  watchlist_id: string;
  name: string;
  description?: string;
  symbol_count: number;
}

interface WatchlistSymbol {
  symbol: string;
  added_at: Date;
}

interface WatchlistManagerProps {
  watchlists: Watchlist[];
  onCreateWatchlist: (name: string, description?: string) => void;
  onDeleteWatchlist: (watchlistId: string) => void;
  onAddSymbol: (watchlistId: string, symbol: string) => void;
  onRemoveSymbol: (watchlistId: string, symbol: string) => void;
  onViewWatchlist: (watchlistId: string) => void;
}

export function WatchlistManager({
  watchlists,
  onCreateWatchlist,
  onDeleteWatchlist,
  onAddSymbol,
  onRemoveSymbol,
  onViewWatchlist,
}: WatchlistManagerProps) {
  const [newWatchlistName, setNewWatchlistName] = useState("");
  const [showCreateForm, setShowCreateForm] = useState(false);

  const handleCreate = () => {
    if (newWatchlistName.trim()) {
      onCreateWatchlist(newWatchlistName.trim());
      setNewWatchlistName("");
      setShowCreateForm(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Watchlists</CardTitle>
          <Button
            size="sm"
            onClick={() => setShowCreateForm(!showCreateForm)}
          >
            <Plus className="h-4 w-4 mr-2" />
            New Watchlist
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Create Form */}
        {showCreateForm && (
          <div className="flex gap-2 p-4 border rounded-lg bg-muted/50">
            <Input
              placeholder="Watchlist name..."
              value={newWatchlistName}
              onChange={(e) => setNewWatchlistName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            />
            <Button onClick={handleCreate}>Create</Button>
            <Button
              variant="ghost"
              onClick={() => {
                setShowCreateForm(false);
                setNewWatchlistName("");
              }}
            >
              Cancel
            </Button>
          </div>
        )}

        {/* Watchlist Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {watchlists.map((watchlist) => (
            <div
              key={watchlist.watchlist_id}
              className="p-4 border rounded-lg hover:border-primary transition-colors cursor-pointer"
              onClick={() => onViewWatchlist(watchlist.watchlist_id)}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <h3 className="font-semibold">{watchlist.name}</h3>
                  {watchlist.description && (
                    <p className="text-sm text-muted-foreground mt-1">
                      {watchlist.description}
                    </p>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteWatchlist(watchlist.watchlist_id);
                  }}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="secondary">
                  {watchlist.symbol_count} symbol{watchlist.symbol_count !== 1 ? "s" : ""}
                </Badge>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    onViewWatchlist(watchlist.watchlist_id);
                  }}
                >
                  <Eye className="h-4 w-4 mr-1" />
                  View
                </Button>
              </div>
            </div>
          ))}
        </div>

        {watchlists.length === 0 && !showCreateForm && (
          <div className="text-center py-12 text-muted-foreground">
            <Search className="h-12 w-12 mx-auto mb-4 opacity-20" />
            <p>No watchlists yet</p>
            <Button
              onClick={() => setShowCreateForm(true)}
              className="mt-4"
            >
              Create Your First Watchlist
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
