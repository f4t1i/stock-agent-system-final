import { useState } from "react";
import { trpc } from "@/lib/trpc";
import { AlertsPanel } from "@/components/alerts/AlertsPanel";
import { WatchlistManager } from "@/components/alerts/WatchlistManager";
import { AlertForm } from "@/components/alerts/AlertForm";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";

export default function Alerts() {
  const [showAlertForm, setShowAlertForm] = useState(false);
  const { toast } = useToast();

  // Queries
  const { data: alerts = [], refetch: refetchAlerts } = trpc.alerts.listAlerts.useQuery();
  const { data: watchlists = [], refetch: refetchWatchlists } = trpc.watchlist.listWatchlists.useQuery();

  // Mutations
  const createAlert = trpc.alerts.createAlert.useMutation({
    onSuccess: () => {
      refetchAlerts();
      setShowAlertForm(false);
      toast({ title: "Alert created successfully" });
    },
  });

  const toggleAlert = trpc.alerts.toggleAlert.useMutation({
    onSuccess: () => {
      refetchAlerts();
      toast({ title: "Alert updated" });
    },
  });

  const deleteAlert = trpc.alerts.deleteAlert.useMutation({
    onSuccess: () => {
      refetchAlerts();
      toast({ title: "Alert deleted" });
    },
  });

  const createWatchlist = trpc.watchlist.createWatchlist.useMutation({
    onSuccess: () => {
      refetchWatchlists();
      toast({ title: "Watchlist created" });
    },
  });

  const deleteWatchlist = trpc.watchlist.deleteWatchlist.useMutation({
    onSuccess: () => {
      refetchWatchlists();
      toast({ title: "Watchlist deleted" });
    },
  });

  const addSymbol = trpc.watchlist.addSymbol.useMutation({
    onSuccess: () => {
      refetchWatchlists();
      toast({ title: "Symbol added" });
    },
  });

  const removeSymbol = trpc.watchlist.removeSymbol.useMutation({
    onSuccess: () => {
      refetchWatchlists();
      toast({ title: "Symbol removed" });
    },
  });

  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Alerts & Watchlists</h1>
        <p className="text-muted-foreground mt-2">
          Monitor stocks and get notified when conditions are met
        </p>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="alerts" className="space-y-4">
        <TabsList>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="watchlists">Watchlists</TabsTrigger>
        </TabsList>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-4">
          {showAlertForm ? (
            <AlertForm
              onSubmit={(alert) => createAlert.mutate(alert)}
              onCancel={() => setShowAlertForm(false)}
            />
          ) : (
            <AlertsPanel
              alerts={alerts}
              onToggle={(alertId, enabled) => toggleAlert.mutate({ alert_id: alertId, enabled })}
              onEdit={(alertId) => {
                // TODO: Implement edit
                toast({ title: "Edit not implemented yet" });
              }}
              onDelete={(alertId) => deleteAlert.mutate({ alert_id: alertId })}
              onCreateNew={() => setShowAlertForm(true)}
            />
          )}
        </TabsContent>

        {/* Watchlists Tab */}
        <TabsContent value="watchlists">
          <WatchlistManager
            watchlists={watchlists}
            onCreateWatchlist={(name, description) =>
              createWatchlist.mutate({ name, description })
            }
            onDeleteWatchlist={(watchlistId) =>
              deleteWatchlist.mutate({ watchlist_id: watchlistId })
            }
            onAddSymbol={(watchlistId, symbol) =>
              addSymbol.mutate({ watchlist_id: watchlistId, symbol })
            }
            onRemoveSymbol={(watchlistId, symbol) =>
              removeSymbol.mutate({ watchlist_id: watchlistId, symbol })
            }
            onViewWatchlist={(watchlistId) => {
              // TODO: Implement view
              toast({ title: "View not implemented yet" });
            }}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
