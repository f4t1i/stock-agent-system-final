/**
 * Custom Hook for WebSocket Integration
 */
import { useEffect, useState, useCallback } from 'react';
import { wsClient, WebSocketEvent, ConnectionState } from '@/lib/websocket';
import { useAnalysisStore } from '@/stores/analysisStore';
import { toast } from 'sonner';

export function useWebSocket() {
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const { setCurrentAnalysis, addToHistory } = useAnalysisStore();

  const isConnected = connectionState === 'connected';
  const canReconnect = connectionState === 'max_retries_reached';

  const handleWebSocketEvent = useCallback((event: WebSocketEvent) => {
    switch (event.type) {
      case 'analysis_complete':
        setCurrentAnalysis(event.data);
        addToHistory(event.data);
        toast.success(`Real-time analysis update: ${event.data.symbol}`);
        break;

      case 'training_update':
        toast.info('Training metrics updated');
        break;

      case 'agent_status':
        console.log('Agent status update:', event.data);
        break;

      case 'system_health':
        if (event.data.status === 'connected') {
          toast.success('Connected to real-time updates');
        } else if (event.data.status === 'disconnected') {
          // Don't show toast for normal disconnects
        } else if (event.data.status === 'error') {
          toast.error(event.data.message, {
            duration: 10000,
            action: connectionState === 'max_retries_reached' ? {
              label: 'Reconnect',
              onClick: () => wsClient.reconnect(),
            } : undefined,
          });
        }
        break;
    }
  }, [setCurrentAnalysis, addToHistory]);

  useEffect(() => {
    // Connect to WebSocket
    wsClient.connect();

    // Subscribe to connection state changes
    const unsubscribeState = wsClient.onStateChange((state) => {
      setConnectionState(state);
    });

    // Subscribe to events
    const unsubscribe = wsClient.subscribe(handleWebSocketEvent);

    // Cleanup on unmount
    return () => {
      unsubscribe();
      unsubscribeState();
    };
  }, [handleWebSocketEvent]);

  const reconnect = useCallback(() => {
    wsClient.reconnect();
  }, []);

  return {
    isConnected,
    connectionState,
    canReconnect,
    reconnect,
    emit: wsClient.emit.bind(wsClient),
  };
}
