/**
 * Custom Hooks for Backtesting
 */
import { useMutation, useQuery } from '@tanstack/react-query';
import { api, BacktestRequest, BacktestResponse } from '@/lib/api';
import { toast } from 'sonner';

/**
 * Hook to run backtest
 */
export function useRunBacktest() {
  return useMutation({
    mutationFn: (request: BacktestRequest) => api.backtest(request),
    onSuccess: (data) => {
      toast.success(`Backtest complete for ${data.symbol}`);
    },
    onError: (error: Error) => {
      toast.error(`Backtest failed: ${error.message}`);
    },
  });
}

/**
 * Hook to get backtest history
 */
export function useBacktestHistory() {
  return useQuery({
    queryKey: ['backtest-history'],
    queryFn: async () => {
      // Mock data for now
      return [];
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
}
