/**
 * Custom Hooks for OpenBB Integration
 */
import { useQuery, useMutation } from '@tanstack/react-query';
import { openbbClient, OpenBBQueryRequest, OpenBBDataRequest } from '@/lib/openbb';
import { toast } from 'sonner';

/**
 * Hook for natural language queries to OpenBB
 * Example: "Should I buy AAPL?", "What's the outlook for TSLA?"
 */
export function useOpenBBQuery() {
  return useMutation({
    mutationFn: (request: OpenBBQueryRequest) => openbbClient.query(request),
    onSuccess: (data) => {
      toast.success('OpenBB query completed');
    },
    onError: (error: Error) => {
      toast.error(`OpenBB query failed: ${error.message}`);
    },
  });
}

/**
 * Hook to get financial data from OpenBB
 */
export function useOpenBBData(symbol: string, dataType: 'price' | 'fundamentals' | 'news' | 'technical' | 'all' = 'all') {
  return useQuery({
    queryKey: ['openbb-data', symbol, dataType],
    queryFn: () => openbbClient.getData({ symbol, data_type: dataType }),
    enabled: !!symbol,
    staleTime: 60 * 1000, // 1 minute
  });
}

/**
 * Hook to get real-time price
 */
export function useOpenBBPrice(symbol: string) {
  return useQuery({
    queryKey: ['openbb-price', symbol],
    queryFn: () => openbbClient.getPrice(symbol),
    enabled: !!symbol,
    refetchInterval: 5000, // Refetch every 5 seconds
    staleTime: 1000,
  });
}

/**
 * Hook to get company fundamentals
 */
export function useOpenBBFundamentals(symbol: string) {
  return useQuery({
    queryKey: ['openbb-fundamentals', symbol],
    queryFn: () => openbbClient.getFundamentals(symbol),
    enabled: !!symbol,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook to get latest news
 */
export function useOpenBBNews(symbol: string, limit: number = 10) {
  return useQuery({
    queryKey: ['openbb-news', symbol, limit],
    queryFn: () => openbbClient.getNews(symbol, limit),
    enabled: !!symbol,
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
}
