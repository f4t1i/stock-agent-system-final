/**
 * Custom Hooks for Stock Analysis
 * 
 * React Query hooks for data fetching and state management
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, StockAnalysisRequest, StockAnalysisResponse, BatchAnalysisRequest } from '@/lib/api';
import { useAnalysisStore } from '@/stores/analysisStore';
import { toast } from 'sonner';

/**
 * Hook to analyze a single stock
 */
export function useAnalyzeStock() {
  const queryClient = useQueryClient();
  const { setCurrentAnalysis, addToHistory, setIsAnalyzing, setError } = useAnalysisStore();

  return useMutation({
    mutationFn: (request: StockAnalysisRequest) => api.analyzeStock(request),
    onMutate: () => {
      setIsAnalyzing(true);
      setError(null);
    },
    onSuccess: (data) => {
      setCurrentAnalysis(data);
      addToHistory(data);
      toast.success(`Analysis complete for ${data.symbol}`);
      
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
    },
    onError: (error: Error) => {
      setError(error.message);
      toast.error(`Analysis failed: ${error.message}`);
    },
    onSettled: () => {
      setIsAnalyzing(false);
    },
  });
}

/**
 * Hook to analyze multiple stocks in batch
 */
export function useBatchAnalysis() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: BatchAnalysisRequest) => api.analyzeBatch(request),
    onSuccess: () => {
      toast.success('Batch analysis complete');
      queryClient.invalidateQueries({ queryKey: ['analysis-history'] });
    },
    onError: (error: Error) => {
      toast.error(`Batch analysis failed: ${error.message}`);
    },
  });
}

/**
 * Hook to get analysis history
 */
export function useAnalysisHistory() {
  const { analysisHistory } = useAnalysisStore();

  return useQuery({
    queryKey: ['analysis-history'],
    queryFn: async () => {
      // Return from store for now, could fetch from API
      return analysisHistory;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook to get current analysis
 */
export function useCurrentAnalysis() {
  const { currentAnalysis } = useAnalysisStore();
  
  return {
    data: currentAnalysis,
    isLoading: false,
  };
}
