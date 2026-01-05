/**
 * Custom Hooks for Training
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, TrainingMetrics } from '@/lib/api';
import { toast } from 'sonner';

/**
 * Hook to get training metrics
 */
export function useTrainingMetrics() {
  return useQuery({
    queryKey: ['training-metrics'],
    queryFn: () => api.getTrainingMetrics(),
    refetchInterval: 5000, // Refetch every 5 seconds when training
    staleTime: 1000,
  });
}

/**
 * Hook to start training
 */
export function useStartTraining() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => api.startTraining(),
    onSuccess: () => {
      toast.success('Training started');
      queryClient.invalidateQueries({ queryKey: ['training-metrics'] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to start training: ${error.message}`);
    },
  });
}

/**
 * Hook to stop training
 */
export function useStopTraining() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => api.stopTraining(),
    onSuccess: () => {
      toast.success('Training stopped');
      queryClient.invalidateQueries({ queryKey: ['training-metrics'] });
    },
    onError: (error: Error) => {
      toast.error(`Failed to stop training: ${error.message}`);
    },
  });
}
