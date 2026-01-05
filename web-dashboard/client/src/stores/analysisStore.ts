/**
 * Zustand Store for Stock Analysis State Management
 */
import { create } from 'zustand';
import { StockAnalysisResponse } from '@/lib/api';

interface AnalysisState {
  // Current analysis
  currentAnalysis: StockAnalysisResponse | null;
  
  // Analysis history
  analysisHistory: StockAnalysisResponse[];
  
  // Loading states
  isAnalyzing: boolean;
  
  // Error state
  error: string | null;
  
  // Actions
  setCurrentAnalysis: (analysis: StockAnalysisResponse) => void;
  addToHistory: (analysis: StockAnalysisResponse) => void;
  setIsAnalyzing: (isAnalyzing: boolean) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
  clearHistory: () => void;
}

export const useAnalysisStore = create<AnalysisState>((set) => ({
  currentAnalysis: null,
  analysisHistory: [],
  isAnalyzing: false,
  error: null,

  setCurrentAnalysis: (analysis) => set({ currentAnalysis: analysis }),
  
  addToHistory: (analysis) => set((state) => ({
    analysisHistory: [analysis, ...state.analysisHistory].slice(0, 50), // Keep last 50
  })),
  
  setIsAnalyzing: (isAnalyzing) => set({ isAnalyzing }),
  
  setError: (error) => set({ error }),
  
  clearError: () => set({ error: null }),
  
  clearHistory: () => set({ analysisHistory: [] }),
}));
