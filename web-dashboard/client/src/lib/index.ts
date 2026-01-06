/**
 * Library Utilities - Barrel Export
 */

// Technical Indicators
export {
  calculateSMA,
  calculateEMA,
  calculateRSI,
  calculateMACD,
  calculateBollingerBands,
  calculateATR,
  calculateStochastic,
  calculateOBV,
  calculateVWAP,
  calculateParabolicSAR,
  calculateFibonacci,
} from './technicalIndicators';

export type {
  PriceData,
  MACDResult,
  BollingerBandsResult,
  StochasticResult,
  FibonacciLevels,
} from './technicalIndicators';

// Data Transformers
export {
  toCandlestickData,
  aggregateByInterval,
  calculateReturns,
  normalizeToPercentage,
  fillMissingPeriods,
  calculateMovingStats,
  groupByCategory,
  calculateCorrelation,
  calculateCorrelationMatrix,
  resampleData,
  filterByDateRange,
  calculatePercentageBreakdown,
  calculateDrawdown,
  rollingWindow,
} from './dataTransformers';

export type {
  RawPriceData,
  TimeInterval,
  ReturnData,
  MovingStats,
  PercentageBreakdown,
  DrawdownData,
} from './dataTransformers';

// WebSocket
export { wsClient, WebSocketClient, Channel, MessageType } from './websocket';
export type {
  ConnectionState,
  PriceUpdateMessage,
  AlertTriggeredMessage,
  WebSocketMessage,
} from './websocket';
