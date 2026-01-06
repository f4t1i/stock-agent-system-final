/**
 * Technical Indicators Library
 *
 * Calculations for common technical analysis indicators:
 * - Moving Averages (SMA, EMA)
 * - RSI (Relative Strength Index)
 * - MACD (Moving Average Convergence Divergence)
 * - Bollinger Bands
 * - ATR (Average True Range)
 * - Stochastic Oscillator
 */

export interface PriceData {
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

/**
 * Simple Moving Average (SMA)
 */
export function calculateSMA(data: number[], period: number): number[] {
  const sma: number[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      sma.push(NaN);
      continue;
    }

    const sum = data.slice(i - period + 1, i + 1).reduce((acc, val) => acc + val, 0);
    sma.push(sum / period);
  }

  return sma;
}

/**
 * Exponential Moving Average (EMA)
 */
export function calculateEMA(data: number[], period: number): number[] {
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);

  // Start with SMA for first value
  let prevEMA = data.slice(0, period).reduce((acc, val) => acc + val, 0) / period;

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      ema.push(NaN);
      continue;
    }

    if (i === period - 1) {
      ema.push(prevEMA);
    } else {
      const currentEMA = (data[i] - prevEMA) * multiplier + prevEMA;
      ema.push(currentEMA);
      prevEMA = currentEMA;
    }
  }

  return ema;
}

/**
 * Relative Strength Index (RSI)
 */
export function calculateRSI(data: number[], period: number = 14): number[] {
  const rsi: number[] = [];

  if (data.length < period + 1) {
    return data.map(() => NaN);
  }

  // Calculate price changes
  const changes: number[] = [];
  for (let i = 1; i < data.length; i++) {
    changes.push(data[i] - data[i - 1]);
  }

  // Separate gains and losses
  const gains = changes.map((change) => (change > 0 ? change : 0));
  const losses = changes.map((change) => (change < 0 ? -change : 0));

  // Calculate average gain and loss for first RSI value
  let avgGain =
    gains.slice(0, period).reduce((acc, val) => acc + val, 0) / period;
  let avgLoss =
    losses.slice(0, period).reduce((acc, val) => acc + val, 0) / period;

  rsi.push(NaN); // First data point has no RSI

  for (let i = 0; i < changes.length; i++) {
    if (i < period - 1) {
      rsi.push(NaN);
      continue;
    }

    if (i === period - 1) {
      const rs = avgGain / (avgLoss || 0.0001);
      rsi.push(100 - 100 / (1 + rs));
    } else {
      // Smoothed averages
      avgGain = (avgGain * (period - 1) + gains[i]) / period;
      avgLoss = (avgLoss * (period - 1) + losses[i]) / period;

      const rs = avgGain / (avgLoss || 0.0001);
      rsi.push(100 - 100 / (1 + rs));
    }
  }

  return rsi;
}

/**
 * MACD (Moving Average Convergence Divergence)
 */
export interface MACDResult {
  macd: number[];
  signal: number[];
  histogram: number[];
}

export function calculateMACD(
  data: number[],
  fastPeriod: number = 12,
  slowPeriod: number = 26,
  signalPeriod: number = 9
): MACDResult {
  const emaFast = calculateEMA(data, fastPeriod);
  const emaSlow = calculateEMA(data, slowPeriod);

  // MACD line
  const macd = emaFast.map((fast, i) => {
    if (isNaN(fast) || isNaN(emaSlow[i])) return NaN;
    return fast - emaSlow[i];
  });

  // Signal line (EMA of MACD)
  const validMACD = macd.filter((val) => !isNaN(val));
  const signalEMA = calculateEMA(validMACD, signalPeriod);

  // Align signal with macd array
  const signal: number[] = [];
  let signalIndex = 0;
  for (let i = 0; i < macd.length; i++) {
    if (isNaN(macd[i])) {
      signal.push(NaN);
    } else {
      signal.push(signalEMA[signalIndex] || NaN);
      signalIndex++;
    }
  }

  // Histogram
  const histogram = macd.map((macdVal, i) => {
    if (isNaN(macdVal) || isNaN(signal[i])) return NaN;
    return macdVal - signal[i];
  });

  return { macd, signal, histogram };
}

/**
 * Bollinger Bands
 */
export interface BollingerBandsResult {
  upper: number[];
  middle: number[];
  lower: number[];
}

export function calculateBollingerBands(
  data: number[],
  period: number = 20,
  stdDev: number = 2
): BollingerBandsResult {
  const middle = calculateSMA(data, period);
  const upper: number[] = [];
  const lower: number[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      upper.push(NaN);
      lower.push(NaN);
      continue;
    }

    const slice = data.slice(i - period + 1, i + 1);
    const mean = middle[i];
    const variance =
      slice.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / period;
    const standardDeviation = Math.sqrt(variance);

    upper.push(mean + stdDev * standardDeviation);
    lower.push(mean - stdDev * standardDeviation);
  }

  return { upper, middle, lower };
}

/**
 * Average True Range (ATR)
 */
export function calculateATR(data: PriceData[], period: number = 14): number[] {
  const atr: number[] = [];

  if (data.length < 2) {
    return data.map(() => NaN);
  }

  // Calculate True Range for each period
  const trueRanges: number[] = [NaN]; // First candle has no TR

  for (let i = 1; i < data.length; i++) {
    const high = data[i].high;
    const low = data[i].low;
    const prevClose = data[i - 1].close;

    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );

    trueRanges.push(tr);
  }

  // Calculate ATR using EMA of TR
  const atrValues = calculateEMA(trueRanges, period);

  return atrValues;
}

/**
 * Stochastic Oscillator
 */
export interface StochasticResult {
  k: number[];
  d: number[];
}

export function calculateStochastic(
  data: PriceData[],
  kPeriod: number = 14,
  dPeriod: number = 3
): StochasticResult {
  const k: number[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < kPeriod - 1) {
      k.push(NaN);
      continue;
    }

    const slice = data.slice(i - kPeriod + 1, i + 1);
    const highest = Math.max(...slice.map((d) => d.high));
    const lowest = Math.min(...slice.map((d) => d.low));
    const current = data[i].close;

    const kValue =
      highest !== lowest ? ((current - lowest) / (highest - lowest)) * 100 : 50;

    k.push(kValue);
  }

  // %D is SMA of %K
  const validK = k.filter((val) => !isNaN(val));
  const dSMA = calculateSMA(validK, dPeriod);

  // Align d with k array
  const d: number[] = [];
  let dIndex = 0;
  for (let i = 0; i < k.length; i++) {
    if (isNaN(k[i])) {
      d.push(NaN);
    } else {
      d.push(dSMA[dIndex] || NaN);
      dIndex++;
    }
  }

  return { k, d };
}

/**
 * On-Balance Volume (OBV)
 */
export function calculateOBV(data: PriceData[]): number[] {
  const obv: number[] = [0];

  for (let i = 1; i < data.length; i++) {
    const prevOBV = obv[i - 1];
    const volume = data[i].volume || 0;

    if (data[i].close > data[i - 1].close) {
      obv.push(prevOBV + volume);
    } else if (data[i].close < data[i - 1].close) {
      obv.push(prevOBV - volume);
    } else {
      obv.push(prevOBV);
    }
  }

  return obv;
}

/**
 * Volume Weighted Average Price (VWAP)
 */
export function calculateVWAP(data: PriceData[]): number[] {
  const vwap: number[] = [];
  let cumulativeTPV = 0; // Typical Price * Volume
  let cumulativeVolume = 0;

  for (let i = 0; i < data.length; i++) {
    const typicalPrice = (data[i].high + data[i].low + data[i].close) / 3;
    const volume = data[i].volume || 0;

    cumulativeTPV += typicalPrice * volume;
    cumulativeVolume += volume;

    vwap.push(cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : NaN);
  }

  return vwap;
}

/**
 * Fibonacci Retracement Levels
 */
export interface FibonacciLevels {
  high: number;
  low: number;
  levels: {
    '0': number;
    '23.6': number;
    '38.2': number;
    '50': number;
    '61.8': number;
    '100': number;
  };
}

export function calculateFibonacci(
  data: number[],
  startIndex: number = 0,
  endIndex?: number
): FibonacciLevels {
  const end = endIndex ?? data.length - 1;
  const slice = data.slice(startIndex, end + 1);

  const high = Math.max(...slice);
  const low = Math.min(...slice);
  const diff = high - low;

  return {
    high,
    low,
    levels: {
      '0': high,
      '23.6': high - diff * 0.236,
      '38.2': high - diff * 0.382,
      '50': high - diff * 0.5,
      '61.8': high - diff * 0.618,
      '100': low,
    },
  };
}

/**
 * Parabolic SAR (Stop and Reverse)
 */
export function calculateParabolicSAR(
  data: PriceData[],
  acceleration: number = 0.02,
  maximum: number = 0.2
): number[] {
  if (data.length < 2) {
    return data.map(() => NaN);
  }

  const sar: number[] = [data[0].low]; // Start with low of first candle
  let isUptrend = true;
  let ep = data[0].high; // Extreme point
  let af = acceleration;

  for (let i = 1; i < data.length; i++) {
    const prevSAR = sar[i - 1];
    let newSAR = prevSAR + af * (ep - prevSAR);

    if (isUptrend) {
      // Uptrend
      if (data[i].low < newSAR) {
        // Trend reversal
        isUptrend = false;
        newSAR = ep;
        ep = data[i].low;
        af = acceleration;
      } else {
        if (data[i].high > ep) {
          ep = data[i].high;
          af = Math.min(af + acceleration, maximum);
        }
      }
    } else {
      // Downtrend
      if (data[i].high > newSAR) {
        // Trend reversal
        isUptrend = true;
        newSAR = ep;
        ep = data[i].high;
        af = acceleration;
      } else {
        if (data[i].low < ep) {
          ep = data[i].low;
          af = Math.min(af + acceleration, maximum);
        }
      }
    }

    sar.push(newSAR);
  }

  return sar;
}
